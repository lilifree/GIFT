"""OPSD + GRPO + UFT: token-level progressive spatio-temporal curriculum loss.

Additive baseline module (the 'OPSD + GRPO + UFT' baseline family discussed in the
GIFT paper, Sec. 2 / App. G): a token-level progressive curriculum on top of the
RLCSD base, WITHOUT modifying any original repo file. The GIFT method itself lives in
``scripts/gift/main/gift_runtime/patch.py``.

Idea (PDF, sec. 3): start training with OPSD (on-policy self-distillation) dense
loss on ALL tokens + GRPO outcome loss; as training progresses, progressively
REMOVE OPSD from the *tail* (suffix) tokens, growing the pure-RL free-exploration
region from back to front, until only GRPO remains. This is a token-level
spatio-temporal curriculum (UFT-style annealing, but on-policy / OPSD-based).

Implementation strategy (additive only):
  * RLCSD's ``loss_mode == "rlcsd"`` already routes the teacher privileged
    context construction (src.self_distill_main) AND the actor teacher-forward +
    loss_kwargs dispatch (third_party/verl/.../dp_actor.py) AND the registered
    policy loss (core_algos.compute_policy_loss_rlcsd). We keep
    ``loss_mode == "rlcsd"`` and **override only the registry entry** at runtime
    with a curriculum variant that is a strict superset: when ``uft_enable`` is
    falsy it delegates to the original ``compute_policy_loss_rlcsd`` unchanged
    (byte-identical behavior), so existing configs are unaffected.
  * The override is installed lazily: a meta-path finder (installed by the
    site-packages .pth at interpreter startup, cheaply, with no torch import)
    intercepts the first import of ``verl.trainer.ppo.core_algos`` and, right
    after the module body runs (which registers the original rlcsd loss), swaps
    ``POLICY_LOSS_REGISTRY["rlcsd"]`` for the curriculum function below. This
    runs inside every process that actually imports core_algos — i.e. the verl
    actor workers where the loss is computed — without editing any vendored file.

The curriculum variant reuses RLCSD's contrastive evidence ratio e_ctr and
sign-preserving modulated advantage (``mod_adv``) for the OPSD path, then splits
each response into two regions along the token axis using a schedule-driven
boundary K(t):

  * prefix region  (normalized pos p <= boundary(t))  -> GRPO + OPSD (mod_adv)
  * suffix region  (p > boundary(t), the "free exploration" tail) -> pure GRPO
    (the plain ``advantages`` surrogate, NOT the OPSD-modulated one)

``boundary(t)`` anneals 1 -> uft_min_prefix_frac over ``uft_mask_anneal_steps``
via cosine (default) or linear schedule, with an optional warmup at full OPSD and
an optional soft (smoothstep) transition window instead of a hard step. An
optional positive-advantage gate (PDF sec. 6.2 / SDPG) disables OPSD on
trajectories whose outcome advantage <= 0. The per-token OPSD magnitude is
already bounded by RLCSD's residual clip (PDF sec. 6.3 pointwise-KL-clipping
defense). RLCSD's contrastive hints (PDF sec. 6.4) are inherited as-is.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import dataclasses
import math
import sys
import warnings

_TARGET_MODULE = "verl.trainer.ppo.core_algos"

# verl.workers.config.actor.PolicyLossConfig is a strict @dataclass: Hydra rejects
# `+policy_loss.uft_*` overrides with "unexpected keyword argument". We wrap its
# __init__ (in an import hook, no vendored file edited) so unknown kwargs are
# accepted and stored on the instance — BaseConfig.get/hgetattr then exposes them
# to _sd_cfg/_cfg at loss time.
_CONFIG_MODULE = "verl.workers.config.actor"
_CONFIG_CLASS = "PolicyLossConfig"

# dp_actor.DataParallelPPOActor.update_policy reads global_steps from
# data.meta_info, but on the RLCSD path the actor-update batch's meta_info does
# NOT carry global_steps (it's only set on gen_batch.meta_info at ray_trainer:1584
# and does not flow back into the actor batch). So global_steps arrives as None
# inside the loss, which would freeze the UFT anneal schedule. We wrap
# update_policy to inject a per-worker monotonic step counter into meta_info
# (update_policy is called exactly once per training step), so the curriculum
# anneals correctly across steps.
_DP_ACTOR_MODULE = "verl.workers.actor.dp_actor"
_DP_ACTOR_CLASS = "DataParallelPPOActor"


# --------------------------------------------------------------------------- #
# Lazy install: wrap core_algos's loader so the override is applied the moment
# core_algos is imported (which happens in actor workers, never at startup).
# --------------------------------------------------------------------------- #
class _AfterExecLoader(importlib.abc.Loader):
    """Delegates to the real loader, then runs a callback after exec_module."""

    def __init__(self, real_loader, after_exec):
        self._real = real_loader
        self._after_exec = after_exec

    def create_module(self, spec):
        if hasattr(self._real, "create_module"):
            return self._real.create_module(spec)
        return None

    def exec_module(self, module):
        self._real.exec_module(module)
        try:
            self._after_exec(module)
        except Exception as exc:  # never break the real import
            warnings.warn(f"[opsd_grpo_uft] override install failed: {exc!r}")

    def get_filename(self, name):
        return self._real.get_filename(name)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _UftMetaFinder(importlib.abc.MetaPathFinder):
    """Intercept the first import of core_algos and wrap its loader."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == _CONFIG_MODULE:
            return self._wrap_spec(fullname, path, target, _patch_policy_loss_config)
        if fullname == _TARGET_MODULE:
            return self._wrap_spec(fullname, path, target, _install_override)
        if fullname == _DP_ACTOR_MODULE:
            return self._wrap_spec(fullname, path, target, _patch_dp_actor_global_steps)
        return None

    @staticmethod
    def _wrap_spec(fullname, path, target, after_exec):
        for finder in sys.meta_path:
            if finder is _UftMetaFinder._sentinel:
                continue
            if isinstance(finder, _UftMetaFinder):
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except (AttributeError, ImportError):
                spec = None
            if spec is not None and spec.loader is not None:
                spec.loader = _AfterExecLoader(spec.loader, after_exec)
                return spec
        return None

    _sentinel = None  # placeholder, never an instance


def _patch_policy_loss_config(actor_cfg_module):
    """Wrap PolicyLossConfig.__init__ to accept (and store) the uft_* kwargs.

    verl's PolicyLossConfig is a strict @dataclass, so Hydra instantiation of
    `+actor_rollout_ref.actor.policy_loss.uft_enable=...` raises
    "unexpected keyword argument". We wrap __init__ to forward known fields to the
    dataclass-generated init and stash the rest on the instance (via
    object.__setattr__, bypassing BaseConfig's frozen-field check for new attrs).
    """
    cls = getattr(actor_cfg_module, _CONFIG_CLASS, None)
    if cls is None:
        return
    if getattr(cls, "_uft_init_wrapped", False):
        return
    try:
        orig_init = cls.__init__
    except AttributeError:
        return
    try:
        fields = dataclasses.fields(cls)
    except TypeError:
        fields = []
    known = {f.name for f in fields}

    def _wrapped_init(self, **kwargs):
        base, extra = {}, {}
        for k, v in kwargs.items():
            (base if k in known else extra)[k] = v
        orig_init(self, **base)
        for k, v in extra.items():
            object.__setattr__(self, k, v)

    cls.__init__ = _wrapped_init
    cls._uft_init_wrapped = True


def _patch_dp_actor_global_steps(dp_actor_module):
    """Inject a per-worker step counter into update_policy's meta_info.

    The RLCSD actor-update batch's meta_info lacks `global_steps`, so the loss
    schedule would see None. update_policy is called once per training step, so a
    per-instance counter on the actor object matches the true global step.
    """
    cls = getattr(dp_actor_module, _DP_ACTOR_CLASS, None)
    if cls is None:
        return
    if getattr(cls, "_uft_global_steps_wrapped", False):
        return
    orig_update = cls.update_policy

    def _update_policy_wrapped(self, data, *args, **kwargs):
        try:
            mi = getattr(data, "meta_info", None)
            if mi is not None and mi.get("global_steps") is None:
                cur = int(getattr(self, "_uft_local_global_steps", 0)) + 1
                self._uft_local_global_steps = cur
                mi["global_steps"] = cur
        except Exception:
            pass
        return orig_update(self, data, *args, **kwargs)

    cls.update_policy = _update_policy_wrapped
    cls._uft_global_steps_wrapped = True


def _install_override(core_algos):
    """Swap POLICY_LOSS_REGISTRY['rlcsd'] for the curriculum variant."""
    registry = getattr(core_algos, "POLICY_LOSS_REGISTRY", None)
    if registry is None or "rlcsd" not in registry:
        return
    if getattr(core_algos, "_uft_override_installed", False):
        return
    original = registry["rlcsd"]
    registry["rlcsd"] = lambda **kwargs: _compute_policy_loss_rlcsd_uft(
        original=original, core_algos=core_algos, **kwargs)
    core_algos._uft_override_installed = True


# Install the finder at first import of THIS module (the .pth imports this module
# cheaply at interpreter startup; no torch/verl import happens here).
if not any(isinstance(f, _UftMetaFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _UftMetaFinder())


# --------------------------------------------------------------------------- #
# Config helpers (mirror core_algos._sd_cfg so we can read policy_loss keys).
# --------------------------------------------------------------------------- #
def _cfg(config, key, default):
    if config is None:
        return default
    pl = getattr(config, "policy_loss", None)
    if pl is None:
        return default
    try:
        if hasattr(pl, "get"):
            value = pl.get(key, default)
        else:
            value = getattr(pl, key, default)
    except Exception:
        return default
    return default if value is None else value


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


# --------------------------------------------------------------------------- #
# The curriculum schedule -> per-token OPSD position mask.
# --------------------------------------------------------------------------- #
def _uft_progress(global_steps, warmup_steps, anneal_steps):
    """Training progress in [0,1] for the OPSD-prefix boundary anneal."""
    if global_steps is None:
        step_index = 0
    else:
        step_index = max(int(global_steps) - 1, 0)  # global_steps is 1-based
    if anneal_steps is None or anneal_steps <= 0:
        return 0.0
    if step_index <= warmup_steps:
        return 0.0
    return min((step_index - warmup_steps) / float(anneal_steps), 1.0)


def _boundary(progress, min_prefix_frac, schedule):
    """OPSD-prefix boundary in [min_prefix_frac, 1]; 1 = OPSD everywhere."""
    span = max(1.0 - min_prefix_frac, 0.0)
    if schedule == "linear":
        return min_prefix_frac + span * (1.0 - progress)
    # default cosine: 1 -> min_prefix_frac over progress 0 -> 1
    return min_prefix_frac + span * 0.5 * (1.0 + math.cos(math.pi * progress))


def _positional_oposd_mask(response_mask, global_steps, config, core_algos):
    """Return per-token mask in (bs, L): 1 where OPSD applies, 0 in free-tail.

    Built from each sequence's own response length so it is meaningful even when
    responses are right-padded to a fixed L. Tokens outside response_mask are 0.
    """
    _positional_oposd_mask._diag_logged = getattr(
        _positional_oposd_mask, "_diag_logged", False)
    torch = sys.modules.get("torch")
    if torch is None:  # core_algos imports torch, so it is always available here
        raise RuntimeError("torch not imported before UFT mask computation")

    device = response_mask.device
    bs, L = response_mask.shape
    positions = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)  # (1,L)
    seq_len = response_mask.to(torch.float32).sum(dim=-1, keepdim=True).clamp(min=1.0)  # (bs,1)
    # normalized position per sequence: head=0, tail=1 (padding -> p>1, masked out later)
    p = positions / seq_len  # (bs,L)

    warmup = _as_int(_cfg(config, "uft_warmup_steps", 0), 0)
    anneal = _as_int(_cfg(config, "uft_mask_anneal_steps", 0), 0)
    progress = _uft_progress(global_steps, warmup, anneal)
    if not _positional_oposd_mask._diag_logged:
        _positional_oposd_mask._diag_logged = True
        try:
            import os as _os
            print(f"[opsd_grpo_uft] DIAG first-call: global_steps={global_steps!r} "
                  f"warmup={warmup!r} anneal={anneal!r} progress={progress!r} "
                  f"cfg_type={type(config).__name__} has_uft_anneal="
                  f"{_cfg(config, 'uft_mask_anneal_steps', '__MISSING__')!r} "
                  f"uft_enable={_cfg(config, 'uft_enable', '__MISSING__')!r} "
                  f"pid={_os.getpid()}", flush=True)
        except Exception:
            pass
    min_prefix = max(min(_as_float(_cfg(config, "uft_min_prefix_frac", 0.0), 0.0), 1.0), 0.0)
    schedule = str(_cfg(config, "uft_anneal_schedule", "cosine")).lower()
    boundary = _boundary(progress, min_prefix, schedule)

    mode = str(_cfg(config, "uft_mask_mode", "soft")).lower()
    if mode == "hard":
        posmask = (p <= boundary).to(torch.float32)
    else:
        # soft smoothstep transition over a window of width `soft_window` (frac of
        # normalized length), centered on `boundary`:
        #   p <= boundary - w/2  -> 1  (full OPSD)
        #   p >= boundary + w/2  -> 0  (free exploration)
        w = max(min(_as_float(_cfg(config, "uft_soft_window", 0.1), 0.1), 1.0), 0.0)
        half = w / 2.0
        lo = boundary - half
        hi = boundary + half
        t = (p - lo) / (hi - lo + 1e-8)
        t = t.clamp(0.0, 1.0)
        posmask = t * t * (3.0 - 2.0 * t)  # smoothstep

    posmask = posmask * response_mask.to(torch.float32)

    # Optional positive-advantage gating (PDF sec.6.2): drop OPSD on trajectories
    # whose outcome advantage <= 0, letting pure GRPO handle them.
    if _as_bool(_cfg(config, "uft_pos_adv_gate", False), False):
        advantages = None  # set by caller path; handled in loss fn instead
        # gating uses per-trajectory scalar advantage, applied in the loss fn
        pass

    diag = {
        "uft_progress": float(progress),
        "uft_boundary": float(boundary),
        "uft_posmask_mean": float(posmask.mean().item()),
    }
    return posmask, diag


# --------------------------------------------------------------------------- #
# Curriculum loss (registered under "rlcsd", delegates to original when disabled).
# --------------------------------------------------------------------------- #
def _compute_policy_loss_rlcsd_uft(original, core_algos, **kwargs):
    config = kwargs.get("config", None)
    if not _as_bool(_cfg(config, "uft_enable", False), False):
        return original(**kwargs)

    torch = sys.modules.get("torch")
    verl_F = getattr(core_algos, "verl_F", None)
    _sd_token_mean = getattr(core_algos, "_sd_token_mean")
    _sd_cfg = getattr(core_algos, "_sd_cfg")
    _sd_distribution_metrics = getattr(core_algos, "_sd_distribution_metrics")
    _sd_teacher_metrics = getattr(core_algos, "_sd_teacher_metrics")

    log_prob = kwargs["log_prob"]
    old_log_prob = kwargs["old_log_prob"]
    advantages = kwargs["advantages"]
    response_mask = kwargs["response_mask"]

    teacher_correct_lp = kwargs.get("teacher_log_probs")
    teacher_wrong_multi_lp = kwargs.get("teacher_wrong_multi_log_probs")
    teacher_wrong_multi_valid_mask = kwargs.get("teacher_wrong_multi_valid_mask")
    assert teacher_correct_lp is not None, "rlcsd requires teacher_log_probs."
    assert (teacher_wrong_multi_lp is not None
            and teacher_wrong_multi_valid_mask is not None), (
        "rlcsd requires teacher_wrong_multi_log_probs and teacher_wrong_multi_valid_mask.")

    rollout_is_weights = kwargs.get("rollout_is_weights", None)
    global_steps = kwargs.get("global_steps", None)

    epsilon = float(_sd_cfg(config, "epsilon", 0.2))
    tau = max(float(_sd_cfg(config, "rlcsd_tau", _sd_cfg(config, "tau", 0.03))), 1e-6)
    beta = float(_sd_cfg(config, "rlcsd_beta", 0.2))
    lam = float(_sd_cfg(config, "rlcsd_lam", _sd_cfg(config, "lam", 0.5)))
    delta = max(float(_sd_cfg(config, "rlcsd_delta", 0.02)), 0.0)
    eta = max(float(_sd_cfg(config, "rlcsd_eta", 1.0)), 0.0)
    residual_clip_low = float(_sd_cfg(config, "rlcsd_residual_clip_low", -0.5))
    residual_clip_high = float(_sd_cfg(config, "rlcsd_residual_clip_high", 0.5))
    if residual_clip_low > residual_clip_high:
        residual_clip_low, residual_clip_high = residual_clip_high, residual_clip_low

    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

    with torch.no_grad():
        valid_mask_bool = teacher_wrong_multi_valid_mask.to(dtype=torch.bool)
        masked_w_lp = teacher_wrong_multi_lp.masked_fill(
            ~valid_mask_bool.unsqueeze(-1), float("-inf"))
        K_valid = valid_mask_bool.to(dtype=teacher_wrong_multi_lp.dtype).sum(dim=1).clamp(min=1.0)
        log_marginal_w = torch.logsumexp(masked_w_lp, dim=1) - torch.log(K_valid).unsqueeze(-1)

        e_ctr = teacher_correct_lp - log_marginal_w
        s_t = torch.tanh(e_ctr / tau)
        preclip_residual = beta * lam * s_t
        residual = preclip_residual.clamp(min=residual_clip_low, max=residual_clip_high)
        selected = (residual.abs() > delta) & response_mask.bool()
        selected_float = selected.float()
        selected_count = selected_float.sum()
        residual_for_adv = selected_float * residual
        response_token_count = response_mask.sum()
        raw_mod_adv = advantages + residual_for_adv
        mod_adv = torch.where(
            advantages > 0,
            raw_mod_adv.clamp_min(0.0),
            torch.where(advantages < 0, raw_mod_adv.clamp_max(0.0),
                        torch.zeros_like(raw_mod_adv)))
        K_valid_mean = K_valid.mean()

    # ---------- UFT curriculum: position-based OPSD mask on the token axis ----
    posmask, uft_diag = _positional_oposd_mask(
        response_mask, global_steps, config, core_algos)

    # OPSD applies only on (selected AND prefix); the OPSD-modulated advantage is
    # used there. Everywhere else — the free-exploration tail (selected but masked
    # out) plus the originally non-selected tokens — uses the pure-GRPO surrogate
    # on the plain outcome advantage. As posmask -> 0 the whole sequence collapses
    # to pure GRPO; at posmask ~ 1 this reduces to vanilla RLCSD.
    opsd_active = selected_float * posmask                      # (bs,L) OPSD region
    pos_adv_gate = _as_bool(_cfg(config, "uft_pos_adv_gate", False), False)
    if pos_adv_gate:
        seq_len = response_mask.to(torch.float32).sum(dim=-1, keepdim=True).clamp(min=1.0)
        adv_scalar = (advantages.to(torch.float32) * response_mask.to(torch.float32)).sum(
            dim=-1, keepdim=True) / seq_len                      # (bs,1)
        opsd_active = opsd_active * (adv_scalar > 0).to(torch.float32)

    grpo_mask = response_mask.to(torch.float32) * (1.0 - opsd_active)
    opsd_mask = response_mask.to(torch.float32) * opsd_active

    ratio_clamped = ratio.clamp(1 - epsilon, 1 + epsilon)
    grpo_per_token_loss = -torch.min(ratio * advantages, ratio_clamped * advantages)
    opsd_per_token_loss = -torch.min(ratio * mod_adv, ratio_clamped * mod_adv)
    if rollout_is_weights is not None:
        grpo_per_token_loss = grpo_per_token_loss * rollout_is_weights
        opsd_per_token_loss = opsd_per_token_loss * rollout_is_weights

    grpo_loss = _sd_token_mean(grpo_per_token_loss, grpo_mask)
    opsd_loss = _sd_token_mean(opsd_per_token_loss, opsd_mask)
    loss = grpo_loss + eta * opsd_loss

    opsd_token_count = opsd_mask.sum()
    metrics = {
        "pg_loss": loss.item(),
        "pg_clipfrac": verl_F.masked_mean(
            ((ratio - 1).abs() > epsilon).float(), response_mask).item(),
        "ppo_kl": ppo_kl.item(),
        "rlcsd_selected_token_count": selected_count.item(),
        "rlcsd_response_token_count": response_token_count.item(),
        "rlcsd_K_valid_mean": K_valid_mean.item(),
        "uft_opsd_token_count": float(opsd_token_count.item()),
        "uft_grpo_token_count": float(grpo_mask.sum().item()),
        "uft_opsd_frac": float((opsd_token_count / response_token_count.clamp(min=1.0)).item()),
    }
    metrics.update(uft_diag)
    metrics.update(_sd_distribution_metrics(e_ctr, response_mask, "rlcsd_e_ctr",
                                            include_percentiles=False))
    metrics.update(_sd_distribution_metrics(residual, opsd_mask, "rlcsd_selected_residual"))
    metrics.update(_sd_teacher_metrics(teacher_correct_lp, log_prob, response_mask,
                                       kwargs.get("teacher_entropy")))
    return loss, metrics