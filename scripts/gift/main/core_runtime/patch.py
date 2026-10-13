"""Shared runtime of GIFT: token-selection plumbing and the masked objective.

This package holds the parts of the method that are independent of the scoring
rule, so that ``gift_runtime`` only has to override the score, the selection
scope and the loss bookkeeping:

* geometry of the sparse teacher-top-k-plus-tail forward KL (Eq. 1-2);
* the coverage schedule ``kappa_t`` (Eq. 10 / App. H.3 Eq. 30);
* the objective ``L^GIFT = L^GRPO + lambda/N_valid * sum m_i(t) w_i l_i^sKL``
  (Eq. 11) with the all-valid-token normaliser;
* the no-grad scoring pass plus FIFO mask queue shared with the training pass.

The vendored verl sources are never edited.  The module is loaded through
``sitecustomize`` in the driver and in every Ray worker and patches only the
in-memory Python objects used by the run.

Default scoring rule
--------------------
On its own this runtime scores eligible tokens by their orthogonal residual
``||d_perp||`` (``direction_residual``) behind a positive-cosine gate, or by
the Gram area ``||r|| * ||d_perp||`` (``magnitude_residual``).  ``gift_runtime``
replaces that rule with the piecewise score of Eq. 8.

Default schedule
----------------
* steps 1..H: the budget is ``initial_retention`` of the valid tokens;
* steps H+1..T_exit-1: the budget decreases linearly toward zero;
* steps T_exit..end: K=0, so the objective is pure GRPO;
* the OPSD coefficient ``lambda`` is always 1.0 and is never annealed.

Exact ranking across dynamic micro-batches requires seeing all scores before
any selected-token loss is evaluated.  The runtime therefore performs one
no-gradient teacher/student scoring pass per PPO mini-batch, joins rows by
``uid``, builds the masks, and then lets the ordinary differentiable pass apply
GRPO plus the masked forward KL.  No extra backward pass is introduced.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any

import yaml


_ENABLE_ENV = "GIFT_CORE_ENABLE"
_CONFIG_ENV = "GIFT_CORE_CONFIG"
_PATCHED = False
_SETTINGS: "CoreSettings | None" = None
_STATE = threading.local()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be boolean, got {raw!r}")


@dataclass(frozen=True)
class CoreSettings:
    group_size: int
    total_training_steps: int
    hold_steps: int
    exit_step: int
    initial_retention: float
    score_mode: str
    cosine_min: float
    norm_eps: float
    base_lambda: float
    top_k_distill: int

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CoreSettings":
        obj = cls(
            group_size=int(raw.get("group_size", -1)),
            total_training_steps=int(raw.get("gift_total_training_steps", -1)),
            hold_steps=int(raw.get("gift_hold_steps", -1)),
            exit_step=int(raw.get("gift_exit_step", -1)),
            initial_retention=float(raw.get("gift_initial_retention", -1.0)),
            score_mode=str(raw.get("gift_score_mode", "")).strip().lower(),
            cosine_min=float(raw.get("gift_cosine_min", 0.0)),
            norm_eps=float(raw.get("gift_norm_eps", 1.0e-8)),
            base_lambda=float(raw.get("opsd_grpo_lambda", -1.0)),
            top_k_distill=int(raw.get("top_k_distill", -1)),
        )
        obj.validate(raw)
        return obj

    def validate(self, raw: dict[str, Any] | None = None) -> None:
        if self.group_size != 8:
            raise ValueError(f"prompt-group experiment requires group_size=8, got {self.group_size}")
        if self.total_training_steps <= 0:
            raise ValueError(
                f"total_training_steps must be positive, got {self.total_training_steps}"
            )
        if not (0 < self.hold_steps < self.exit_step <= self.total_training_steps):
            raise ValueError(
                "require 0 < hold_steps < exit_step <= total_training_steps, got "
                f"{self.hold_steps}/{self.exit_step}/{self.total_training_steps}"
            )
        if not (0.0 < self.initial_retention <= 1.0):
            raise ValueError("gift_initial_retention must be in (0, 1]")
        if self.score_mode not in {
            "direction_residual",
            "magnitude_residual",
            "piecewise_conflict_residual",
        }:
            raise ValueError(
                "gift_score_mode must be direction_residual, magnitude_residual or "
                "piecewise_conflict_residual"
            )
        if not (0.0 <= self.cosine_min < 1.0):
            raise ValueError("gift_cosine_min must be in [0, 1)")
        if self.norm_eps <= 0.0:
            raise ValueError("gift_norm_eps must be positive")
        if self.base_lambda != 1.0:
            raise ValueError(
                f"token-count-only annealing requires opsd_grpo_lambda=1.0, got {self.base_lambda}"
            )
        if self.top_k_distill <= 0:
            raise ValueError("GIFT scoring requires sparse top_k_distill > 0")
        if raw is None:
            return
        expected = {
            "gift_enabled": True,
            "gift_schedule": "linear_token_budget_only",
            "method": "opsd_grpo_curriculum",
            "teacher_mode": "fixed",
            "use_lora": False,
            "full_logit_distill": True,
            "distill_add_tail": True,
            "beta": 0.0,
            "jsd_token_clip": 0.0,
            "gift_support": "teacher_topk_plus_tail_exact_full_logit_geometry",
            "val_before_train": False,
            "save_steps": 10,
            "eval_steps": 10,
        }
        mismatches = {
            key: {"actual": raw.get(key), "expected": expected_value}
            for key, expected_value in expected.items()
            if raw.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(f"core config mismatch: {mismatches}")


def load_settings() -> CoreSettings:
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS
    config_text = os.environ.get(_CONFIG_ENV, "").strip()
    if not config_text:
        raise RuntimeError(f"{_CONFIG_ENV} is required when {_ENABLE_ENV}=1")
    config_path = Path(config_text).expanduser().resolve()
    if not config_path.is_file():
        raise RuntimeError(f"core config does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"core config must be a mapping: {config_path}")
    _SETTINGS = CoreSettings.from_mapping(raw)
    return _SETTINGS


def _cfg(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    policy_loss = getattr(config, "policy_loss", None)
    if policy_loss is not None:
        if hasattr(policy_loss, "get"):
            value = policy_loss.get(key, default)
        else:
            value = getattr(policy_loss, key, default)
        return default if value is None else value
    if hasattr(config, "get"):
        value = config.get(key, default)
        return default if value is None else value
    value = getattr(config, key, default)
    return default if value is None else value


def coverage_ratio(config: Any, global_steps) -> float:
    """Coverage kappa_t (paper Eq. 10): ``initial_retention`` through step H=``hold_steps``,
    then linear to zero at T_exit=``exit_step``.  H, T_exit and the total step count are
    read from the config; the paper's own run used H=10, T_exit=78 over 96 steps.
    initial_retention is 1.0 for GIFT and 0.5 in the ablation YAMLs."""
    if global_steps is None:
        raise RuntimeError("GIFT requires data.meta_info['global_steps']")
    hold_raw = _cfg(config, "gift_hold_steps", None)
    exit_raw = _cfg(config, "gift_exit_step", None)
    initial_raw = _cfg(config, "gift_initial_retention", None)
    if hold_raw is None or exit_raw is None or initial_raw is None:
        settings = load_settings()
        hold_steps = settings.hold_steps
        exit_step = settings.exit_step
        initial = settings.initial_retention
    else:
        hold_steps = int(hold_raw)
        exit_step = int(exit_raw)
        initial = float(initial_raw)
    step = int(global_steps)
    if step <= hold_steps:
        return initial
    if step >= exit_step:
        return 0.0
    progress = (step - hold_steps) / float(exit_step - hold_steps)
    return initial * (1.0 - progress)


def opsd_lambda(config: Any, global_steps) -> float:
    """The OPSD coefficient is fixed; token retention is the only annealed axis."""
    if global_steps is None:
        raise RuntimeError("GIFT requires data.meta_info['global_steps']")
    value = _cfg(config, "opsd_grpo_lambda", None)
    return load_settings().base_lambda if value is None else max(float(value), 0.0)


def teacher_active(config: Any, global_steps) -> bool:
    """Skip teacher work once the token budget reaches exactly zero."""
    return (
        opsd_lambda(config, global_steps) > 0.0
        and coverage_ratio(config, global_steps) > 0.0
    )


def sparse_tail_gradient_geometry(
    *,
    student_topk_log_probs,
    teacher_topk_log_probs,
    topk_indices,
    actions,
    student_action_log_probs,
    student_sum_pi_squared,
    old_action_log_probs,
    advantages,
    response_mask,
    clip_ratio_low: float,
    clip_ratio_high: float,
    dual_clip_ratio: float,
    score_mode: str,
    cosine_min: float = 0.0,
    norm_eps: float = 1.0e-8,
    rollout_is_weights=None,
) -> dict[str, Any]:
    """Compute exact logit geometry for the teacher-top-k plus tail KL.

    The sparse OPSD loss keeps every teacher top-k token explicit and merges
    the rest of the vocabulary into one tail probability.  Its full-logit
    gradient on tail token ``j`` is ``alpha * p_student[j]``.  Therefore the
    dot product and both norms need only top-k probabilities, the sampled-token
    probability, and the full-vocabulary sufficient statistic ``sum(p^2)``.
    """
    import torch

    tensors = (
        student_topk_log_probs,
        teacher_topk_log_probs,
        topk_indices,
    )
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("top-k log probabilities and indices must have shape [B, L, K]")
    if student_topk_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError("student and teacher top-k tensors must have identical shapes")
    if topk_indices.shape != student_topk_log_probs.shape:
        raise ValueError("top-k index shape must match top-k log-probability shape")
    expected = student_topk_log_probs.shape[:2]
    for name, tensor in (
        ("actions", actions),
        ("student_action_log_probs", student_action_log_probs),
        ("student_sum_pi_squared", student_sum_pi_squared),
        ("old_action_log_probs", old_action_log_probs),
        ("advantages", advantages),
        ("response_mask", response_mask),
    ):
        if tensor.shape != expected:
            raise ValueError(f"{name} shape {tensor.shape} != expected {expected}")

    p_student_topk = student_topk_log_probs.detach().float().exp()
    p_teacher_topk = teacher_topk_log_probs.detach().float().exp()
    action_match = topk_indices.detach().eq(actions.detach().unsqueeze(-1))
    action_in_topk = action_match.any(dim=-1)
    student_action_prob = student_action_log_probs.detach().float().exp()
    student_topk_mass = p_student_topk.sum(dim=-1)
    teacher_topk_mass = p_teacher_topk.sum(dim=-1)
    student_tail_mass = (1.0 - student_topk_mass).clamp_min(norm_eps)
    # For L = KL(q_topk+tail || p_topk+tail), tail-coordinate gradients are
    # alpha*p_j, alpha=(Q_topk-P_topk)/P_tail, for every non-top-k vocabulary j.
    tail_alpha = (teacher_topk_mass - student_topk_mass) / student_tail_mass
    full_sum_p_squared = student_sum_pi_squared.detach().float()
    tail_sum_p_squared = (
        full_sum_p_squared - p_student_topk.square().sum(dim=-1)
    ).clamp_min(0.0)
    topk_d = p_student_topk - p_teacher_topk
    d0_norm_sq = topk_d.square().sum(dim=-1) + tail_alpha.square() * tail_sum_p_squared

    p_dot_d = (
        (p_student_topk * topk_d).sum(dim=-1)
        + tail_alpha * tail_sum_p_squared
    )
    matched_d = (topk_d * action_match.to(topk_d.dtype)).sum(dim=-1)
    action_d = torch.where(action_in_topk, matched_d, tail_alpha * student_action_prob)
    policy_base_dot_d = p_dot_d - action_d
    policy_base_norm_sq = (
        full_sum_p_squared - 2.0 * student_action_prob + 1.0
    ).clamp_min(0.0)

    ratio = torch.exp(
        (student_action_log_probs.detach().float() - old_action_log_probs.detach().float())
        .clamp(min=-20.0, max=20.0)
    )
    advantage = advantages.detach().float()
    active = torch.ones_like(response_mask, dtype=torch.bool)
    active &= ~((advantage > 0.0) & (ratio > 1.0 + float(clip_ratio_high)))
    active &= ~((advantage < 0.0) & (ratio < 1.0 - float(clip_ratio_low)))
    active &= ~((advantage < 0.0) & (ratio > float(dual_clip_ratio)))

    if rollout_is_weights is None:
        importance_weight = torch.ones_like(advantage)
    else:
        importance_weight = rollout_is_weights.detach().float()
        if importance_weight.shape != expected:
            raise ValueError(
                f"rollout_is_weights shape {importance_weight.shape} != expected {expected}"
            )
    finite_weight = torch.isfinite(importance_weight) & (importance_weight > 0.0)

    policy_scale = advantage * ratio * active.to(advantage.dtype) * importance_weight
    r_norm_sq = policy_scale.square() * policy_base_norm_sq
    d_norm_sq = importance_weight.square() * d0_norm_sq
    dot = policy_scale * importance_weight * policy_base_dot_d
    finite = (
        torch.isfinite(r_norm_sq)
        & torch.isfinite(d_norm_sq)
        & torch.isfinite(dot)
        & torch.isfinite(ratio)
        & torch.isfinite(advantage)
        & finite_weight
    )
    valid = response_mask.detach().bool() & active & finite
    valid &= r_norm_sq > float(norm_eps) ** 2
    valid &= d_norm_sq > float(norm_eps) ** 2

    denominator = torch.sqrt((r_norm_sq * d_norm_sq).clamp_min(0.0))
    cosine = torch.zeros_like(dot)
    cosine[valid] = (dot[valid] / denominator[valid].clamp_min(norm_eps)).clamp(-1.0, 1.0)
    residual_sq = (d_norm_sq - dot.square() / r_norm_sq.clamp_min(float(norm_eps) ** 2)).clamp_min(0.0)
    residual = torch.sqrt(residual_sq)
    r_norm = torch.sqrt(r_norm_sq.clamp_min(0.0))
    d_norm = torch.sqrt(d_norm_sq.clamp_min(0.0))
    eligible = valid & (cosine > float(cosine_min)) & (residual > float(norm_eps))

    if score_mode == "direction_residual":
        score = residual
    elif score_mode == "magnitude_residual":
        # ||r|| * ||d_perp|| = sqrt(||r||^2 ||d||^2 - <r,d>^2).
        score = torch.sqrt((r_norm_sq * d_norm_sq - dot.square()).clamp_min(0.0))
    else:
        raise ValueError(f"unsupported score_mode={score_mode!r}")
    score = torch.where(eligible, score, torch.zeros_like(score))

    return {
        "score": score,
        "eligible": eligible,
        "valid": response_mask.detach().bool(),
        "active": active,
        "cosine": cosine,
        "residual": residual,
        "r_norm": r_norm,
        "d_norm": d_norm,
        "ratio": ratio,
        "advantage": advantage,
        "action_in_topk": action_in_topk,
    }


def build_selection_masks(
    records: list[dict[str, Any]],
    retention_ratio: float,
    *,
    group_size: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Gate on compatibility, then perform the exact Top-K selection."""
    import torch

    if not (0.0 < float(retention_ratio) <= 1.0):
        raise ValueError(f"mask construction expects 0 < retention <= 1, got {retention_ratio}")
    masks = [torch.zeros_like(record["valid"], dtype=torch.float32) for record in records]
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record_idx, record in enumerate(records):
        shape = record["valid"].shape
        for key in (
            "score", "eligible", "active", "cosine", "residual", "r_norm",
            "d_norm", "ratio", "advantage", "action_in_topk",
        ):
            if record[key].shape != shape:
                raise RuntimeError(f"GIFT {key} shape {record[key].shape} != {shape}")
        if shape[0] != len(record["uids"]):
            raise RuntimeError("GIFT uid count does not match tensor rows")
        for row_idx, uid in enumerate(record["uids"]):
            groups[str(uid)].append((record_idx, row_idx))
    if not groups:
        raise RuntimeError("GIFT received no prompt groups")

    total_budget = 0
    total_selected = 0
    total_valid = 0
    zero_eligible_groups = 0
    rollout_selected_counts: list[int] = []
    for uid, row_refs in groups.items():
        if len(row_refs) != int(group_size):
            raise RuntimeError(
                f"uid={uid} is incomplete on this actor rank: got {len(row_refs)} rollouts, "
                f"expected {group_size}; enable group-preserving DP balancing"
            )
        refs: list[tuple[int, int, Any]] = []
        score_chunks = []
        eligible_chunks = []
        valid_count = 0
        for record_idx, row_idx in row_refs:
            valid_row = records[record_idx]["valid"][row_idx].bool()
            positions = torch.nonzero(valid_row, as_tuple=False).squeeze(-1)
            refs.append((record_idx, row_idx, positions))
            score_chunks.append(records[record_idx]["score"][row_idx, positions].float())
            eligible_chunks.append(records[record_idx]["eligible"][row_idx, positions].bool())
            valid_count += int(positions.numel())
        if valid_count <= 0:
            raise RuntimeError(f"uid={uid} has no valid response tokens")

        scores = torch.cat(score_chunks)
        eligible = torch.cat(eligible_chunks)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
        budget = int(math.floor(float(retention_ratio) * valid_count))
        selected_count = min(budget, int(eligible_indices.numel()))
        selected = torch.zeros(valid_count, dtype=torch.bool)
        if selected_count > 0:
            eligible_scores = scores[eligible_indices]
            selected_in_eligible = torch.topk(
                eligible_scores, k=selected_count, largest=True, sorted=False
            ).indices
            selected[eligible_indices[selected_in_eligible]] = True
        if int(eligible_indices.numel()) == 0:
            zero_eligible_groups += 1

        offset = 0
        for record_idx, row_idx, positions in refs:
            row_count = int(positions.numel())
            row_selected = selected[offset : offset + row_count]
            masks[record_idx][row_idx, positions[row_selected]] = 1.0
            rollout_selected_counts.append(int(row_selected.sum().item()))
            offset += row_count
        if offset != valid_count:
            raise AssertionError(f"uid={uid}: reconstructed {offset} tokens, expected {valid_count}")
        actual = sum(int(masks[r][i].sum().item()) for r, i in row_refs)
        if actual != selected_count:
            raise AssertionError(f"uid={uid}: selected {actual}, expected {selected_count}")
        total_budget += budget
        total_selected += selected_count
        total_valid += valid_count

    queue_items: list[dict[str, Any]] = []
    for record, mask in zip(records, masks, strict=True):
        valid = record["valid"].bool()
        selected = mask.bool()
        local_valid = int(valid.sum().item())
        local_selected = int(selected.sum().item())

        def mean_on(key: str, selector) -> float:
            values = record[key].float()[selector]
            return float(values.mean().item()) if values.numel() else 0.0

        queue_items.append(
            {
                "mask": mask,
                "valid_signature": tuple(int(x) for x in valid.sum(dim=1).tolist()),
                "selected": local_selected,
                "valid": local_valid,
                "eligible": int((record["eligible"] & valid).sum().item()),
                "active": int((record["active"] & valid).sum().item()),
                "negative_valid": int(((record["advantage"] < 0) & valid).sum().item()),
                "negative_selected": int(((record["advantage"] < 0) & selected).sum().item()),
                "action_in_topk": int((record["action_in_topk"] & valid).sum().item()),
                "score_mean": mean_on("score", valid),
                "score_selected_mean": mean_on("score", selected),
                "cosine_mean": mean_on("cosine", valid),
                "cosine_selected_mean": mean_on("cosine", selected),
                "residual_mean": mean_on("residual", valid),
                "residual_selected_mean": mean_on("residual", selected),
                "r_norm_mean": mean_on("r_norm", valid),
                "r_norm_selected_mean": mean_on("r_norm", selected),
                "d_norm_mean": mean_on("d_norm", valid),
                "d_norm_selected_mean": mean_on("d_norm", selected),
                "group_count": float(len(groups)),
            }
        )

    rollout_counts = torch.tensor(rollout_selected_counts, dtype=torch.float32)
    rollout_mean = float(rollout_counts.mean().item()) if rollout_counts.numel() else 0.0
    rollout_cv = (
        float(rollout_counts.std(unbiased=False).item()) / rollout_mean
        if rollout_mean > 0.0
        else 0.0
    )
    diagnostics = {
        "group_count": float(len(groups)),
        "selected": float(total_selected),
        "valid": float(total_valid),
        "budget": float(total_budget),
        "actual_retention": float(total_selected / max(total_valid, 1)),
        "budget_fill": float(total_selected / max(total_budget, 1)),
        "zero_eligible_group_fraction": float(zero_eligible_groups / max(len(groups), 1)),
        "zero_token_rollout_fraction": float((rollout_counts == 0).float().mean().item())
        if rollout_counts.numel()
        else 0.0,
        "rollout_allocation_cv": rollout_cv,
    }
    for item in queue_items:
        item.update(
            {
                "budget_fill": diagnostics["budget_fill"],
                "zero_eligible_group_fraction": diagnostics["zero_eligible_group_fraction"],
                "zero_token_rollout_fraction": diagnostics["zero_token_rollout_fraction"],
                "rollout_allocation_cv": diagnostics["rollout_allocation_cv"],
            }
        )
    return queue_items, diagnostics


def _clone_for_device(micro_batch, device):
    batch_keys = list(micro_batch.batch.keys())
    non_tensor_keys = list(micro_batch.non_tensor_batch.keys())
    return micro_batch.select(
        batch_keys=batch_keys,
        non_tensor_batch_keys=non_tensor_keys,
    ).to(device)


def _student_forward_with_sum_pi_squared(actor, model_inputs, *, temperature, topk_indices):
    """Run the sparse scoring forward and expose ``sum(p**2)`` for this call only.

    The vendored actor gates this sufficient statistic behind a frozen config
    field and normally initializes its helper only when that field is enabled
    globally.  Global enablement is incompatible with the prefix grouper used
    to preserve prompt-group IDs.  This scoring call already bypasses the
    prefix-grouper fast path because aligned top-k indices are requested, so it
    is safe to enable the statistic only around this no-grad forward.
    """
    from verl.utils import torch_functional as verl_F

    if not hasattr(actor, "calculate_sum_pi_squared_from_logits"):
        actor.calculate_sum_pi_squared_from_logits = verl_F.calculate_sum_pi_squared_from_logits
    old_value = bool(actor.config.get("calculate_sum_pi_squared", False))
    object.__setattr__(actor.config, "calculate_sum_pi_squared", True)
    try:
        outputs = actor._forward_micro_batch(
            model_inputs,
            temperature=temperature,
            calculate_entropy=False,
            topk_indices=topk_indices,
        )
    finally:
        object.__setattr__(actor.config, "calculate_sum_pi_squared", old_value)
    if "sum_pi_squared" not in outputs:
        raise RuntimeError("GIFT student scoring did not return sum_pi_squared")
    return outputs


def _score_micro_batches(actor, micro_batches, retention_ratio: float) -> list[dict[str, Any]]:
    """Run one no-grad scoring pass and return masks for the training pass."""
    import torch
    from verl.utils.device import get_device_id

    settings = load_settings()
    temperature = float(getattr(actor, "_gift_temperature"))
    pad_token_id = int(getattr(actor, "_gift_pad_token_id"))
    clip_ratio = float(actor.config.clip_ratio)
    clip_ratio_low = float(
        actor.config.clip_ratio_low
        if actor.config.clip_ratio_low is not None
        else clip_ratio
    )
    clip_ratio_high = float(
        actor.config.clip_ratio_high
        if actor.config.clip_ratio_high is not None
        else clip_ratio
    )
    dual_clip_ratio = float(actor.config.get("clip_ratio_c", 3.0))
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for micro_batch in micro_batches:
            work = _clone_for_device(micro_batch, get_device_id())
            try:
                if "uid" not in work.non_tensor_batch:
                    raise RuntimeError(
                        "GIFT requires uid on actor micro-batches; "
                        "actor.use_prefix_grouper must be true"
                    )
                model_inputs = {**work.batch, **work.non_tensor_batch, "pad_token_id": pad_token_id}
                required_teacher = (
                    "teacher_input_ids",
                    "teacher_attention_mask",
                    "teacher_position_ids",
                )
                missing = [name for name in required_teacher if name not in model_inputs]
                if missing:
                    raise RuntimeError(f"GIFT scoring pass is missing teacher tensors: {missing}")
                teacher_inputs = {
                    "responses": model_inputs["responses"],
                    "response_mask": model_inputs["response_mask"],
                    "input_ids": model_inputs["teacher_input_ids"],
                    "attention_mask": model_inputs["teacher_attention_mask"],
                    "position_ids": model_inputs["teacher_position_ids"],
                }
                teacher_outputs = actor._teacher_forward(
                    teacher_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    distill_topk=settings.top_k_distill,
                    align_response_by_mask=True,
                )
                if teacher_outputs is None:
                    raise RuntimeError("GIFT teacher scoring forward returned None")
                teacher_topk = teacher_outputs.get("topk_log_probs")
                teacher_indices = teacher_outputs.get("topk_indices")
                if teacher_topk is None or teacher_indices is None:
                    raise RuntimeError("GIFT teacher scoring did not return sparse top-k outputs")

                student_outputs = _student_forward_with_sum_pi_squared(
                    actor,
                    model_inputs,
                    temperature=temperature,
                    topk_indices=teacher_indices,
                )
                student_topk = student_outputs.get("topk_log_probs")
                if student_topk is None:
                    raise RuntimeError("GIFT student scoring did not return sparse top-k outputs")
                geometry = sparse_tail_gradient_geometry(
                    student_topk_log_probs=student_topk,
                    teacher_topk_log_probs=teacher_topk,
                    topk_indices=teacher_indices,
                    actions=model_inputs["responses"],
                    student_action_log_probs=student_outputs["log_probs"],
                    student_sum_pi_squared=student_outputs["sum_pi_squared"],
                    old_action_log_probs=model_inputs["old_log_probs"],
                    advantages=model_inputs["advantages"],
                    response_mask=model_inputs["response_mask"],
                    clip_ratio_low=clip_ratio_low,
                    clip_ratio_high=clip_ratio_high,
                    dual_clip_ratio=dual_clip_ratio,
                    score_mode=settings.score_mode,
                    cosine_min=settings.cosine_min,
                    norm_eps=settings.norm_eps,
                    rollout_is_weights=model_inputs.get("rollout_is_weights"),
                )
                records.append(
                    {
                        "uids": [str(uid) for uid in work.non_tensor_batch["uid"].tolist()],
                        **{
                            key: value.detach().cpu()
                            for key, value in geometry.items()
                        },
                    }
                )
            finally:
                del work

    queue_items, diagnostics = build_selection_masks(
        records,
        retention_ratio,
        group_size=settings.group_size,
    )
    rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
    mini_index = int(getattr(actor, "_gift_mini_index", 0))
    if rank == 0:
        print(
            "[GIFT-core] "
            f"step={getattr(actor, '_gift_step')} mini={mini_index} "
            f"mode={settings.score_mode} groups={int(diagnostics['group_count'])} "
            f"budget_ratio={retention_ratio:.8f} "
            f"selected={int(diagnostics['selected'])}/{int(diagnostics['valid'])} "
            f"fill={diagnostics['budget_fill']:.6f} "
            f"zero_eligible_groups={diagnostics['zero_eligible_group_fraction']:.6f} "
            f"zero_token_rollouts={diagnostics['zero_token_rollout_fraction']:.6f} "
            f"allocation_cv={diagnostics['rollout_allocation_cv']:.6f} lambda=1",
            flush=True,
        )
    return queue_items


def _consume_selection_mask(
    response_mask,
    student_entropy,
    forward_divergence,
    retention_ratio,
    clip_quantile=0.98,
    entropy_eps=1e-6,
    min_selected_tokens=1,
):
    """Consume the precomputed mask; legacy selector arguments are ignored."""
    import torch

    queue = getattr(_STATE, "mask_queue", None)
    if queue is None or not queue:
        raise RuntimeError("GIFT mask queue is empty; scoring/training pass order diverged")
    item = queue.popleft()
    mask = item["mask"].to(device=response_mask.device, dtype=torch.float32)
    if mask.shape != response_mask.shape:
        raise RuntimeError(f"GIFT mask shape {mask.shape} != response mask {response_mask.shape}")
    valid = response_mask.detach().bool()
    signature = tuple(int(x) for x in valid.sum(dim=1).cpu().tolist())
    if signature != item["valid_signature"]:
        raise RuntimeError(
            f"GIFT micro-batch order changed: expected {item['valid_signature']}, got {signature}"
        )
    if bool((mask.bool() & ~valid).any().item()):
        raise RuntimeError("GIFT selected a padded token")
    selected = int(mask.sum().item())
    if selected != int(item["selected"]):
        raise RuntimeError(f"GIFT selected count changed: {selected} != {item['selected']}")
    metrics = {
        "gift/actual_retention_ratio": float(selected / max(int(item["valid"]), 1)),
        "gift/selected_token_count_local": float(selected),
        "gift/valid_token_count_local": float(item["valid"]),
        "gift/eligible_ratio": float(item["eligible"] / max(int(item["valid"]), 1)),
        "gift/ppo_active_ratio": float(item["active"] / max(int(item["valid"]), 1)),
        "gift/negative_adv_valid_ratio": float(
            item["negative_valid"] / max(int(item["valid"]), 1)
        ),
        "gift/negative_adv_selected_ratio": float(
            item["negative_selected"] / max(selected, 1)
        ),
        "gift/action_in_teacher_topk_ratio": float(
            item["action_in_topk"] / max(int(item["valid"]), 1)
        ),
        "gift/score_mean": float(item["score_mean"]),
        "gift/score_selected_mean": float(item["score_selected_mean"]),
        "gift/cosine_mean": float(item["cosine_mean"]),
        "gift/cosine_selected_mean": float(item["cosine_selected_mean"]),
        "gift/residual_mean": float(item["residual_mean"]),
        "gift/residual_selected_mean": float(item["residual_selected_mean"]),
        "gift/grpo_norm_mean": float(item["r_norm_mean"]),
        "gift/grpo_norm_selected_mean": float(item["r_norm_selected_mean"]),
        "gift/opsd_norm_mean": float(item["d_norm_mean"]),
        "gift/opsd_norm_selected_mean": float(item["d_norm_selected_mean"]),
        "gift/group_scope": 1.0,
        "gift/group_count_local": float(item["group_count"]),
        "gift/budget_fill_ratio": float(item["budget_fill"]),
        "gift/zero_eligible_group_fraction": float(
            item["zero_eligible_group_fraction"]
        ),
        "gift/zero_token_rollout_fraction": float(
            item["zero_token_rollout_fraction"]
        ),
        "gift/rollout_allocation_cv": float(item["rollout_allocation_cv"]),
        "gift/uses_entropy": 0.0,
        "gift/uses_advantage_magnitude": float(
            load_settings().score_mode == "magnitude_residual"
        ),
    }
    return mask, metrics


def _patch_core(module: ModuleType) -> None:
    if getattr(module, "_gift_overlay", False):
        return

    def opsd_grpo_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        loss_agg_mode="token-mean",
        config=None,
        rollout_is_weights=None,
        **kwargs,
    ):
        import torch

        if config is None:
            raise RuntimeError("GIFT loss requires actor config")
        if loss_agg_mode != "token-mean":
            raise RuntimeError(
                "GIFT all-valid-token normalization requires loss_agg_mode=token-mean"
            )
        global_steps = kwargs.get("global_steps")
        retention = coverage_ratio(config, global_steps)
        opsd_coef = opsd_lambda(config, global_steps)
        grpo_loss, metrics = module.compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )
        schedule_metrics = {
            "gift/target_budget_ratio": float(retention),
            "gift/lambda": float(opsd_coef),
            "gift/teacher_forward_enabled": float(retention > 0.0 and opsd_coef > 0.0),
            "gift/score_mode_strength": float(
                load_settings().score_mode == "magnitude_residual"
            ),
            "gift/all_valid_token_denominator": 1.0,
        }
        if retention <= 0.0 or opsd_coef <= 0.0:
            metrics.update(
                {
                    "grpo_loss": float(grpo_loss.detach().item()),
                    "opsd_loss": 0.0,
                    "gift/weighted_opsd_loss": 0.0,
                    "gift/actual_retention_ratio": 0.0,
                    "gift/selected_token_count_global": 0.0,
                }
            )
            metrics.update(schedule_metrics)
            return grpo_loss, metrics

        teacher_lp = kwargs.get("teacher_log_probs")
        if teacher_lp is None:
            raise RuntimeError("GIFT is active but teacher_log_probs is missing")
        if not module._sd_bool_cfg(config, "full_logit_distill", True):
            raise AssertionError("GIFT requires full_logit_distill=true")

        top_k_distill = module._sd_int_cfg(config, "top_k_distill", 0)
        add_tail = module._sd_bool_cfg(config, "distill_add_tail", True)
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill <= 0 or student_topk is None or teacher_topk is None:
            raise RuntimeError("GIFT requires aligned sparse teacher/student top-k logits")
        student_distill, teacher_distill = module._sd_prepare_sparse_topk_log_probs(
            student_topk_log_probs=student_topk,
            teacher_topk_log_probs=teacher_topk.detach(),
            add_tail=add_tail,
        )
        selected_mask, selection_metrics = _consume_selection_mask(
            response_mask=response_mask,
            student_entropy=None,
            forward_divergence=None,
            retention_ratio=retention,
        )
        beta = float(module._sd_cfg(config, "beta", 0.0))
        jsd_token_clip = float(module._sd_cfg(config, "jsd_token_clip", 0.0))
        if beta != 0.0 or jsd_token_clip > 0.0:
            raise RuntimeError("GIFT requires ordinary unclipped forward KL")
        opsd_per_token = module._sd_generalized_jsd(
            student_distill,
            teacher_distill,
            mix=0.0,
            vocab_clip=None,
        )
        if rollout_is_weights is not None:
            opsd_per_token = opsd_per_token * rollout_is_weights
        # Keep every selected token's coefficient fixed at lambda/N_valid.
        # Shrinking the mask, rather than a lambda schedule, removes OPSD.
        opsd_loss = module.agg_loss(
            loss_mat=opsd_per_token * selected_mask.to(opsd_per_token.dtype),
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        global_selected = selected_mask.detach().float().sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_selected, op=torch.distributed.ReduceOp.SUM)
        total_loss = grpo_loss + opsd_coef * opsd_loss
        metrics.update(
            {
                "grpo_loss": float(grpo_loss.detach().item()),
                "opsd_loss": float(opsd_loss.detach().item()),
                "gift/weighted_opsd_loss": float(
                    (opsd_coef * opsd_loss).detach().item()
                ),
                "gift/selected_token_count_global": float(global_selected.item()),
                "distill_beta": 0.0,
                "distill_topk": float(top_k_distill),
            }
        )
        metrics.update(schedule_metrics)
        metrics.update(selection_metrics)
        metrics.update(
            module._sd_teacher_metrics(
                teacher_lp,
                log_prob,
                response_mask,
                kwargs.get("teacher_entropy"),
            )
        )
        return total_loss, metrics

    opsd_grpo_policy_loss._is_opsd_grpo_curriculum = True
    opsd_grpo_policy_loss._is_opsd_grpo_overlay = True
    module.opsd_grpo_teacher_active = teacher_active
    module.compute_policy_loss_opsd_grpo_curriculum = opsd_grpo_policy_loss
    module.POLICY_LOSS_REGISTRY["opsd_grpo_curriculum"] = opsd_grpo_policy_loss
    module._gift_overlay = True


def _patch_actor(module: ModuleType) -> None:
    cls = module.DataParallelPPOActor
    if getattr(cls, "_gift_core_overlay", False):
        return
    from verl.trainer.ppo import core_algos

    original_update = cls.update_policy
    original_prepare = cls._prepare_update_micro_batches

    def update_policy(self, data):
        step = data.meta_info.get("global_steps")
        retention = coverage_ratio(self.config, step)
        active = teacher_active(self.config, step)
        if not bool(self.config.get("use_prefix_grouper", False)):
            raise RuntimeError("GIFT requires actor.use_prefix_grouper=true")
        if "uid" not in data.non_tensor_batch:
            raise RuntimeError("GIFT requires uid in the actor update batch")
        self._gift_step = int(step)
        self._gift_retention = float(retention)
        self._gift_temperature = float(data.meta_info["temperature"])
        self._gift_pad_token_id = int(data.meta_info.get("pad_token_id", 0))
        self._gift_active = bool(active)
        self._gift_mini_index = 0
        _STATE.mask_queue = deque()
        try:
            result = original_update(self, data)
            if getattr(_STATE, "mask_queue", None):
                raise RuntimeError(
                    f"GIFT training pass left {len(_STATE.mask_queue)} masks unconsumed"
                )
            return result
        finally:
            _STATE.mask_queue = deque()
            self._gift_active = False

    def prepare_update_micro_batches(self, mini_batch, needs_teacher_forward: bool):
        micro_batches = original_prepare(self, mini_batch, needs_teacher_forward)
        if not bool(getattr(self, "_gift_active", False)):
            return micro_batches
        if not needs_teacher_forward:
            raise RuntimeError("GIFT active step unexpectedly disabled teacher forward")
        if getattr(_STATE, "mask_queue", None):
            raise RuntimeError("previous GIFT mini-batch masks were not fully consumed")
        queue_items = _score_micro_batches(
            self,
            micro_batches,
            retention_ratio=float(self._gift_retention),
        )
        if len(queue_items) != len(micro_batches):
            raise RuntimeError(
                "GIFT produced "
                f"{len(queue_items)} masks for {len(micro_batches)} micro-batches"
            )
        _STATE.mask_queue = deque(queue_items)
        self._gift_mini_index += 1
        return micro_batches

    cls.update_policy = update_policy
    cls._prepare_update_micro_batches = prepare_update_micro_batches
    cls._gift_core_overlay = True
    # dp_actor imports these names by value.  Assign explicitly as a guard for
    # environments where the actor module was imported before this callback.
    module.opsd_grpo_teacher_active = core_algos.opsd_grpo_teacher_active


def _patch_self_distill(module: ModuleType) -> None:
    from verl.trainer.ppo import core_algos

    module.opsd_grpo_teacher_active = core_algos.opsd_grpo_teacher_active


class _PatchFinder:
    def __init__(self, callbacks: dict[str, Any]):
        self._callbacks = callbacks

    def find_spec(self, fullname, path=None, target=None):
        import importlib.machinery as machinery

        callback = self._callbacks.get(fullname)
        if callback is None:
            return None
        spec = machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        original_loader = spec.loader

        class _CallbackLoader:
            def create_module(self, spec):
                if hasattr(original_loader, "create_module"):
                    return original_loader.create_module(spec)
                return None

            def exec_module(self, loaded_module):
                original_loader.exec_module(loaded_module)
                callback(loaded_module)

        spec.loader = _CallbackLoader()
        return spec


def apply_runtime_patches() -> None:
    global _PATCHED
    if _PATCHED or not _env_flag(_ENABLE_ENV, default=False):
        return
    settings = load_settings()
    callbacks = {
        "verl.trainer.ppo.core_algos": _patch_core,
        "verl.workers.actor.dp_actor": _patch_actor,
        "src.self_distill_main": _patch_self_distill,
    }
    for module_name, callback in callbacks.items():
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            callback(loaded)
    pending = {name: callback for name, callback in callbacks.items() if name not in sys.modules}
    if pending:
        sys.meta_path.insert(0, _PatchFinder(pending))
    _PATCHED = True
    print(
        "[GIFT-core] installed isolated overlay: "
        f"group_size={settings.group_size} hold={settings.hold_steps} "
        f"exit={settings.exit_step} score={settings.score_mode} "
        f"kappa0={settings.initial_retention} cosine_min={settings.cosine_min} "
        "entropy=off lambda=constant-1",
        file=sys.stderr,
        flush=True,
    )
