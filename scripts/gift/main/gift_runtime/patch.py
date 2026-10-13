"""GIFT: Gradient-Informed Filtering -- piecewise gradient ranking with coverage annealing.

Paper correspondence (GIFT, ICLR 2027 submission):

* Sec. 3.1, Eq. 1-2   r_i, d_i, a_i = <d_i, r_hat_i>, b_i = ||d_i - a_i r_hat_i||
                      -> ``sparse_tail_gradient_geometry`` (here ``a`` = ``parallel``,
                      ``b`` = ``residual``)
* Sec. 3.4, Eq. 8     piecewise score s_i -> ``score`` in ``sparse_tail_gradient_geometry``
* Sec. 3.3, Eq. 7 /   exact-quota Top-K selection (Theorem 3), App. H.3 Eq. 29
                      -> ``build_gift_masks`` (K = floor(kappa_t * |I_m|))
* Sec. 3.4, Eq. 10 /  coverage schedule kappa_t (H, T_exit, total_training_steps
  App. H.3 Eq. 30     are config-driven; the paper's own run used 10, 78, 96)
                      -> ``coverage_ratio`` in ``core_runtime``
* Sec. 3.5, Eq. 11    L^GIFT = L^GRPO + lambda/N_valid * sum m_i(t) w_i l_i^sKL, lambda=1
                      -> ``opsd_grpo_policy_loss`` in ``core_runtime``
* App. H.4 scopes     ``selection_scope``: ``prompt_group`` = "Per-prompt";
                      ``micro_batch`` = "Training sub-batch" (the GIFT default)

GIFT piecewise gradient ranking with token-count-only annealing.

This is an isolated overlay on the shared ``core_runtime`` package (same
directory).  It never edits the project or vendored ``verl`` sources.

For each valid response token, let ``r`` be the clipped GRPO loss gradient and
``d`` the sparse teacher-top-k-plus-tail forward-KL loss gradient in output
logit space.  With ``r_hat = r / ||r||``:

    a = <d, r_hat>
    b = ||d - a r_hat||
    score = a       when a < 0
            b       when a >= 0

Thus every non-conflicting token ranks ahead of every conflicting token;
conflicting tokens are ordered from weak to strong conflict, while compatible
tokens are ordered by orthogonal teacher information.  The score and mask are
detached.  Selected tokens still receive the original, unprojected forward KL.

Two selection scopes are supported by otherwise identical configs:

* ``prompt_group``: the eight rollouts of each prompt share an independent
  Top-K budget;
* ``micro_batch``: all valid response tokens in one actual actor training
  microbatch share one Top-K budget, allowing budget transfer across prompts.

Schedule: kappa=1 through step ``gift_hold_steps``, linear token-count decay
until ``gift_exit_step``, then kappa=0 for the rest of the ``gift_total_training_steps``
run.  The OPSD coefficient remains exactly one.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import threading
from typing import Any

import yaml

from core_runtime import patch as core


_ENABLE_ENV = "GIFT_ENABLE"
_CONFIG_ENV = "GIFT_CONFIG"
_PATCHED = False
_SETTINGS: "GiftSettings | None" = None
_LOCK = threading.Lock()

# Keep references to the validated implementation before replacing the public
# hooks used dynamically by its scoring and loss wrappers.
_CORE_GEOMETRY = core.sparse_tail_gradient_geometry
_CORE_CONSUMER = core._consume_selection_mask


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
class GiftSettings:
    group_size: int
    total_training_steps: int
    hold_steps: int
    exit_step: int
    initial_retention: float
    score_mode: str
    selection_scope: str
    norm_eps: float
    base_lambda: float
    top_k_distill: int

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "GiftSettings":
        obj = cls(
            group_size=int(raw.get("group_size", -1)),
            total_training_steps=int(raw.get("gift_total_training_steps", -1)),
            hold_steps=int(raw.get("gift_hold_steps", -1)),
            exit_step=int(raw.get("gift_exit_step", -1)),
            initial_retention=float(raw.get("gift_initial_retention", -1.0)),
            score_mode=str(raw.get("gift_score_mode", "")).strip().lower(),
            selection_scope=str(
                raw.get("gift_selection_scope", "")
            ).strip().lower(),
            norm_eps=float(raw.get("gift_norm_eps", 1.0e-8)),
            base_lambda=float(raw.get("opsd_grpo_lambda", -1.0)),
            top_k_distill=int(raw.get("top_k_distill", -1)),
        )
        obj.validate(raw)
        return obj

    def validate(self, raw: dict[str, Any] | None = None) -> None:
        if self.group_size != 8:
            raise ValueError(f"GIFT requires group_size=8, got {self.group_size}")
        if self.total_training_steps <= 0:
            raise ValueError(
                f"total_training_steps must be positive, got {self.total_training_steps}"
            )
        if not (0 < self.hold_steps < self.exit_step <= self.total_training_steps):
            raise ValueError(
                "require 0 < hold_steps < exit_step <= total_training_steps, got "
                f"{self.hold_steps}/{self.exit_step}/{self.total_training_steps}"
            )
        if self.initial_retention != 1.0:
            raise ValueError(
                "GIFT full-start schedule requires initial_retention=1.0, got "
                f"{self.initial_retention}"
            )
        if self.score_mode != "piecewise_conflict_residual":
            raise ValueError(
                "gift_score_mode must be piecewise_conflict_residual"
            )
        if self.selection_scope not in {"prompt_group", "micro_batch"}:
            raise ValueError(
                "gift_selection_scope must be prompt_group or micro_batch"
            )
        if self.norm_eps <= 0.0:
            raise ValueError("gift_norm_eps must be positive")
        if self.base_lambda != 1.0:
            raise ValueError("GIFT token-only annealing requires opsd_grpo_lambda=1.0")
        if self.top_k_distill <= 0:
            raise ValueError("GIFT requires sparse top_k_distill > 0")
        if raw is None:
            return
        expected = {
            "gift_piecewise_enabled": True,
            "gift_enabled": True,
            "gift_schedule": "linear_token_budget_only",
            "gift_piecewise_zero_grpo_policy": "neutral_score_zero",
            "gift_piecewise_zero_opsd_policy": "exclude",
            "method": "opsd_grpo_curriculum",
            "teacher_mode": "fixed",
            "use_lora": False,
            "full_logit_distill": True,
            "distill_add_tail": True,
            "beta": 0.0,
            "jsd_token_clip": 0.0,
            "gift_support": (
                "teacher_topk_plus_tail_exact_full_logit_geometry"
            ),
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
            raise ValueError(f"GIFT config mismatch: {mismatches}")


def _read_config() -> dict[str, Any]:
    config_text = os.environ.get(_CONFIG_ENV, "").strip()
    if not config_text:
        raise RuntimeError(f"{_CONFIG_ENV} is required when {_ENABLE_ENV}=1")
    path = Path(config_text).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"GIFT config does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"GIFT config must be a mapping: {path}")
    return raw


def load_settings() -> GiftSettings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = GiftSettings.from_mapping(_read_config())
    return _SETTINGS


def _validate_base_settings(
    self: core.CoreSettings,
    raw: dict[str, Any] | None = None,
) -> None:
    """Make the base runtime accept only this overlay's stricter contract."""
    if raw is None:
        if self.group_size != 8:
            raise ValueError(f"GIFT base settings mismatch: {self}")
        return
    gift = GiftSettings.from_mapping(raw)
    for field in (
        "group_size",
        "total_training_steps",
        "hold_steps",
        "exit_step",
        "initial_retention",
        "score_mode",
        "norm_eps",
        "base_lambda",
        "top_k_distill",
    ):
        if getattr(self, field) != getattr(gift, field):
            raise ValueError(
                f"GIFT/base setting diverged for {field}: "
                f"{getattr(self, field)!r} != {getattr(gift, field)!r}"
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
    """Compute exact sparse-gradient geometry and the GIFT piecewise score."""
    import torch

    if score_mode != "piecewise_conflict_residual":
        raise ValueError(f"GIFT received unsupported score_mode={score_mode!r}")
    # The base routine contains the autograd-calibrated exact geometry for the
    # sparse teacher-top-k plus tail loss.  Its gate/score are deliberately
    # ignored and replaced below.
    result = _CORE_GEOMETRY(
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        topk_indices=topk_indices,
        actions=actions,
        student_action_log_probs=student_action_log_probs,
        student_sum_pi_squared=student_sum_pi_squared,
        old_action_log_probs=old_action_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        dual_clip_ratio=dual_clip_ratio,
        score_mode="direction_residual",
        cosine_min=-1.0,
        norm_eps=norm_eps,
        rollout_is_weights=rollout_is_weights,
    )

    valid_token = result["valid"].bool()
    r_norm = result["r_norm"].float()
    d_norm = result["d_norm"].float()
    cosine = result["cosine"].float()
    residual_raw = result["residual"].float()
    finite = (
        torch.isfinite(r_norm)
        & torch.isfinite(d_norm)
        & torch.isfinite(cosine)
        & torch.isfinite(residual_raw)
        & torch.isfinite(result["ratio"].float())
        & torch.isfinite(result["advantage"].float())
    )
    teacher_nonzero = d_norm > float(norm_eps)
    eligible = valid_token & finite & teacher_nonzero
    direction_defined = (
        eligible
        & result["active"].bool()
        & (r_norm > float(norm_eps))
    )

    parallel = torch.zeros_like(d_norm)
    parallel[direction_defined] = (
        cosine[direction_defined] * d_norm[direction_defined]
    )
    residual = torch.zeros_like(residual_raw)
    residual[direction_defined] = residual_raw[direction_defined]
    conflict = direction_defined & (parallel < 0.0)
    neutral_grpo = eligible & ~direction_defined

    score = torch.zeros_like(d_norm)
    score[direction_defined] = torch.where(
        conflict[direction_defined],
        parallel[direction_defined],  # equals -|a| on the conflict branch
        residual[direction_defined],
    )
    score = torch.where(eligible, score, torch.zeros_like(score))

    result.update(
        {
            "score": score,
            "eligible": eligible,
            "residual": residual,
            "parallel": parallel,
            "conflict": conflict,
            "neutral_grpo": neutral_grpo,
            "direction_defined": direction_defined,
            "teacher_zero": valid_token & finite & ~teacher_nonzero,
        }
    )
    return result


def _selection_scopes(
    records: list[dict[str, Any]],
    selection_scope: str,
    group_size: int,
) -> tuple[list[tuple[str, list[tuple[int, int]]]], int]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record_idx, record in enumerate(records):
        if record["valid"].shape[0] != len(record["uids"]):
            raise RuntimeError("GIFT uid count does not match tensor rows")
        for row_idx, uid in enumerate(record["uids"]):
            groups[str(uid)].append((record_idx, row_idx))
    if not groups:
        raise RuntimeError("GIFT received no prompt groups")
    incomplete = {
        uid: len(refs) for uid, refs in groups.items() if len(refs) != int(group_size)
    }
    if incomplete:
        raise RuntimeError(
            "GIFT requires complete prompt groups on each actor rank; "
            f"incomplete={incomplete}"
        )
    if selection_scope == "prompt_group":
        return [(f"group:{uid}", refs) for uid, refs in groups.items()], len(groups)
    if selection_scope == "micro_batch":
        scopes = []
        for record_idx, record in enumerate(records):
            refs = [(record_idx, row_idx) for row_idx in range(record["valid"].shape[0])]
            if refs:
                scopes.append((f"micro_batch:{record_idx}", refs))
        return scopes, len(groups)
    raise ValueError(f"unknown GIFT selection scope: {selection_scope!r}")


def build_gift_masks(
    records: list[dict[str, Any]],
    retention_ratio: float,
    *,
    group_size: int = 8,
    selection_scope: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Build exact Top-K masks in either prompt-group or microbatch scope."""
    import torch

    if not (0.0 < float(retention_ratio) <= 1.0):
        raise ValueError(f"GIFT mask construction expects 0 < kappa <= 1, got {retention_ratio}")
    scope_name = selection_scope or load_settings().selection_scope
    required_keys = (
        "score",
        "eligible",
        "active",
        "cosine",
        "residual",
        "parallel",
        "conflict",
        "neutral_grpo",
        "direction_defined",
        "teacher_zero",
        "r_norm",
        "d_norm",
        "ratio",
        "advantage",
        "action_in_topk",
    )
    masks = [torch.zeros_like(record["valid"], dtype=torch.float32) for record in records]
    for record in records:
        shape = record["valid"].shape
        for key in required_keys:
            if record[key].shape != shape:
                raise RuntimeError(f"GIFT {key} shape {record[key].shape} != {shape}")

    scopes, group_count = _selection_scopes(records, scope_name, group_size)
    total_budget = 0
    total_selected = 0
    total_valid = 0
    zero_eligible_scopes = 0

    for scope_id, row_refs in scopes:
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
            raise RuntimeError(f"{scope_id} has no valid response tokens")

        scores = torch.cat(score_chunks)
        eligible = torch.cat(eligible_chunks)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
        budget = int(math.floor(float(retention_ratio) * valid_count))
        selected_count = min(budget, int(eligible_indices.numel()))
        selected = torch.zeros(valid_count, dtype=torch.bool)
        if selected_count:
            selected_local = torch.topk(
                scores[eligible_indices],
                k=selected_count,
                largest=True,
                sorted=False,
            ).indices
            selected[eligible_indices[selected_local]] = True
        if not int(eligible_indices.numel()):
            zero_eligible_scopes += 1

        offset = 0
        for record_idx, row_idx, positions in refs:
            row_count = int(positions.numel())
            row_selected = selected[offset : offset + row_count]
            masks[record_idx][row_idx, positions[row_selected]] = 1.0
            offset += row_count
        if offset != valid_count:
            raise AssertionError(f"{scope_id}: reconstructed {offset}, expected {valid_count}")
        actual = sum(int(masks[r][i].sum().item()) for r, i in row_refs)
        if actual != selected_count:
            raise AssertionError(f"{scope_id}: selected {actual}, expected {selected_count}")
        total_budget += budget
        total_selected += selected_count
        total_valid += valid_count

    rollout_selected_counts = [
        int(mask[row_idx].sum().item())
        for mask in masks
        for row_idx in range(mask.shape[0])
    ]
    rollout_counts = torch.tensor(rollout_selected_counts, dtype=torch.float32)
    rollout_mean = float(rollout_counts.mean().item()) if rollout_counts.numel() else 0.0
    rollout_cv = (
        float(rollout_counts.std(unbiased=False).item()) / rollout_mean
        if rollout_mean > 0.0
        else 0.0
    )

    diagnostics = {
        "group_count": float(group_count),
        "scope_count": float(len(scopes)),
        "selected": float(total_selected),
        "valid": float(total_valid),
        "budget": float(total_budget),
        "actual_retention": float(total_selected / max(total_valid, 1)),
        "budget_fill": float(total_selected / max(total_budget, 1)),
        # Kept for the base logger; in microbatch mode this is per-scope.
        "zero_eligible_group_fraction": float(zero_eligible_scopes / max(len(scopes), 1)),
        "zero_eligible_scope_fraction": float(zero_eligible_scopes / max(len(scopes), 1)),
        "zero_token_rollout_fraction": (
            float((rollout_counts == 0).float().mean().item())
            if rollout_counts.numel()
            else 0.0
        ),
        "rollout_allocation_cv": rollout_cv,
    }

    queue_items: list[dict[str, Any]] = []
    for record, mask in zip(records, masks, strict=True):
        valid = record["valid"].bool()
        selected = mask.bool()
        local_valid = int(valid.sum().item())
        local_selected = int(selected.sum().item())

        def count_on(key: str, selector) -> int:
            return int((record[key].bool() & selector).sum().item())

        def mean_on(key: str, selector) -> float:
            values = record[key].float()[selector]
            return float(values.mean().item()) if values.numel() else 0.0

        item = {
            "mask": mask,
            "valid_signature": tuple(int(x) for x in valid.sum(dim=1).tolist()),
            "selected": local_selected,
            "valid": local_valid,
            "eligible": count_on("eligible", valid),
            "active": count_on("active", valid),
            "negative_valid": int(((record["advantage"] < 0) & valid).sum().item()),
            "negative_selected": int(
                ((record["advantage"] < 0) & selected).sum().item()
            ),
            "action_in_topk": count_on("action_in_topk", valid),
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
            "parallel_mean": mean_on("parallel", valid),
            "parallel_selected_mean": mean_on("parallel", selected),
            "conflict_valid": count_on("conflict", valid),
            "conflict_selected": count_on("conflict", selected),
            "neutral_valid": count_on("neutral_grpo", valid),
            "neutral_selected": count_on("neutral_grpo", selected),
            "direction_defined_valid": count_on("direction_defined", valid),
            "teacher_zero_valid": count_on("teacher_zero", valid),
            "scope_is_group": float(scope_name == "prompt_group"),
            "scope_count": diagnostics["scope_count"],
            "group_count": diagnostics["group_count"],
            "budget_fill": diagnostics["budget_fill"],
            "zero_eligible_group_fraction": diagnostics[
                "zero_eligible_group_fraction"
            ],
            "zero_eligible_scope_fraction": diagnostics[
                "zero_eligible_scope_fraction"
            ],
            "zero_token_rollout_fraction": diagnostics[
                "zero_token_rollout_fraction"
            ],
            "rollout_allocation_cv": diagnostics["rollout_allocation_cv"],
        }
        queue_items.append(item)
    return queue_items, diagnostics


def _consume_gift_mask(*args, **kwargs):
    queue = getattr(core._STATE, "mask_queue", None)
    if queue is None or not queue:
        raise RuntimeError("GIFT mask queue is empty")
    item = queue[0]
    mask, metrics = _CORE_CONSUMER(*args, **kwargs)
    selected = int(item["selected"])
    valid = max(int(item["valid"]), 1)
    metrics.update(
        {
            "gift/group_scope": float(item["scope_is_group"]),
            "gift/scope_is_group": float(item["scope_is_group"]),
            "gift/scope_count_local": float(item["scope_count"]),
            "gift/conflict_valid_ratio": float(
                item["conflict_valid"] / valid
            ),
            "gift/conflict_selected_ratio": float(
                item["conflict_selected"] / max(selected, 1)
            ),
            "gift/nonconflict_selected_ratio": float(
                (selected - item["conflict_selected"]) / max(selected, 1)
            ),
            "gift/neutral_grpo_valid_ratio": float(
                item["neutral_valid"] / valid
            ),
            "gift/neutral_grpo_selected_ratio": float(
                item["neutral_selected"] / max(selected, 1)
            ),
            "gift/direction_defined_ratio": float(
                item["direction_defined_valid"] / valid
            ),
            "gift/teacher_zero_excluded_ratio": float(
                item["teacher_zero_valid"] / valid
            ),
            "gift/parallel_mean": float(item["parallel_mean"]),
            "gift/parallel_selected_mean": float(
                item["parallel_selected_mean"]
            ),
            "gift/zero_eligible_scope_fraction": float(
                item["zero_eligible_scope_fraction"]
            ),
            "gift/ranking_uses_entropy": 0.0,
            "gift/loss_uses_projected_gradient": 0.0,
            "gift/lambda_is_constant": 1.0,
        }
    )
    return mask, metrics


def apply_runtime_patches() -> None:
    """Install GIFT before delegating actor/loss patching to the base runtime."""
    global _PATCHED
    if _PATCHED or not _env_flag(_ENABLE_ENV, default=False):
        return
    with _LOCK:
        if _PATCHED:
            return
        settings = load_settings()
        base_config = os.environ.get("GIFT_CORE_CONFIG", "").strip()
        gift_config = os.environ.get(_CONFIG_ENV, "").strip()
        if not _env_flag("GIFT_CORE_ENABLE", default=False):
            raise RuntimeError("GIFT requires GIFT_CORE_ENABLE=1")
        if Path(base_config).resolve() != Path(gift_config).resolve():
            raise RuntimeError(
                "the GIFT and core runtimes must use the same config"
            )
        if core._PATCHED:
            raise RuntimeError(
                "the core runtime was patched before GIFT; check PYTHONPATH ordering"
            )

        core.CoreSettings.validate = _validate_base_settings
        core.sparse_tail_gradient_geometry = sparse_tail_gradient_geometry
        core.build_selection_masks = build_gift_masks
        core._consume_selection_mask = _consume_gift_mask
        core.apply_runtime_patches()
        _PATCHED = True
        print(
            "[GIFT] installed isolated overlay: "
            f"scope={settings.selection_scope} group_size={settings.group_size} "
            "score=piecewise(-|parallel| if conflict else residual) "
            f"kappa=1@1..{settings.hold_steps}->0@{settings.exit_step} "
            f"(of {settings.total_training_steps} steps) lambda=1 "
            "entropy=off projected_loss=off",
            file=sys.stderr,
            flush=True,
        )


__all__ = [
    "GiftSettings",
    "apply_runtime_patches",
    "build_gift_masks",
    "load_settings",
    "sparse_tail_gradient_geometry",
]
