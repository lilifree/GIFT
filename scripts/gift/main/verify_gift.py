#!/usr/bin/env python3
"""Fail-closed validation for the GIFT configs.

Hyperparameters are compared exactly.  Path keys are checked only for presence and,
with ``--check-persistent``, for existence: the YAML is the single source of truth for
where the weights, the parquet files and the output directory live, so a checkout can
relocate them by editing the YAML alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from core_runtime import patch as core
from gift_runtime.patch import GiftSettings


REPO_ROOT = Path(__file__).resolve().parents[3]
PATH_KEYS = ("model_path", "train_files", "val_files", "output_dir")
MODEL_FILES = ("config.json", "tokenizer_config.json")


def resolve_path(value: Any) -> Path:
    """Config paths are relative to the repository root unless absolute."""
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"config must be a mapping: {path}")
    return raw


def validate_config(path: Path, *, check_persistent: bool = False) -> dict[str, Any]:
    raw = read_config(path)
    settings = GiftSettings.from_mapping(raw)

    # Path keys are free-form (relative to the repo root, or absolute).
    missing_paths = [key for key in PATH_KEYS if not str(raw.get(key, "")).strip()]
    if missing_paths:
        raise RuntimeError(f"GIFT config is missing path keys: {missing_paths}")

    # Run identifiers (project/experiment names) and the step schedule are free;
    # only the method hyperparameters are pinned.
    expected = {
        "gift_selection_scope": "micro_batch",
        "seed": 42,
        "per_device_batch_size": 32,
        "group_size": 8,
        "ppo_mini_batch_size": 64,
        "max_train_samples": 60000,
        "max_prompt_length": 2048,
        "max_completion_length": 16384,
        "actor_max_token_len_per_gpu": 24576,
        "save_steps": 10,
        "eval_steps": 10,
        "val_before_train": False,
        "opsd_grpo_lambda": 1.0,
        "beta": 0.0,
        "jsd_token_clip": 0.0,
        "top_k_distill": 100,
        "gift_initial_retention": 1.0,
        "gift_cosine_min": 0.0,
        "gift_piecewise_zero_grpo_policy": "neutral_score_zero",
        "gift_piecewise_zero_opsd_policy": "exclude",
        "gift_support": "teacher_topk_plus_tail_exact_full_logit_geometry",
    }
    mismatches = {
        key: {"actual": raw.get(key), "expected": value}
        for key, value in expected.items()
        if raw.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"GIFT config mismatch: {mismatches}")

    # The schedule itself (H, T_exit, total steps) is config-driven -- check the
    # closed-form kappa_t (Eq. 10) holds for whatever values this config picks,
    # rather than pinning one specific step count.
    total_steps = settings.total_training_steps
    hold_steps = settings.hold_steps
    exit_step = settings.exit_step
    initial = settings.initial_retention

    ratios = [core.coverage_ratio(raw, step) for step in range(1, total_steps + 1)]
    if ratios[:hold_steps] != [initial] * hold_steps:
        raise RuntimeError(f"steps 1..{hold_steps} must use a full token budget")
    if any(ratio != 0.0 for ratio in ratios[exit_step - 1 :]):
        raise RuntimeError(f"steps {exit_step}..{total_steps} must fully withdraw the teacher")
    for step in range(hold_steps + 1, exit_step):
        expected_ratio = initial * (exit_step - step) / (exit_step - hold_steps)
        if abs(ratios[step - 1] - expected_ratio) > 1.0e-12:
            raise RuntimeError(
                f"unexpected retention at step {step}: {ratios[step - 1]} != {expected_ratio}"
            )
    if any(right > left + 1.0e-12 for left, right in zip(ratios, ratios[1:])):
        raise RuntimeError("GIFT token budget must be monotone non-increasing")
    if any(core.opsd_lambda(raw, step) != 1.0 for step in range(1, total_steps + 1)):
        raise RuntimeError("GIFT OPSD lambda must remain exactly 1.0")

    if check_persistent:
        model_dir = resolve_path(raw["model_path"])
        required = [model_dir / name for name in MODEL_FILES]
        required += [resolve_path(raw["train_files"]), resolve_path(raw["val_files"])]
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise RuntimeError(
                "missing model/data files referenced by the config: " + ", ".join(missing)
            )
        output_dir = resolve_path(raw["output_dir"])
        if output_dir.exists():
            raise RuntimeError(f"output directory already exists: {output_dir}")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-persistent", action="store_true")
    args = parser.parse_args()
    raw = validate_config(args.config.resolve(), check_persistent=args.check_persistent)
    settings = GiftSettings.from_mapping(raw)
    mid_step = (settings.hold_steps + settings.exit_step) // 2
    sample_steps = sorted(
        {
            1,
            settings.hold_steps,
            settings.hold_steps + 1,
            mid_step,
            settings.exit_step - 1,
            settings.exit_step,
            settings.total_training_steps,
        }
    )
    samples = ", ".join(
        f"s{step}={core.coverage_ratio(raw, step):.6f}" for step in sample_steps
    )
    print(
        "[GIFT] verified: "
        f"scope={raw['gift_selection_scope']} group=8 lambda=1 "
        f"val_before_train=false retention({samples})"
    )


if __name__ == "__main__":
    main()
