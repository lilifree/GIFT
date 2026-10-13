#!/usr/bin/env python3
"""Fail-closed checks for the Qwen3-4B/1.7B GIFT math reproductions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core_runtime import patch as core
from gift_runtime.patch import GiftSettings


HERE = Path(__file__).resolve().parent
REFERENCE_CONFIG = (
    HERE / "qwen3_8b_gift_microbatch_token_anneal.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_FILES = ("config.json", "tokenizer_config.json")


REFERENCE_MODEL = "Qwen3-8B"


def _model_tokens(model_name: str) -> tuple[str, str, str]:
    """Spellings of a model name used in config identifiers, e.g. Qwen3-1.7B ->
    ("Qwen3-1.7B", "qwen3_1_7b", "qwen3-1-7b")."""
    lower = model_name.lower()
    return model_name, lower.replace(".", "_").replace("-", "_"), lower.replace(".", "-")


@dataclass(frozen=True)
class ModelSpec:
    model_name: str

    def scale(self, value: str) -> str:
        """Rewrite an 8B identifier (file, project, experiment or output name) for this model.

        Identifiers are derived from the 8B reference instead of being spelled out here,
        so renaming a run or changing its step count only touches the 8B YAML.
        """
        for ref_token, token in zip(_model_tokens(REFERENCE_MODEL), _model_tokens(self.model_name)):
            value = value.replace(ref_token, token)
        return value

    @property
    def config_name(self) -> str:
        return self.scale(REFERENCE_CONFIG.name)


def resolve_path(value: Any) -> Path:
    """Config paths are relative to the repository root unless absolute."""
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


SPECS = {name: ModelSpec(model_name=name) for name in ("Qwen3-4B", "Qwen3-1.7B")}

MODEL_SPECIFIC_KEYS = {
    "model_path",
    "output_dir",
    "project_name",
    "experiment_name",
}


def read_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"config must be a mapping: {path}")
    return raw


def _resolve_spec(path: Path, raw: dict[str, Any]) -> ModelSpec:
    model_name = Path(str(raw.get("model_path", ""))).name
    try:
        spec = SPECS[model_name]
    except KeyError as exc:
        raise RuntimeError(f"unsupported model for scale reproduction: {model_name!r}") from exc
    if path.name != spec.config_name:
        raise RuntimeError(
            f"config/model mismatch: {path.name!r} != {spec.config_name!r}"
        )
    return spec


def validate_config(path: Path, *, check_persistent: bool = False) -> dict[str, Any]:
    path = path.resolve()
    raw = read_config(path)
    reference = read_config(REFERENCE_CONFIG)
    spec = _resolve_spec(path, raw)
    settings = GiftSettings.from_mapping(raw)
    reference_output = Path(str(reference["output_dir"]))

    common_mismatches = {
        key: {"actual": raw.get(key), "reference_8b": value}
        for key, value in reference.items()
        if key not in MODEL_SPECIFIC_KEYS and raw.get(key) != value
    }
    extra_keys = set(raw) - set(reference)
    missing_keys = set(reference) - set(raw)
    if common_mismatches or extra_keys or missing_keys:
        raise RuntimeError(
            "small-model config must differ from 8B only in model identity fields: "
            f"mismatches={common_mismatches}, extra={sorted(extra_keys)}, "
            f"missing={sorted(missing_keys)}"
        )

    expected_model_values = {
        "project_name": spec.scale(str(reference["project_name"])),
        "experiment_name": spec.scale(str(reference["experiment_name"])),
    }
    model_mismatches = {
        key: {"actual": raw.get(key), "expected": value}
        for key, value in expected_model_values.items()
        if raw.get(key) != value
    }
    # Paths are free-form, but the 4B/1.7B runs must keep the 8B data files,
    # the 8B output root, and a model directory named after the model.
    path_mismatches = {}
    for key in ("train_files", "val_files"):
        if resolve_path(raw[key]) != resolve_path(reference[key]):
            path_mismatches[key] = {
                "actual": str(raw[key]),
                "expected_8b": str(reference[key]),
            }
    model_dir = resolve_path(raw["model_path"])
    if model_dir.name != spec.model_name:
        path_mismatches["model_path"] = {
            "actual": str(raw["model_path"]),
            "expected_dir_name": spec.model_name,
        }
    if Path(str(raw["output_dir"])).parent != reference_output.parent:
        path_mismatches["output_dir"] = {
            "actual": str(raw["output_dir"]),
            "expected_root": str(reference_output.parent),
        }
    expected_output_name = spec.scale(reference_output.name)
    if Path(str(raw["output_dir"])).name != expected_output_name:
        path_mismatches["output_dir_name"] = {
            "actual": Path(str(raw["output_dir"])).name,
            "expected": expected_output_name,
        }
    if model_mismatches or path_mismatches:
        raise RuntimeError(
            "model-specific config mismatch: "
            f"{model_mismatches} {path_mismatches}"
        )

    if settings.selection_scope != "micro_batch":
        raise RuntimeError(f"selection scope must be micro_batch: {settings.selection_scope}")
    # The schedule (H, T_exit, total steps) is config-driven; the field-by-field
    # diff above already pins it to whatever the 8B reference config specifies.
    total_steps = settings.total_training_steps
    hold_steps = settings.hold_steps
    exit_step = settings.exit_step
    initial = settings.initial_retention

    ratios = [core.coverage_ratio(raw, step) for step in range(1, total_steps + 1)]
    if ratios[:hold_steps] != [initial] * hold_steps:
        raise RuntimeError(f"steps 1..{hold_steps} must retain all valid response tokens")
    if any(ratio != 0.0 for ratio in ratios[exit_step - 1 :]):
        raise RuntimeError(f"steps {exit_step}..{total_steps} must fully withdraw the teacher")
    for step in range(hold_steps + 1, exit_step):
        expected_ratio = initial * (exit_step - step) / (exit_step - hold_steps)
        actual = ratios[step - 1]
        if abs(actual - expected_ratio) > 1.0e-12:
            raise RuntimeError(
                f"unexpected retention at step {step}: {actual} != {expected_ratio}"
            )
    if any(right > left + 1.0e-12 for left, right in zip(ratios, ratios[1:])):
        raise RuntimeError("token budget must be monotone non-increasing")
    if any(core.opsd_lambda(raw, step) != 1.0 for step in range(1, total_steps + 1)):
        raise RuntimeError("OPSD lambda must stay fixed at 1.0")

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
    raw = validate_config(args.config, check_persistent=args.check_persistent)
    settings = GiftSettings.from_mapping(raw)
    model_name = Path(str(raw["model_path"])).name
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
        "[GIFT-MODEL-SCALE-MATH] verified: "
        f"model={model_name} scope=micro_batch group=8 lambda=1 "
        f"val_before_train=false retention({samples})"
    )


if __name__ == "__main__":
    main()
