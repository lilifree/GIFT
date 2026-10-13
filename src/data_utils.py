"""Dataset loading, preprocessing, and privileged context construction."""

import json
import os
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset, concatenate_datasets

from src.opsd_format import extract_boxed_answer, grade_boxed_answer, strip_legacy_math_prompt


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_openthoughts_114k_math_filtered(data_dir: Optional[str] = None) -> Dataset:
    """Load the locally filtered OpenThoughts math training dataset."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    local_path = data_dir / "openthoughts_114k_math_filtered"
    if local_path.exists():
        return Dataset.load_from_disk(str(local_path))

    raise FileNotFoundError(
        f"Filtered OpenThoughts dataset not found at {local_path}. "
        "Create it first with `python scripts/filter_openthoughts_math.py`."
    )


def load_dapo_math_17k(data_dir: Optional[str] = None) -> Dataset:
    """Load DAPO-Math-17k training dataset.

    Format: {data_source, prompt: [{content, role}], ability,
             reward_model: {ground_truth, style}, extra_info: {index}}
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    local_path = data_dir / "dapo_math_17k"
    if local_path.exists():
        ds = Dataset.load_from_disk(str(local_path))
    else:
        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds.save_to_disk(str(local_path))
    return ds


def load_amc23(data_dir: Optional[str] = None) -> Dataset:
    """Load AMC 2022-2023 test dataset.

    Format: {id, problem, answer (float), url}
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    local_path = data_dir / "amc23"
    if local_path.exists():
        ds = Dataset.load_from_disk(str(local_path))
    else:
        ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
        ds.save_to_disk(str(local_path))
    return ds


def load_aime24(data_dir: Optional[str] = None) -> Dataset:
    """Load AIME 2024 test dataset.

    Uses AI-MO/aimo-validation-aime, filtered for 2024 problems.
    Format: {id, problem, answer (float), url}
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    local_path = data_dir / "aime24"
    if local_path.exists():
        ds = Dataset.load_from_disk(str(local_path))
    else:
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        # Filter for 2024 by URL
        ds_2024 = ds.filter(lambda x: "2024" in str(x.get("url", "")))
        if len(ds_2024) == 0:
            # Fallback: use all AIME data
            print("Warning: Could not filter AIME 2024, using all AIME data")
            ds_2024 = ds
        ds_2024.save_to_disk(str(local_path))
        ds = ds_2024
    return ds


def load_aime25(data_dir: Optional[str] = None) -> Dataset:
    """Load AIME 2025 test dataset.

    Uses opencompass/AIME2025. Format: {question, answer}
    Two configs: AIME2025-I (15) and AIME2025-II (15), total 30 problems.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    local_path = data_dir / "aime25"
    if local_path.exists():
        ds = Dataset.load_from_disk(str(local_path))
    else:
        try:
            ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
            ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
            ds = concatenate_datasets([ds1, ds2])
        except Exception:
            # Fallback: try different dataset source
            try:
                ds = load_dataset("MathArena/aime_2025", split="train")
            except Exception:
                ds = load_dataset("yentinglin/aime_2025", split="train")
        ds.save_to_disk(str(local_path))
    return ds


def load_deepmath_filtered(data_dir: Optional[str] = None) -> Dataset:
    """Load the locally filtered DeepMath training dataset."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    local_path = data_dir / "deepmath_filtered"
    if local_path.exists():
        return Dataset.load_from_disk(str(local_path))

    raise FileNotFoundError(
        f"Filtered DeepMath dataset not found at {local_path}. "
        "Create it first with `python scripts/filter_deepmath.py`."
    )


def load_kk_dataset(name: str, data_dir: Optional[str] = None) -> Dataset:
    """Load a locally generated Knights & Knaves dataset by short name.

    Supported names: kk_3to7, kk_3to7_test, kk_8, kk_9, kk_10.
    Generate with `python scripts/generate_kk.py`.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    local_path = data_dir / name
    if local_path.exists():
        return Dataset.load_from_disk(str(local_path))
    raise FileNotFoundError(
        f"K&K dataset not found at {local_path}. "
        "Generate it first with `python scripts/generate_kk.py`."
    )


def load_training_dataset(dataset_name: str, data_dir: Optional[str] = None) -> Dataset:
    """Load a supported training dataset by name."""
    if dataset_name == "dapo_math_17k":
        return load_dapo_math_17k(data_dir)
    if dataset_name == "openthoughts_114k_math_filtered":
        return load_openthoughts_114k_math_filtered(data_dir)
    if dataset_name == "deepmath_filtered":
        return load_deepmath_filtered(data_dir)
    if dataset_name in ("kk_3to7", "kk_4to8"):
        return load_kk_dataset(dataset_name, data_dir)

    raise ValueError(
        f"Unsupported training dataset: {dataset_name}. "
        "Supported values: dapo_math_17k, openthoughts_114k_math_filtered, "
        "deepmath_filtered, kk_3to7, kk_4to8"
    )


def extract_answer_from_boxed(text: str) -> Optional[str]:
    """Extract answer from the final \\boxed{...} expression."""
    return extract_boxed_answer(text)


def normalize_answer(answer: str) -> str:
    """Normalize answer string for light-weight display or debugging."""
    if answer is None:
        return ""
    return str(answer).strip().strip("$").strip()


def check_answer(prediction: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    return grade_boxed_answer(prediction, ground_truth)


def normalize_dataset(ds: Dataset, dataset_name: str) -> list[dict]:
    """Normalize dataset to unified format: [{problem, answer, solution, source}, ...]"""
    results = []
    for item in ds:
        solution = ""
        if dataset_name == "dapo_math_17k":
            # DAPO format: prompt is a list of {content, role}
            prompt_data = item.get("prompt", [])
            if isinstance(prompt_data, list) and len(prompt_data) > 0:
                # Extract content from last message (user message)
                problem = prompt_data[-1].get("content", "")
            else:
                problem = str(prompt_data)
            problem = strip_legacy_math_prompt(problem)

            # Answer is in reward_model.ground_truth
            reward_model = item.get("reward_model", {})
            if isinstance(reward_model, dict):
                answer = reward_model.get("ground_truth", "")
            else:
                answer = str(reward_model)
            extra_info = item.get("extra_info", {})
            if isinstance(extra_info, dict):
                solution = str(extra_info.get("solution", ""))

        elif dataset_name == "openthoughts_114k_math_filtered":
            problem = strip_legacy_math_prompt(item.get("problem", ""))
            answer = str(item.get("answer", ""))
            solution = str(item.get("solution", ""))

        elif dataset_name == "deepmath_filtered":
            problem = strip_legacy_math_prompt(item.get("problem", ""))
            answer = str(item.get("answer", ""))
            solution = str(item.get("solution", ""))

        elif dataset_name in ("kk_3to7", "kk_3to7_test", "kk_4to8", "kk_4to8_test",
                              "kk_8", "kk_9", "kk_10", "kk_11"):
            # K&K HF format: {problem, solution, answer}. The puzzle text already
            # contains the boxed-answer instruction, so do NOT strip it via the
            # math prompt stripper (which only removes the math-specific suffix).
            problem = str(item.get("problem", "")).strip()
            answer = str(item.get("answer", ""))
            solution = str(item.get("solution", ""))

        elif dataset_name in ("amc23", "aime24"):
            # AI-MO format: {problem, answer (float), url}
            problem = strip_legacy_math_prompt(item.get("problem", ""))
            answer = str(item.get("answer", ""))
            solution = str(item.get("solution", ""))

        elif dataset_name == "aime25":
            # opencompass format: {question, answer}
            problem = strip_legacy_math_prompt(item.get("question", item.get("problem", "")))
            answer = str(item.get("answer", ""))

        else:
            problem = item.get("problem", item.get("question", item.get("prompt", "")))
            answer = str(item.get("answer", item.get("solution", "")))

        if problem:
            results.append({
                "problem": problem.strip(),
                "answer": str(answer).strip(),
                "solution": str(solution).strip(),
                "source": dataset_name,
            })
    return results


def prepare_training_data(
    data_dir: Optional[str] = None,
    dataset_name: str = "dapo_math_17k",
) -> list[dict]:
    """Load and prepare a supported training dataset."""
    ds = load_training_dataset(dataset_name, data_dir)
    return normalize_dataset(ds, dataset_name)


def prepare_eval_data(data_dir: Optional[str] = None) -> dict[str, list[dict]]:
    """Load and prepare all evaluation datasets."""
    eval_sets = {}

    try:
        ds = load_amc23(data_dir)
        eval_sets["amc23"] = normalize_dataset(ds, "amc23")
    except Exception as e:
        print(f"Warning: Failed to load AMC23: {e}")

    try:
        ds = load_aime24(data_dir)
        eval_sets["aime24"] = normalize_dataset(ds, "aime24")
    except Exception as e:
        print(f"Warning: Failed to load AIME24: {e}")

    try:
        ds = load_aime25(data_dir)
        eval_sets["aime25"] = normalize_dataset(ds, "aime25")
    except Exception as e:
        print(f"Warning: Failed to load AIME25: {e}")

    return eval_sets
