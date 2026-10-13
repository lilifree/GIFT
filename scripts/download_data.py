"""Download GIFT training/eval parquet files from a HuggingFace dataset repo.

The verl launcher (`scripts/_gpu_run_verl.sh`) expects each dataset to live at
`data/gift/<DATASET_NAME>/{train,val}.parquet`. This script mirrors the upstream
HF repo layout into that directory.

Example:
    # download every dataset referenced by the shipped configs
    python scripts/download_data.py --all

    # only the kk_4to8 train split
    python scripts/download_data.py --dataset kk_4to8 --split train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "Leyiii/RLCSD"
# Anchored at the repository root, where the configs resolve their relative paths.
DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parents[1] / "data" / "gift"

# (dataset_name, split). dataset_name matches `train_dataset` / `val_dataset` in configs.
DATASETS = [
    ("deepmath_filtered_level5_7",            "train"),  # Qwen3-1.7B math
    ("deepmath_filtered_level6_8",            "train"),  # Qwen3-4B math
    ("deepmath_filtered_level7_10",           "train"),  # Qwen3-8B + Olmo math
    ("amc23+aime24+aime25",                   "val"),    # math eval
    ("kk_4to8",                               "train"),  # logic train
    ("kk_4to8_test+kk_9+kk_10+kk_11",         "val"),    # logic eval
]


def download(repo_id: str, dataset_name: str, split: str, dst_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    remote_path = f"{dataset_name}/{split}.parquet"
    local_dir = dst_root / dataset_name
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        repo_type="dataset",
        local_dir=str(dst_root),
    )
    return Path(local_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo to pull from")
    p.add_argument("--dst", default=str(DEFAULT_LOCAL_ROOT), help="Local data root (default <repo root>/data/gift)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="Download every dataset")
    g.add_argument("--dataset", help="Single dataset name (e.g. kk_4to8)")
    p.add_argument("--split", choices=("train", "val"), help="Split to download (with --dataset)")
    args = p.parse_args()

    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    if args.all or (not args.dataset):
        targets = DATASETS
    else:
        if args.split is None:
            print("--split is required when --dataset is given", file=sys.stderr)
            return 2
        targets = [(args.dataset, args.split)]

    for name, split in targets:
        try:
            path = download(args.repo, name, split, dst_root)
            print(f"OK   {name}/{split}.parquet -> {path}")
        except Exception as exc:
            print(f"FAIL {name}/{split}.parquet: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
