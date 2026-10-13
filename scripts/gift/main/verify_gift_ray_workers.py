#!/usr/bin/env python3
"""Verify the GIFT overlay, paths, and config on every Ray node."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--expected-nodes", required=True, type=int)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config = args.config.resolve()
    gift_dir = repo_root / "scripts/gift/main"
    required = (
        config,
        repo_root / "third_party/verl/verl/__init__.py",
        gift_dir / "gift_runtime/__init__.py",
        gift_dir / "gift_runtime/patch.py",
        gift_dir / "core_runtime/__init__.py",
        gift_dir / "core_runtime/patch.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Ray GIFT preflight source is missing: " + ", ".join(missing))

    worker_env = {
        "PYTHONPATH": str(os.environ.get("PYTHONPATH", "")),
        "GIFT_ENABLE": "1",
        "GIFT_CONFIG": str(config),
        "GIFT_CORE_ENABLE": "1",
        "GIFT_CORE_CONFIG": str(config),
    }
    runtime_env: dict[str, object] = {
        "env_vars": worker_env,
        "worker_process_setup_hook": (
            "gift_runtime.patch.apply_runtime_patches"
        ),
    }

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(
        address=args.address,
        ignore_reinit_error=True,
        log_to_driver=True,
        runtime_env=runtime_env,
    )
    try:
        nodes = sorted(
            (node for node in ray.nodes() if node.get("Alive")),
            key=lambda node: node["NodeID"],
        )
        if len(nodes) != args.expected_nodes:
            raise RuntimeError(
                f"expected {args.expected_nodes} live Ray nodes, found {len(nodes)}"
            )

        @ray.remote(num_cpus=0, max_calls=1)
        def verify_worker(expected_repo: str, expected_config: str) -> dict[str, str]:
            from pathlib import Path as WorkerPath
            import os as worker_os

            import verl
            import core_runtime
            import gift_runtime
            from core_runtime import patch as core
            from gift_runtime import patch as gift
            from verl.trainer.ppo import core_algos
            from verl.trainer.ppo.core_algos import get_policy_loss_fn
            from verl.workers.actor.dp_actor import DataParallelPPOActor

            gift.apply_runtime_patches()
            expected_root = WorkerPath(expected_repo).resolve()
            actual_verl = WorkerPath(verl.__file__).resolve()
            expected_verl = expected_root / "third_party/verl"
            if expected_verl not in actual_verl.parents:
                raise RuntimeError(
                    f"wrong verl import: expected under {expected_verl}, got {actual_verl}"
                )
            expected_gift = expected_root / "scripts/gift/main"
            actual_gift = WorkerPath(gift_runtime.__file__).resolve().parent.parent
            if actual_gift != expected_gift:
                raise RuntimeError(f"wrong GIFT runtime: {actual_gift} != {expected_gift}")
            expected_core = expected_root / "scripts/gift/main"
            actual_core = WorkerPath(core_runtime.__file__).resolve().parent.parent
            if actual_core != expected_core:
                raise RuntimeError(f"wrong core runtime: {actual_core} != {expected_core}")
            actual_config = WorkerPath(
                worker_os.environ["GIFT_CONFIG"]
            ).resolve()
            if actual_config != WorkerPath(expected_config).resolve():
                raise RuntimeError(
                    f"wrong GIFT config: {actual_config} != {expected_config}"
                )

            settings = gift.load_settings()
            loss_fn = get_policy_loss_fn("opsd_grpo_curriculum")
            if settings.group_size != 8:
                raise RuntimeError(f"runtime contract mismatch: {settings}")
            if core.sparse_tail_gradient_geometry is not gift.sparse_tail_gradient_geometry:
                raise RuntimeError("GIFT geometry override is not active")
            if core.build_selection_masks is not gift.build_gift_masks:
                raise RuntimeError("GIFT scope selector override is not active")
            if not getattr(core_algos, "_gift_overlay", False):
                raise RuntimeError("core overlay is not active")
            if not getattr(DataParallelPPOActor, "_gift_core_overlay", False):
                raise RuntimeError("actor overlay is not active")
            if not getattr(loss_fn, "_is_opsd_grpo_overlay", False):
                raise RuntimeError("GIFT policy loss wrapper is not registered")
            if core.opsd_lambda({}, 1) != 1.0:
                raise RuntimeError("GIFT lambda is not fixed at one")
            if core.coverage_ratio({}, settings.exit_step) != 0.0:
                raise RuntimeError("GIFT exit step is not a zero-token exit")
            return {
                "node": worker_os.uname().nodename,
                "verl": str(actual_verl),
                "gift_runtime": str(actual_gift),
                "scope": settings.selection_scope,
                "overlay": "on",
            }

        tickets = [
            verify_worker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False
                )
            ).remote(str(repo_root), str(config))
            for node in nodes
        ]
        for result in ray.get(tickets):
            print(f"[GIFT-ray-worker] ok: {result}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    sys.exit(main() or 0)
