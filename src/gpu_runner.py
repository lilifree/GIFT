"""GPU task runner for RLCSD.

NEW file — nothing original is modified. Defines a SelfDistillTaskRunner
subclass in an importable module so Ray serializes the actor by reference
(a class defined in the `python -m` entry module would get __module__='__main__'
and be pickled by value, which crashes the worker on reconstruction — see
src/gpu_entry.py). Also applies a GPU-specific runtime guard inside the worker.
"""

import os
from pathlib import Path
import shutil
import time
import traceback

import src.self_distill_main  # noqa: F401  (side effects: _apply_patches)

from src.self_distill_main import SelfDistillTaskRunner
from verl.trainer.ppo import metric_utils as _metric_utils
from verl.trainer.ppo import ray_trainer as _ray_trainer


class GpuSelfDistillTaskRunner(SelfDistillTaskRunner):
    """SelfDistillTaskRunner + a divide-by-zero guard on timing metrics.

    RLCSD drops whole rollout groups that lack a correct (or an incorrect)
    sibling. On a cold model the first steps can yield an empty actor batch;
    verl's compute_timing_metrics then divides by num_response_tokens=0 and
    crashes trainer.fit(). An empty step should be skipped, not fatal.
    """

    def run(self, config):
        _orig_ctm = _metric_utils.compute_timing_metrics

        def _safe_ctm(batch, timing_raw):
            try:
                return _orig_ctm(batch, timing_raw)
            except ZeroDivisionError:
                return {}

        _ray_trainer.compute_timing_metrics = _safe_ctm
        _metric_utils.compute_timing_metrics = _safe_ctm
        try:
            result = super().run(config)
        except BaseException:
            # Keep the original training traceback even if persistent storage
            # itself is temporarily unavailable during best-effort archival.
            try:
                self._archive_actor_logs()
            except BaseException:
                print("ERROR: failed to archive Ray TaskRunner logs")
                traceback.print_exc()
            raise
        else:
            # Archive closed actor-owned files without allowing a transient
            # logging-path outage to invalidate a completed training run.
            self._archive_actor_logs()
            return result

    @staticmethod
    def _archive_actor_logs(attempts: int = 5) -> bool:
        """Best-effort copy of closed actor-owned logs to persistence."""
        local_dir_value = os.environ.get("VERL_RUN_LOG_DIR")
        archive_root_value = os.environ.get("VERL_LOG_ARCHIVE_DIR")
        if not local_dir_value or not archive_root_value:
            return True

        local_dir = Path(local_dir_value)
        archive_target = Path(archive_root_value) / local_dir.name
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                archive_target.mkdir(parents=True, exist_ok=True)
                for name in ("metrics.jsonl", "tensorboard_log", "rollout_details"):
                    source = local_dir / name
                    if not source.exists():
                        continue
                    destination = archive_target / name
                    if source.is_dir():
                        shutil.copytree(source, destination, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source, destination)
                print(f"Archived Ray TaskRunner logs: {archive_target}", flush=True)
                return True
            except OSError as exc:
                last_error = exc
                print(
                    "Warning: Ray TaskRunner log archive failed "
                    f"(attempt {attempt}/{attempts}): {exc}",
                    flush=True,
                )
                if attempt < attempts:
                    time.sleep(attempt)

        print(
            "Warning: continuing after actor-log archive failure; training "
            f"status is unchanged: {last_error}",
            flush=True,
        )
        return False
