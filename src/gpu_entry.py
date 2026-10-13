"""GPU entry shim for RLCSD self-distillation training.

NEW file — original src/self_distill_main.py is untouched. Reimplements only the
tiny `__main__` block of self_distill_main.py (which lives behind
`if __name__ == "__main__"` and so is not importable), but imports
`src.self_distill_main` as a NORMAL module first.

Why: on this GPU box, `python -m src.self_distill_main` runs the module as
`__main__`, so `SelfDistillTaskRunner.__module__` becomes `'__main__'`. Ray then
pickles the actor class BY VALUE; the worker reconstructs it via
cloudpickle._make_skeleton_class, which hits a metaclass conflict (and, as a
side effect, imports verl.utils.device at the wrong time, producing the
"changing CUDA_VISIBLE_DEVICES after program start" warning). Importing the
module normally keeps `__module__='src.self_distill_main'` so Ray serializes
the actor BY REFERENCE and the worker simply imports it.

Usage (same hydra overrides as src.self_distill_main):
    python -m src.gpu_entry <hydra overrides...>
"""

# Importing the module (not running it as __main__) keeps SelfDistillTaskRunner's
# __module__ correct and runs _apply_patches().
import src.self_distill_main  # noqa: F401  (side effects: monkey-patches)

import hydra
import ray

from src.gpu_runner import GpuSelfDistillTaskRunner
from src.self_distill_main import _ensure_legacy_worker_impl
from verl.trainer.main_ppo import migrate_legacy_reward_impl, run_ppo
from verl.utils.device import auto_set_device


@hydra.main(
    config_path="../third_party/verl/verl/trainer/config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    config = _ensure_legacy_worker_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(GpuSelfDistillTaskRunner))


if __name__ == "__main__":
    main()