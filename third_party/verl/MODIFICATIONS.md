# Modifications to the vendored verl

This directory is a copy of [verl](https://github.com/volcengine/verl) (version `0.8.0.dev`,
Apache-2.0, see `LICENSE` and `Notice.txt`), vendored so that the GIFT experiments in
`scripts/gift/` and the training entry points in `src/` can run without patching an installed
package. The upstream license and copyright notices are retained unchanged.

**This copy is not pristine.** The files below carry additions made by the GIFT authors. The
GIFT method itself is implemented as a runtime overlay (`scripts/gift/main/gift_runtime/` and
`scripts/gift/main/core_runtime/`) that patches these hooks in memory at import time; the
changes listed here are the hooks it attaches to, plus the baselines that the paper compares
against.

## Files that differ from upstream

| File | What was added |
|---|---|
| `verl/trainer/ppo/core_algos.py` | Policy-loss implementations and their registry entries: `opsd`, `opsd_ectr`, `opsd_grpo_curriculum` (the GIFT base), `tip_opsd_grpo_curriculum` and `tip_soft_or_opsd_grpo_curriculum` (the TIP baselines of the paper, Sec. 2 / Table 3), `rlsd`, `rlsd_ectr`, `rlcsd`, `sdpo`, `srpo`. Plus the shared helpers `_sd_cfg` / `_sd_bool_cfg` / `_sd_int_cfg`, `_sd_prepare_sparse_topk_log_probs`, `_sd_prepare_distill_log_probs`, `_sd_generalized_jsd`, `_sd_teacher_metrics`. |
| `verl/workers/actor/dp_actor.py` | The actor-side plumbing the overlays need: `use_prefix_grouper` (keeps GRPO group ids intact across dynamic micro-batches), `calculate_sum_pi_squared` (the `Σπ²` sufficient statistic used by the exact sparse gradient geometry), `_teacher_forward`, sparse teacher top-k capture, and the per-loss-mode teacher-activation dispatch. |
| `verl/workers/config/actor.py` | The extra `PolicyLossConfig` fields those losses read (sparse-distillation switches, the `tip_*` schedule keys, `use_prefix_grouper`, `calculate_sum_pi_squared`). |
| `verl/utils/torch_functional.py` | `calculate_sum_pi_squared_from_logits`, an exact `Σπ²` from logits. |
| `verl/trainer/ppo/ray_trainer.py`, `verl/workers/fsdp_workers.py`, `verl/trainer/ppo/metric_utils.py`, `verl/utils/metric/utils.py`, `verl/workers/utils/losses.py`, `verl/trainer/config/**` | Wiring: config plumbing, metric aggregation and dispatch for the modes above. |

Two identifiers worth knowing when reading the paper alongside this tree:

* **`rlcsd` / `rlcsd_*`** — the RLCSD baseline (Pan et al., 2026). Implemented here because the
  GIFT overlays reuse its teacher-context construction and contrastive evidence ratio.
* **`tip_*`** — the TIP baseline (Xu et al., 2026), whose token-selection criterion is the `Q3`
  score `divergence × (1 − normalized entropy)` used in the paper's Table 3. `tip_soft_or_*` is
  the same family's Soft-OR→Q3 mixing variant.

The `Q3` name is the paper's, not an internal codename: the paper cites the criterion as
"the Q3 criterion in TIP". Earlier revisions of this tree used `tip_q3_*` and `tip_mix_*`; both
were renamed to `tip_*` and `tip_soft_or_*` before release, and no trace of the old names
remains.
