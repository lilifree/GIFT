#!/usr/bin/env bash
# Launch either GIFT selection scope on any GPU layout.  All ranking code is
# injected at runtime; project and vendored verl sources remain untouched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${GIFT_CONFIG:-${1:-}}"
if [[ -z "${CONFIG}" ]]; then
    echo "Pass a GIFT YAML config as the first argument" >&2
    exit 1
fi
if [[ $# -gt 0 && "$1" == *.yaml ]]; then
    shift
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "GIFT config does not exist: ${CONFIG}" >&2
    exit 1
fi
CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"

read -r GROUP_SIZE CONFIG_MAX_STEPS SELECTION_SCOPE CONFIG_NNODES CONFIG_GPUS_PER_NODE PPO_MINI_BS MODEL_NAME < <(
    python3 - "${CONFIG}" <<'PY'
from pathlib import Path
import sys
import yaml

values = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    values.get("group_size", ""),
    values.get("gift_total_training_steps", ""),
    values.get("gift_selection_scope", ""),
    values.get("nnodes", 1),
    values.get("n_gpus_per_node", 8),
    values.get("ppo_mini_batch_size", ""),
    Path(str(values.get("model_path", ""))).name or "model",
)
PY
)
MAX_STEPS="${GIFT_MAX_STEPS:-${CONFIG_MAX_STEPS}}"
if [[ "${GROUP_SIZE}" != "8" ]]; then
    echo "GIFT requires group_size=8, got ${GROUP_SIZE}" >&2
    exit 1
fi
if [[ -z "${MAX_STEPS}" ]]; then
    echo "GIFT requires gift_total_training_steps in the config (or GIFT_MAX_STEPS)" >&2
    exit 1
fi
if [[ "${MAX_STEPS}" != "${CONFIG_MAX_STEPS}" ]]; then
    echo "GIFT_MAX_STEPS=${MAX_STEPS} must equal gift_total_training_steps=${CONFIG_MAX_STEPS}" >&2
    exit 1
fi
if [[ "${SELECTION_SCOPE}" != "prompt_group" && "${SELECTION_SCOPE}" != "micro_batch" ]]; then
    echo "Invalid GIFT selection scope: ${SELECTION_SCOPE}" >&2
    exit 1
fi

export GIFT_ENABLE=1
export GIFT_CONFIG="${CONFIG}"
export GIFT_CORE_ENABLE=1
export GIFT_CORE_CONFIG="${CONFIG}"

export PYTHONPATH="${SCRIPT_DIR}/gift_bootstrap:${SCRIPT_DIR}:${REPO_ROOT}/third_party/verl:${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

cd "${REPO_ROOT}"

NODE_RANK="${NODE_RANK:-${RANK:-0}}"
# Topology comes from the config (nnodes, n_gpus_per_node); NUM_MACHINES / NUM_GPUS override it.
NNODES="${NUM_MACHINES:-${CONFIG_NNODES}}"
GPUS_PER_NODE="${NUM_GPUS:-${CONFIG_GPUS_PER_NODE}}"
RAY_HEAD_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_GCS_PORT="${RAY_GCS_PORT:-${MASTER_PORT:-6379}}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/gift-ray}"
# Leave the object store size to Ray unless RAY_OBJECT_STORE_MEMORY (bytes) is set.
RAY_STORE_ARGS=()
if [[ -n "${RAY_OBJECT_STORE_MEMORY:-}" ]]; then
    RAY_STORE_ARGS=(--object-store-memory="${RAY_OBJECT_STORE_MEMORY}")
fi

if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ && "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "nnodes and n_gpus_per_node must be positive integers, got ${NNODES} x ${GPUS_PER_NODE}" >&2
    exit 1
fi
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
# Each actor rank must receive whole prompt groups (GIFT selection and verl's
# PrefixGrouper balancing both rely on it), so the GPU count must divide the
# mini-batch prompt count.
if [[ -n "${PPO_MINI_BS}" ]] && (( PPO_MINI_BS % TOTAL_GPUS != 0 )); then
    echo "ppo_mini_batch_size=${PPO_MINI_BS} must be divisible by the total GPU count ${NNODES} x ${GPUS_PER_NODE} = ${TOTAL_GPUS}" >&2
    exit 1
fi

echo "[GIFT] node=${NODE_RANK}/${NNODES} scope=${SELECTION_SCOPE} config=${CONFIG}"

PYTHON_SITE_DIR="$(python3 - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
RAY_IMPORT_PTH="${PYTHON_SITE_DIR}/zzzz_gift_${BASHPID}.pth"
python3 - "${REPO_ROOT}" "${RAY_IMPORT_PTH}" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
hook_path = Path(sys.argv[2])
gift_dir = repo_root / "scripts/gift/main"
import_paths = [
    gift_dir / "gift_bootstrap",
    gift_dir,
    repo_root / "third_party/verl",
    repo_root,
]
for path in import_paths:
    if not path.is_dir():
        raise RuntimeError(f"required import path is missing: {path}")
hook_path.write_text(
    f"import sys; sys.path[:0] = {[str(path) for path in import_paths]!r}\n",
    encoding="utf-8",
)
print(f"Installed temporary GIFT Ray import path: {hook_path}")
PY
trap 'rm -f -- "${RAY_IMPORT_PTH}"' EXIT

if [[ "${NODE_RANK}" == "0" && "${GIFT_RUN_PREFLIGHT:-1}" == "1" ]]; then
    python3 "${SCRIPT_DIR}/verify_gift.py" \
        --config "${CONFIG}" --check-persistent
    python3 "${SCRIPT_DIR}/test_gift.py"
    python3 - "${REPO_ROOT}/third_party/verl" "${SCRIPT_DIR}" <<'PY'
from pathlib import Path
import sys

expected_verl = Path(sys.argv[1]).resolve()
expected_gift = Path(sys.argv[2]).resolve()
import verl
import core_runtime
import gift_runtime
from core_runtime import patch as core
from gift_runtime import patch as gift
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.actor.dp_actor import DataParallelPPOActor

gift.apply_runtime_patches()
actual_verl = Path(verl.__file__).resolve()
actual_core = Path(core_runtime.__file__).resolve().parent.parent
actual_gift = Path(gift_runtime.__file__).resolve().parent.parent
if expected_verl not in actual_verl.parents:
    raise RuntimeError(f"wrong verl: expected under {expected_verl}, got {actual_verl}")
if actual_core != expected_gift:
    raise RuntimeError(f"wrong core runtime: {actual_core} != {expected_gift}")
if actual_gift != expected_gift:
    raise RuntimeError(f"wrong GIFT runtime: {actual_gift} != {expected_gift}")
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
print(
    f"[GIFT] local runtime verified: verl={actual_verl} "
    f"scope={settings.selection_scope} steps={settings.total_training_steps} overlay=on"
)
PY
fi

RAY_ENV_ARGS=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GIFT_ENABLE='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GIFT_CONFIG='${CONFIG}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GIFT_CORE_ENABLE='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GIFT_CORE_CONFIG='${CONFIG}'"
    "+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=gift_runtime.patch.apply_runtime_patches"
)

python3 - "${RAY_ENV_ARGS[@]}" <<'PY'
from omegaconf import OmegaConf
import sys

dotlist = [
    arg[1:] if arg.startswith("+") else arg
    for arg in sys.argv[1:]
    if ".runtime_env.env_vars." in arg
]
config = OmegaConf.from_dotlist(dotlist)
env_vars = OmegaConf.to_container(
    config.ray_kwargs.ray_init.runtime_env.env_vars, resolve=True
)
if not isinstance(env_vars, dict):
    raise TypeError(f"Ray env_vars must be a mapping, got {type(env_vars)!r}")
invalid = {k: type(v).__name__ for k, v in env_vars.items() if not isinstance(v, str)}
if invalid:
    raise TypeError(f"Ray env_vars must all be strings, invalid={invalid}")
print("[GIFT] verified Hydra Ray env_vars string types")
PY

if [[ "${GIFT_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[GIFT] preflight-only completed"
    exit 0
fi

mkdir -p "${RAY_TMPDIR}"
if [[ "${NODE_RANK}" == "0" ]]; then
    ray stop --force >/dev/null 2>&1 || true
    head_started=0
    for attempt in $(seq 1 12); do
        if ray start --head \
            --port="${RAY_GCS_PORT}" \
            --node-ip-address="${RAY_HEAD_ADDR}" \
            --num-gpus="${GPUS_PER_NODE}" \
            ${RAY_STORE_ARGS[@]+"${RAY_STORE_ARGS[@]}"} \
            --temp-dir="${RAY_TMPDIR}" \
            --include-dashboard=false; then
            head_started=1
            break
        fi
        echo "[GIFT] Ray head start failed (${attempt}/12); retrying" >&2
        ray stop --force >/dev/null 2>&1 || true
        sleep 5
    done
    if [[ "${head_started}" != "1" ]]; then
        echo "GIFT Ray head failed to start on ${RAY_HEAD_ADDR}:${RAY_GCS_PORT}" >&2
        exit 1
    fi
else
    joined=0
    for attempt in $(seq 1 60); do
        if ray start --address="${RAY_HEAD_ADDR}:${RAY_GCS_PORT}" \
            --num-gpus="${GPUS_PER_NODE}" \
            ${RAY_STORE_ARGS[@]+"${RAY_STORE_ARGS[@]}"} \
            --temp-dir="${RAY_TMPDIR}"; then
            joined=1
            break
        fi
        echo "Node ${NODE_RANK} waiting for Ray head (${attempt}/60)"
        sleep 5
    done
    if [[ "${joined}" != "1" ]]; then
        echo "Node ${NODE_RANK} failed to join Ray head" >&2
        exit 1
    fi
fi

ACTUAL_NODES="$(python3 - "${NNODES}" <<'PY'
import sys
import time
import ray

want = int(sys.argv[1])
ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
alive = 0
for _ in range(120):
    alive = sum(1 for node in ray.nodes() if node.get("Alive"))
    if alive >= want:
        break
    time.sleep(5)
print(alive)
ray.shutdown()
PY
)"
echo "[GIFT] Ray sees ${ACTUAL_NODES}/${NNODES} live nodes"
if (( ACTUAL_NODES < NNODES )); then
    echo "Ray cluster did not reach ${NNODES} live nodes" >&2
    exit 1
fi

if [[ "${NODE_RANK}" != "0" ]]; then
    echo "[GIFT] worker node ${NODE_RANK} is ready"
    tail -f /dev/null
    exit 0
fi

export RAY_ADDRESS="${RAY_HEAD_ADDR}:${RAY_GCS_PORT}"
python3 "${SCRIPT_DIR}/verify_gift_ray_workers.py" \
    --address "${RAY_ADDRESS}" \
    --expected-nodes "${NNODES}" \
    --repo-root "${REPO_ROOT}" \
    --config "${CONFIG}"

echo "Starting ${MODEL_NAME} GIFT piecewise ranking on ${NNODES} x ${GPUS_PER_NODE} GPU"
echo "scope=${SELECTION_SCOPE}; entropy=off; unprojected KL; kappa 1->0; lambda=1"
bash "${REPO_ROOT}/scripts/_gpu_run_verl.sh" "${CONFIG}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.total_training_steps="${MAX_STEPS}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_prefix_grouper=True \
    trainer.balance_batch=True \
    data.shuffle=True \
    +data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    "${RAY_ENV_ARGS[@]}" \
    "$@"
