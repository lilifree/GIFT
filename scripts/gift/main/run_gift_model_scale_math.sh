#!/usr/bin/env bash
# Model-scale entrypoint for Qwen3-4B and Qwen3-1.7B GIFT microbatch token-only runs.
# It performs model-aware checks, then delegates to the already-tested GIFT runtime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${GIFT_CONFIG:-${1:-}}"
if [[ -z "${CONFIG}" ]]; then
    echo "Pass a Qwen3-4B or Qwen3-1.7B GIFT YAML config" >&2
    exit 1
fi
if [[ $# -gt 0 && "$1" == *.yaml ]]; then
    shift
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "GIFT model-scale config does not exist: ${CONFIG}" >&2
    exit 1
fi
CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"

read -r MODEL_NAME SELECTION_SCOPE VAL_BEFORE_TRAIN TOTAL_STEPS < <(
    python3 - "${CONFIG}" <<'PY'
from pathlib import Path
import sys
import yaml

values = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    Path(str(values.get("model_path", ""))).name,
    values.get("gift_selection_scope", ""),
    str(values.get("val_before_train", "")).lower(),
    values.get("gift_total_training_steps", ""),
)
PY
)
if [[ "${MODEL_NAME}" != "Qwen3-4B" && "${MODEL_NAME}" != "Qwen3-1.7B" ]]; then
    echo "Unsupported model for this launcher: ${MODEL_NAME}" >&2
    exit 1
fi
if [[ "${SELECTION_SCOPE}" != "micro_batch" ]]; then
    echo "Expected micro_batch selection, got ${SELECTION_SCOPE}" >&2
    exit 1
fi
if [[ "${VAL_BEFORE_TRAIN}" != "false" ]]; then
    echo "Step-0 validation must remain disabled" >&2
    exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
if [[ "${NODE_RANK}" == "0" && "${GIFT_RUN_PREFLIGHT:-1}" == "1" ]]; then
    python3 "${SCRIPT_DIR}/verify_gift_model_scale_math.py" \
        --config "${CONFIG}" --check-persistent
    python3 "${SCRIPT_DIR}/test_gift_model_scale_math.py"
fi

# The shared launcher is model-agnostic apart from its 8B-only preflight.
# The model-aware checks above replace that preflight.
export GIFT_RUN_PREFLIGHT=0
export GIFT_CONFIG="${CONFIG}"
echo "[GIFT-MODEL-SCALE-MATH] starting ${MODEL_NAME}; microbatch token-only; ${TOTAL_STEPS} steps"
exec bash "${SCRIPT_DIR}/run_gift.sh" \
    "${CONFIG}" "$@"
