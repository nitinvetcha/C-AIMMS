#!/bin/bash
# ITERRET OFFLINE PHASE -- build the experience bank from the 10% bootstrap split.
#
#   cd <repo> && sbatch scripts/run_offline_bank.slurm
#   sbatch --partition=ada scripts/run_offline_bank.slurm     # different partition
#
# Produces:
#   $CAIMMS_OUTPUT_DIR/experience_bank.json         <- consumed by the ablation
#   $CAIMMS_OUTPUT_DIR/experience_bank_trace.json   <- per-step audit trail
#
# Needs ONE GPU (vLLM only -- no adapter, no second model). Typically <1h:
# conversation 0's CTC graph is already cached from earlier runs, so the cost
# is 10 seed trajectories (<=11 LLM calls each) plus rubric scoring/distillation.
#
#SBATCH --job-name=caimms_offline_bank
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --requeue
# Relative to the submit dir (the repo root), so logs land in the workspace
# outputs/ and NOT inside the repo -- an `rsync --delete` of the repo has
# already deleted the log of a running job once.
#SBATCH --output=../outputs/caimms_offline_bank_%j.log

set -euo pipefail

# Resolve the repo from wherever sbatch was run, then let env.sh derive paths.
REPO="${CAIMMS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if   [ -f "${REPO}/env.sh" ];    then :
elif [ -f "${REPO}/../env.sh" ]; then REPO="$(cd "${REPO}/.." && pwd)"
else
    echo "ERROR: cannot find env.sh from '${REPO}'."
    echo "       Submit from the repo root:  cd <repo> && sbatch scripts/run_offline_bank.slurm"
    exit 1
fi
source "${REPO}/env.sh"

# ── the paper's protocol, pinned here so the job is self-documenting ──────────
export CAIMMS_BOOTSTRAP_SAMPLES="${CAIMMS_BOOTSTRAP_SAMPLES:-1}"        # 10% of 10 convs
export CAIMMS_BOOTSTRAP_SEED_QUESTIONS="${CAIMMS_BOOTSTRAP_SEED_QUESTIONS:-10}"
export CAIMMS_MAX_ITERATIONS="${CAIMMS_MAX_ITERATIONS:-5}"              # n_max = 5
export CAIMMS_BANK_MODULE_MODE="${CAIMMS_BANK_MODULE_MODE:-native}"

OUTPUT_DIR="${CAIMMS_OUTPUT_DIR}"
SERVER_LOG="${OUTPUT_DIR}/offline_bank_server_${SLURM_JOB_ID}.log"
RUN_LOG="${OUTPUT_DIR}/offline_bank_run_${SLURM_JOB_ID}.log"
MAX_WAIT=600
INTERVAL=10
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "  ITERRET OFFLINE PHASE (experience bank)"
echo "  Job: ${SLURM_JOB_ID} | Node: $(hostname) | $(date)"
echo "  repo      : ${CAIMMS_ROOT}"
echo "  bootstrap : ${CAIMMS_BOOTSTRAP_SAMPLES} conv x ${CAIMMS_BOOTSTRAP_SEED_QUESTIONS} seed questions"
echo "  bank out  : ${OUTPUT_DIR}/experience_bank.json"
echo "=============================================="

caimms_activate
cd "${CAIMMS_ROOT}/delta-Mem"

# MiniLM must load: the bank's own retrieval and the graph's semantic ranking
# both silently downgrade to keyword overlap if it doesn't, without changing
# the exit code.
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode("warmup")
print("preflight: MiniLM encoder OK")
PY

SERVER_PID=""
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[1/3] Starting vLLM on GPU 0 (port ${VLLM_PORT})..."
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "${CAIMMS_MODEL_PATH}" \
    --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --port "${VLLM_PORT}" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --enforce-eager \
    --disable-log-requests \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[2/3] Waiting for server (up to ${MAX_WAIT}s)..."
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM died during startup. Real cause:"
        grep -nE "OutOfMemory|CUDA|Error|Traceback|Address already in use" "${SERVER_LOG}" | tail -15
        echo "--- last 25 lines ---"; tail -25 "${SERVER_LOG}"
        exit 1
    fi
    sleep ${INTERVAL}; ELAPSED=$((ELAPSED + INTERVAL))
    if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
        echo "ERROR: server did not come up within ${MAX_WAIT}s."
        grep -nE "OutOfMemory|CUDA|Error|Traceback" "${SERVER_LOG}" | tail -15
        tail -25 "${SERVER_LOG}"
        exit 1
    fi
    echo "      ...${ELAPSED}s"
done
echo "      Server online."

echo "[3/3] Building experience bank..."
set +e
python3 -u -m deltamem.workmem.build_bootstrap_bank 2>&1 | tee -a "${RUN_LOG}"
EXIT=${PIPESTATUS[0]}
set -e

echo "=============================================="
echo "  Done at $(date) | exit ${EXIT}"
echo "  bank : ${OUTPUT_DIR}/experience_bank.json"
echo "  trace: ${OUTPUT_DIR}/experience_bank_trace.json"
echo "  Next : sbatch scripts/run_bank_ablation.slurm"
echo "=============================================="
exit "${EXIT}"
