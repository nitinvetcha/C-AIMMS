#!/bin/bash
# ITERRET ABLATION *WITH* THE EXPERIENCE BANK, on the paper's held-out split.
#
#   cd <repo> && sbatch scripts/run_bank_ablation.slurm
#
# Protocol (ITERRET paper, Sec. 3):
#   * conversation 0 built the bank (run_offline_bank.slurm) and is HELD OUT
#   * conversations 1-9 are scored = 1,388 questions
#     (multi 250 / temporal 284 / open 83 / single 771 -- matches the paper's
#      own per-category Ns exactly)
#   * n_max = 5, adversarial category excluded
#
# Writes to its OWN checkpoint file. That is not cosmetic: the ablation resumes
# by (sample_idx, q_idx), so pointing this at the no-bank run's .jsonl would
# make it skip every question already there and silently report the OLD arm's
# scores as the new arm's.
#
#SBATCH --job-name=caimms_bank_abl
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=../outputs/caimms_bank_ablation_%j.log

set -euo pipefail

REPO="${CAIMMS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if   [ -f "${REPO}/env.sh" ];    then :
elif [ -f "${REPO}/../env.sh" ]; then REPO="$(cd "${REPO}/.." && pwd)"
else
    echo "ERROR: cannot find env.sh from '${REPO}'."
    echo "       Submit from the repo root:  cd <repo> && sbatch scripts/run_bank_ablation.slurm"
    exit 1
fi
source "${REPO}/env.sh"

# ── the paper's protocol ──────────────────────────────────────────────────────
export CAIMMS_BOOTSTRAP_SAMPLES="${CAIMMS_BOOTSTRAP_SAMPLES:-1}"   # hold out conv 0
export CAIMMS_REQUIRE_BANK=1                                       # never silently run bankless
export CAIMMS_BANK_PATH="${CAIMMS_BANK_PATH:-${CAIMMS_OUTPUT_DIR}/experience_bank.json}"
export CAIMMS_ABLATION_OUTPUT="${CAIMMS_ABLATION_OUTPUT:-${CAIMMS_OUTPUT_DIR}/ablation_bank_heldout.jsonl}"

OUTPUT_DIR="${CAIMMS_OUTPUT_DIR}"
SERVER_LOG="${OUTPUT_DIR}/bank_ablation_server_${SLURM_JOB_ID}.log"
RUN_LOG="${OUTPUT_DIR}/bank_ablation_run_${SLURM_JOB_ID}.log"
MAX_WAIT=600
INTERVAL=10
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "  C-AIMMS ABLATION + EXPERIENCE BANK (held-out)"
echo "  Job: ${SLURM_JOB_ID} | Node: $(hostname) | $(date)"
echo "  repo  : ${CAIMMS_ROOT}"
echo "  bank  : ${CAIMMS_BANK_PATH}"
echo "  out   : ${CAIMMS_ABLATION_OUTPUT}"
echo "  split : hold out first ${CAIMMS_BOOTSTRAP_SAMPLES} conversation(s) -> 1388 questions"
echo "=============================================="

# Fail BEFORE allocating vLLM if the offline phase never ran.
if [ ! -f "${CAIMMS_BANK_PATH}" ]; then
    echo "ERROR: no experience bank at ${CAIMMS_BANK_PATH}"
    echo "       Run the offline phase first:  sbatch scripts/run_offline_bank.slurm"
    exit 1
fi
python3 - <<PY
import json, sys
d = json.load(open("${CAIMMS_BANK_PATH}"))
p, r = len(d.get("planning_bank", [])), len(d.get("reflection_bank", []))
print(f"preflight: bank has {p} planning + {r} reflection entries")
if p + r == 0:
    sys.exit("ERROR: bank is empty -- this would silently be the no-bank arm.")
PY

caimms_activate
cd "${CAIMMS_ROOT}/delta-Mem"

NCACHED="$(ls "${OUTPUT_DIR}/graph_cache"/sample_*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "graph cache: ${NCACHED} conversation(s) cached -- these will NOT be rebuilt"

python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode("warmup")
print("preflight: MiniLM retrieval encoder OK")
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

echo "[3/3] Starting held-out ablation on GPU 1 (base model, bank-guided)..."
set +e
CUDA_VISIBLE_DEVICES=1 python3 -u -m deltamem.workmem.eval_locomo_ablation 2>&1 | tee -a "${RUN_LOG}"
EVAL_EXIT=${PIPESTATUS[0]}
set -e

ROWS="$(wc -l < "${CAIMMS_ABLATION_OUTPUT}" 2>/dev/null || echo 0)"
echo "=============================================="
echo "  Done at $(date) | exit ${EVAL_EXIT} | rows: ${ROWS} (expect 1388)"
echo "  Score: python3 ${CAIMMS_ROOT}/scripts/score_calculator.py ${CAIMMS_ABLATION_OUTPUT}"
echo "=============================================="
exit "${EVAL_EXIT}"
