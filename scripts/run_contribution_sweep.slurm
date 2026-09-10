#!/bin/bash
#SBATCH --job-name=caimms_contribution
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=../outputs/contribution_%j.log
#
# The actual delta-mem-on-vs-off measurement. The token-validity mask fix
# it depends on is now permanent in delta_impl.py (measured: mean_base_o_norm
# 0.0000 -> 19.9964 on n=30). Replays SWEEP_N questions' worth of evidence
# already stored in a completed run, with the mask fix applied, comparing
# delta_scaling=0.0 (delta-mem neutralised) against 2.0 (shipped) on
# byte-identical evidence -- one model load, replay_delta_scaling.py sweeps
# both scales internally.
#
# No vLLM: replay_delta_scaling.py never calls IterRet, it replays stored
# retrieval.final_evidence_ids against the cached CTC graphs. One GPU.
#
#   sbatch scripts/run_contribution_sweep.slurm                    # default: 300 questions
#   sbatch --export=ALL,SWEEP_N=600 scripts/run_contribution_sweep.slurm
#
set -euo pipefail

REPO="${CAIMMS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if   [ -f "${REPO}/env.sh" ];    then :
elif [ -f "${REPO}/../env.sh" ]; then REPO="$(cd "${REPO}/.." && pwd)"
else
    echo "ERROR: cannot find env.sh from '${REPO}'."
    echo "       Submit from the repo root:  cd <repo> && sbatch scripts/run_contribution_sweep.slurm"
    exit 1
fi
source "${REPO}/env.sh"

SOURCE_RUN="${CAIMMS_OUTPUT_DIR}/workmem_phase2_write0_full.jsonl"
N="${SWEEP_N:-300}"
OUT="${CAIMMS_OUTPUT_DIR}/replay_scaling_fixed.jsonl"

echo "=============================================="
echo "  C-AIMMS delta-mem contribution sweep"
echo "  Job: ${SLURM_JOB_ID} | Node: $(hostname) | $(date)"
echo "  source evidence : ${SOURCE_RUN}"
echo "  n questions     : ${N}"
echo "  arms            : 0.0 (off) vs 2.0 (shipped)"
echo "  out             : ${OUT}"
echo "=============================================="

[ -f "${SOURCE_RUN}" ] || { echo "ERROR: missing ${SOURCE_RUN}"; exit 1; }

caimms_activate
cd "${CAIMMS_ROOT}/delta-Mem"

REPLAY_MAX_QUESTIONS="${N}" \
python3 -u -m deltamem.workmem.replay_delta_scaling \
    "${SOURCE_RUN}" 0.0,2.0 "${OUT}"

echo
echo "=============================================="
echo "  Done at: $(date)"
echo "  Score it with:"
echo "    python3 scripts/compare_replay_arms.py ${OUT} --baseline 0.0"
echo "=============================================="
