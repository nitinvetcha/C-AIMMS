#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run IterRet/IMR on the LongBench QA tasks, BOTH baseline (fixed chunks) and
# surprise (surprise-bounded document events), so the F1 delta is interpretable.
#
# Assumes a vLLM (or Ollama) server is already serving MODEL at BASE_URL.
# Point SURPRISE_DEVICE at a free GPU for the segmenter.
#
# Usage (defaults shown):
#   BASE_URL=http://localhost:8002/v1 MODEL=Qwen/Qwen3-4B-Instruct-2507 \
#   SURPRISE_DEVICE=cuda:2 GAMMA=1.5 MIN_BLOCK=64 MAX_SAMPLES=30 \
#   LONGBENCH_PATH=data/longbench  bash run_longbench.sh
#
# Run a subset:  TASKS="multifieldqa_en hotpotqa" bash run_longbench.sh
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
BASE_URL="${BASE_URL:-http://localhost:8002/v1}"
SURPRISE_MODEL="${SURPRISE_MODEL:-$MODEL}"
SURPRISE_DEVICE="${SURPRISE_DEVICE:-cuda:0}"
GAMMA="${GAMMA:-1.5}"                       # prose default (NOT dialogue's 3.5)
MIN_BLOCK="${MIN_BLOCK:-64}"               # min event size in tokens (docs want larger than dialogue's 8)
WORDS_PER_CHUNK="${WORDS_PER_CHUNK:-200}"  # baseline fixed-chunk size
MAX_SAMPLES="${MAX_SAMPLES:-30}"           # cap per task (set to 0/empty for all)
LONGBENCH_PATH="${LONGBENCH_PATH:-data/longbench}"
OUT_DIR="${OUT_DIR:-longbench_results}"
PY="${PY:-python}"

# Default: single-doc QA (fast). Add the multi-doc ones for the full QA suite:
#   TASKS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique"
TASKS="${TASKS:-multifieldqa_en qasper hotpotqa 2wikimqa}"

mkdir -p "$OUT_DIR"
COMMON=( --use-real-llm --llm-base-url "$BASE_URL" --llm-model "$MODEL" )
[[ -n "$LONGBENCH_PATH" ]] && COMMON+=( --longbench-path "$LONGBENCH_PATH" )
[[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" != "0" ]] && COMMON+=( --max-samples "$MAX_SAMPLES" )

echo "[runbook] model=$MODEL base_url=$BASE_URL surprise_device=$SURPRISE_DEVICE gamma=$GAMMA tasks=[$TASKS]"

for task in $TASKS; do
  echo "=========================================================="
  echo "[runbook] TASK=$task  ::  BASELINE (fixed ${WORDS_PER_CHUNK}-word chunks)"
  echo "=========================================================="
  $PY -m longbench.run_longbench_eval --task "$task" "${COMMON[@]}" \
    --baseline-words-per-chunk "$WORDS_PER_CHUNK" \
    --output "$OUT_DIR/${task}_baseline.json"

  echo "=========================================================="
  echo "[runbook] TASK=$task  ::  SURPRISE (gamma=$GAMMA, min_block=$MIN_BLOCK)"
  echo "=========================================================="
  $PY -m longbench.run_longbench_eval --task "$task" "${COMMON[@]}" \
    --surprise-segmentation --surprise-model "$SURPRISE_MODEL" --surprise-device "$SURPRISE_DEVICE" \
    --surprise-gamma "$GAMMA" --surprise-min-block-size "$MIN_BLOCK" \
    --output "$OUT_DIR/${task}_surprise.json"
done

echo "[runbook] done. Aggregating:"
$PY -m longbench.summarize "$OUT_DIR"
