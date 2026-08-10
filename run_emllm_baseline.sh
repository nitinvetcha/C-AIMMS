#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Vanilla LongBench baseline for the EM-LLM backbones. Serves each model with
# vLLM in turn, runs every QA task against it (middle-truncated to the model's
# window -- the limited-context setting EM-LLM is designed to beat), then
# aggregates into a model x task qa_f1 table.
#
# Prereqs: `huggingface-cli login` (LLaMA-3 and Mistral are GATED -- accept their
# licenses on HF first, or those two will 401 on download).
#
# Usage:  GPU=0 MAX_SAMPLES=50 LONGBENCH_PATH=data/longbench bash run_emllm_baseline.sh
# Subset: MODELS_TO_RUN="mistral-7b" TASKS="multifieldqa_en hotpotqa" bash run_emllm_baseline.sh
# ---------------------------------------------------------------------------
set -euo pipefail

LONGBENCH_PATH="${LONGBENCH_PATH:-data/longbench}"
TASKS="${TASKS:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
OUT_DIR="${OUT_DIR:-emllm_results}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
PY="${PY:-python}"
MODELS_TO_RUN="${MODELS_TO_RUN:-llama3-8b mistral-7b phi3.5-mini}"
mkdir -p "$OUT_DIR"

# alias -> "HF_id  vLLM_max_model_len". Phi supports 128k but is capped here so a
# single 48GB GPU can hold the KV cache; raise it if you have the memory.
model_spec() {
  case "$1" in
    llama3-8b)   echo "meta-llama/Meta-Llama-3-8B-Instruct 8192" ;;
    mistral-7b)  echo "mistralai/Mistral-7B-Instruct-v0.2 32768" ;;
    phi3.5-mini) echo "microsoft/Phi-3.5-mini-instruct 32768" ;;
    *)           echo "$1 8192" ;;
  esac
}

for alias in $MODELS_TO_RUN; do
  read -r hfid maxlen <<< "$(model_spec "$alias")"
  trunc=$((maxlen - 512))    # leave headroom for generation + chat template
  echo "=========================================================="
  echo "[emllm] serving $alias = $hfid  (vLLM max-model-len $maxlen, truncate to $trunc)"
  echo "=========================================================="
  CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$hfid" --port "$PORT" \
    --gpu-memory-utilization 0.9 --max-model-len "$maxlen" \
    > "$OUT_DIR/vllm_$alias.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

  echo "[emllm] waiting for vLLM ($alias) ..."
  until curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
    kill -0 $VLLM_PID 2>/dev/null || { echo "[emllm] vLLM died; see $OUT_DIR/vllm_$alias.log"; exit 1; }
    sleep 5
  done

  for task in $TASKS; do
    echo "[emllm] --- $alias / $task ---"
    $PY -m emllm_baseline.run_baseline --task "$task" --model "$hfid" \
      --llm-base-url "http://localhost:$PORT/v1" --longbench-path "$LONGBENCH_PATH" \
      --max-length "$trunc" --max-samples "$MAX_SAMPLES" \
      --output "$OUT_DIR/${alias}_${task}.json"
  done

  kill $VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true
  trap - EXIT
done

echo "[emllm] done. Aggregate table:"
$PY -m emllm_baseline.summarize "$OUT_DIR"
