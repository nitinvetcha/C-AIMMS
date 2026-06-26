#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full LoCoMo benchmark for IterRet, on a GPU box, with Qwen3-4B-Instruct as
# BOTH the answering LLM (served by vLLM) and the surprise segmenter.
#
# It runs TWO evals so the F1 delta is interpretable:
#   1) baseline    : per-turn episodes (no surprise segmentation)
#   2) surprise    : surprise-bounded events (Stage 1 + Stage 2 refinement)
#
# Usage:
#   bash run_locomo_gpu.sh
#
# Override anything via env vars, e.g. pin everything to physical GPU 1:
#   GPU=1 SURPRISE_MODEL=Qwen/Qwen2.5-3B-Instruct GAMMA=3.5 bash run_locomo_gpu.sh
#
# Prereqs on the cluster:
#   python -m venv .venv && . .venv/bin/activate
#   pip install -r requirements.txt vllm      # vllm pulls a CUDA torch build
#   # data/locomo10.json must exist (this repo's run downloaded it from
#   #   https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json )
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"          # answering LLM (vLLM)
# Surprise segmenter model (loads via transformers AutoModelForCausalLM, so it
# must be a model_type your transformers knows). On old transformers without
# Qwen3 support (KeyError: 'qwen3'), override to a Qwen2-family model:
#   SURPRISE_MODEL=Qwen/Qwen2.5-3B-Instruct bash run_locomo_gpu.sh
SURPRISE_MODEL="${SURPRISE_MODEL:-$MODEL}"
LOCOMO_PATH="${LOCOMO_PATH:-data/locomo10.json}"
PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}/v1"
GAMMA="${GAMMA:-3.5}"                                    # surprise threshold (calibrate below)
OUT_DIR="${OUT_DIR:-locomo_gpu_results}"
PY="${PY:-python}"

# GPU selection. By default we inherit CUDA_VISIBLE_DEVICES. To pin EVERYTHING
# (vLLM + the surprise model) to ONE physical GPU -- the common case when only
# one card is free -- set GPU=<index>, e.g. GPU=1 to use only physical GPU 1.
GPU="${GPU:-}"
if [[ -n "$GPU" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
# With a single visible GPU both models live on cuda:0, so vLLM must NOT grab the
# whole card -- leave room for the surprise model:
SURPRISE_DEVICE="${SURPRISE_DEVICE:-cuda:0}"
VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.55}"                   # raise toward 0.9 if vLLM has its OWN gpu
VLLM_MAX_LEN="${VLLM_MAX_LEN:-16384}"

mkdir -p "$OUT_DIR"
echo "[runbook] model=$MODEL gamma=$GAMMA CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<inherited>} surprise=$SURPRISE_DEVICE"

# dataset (snap-research/locomo)
if [[ ! -f "$LOCOMO_PATH" ]]; then
  mkdir -p "$(dirname "$LOCOMO_PATH")"
  curl -sL -o "$LOCOMO_PATH" \
    https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
fi

# 1. Serve the answering LLM with vLLM ---------------------------------------
echo "[runbook] starting vLLM on port $PORT ..."
vllm serve "$MODEL" \
  --port "$PORT" --gpu-memory-utilization "$VLLM_MEM_UTIL" --max-model-len "$VLLM_MAX_LEN" \
  > "$OUT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'echo "[runbook] stopping vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true' EXIT

echo "[runbook] waiting for vLLM to be ready (this can take a few minutes on first load)..."
until curl -sf "$BASE_URL/models" >/dev/null 2>&1; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[runbook] vLLM died; see $OUT_DIR/vllm.log"; exit 1; fi
  sleep 5
done
echo "[runbook] vLLM is up."

COMMON=( --locomo-path "$LOCOMO_PATH" --use-real-llm --llm-base-url "$BASE_URL" --llm-model "$MODEL" )

# 2. Calibrate gamma on the first conversation (sanity: aim for events of
#    ~3-10 turns; if everything is 1-turn, raise GAMMA; if one giant event,
#    lower it). Comment this block out once you've picked a value.
echo "[runbook] gamma calibration on conversation 0 ..."
$PY - "$LOCOMO_PATH" "$SURPRISE_MODEL" "$SURPRISE_DEVICE" <<'PYCAL'
import sys, json
from iterret.locomo_data import conversation_to_turns
from iterret.episode_segmenter import SurpriseEpisodeSegmenter
path, model, dev = sys.argv[1], sys.argv[2], sys.argv[3]
turns = conversation_to_turns(json.load(open(path))[0]["conversation"])
seg = SurpriseEpisodeSegmenter(model, gamma=3.5, min_block_size=8, similarity_refinement=True, device=dev)
for g in [2.5, 3.0, 3.5, 4.0, 5.0]:
    seg._tracker_kwargs["surprisal_threshold_gamma"] = g
    ev = seg.segment(turns)
    sizes = [len(e) for e in ev]
    print(f"  gamma={g}: {len(ev)} events  | mean turns/event={sum(sizes)/len(sizes):.1f}  | sizes={sizes[:20]}{'...' if len(sizes)>20 else ''}")
PYCAL

# 3. Baseline: per-turn episodes (no surprise) -------------------------------
echo "[runbook] === BASELINE (per-turn episodes) ==="
$PY run_locomo_eval.py "${COMMON[@]}" \
  --output "$OUT_DIR/results_baseline.json" \
  --save-graphs-dir "$OUT_DIR/graphs_baseline"

# 4. Proposed model: surprise-bounded events (Stage 1 + Stage 2) -------------
echo "[runbook] === PROPOSED (surprise segmentation + refinement, gamma=$GAMMA) ==="
$PY run_locomo_eval.py "${COMMON[@]}" \
  --surprise-segmentation --surprise-model "$SURPRISE_MODEL" --surprise-gamma "$GAMMA" \
  --surprise-device "$SURPRISE_DEVICE" \
  --output "$OUT_DIR/results_surprise.json" \
  --save-graphs-dir "$OUT_DIR/graphs_surprise"

echo "[runbook] done. F1/judge tables are above; full per-question JSON in $OUT_DIR/."
