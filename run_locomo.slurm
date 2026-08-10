#!/usr/bin/env bash
# ===========================================================================
# Full LoCoMo benchmark for IterRet (baseline vs. surprise-segmented), with
# Qwen3-4B-Instruct as BOTH the answering LLM (vLLM) and the surprise segmenter.
#
# Submit:   sbatch run_locomo.slurm
# Override: sbatch --export=ALL,GAMMA=4.0,MODEL=Qwen/Qwen3-4B-Instruct-2507 run_locomo.slurm
#
# GPU layout (2 GPUs, no CUDA_VISIBLE_DEVICES juggling):
#   - vLLM serves the chat model on cuda:0 (Qwen3-4B fits one H200 with room to spare)
#   - the surprise model (plain transformers) runs on cuda:1
# To fit BOTH models on ONE GPU (ok on A100-40GB+):
#   sbatch --gres=gpu:1 --export=ALL,SURPRISE_DEVICE=cuda:0,VLLM_MEM_UTIL=0.55 run_locomo.slurm
# (the vLLM line already honors VLLM_MEM_UTIL if you export it)
# ===========================================================================
#SBATCH --job-name=locomo-iterret
#SBATCH --partition=a100
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-locomo-%j.out
#SBATCH --error=slurm-locomo-%j.out

set -euo pipefail

# ---- knobs (override via --export) ----------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"     # answering LLM (served by vLLM)
# Surprise segmenter model. Defaults to MODEL, but it loads via `transformers`
# (AutoModelForCausalLM), so it must be a model_type your installed transformers
# knows. On an old transformers (e.g. Python 3.8 venvs that predate Qwen3, which
# needs transformers>=4.51) Qwen3 fails with `KeyError: 'qwen3'` -- override to a
# Qwen2-family model, which older transformers supports:
#   sbatch --export=ALL,SURPRISE_MODEL=Qwen/Qwen2.5-3B-Instruct run_locomo.slurm
SURPRISE_MODEL="${SURPRISE_MODEL:-$MODEL}"
LOCOMO_PATH="${LOCOMO_PATH:-data/locomo10.json}"
PORT="${PORT:-8000}"
GAMMA="${GAMMA:-3.5}"                              # surprise threshold (see calibration)
VLLM_TP="${VLLM_TP:-1}"                            # vLLM tensor-parallel size -> cuda:0..TP-1
SURPRISE_DEVICE="${SURPRISE_DEVICE:-cuda:1}"       # dedicated GPU for the surprise model
VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.90}"             # lower (~0.55) when sharing ONE GPU with surprise
VLLM_MAX_LEN="${VLLM_MAX_LEN:-32768}"              # 16384 is too small: IterRet packs ~15.5k input
                                                   # tokens, so 16384-15513<1024 -> HTTP 400s. 32k fixes it.
SEMANTIC="${SEMANTIC:-1}"                          # 1 = new system (embedding cue-match + relevance
                                                   # fallback). Set SEMANTIC=0 for the lexical paper config.
OUT_DIR="${OUT_DIR:-locomo_results_${SLURM_JOB_ID:-local}}"
BASE_URL="http://localhost:${PORT}/v1"

# ---- cluster environment (EDIT to match your site) ------------------------
# Winning env (verified on catanzaro): Python 3.12 + cu12 vLLM 0.10.1.1 + transformers<5.
# Python 3.13 has NO cu12 vLLM wheel (only cu13, which needs a CUDA-13 driver), and
# Python 3.8 venvs are too old for Qwen3 (KeyError: 'qwen3'). So use a conda 3.12 env.
# ONE-TIME setup on a login node (see the header comment block for the full recipe):
#   conda create -y -n caimms312 python=3.12 && conda activate caimms312
#   pip install "vllm==0.10.1.1" "transformers>=4.55,<5" \
#       "langgraph==1.2.5" "openai==2.43.0" "sentence-transformers>=3.0"
# module load cuda/12.8       # if your site requires it; match the node driver
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-caimms312}"
# If compute nodes have NO internet, pre-stage on a login node instead:
#   - data/locomo10.json (curl below), and
#   - HF model caches: HF_HOME=$PWD/.hf_cache and pre-pull MODEL + a
#     sentence-transformers model; then export HF_HOME here.
# ---------------------------------------------------------------------------

mkdir -p "$OUT_DIR"
echo "[slurm] job=$SLURM_JOB_ID model=$MODEL surprise_model=$SURPRISE_MODEL gamma=$GAMMA vllm_tp=$VLLM_TP surprise=$SURPRISE_DEVICE"
nvidia-smi -L || true

# dataset (snap-research/locomo)
if [[ ! -f "$LOCOMO_PATH" ]]; then
  mkdir -p "$(dirname "$LOCOMO_PATH")"
  curl -sL -o "$LOCOMO_PATH" \
    https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
fi

# 1. serve the answering LLM with vLLM (uses cuda:0..VLLM_TP-1) --------------
echo "[slurm] starting vLLM (TP=$VLLM_TP) ..."
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$VLLM_TP" \
  --gpu-memory-utilization "$VLLM_MEM_UTIL" --max-model-len "$VLLM_MAX_LEN" \
  > "$OUT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'echo "[slurm] stopping vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true' EXIT

echo "[slurm] waiting for vLLM (first load can take minutes)..."
until curl -sf "$BASE_URL/models" >/dev/null 2>&1; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "[slurm] vLLM died; see $OUT_DIR/vllm.log"; exit 1; }
  sleep 5
done
echo "[slurm] vLLM ready."

COMMON=( --locomo-path "$LOCOMO_PATH" --use-real-llm --llm-base-url "$BASE_URL" --llm-model "$MODEL" )
[[ "$SEMANTIC" == "1" ]] && COMMON+=( --semantic-retrieval )
echo "[slurm] semantic retrieval: $([[ "$SEMANTIC" == "1" ]] && echo ON || echo OFF)"

# 2. gamma calibration on conversation 0 (aim for ~3-10 turns/event) ---------
echo "[slurm] gamma calibration:"
python - "$LOCOMO_PATH" "$SURPRISE_MODEL" "$SURPRISE_DEVICE" <<'PYCAL'
import sys, json
from iterret.locomo_data import conversation_to_turns
from iterret.episode_segmenter import SurpriseEpisodeSegmenter
path, model, dev = sys.argv[1:4]
turns = conversation_to_turns(json.load(open(path))[0]["conversation"])
seg = SurpriseEpisodeSegmenter(model, gamma=3.5, min_block_size=8, similarity_refinement=True, device=dev)
for g in [2.5, 3.0, 3.5, 4.0, 5.0]:
    seg._tracker_kwargs["surprisal_threshold_gamma"] = g
    s = [len(e) for e in seg.segment(turns)]
    print(f"  gamma={g}: {len(s)} events | mean turns/event={sum(s)/len(s):.1f}")
PYCAL

# 3. BASELINE: per-turn episodes (no surprise) ------------------------------
echo "[slurm] === BASELINE (per-turn episodes) ==="
python run_locomo_eval.py "${COMMON[@]}" \
  --output "$OUT_DIR/results_baseline.json" --save-graphs-dir "$OUT_DIR/graphs_baseline"

# 4. PROPOSED: surprise-bounded events (Stage 1 + Stage 2 refinement) -------
echo "[slurm] === PROPOSED (surprise segmentation + refinement, gamma=$GAMMA) ==="
python run_locomo_eval.py "${COMMON[@]}" \
  --surprise-segmentation --surprise-model "$SURPRISE_MODEL" --surprise-gamma "$GAMMA" \
  --surprise-device "$SURPRISE_DEVICE" \
  --output "$OUT_DIR/results_surprise.json" --save-graphs-dir "$OUT_DIR/graphs_surprise"

echo "[slurm] done. F1/judge tables are above; per-question JSON in $OUT_DIR/."
