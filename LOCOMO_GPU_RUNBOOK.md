# Running the full LoCoMo benchmark on a GPU cluster

Gets a real F1 (and LLM-judge accuracy) for IterRet with Qwen3-4B-Instruct as both
the answering LLM and the surprise segmenter. One command does setup-aware launch:

```bash
bash run_locomo_gpu.sh
```

This runs **two** evals so the number is interpretable:
- **baseline** — per-turn episodes (original IterRet)
- **proposed** — surprise-bounded events (Stage 1 surprise + Stage 2 KV-modularity refinement)

The F1/judge **delta** between them is what "tells about the performance of the proposed model."

## Setup (once, on the cluster)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt vllm          # vllm pulls a CUDA torch build
# dataset (same file this repo already uses):
mkdir -p data && curl -sL -o data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

## GPU sizing

Two Qwen3-4B copies are live at once: vLLM (chat, ~8 GB + KV cache) and the surprise
segmenter (plain `transformers`, ~8 GB). Place them deliberately:

| GPUs | settings |
|------|----------|
| **≥2** (recommended) | `VLLM_GPU=0 SURPRISE_DEVICE=cuda:1` (default-ish); each model gets its own GPU |
| **1 × 40–80 GB** (A100/H100) | `SURPRISE_DEVICE=cuda:0 VLLM_MEM_UTIL=0.55` so both fit on one card |
| **1 × 24 GB** (4090/3090) | tight: `VLLM_MEM_UTIL=0.4 VLLM_MAX_LEN=8192 SURPRISE_DEVICE=cuda:0`; expect pressure |

Example (2 GPUs, pick a calibrated gamma):

```bash
MODEL=Qwen/Qwen3-4B-Instruct-2507 GAMMA=3.5 VLLM_GPU=0 SURPRISE_DEVICE=cuda:1 \
  bash run_locomo_gpu.sh
```

## Gamma calibration (important)

The surprise threshold `--surprise-gamma` controls event size, and the right value is
**model- and data-dependent**. On a 0.5B model over LoCoMo dialogue, `gamma=1.5` (the
module default) put a boundary at *every* turn; `gamma≈4.0` gave coherent multi-turn
events. Qwen3-4B will differ. The script prints an event-size sweep on conversation 0
before running — pick the gamma whose events average ~3–10 turns, set `GAMMA=`, and
re-run. (Aiming for genuine topic-coherent events, not per-turn or one giant blob.)

## Output

- Console: a category-wise F1 + judge(%) table for each eval (baseline, then proposed).
- `locomo_gpu_results/results_baseline.json`, `results_surprise.json` — full per-question
  results (predicted/gold/F1/correct/iterations).
- `locomo_gpu_results/graphs_*/` — saved CTC graphs; inspect the episodic nodes to see
  the surprise-segmented events.

## Notes baked into the code (all GPU-safe)

- Surprise = the EM-LLM quantity `-log P(x_t | x_<t)` (logits shifted right by one, with
  the previous chunk's last logit carried across seams).
- Stage-2 refinement reads keys from the modern `DynamicCache` and computes its Gram
  matrix in **float32** (fp16 there overflows to `inf` and collapses all boundaries).
- Device/dtype: fp16 on CUDA/MPS, fp32 on CPU; `--surprise-device` overrides placement.
