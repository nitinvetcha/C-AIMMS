"""Vanilla long-context LongBench baseline for the EM-LLM backbones.

For each sample: build the official LongBench task prompt, middle-truncate it to
the model's context budget (LongBench's canonical truncation -- keep the head and
tail, drop the middle), send it to the model, generate the task's answer length,
and score with qa_f1. No memory graph, no retrieval -- this is the plain
truncated-context baseline both EM-LLM and IterRet+surprise are compared to.

Run from the repo root, e.g. against a vLLM server:

  python -m emllm_baseline.run_baseline --task multifieldqa_en \
      --model meta-llama/Meta-Llama-3-8B-Instruct \
      --llm-base-url http://localhost:8000/v1 --longbench-path data/longbench \
      --max-samples 50 --output emllm_results/llama3-8b_multifieldqa_en.json

Or every EM-LLM backbone x every QA task via run_emllm_baseline.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from iterret.llm_client import LLMClient, OpenAICompatibleLLMClient, TransformersLLMClient
from longbench.longbench_data import LONGBENCH_QA_TASKS, load_longbench
from longbench.metrics import score_sample
from emllm_baseline.longbench_prompts import (
    DATASET2MAXGEN, DATASET2PROMPT, EMLLM_MODELS, MODEL2MAXLEN,
)


def resolve_model(name: str) -> str:
    """Accept either a short alias (llama3-8b) or a full HF id."""
    return EMLLM_MODELS.get(name, name)


def middle_truncate(prompt: str, tokenizer, max_length: int) -> str:
    """LongBench's canonical truncation: if the tokenized prompt is longer than
    max_length, keep the first and last max_length//2 tokens and drop the middle
    (the instructions live at the head and the question at the tail, so both
    survive)."""
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(ids) <= max_length:
        return prompt
    half = max_length // 2
    return (tokenizer.decode(ids[:half], skip_special_tokens=True)
            + tokenizer.decode(ids[-half:], skip_special_tokens=True))


def make_llm(args: argparse.Namespace, model_id: str, max_gen: int) -> LLMClient:
    if args.local_hf_llm:
        return TransformersLLMClient(model_id, device=args.hf_llm_device, max_tokens=max_gen)
    return OpenAICompatibleLLMClient(base_url=args.llm_base_url, model=model_id, max_tokens=max_gen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vanilla LongBench baseline for EM-LLM backbones.")
    p.add_argument("--task", required=True, choices=LONGBENCH_QA_TASKS)
    p.add_argument("--model", required=True,
                   help=f"HF id or alias {list(EMLLM_MODELS)}.")
    p.add_argument("--longbench-path", default=None, help="Local <task>.jsonl dir/file (else HF download).")
    p.add_argument("--llm-base-url", default="http://localhost:8000/v1",
                   help="OpenAI-compatible (vLLM) endpoint serving --model.")
    p.add_argument("--local-hf-llm", action="store_true",
                   help="Run the model in-process via transformers instead of a server.")
    p.add_argument("--hf-llm-device", default=None)
    p.add_argument("--max-length", type=int, default=None,
                   help="Context budget for middle-truncation (default: per-model from MODEL2MAXLEN, "
                        "else 7500).")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_id = resolve_model(args.model)
    max_gen = DATASET2MAXGEN.get(args.task, 64)
    max_length = args.max_length or MODEL2MAXLEN.get(model_id, 7500)

    from transformers import AutoTokenizer  # for truncation only (lightweight)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm = make_llm(args, model_id, max_gen)
    template = DATASET2PROMPT[args.task]
    print(f"[emllm-baseline] model={model_id} task={args.task} max_length={max_length} max_gen={max_gen} "
          f"backend={'transformers' if args.local_hf_llm else args.llm_base_url}")

    samples = load_longbench(args.task, path=args.longbench_path, max_samples=args.max_samples)
    print(f"[emllm-baseline] {len(samples)} sample(s)")

    results: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        start = time.time()
        prompt = template.format(context=s["context"], input=s["question"])
        prompt = middle_truncate(prompt, tokenizer, max_length)
        try:
            # Single-prompt generation: hand the whole thing to the model as the
            # user turn (vLLM applies the model's chat template); empty system.
            pred = llm.chat("", prompt, temperature=0.0).strip()
        except Exception as exc:  # one bad sample shouldn't abort the sweep
            pred = f"[error: {exc}]"
        f1 = score_sample(pred, s["answers"], args.task)
        results.append({
            "id": s["id"], "task": args.task, "question": s["question"],
            "gold_answers": s["answers"], "predicted_answer": pred,
            "f1": f1, "elapsed_sec": time.time() - start,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(samples):
            running = 100 * sum(r["f1"] for r in results) / len(results)
            print(f"[emllm-baseline] {i + 1}/{len(samples)}  running qa_f1={running:.2f}%")

    qa_f1 = 100 * sum(r["f1"] for r in results) / len(results) if results else 0.0
    print(f"\n==== {model_id}  |  {args.task}  |  N={len(results)}  |  qa_f1={qa_f1:.2f}% ====")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "model": model_id, "task": args.task,
                       "max_length": max_length, "qa_f1": qa_f1, "results": results}, fh, indent=2)
        print(f"[emllm-baseline] wrote {args.output}")


if __name__ == "__main__":
    main()
