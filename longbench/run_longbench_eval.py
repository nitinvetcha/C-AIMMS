"""Evaluate IterRet/IMR (optionally with surprise segmentation) on LongBench.

Mirrors ``run_locomo_eval.py`` but for documents. It reuses the whole IterRet
stack unchanged -- LLM clients, the closed-loop retrieval graph, the offline
experience bank, the surprise CLI flags -- and swaps only the two
document-specific pieces: ``DocumentSurpriseSegmenter`` (surprise on continuous
text) and ``build_ctc_graph_from_document``.

Run from the repo root:

  python -m longbench.run_longbench_eval --task multifieldqa_en \
      --use-real-llm --llm-base-url http://localhost:8000/v1 --llm-model Qwen/Qwen3-4B-Instruct-2507 \
      --surprise-segmentation --surprise-model Qwen/Qwen3-4B-Instruct-2507 \
      --surprise-device cuda:0 --surprise-gamma 1.5 \
      --max-samples 30 --output longbench_results/multifieldqa_surprise.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Make `import iterret` / `import longbench` work whether launched as
# `python -m longbench.run_longbench_eval` or `python longbench/run_longbench_eval.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from iterret import evaluator
from iterret.episode_segmenter import add_surprise_cli_args
from iterret.experience_bank import build_default_embedding_backend, empty_experience_bank
from iterret.graph import build_graph
from iterret.llm_client import (
    LLMClient, MockLLMClient, OpenAICompatibleLLMClient, TransformersLLMClient,
)
from iterret.llm_judge import judge_answer
from iterret.memory_builder import DEFAULT_MAX_CHARS_PER_CALL
from iterret.offline_pipeline import collect_trajectories, construct_experience_banks
from iterret.state import DEFAULT_MAX_ITERATIONS, new_state
from iterret.locomo_data import split_bootstrap_eval

from longbench.doc_memory_builder import build_ctc_graph_from_document
from longbench.doc_segmenter import fixed_word_chunks
from longbench.longbench_data import LONGBENCH_QA_TASKS, load_longbench
from longbench.metrics import score_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate IterRet (+surprise) on a LongBench QA task.")
    p.add_argument("--task", required=True, choices=LONGBENCH_QA_TASKS,
                   help="LongBench QA task to evaluate.")
    p.add_argument("--longbench-path", default=None,
                   help="Local <task>.jsonl file or a directory of them. Omit to download from "
                        "HuggingFace (THUDM/LongBench).")

    # LLM selection -- identical semantics to run_locomo_eval.py.
    p.add_argument("--use-real-llm", action="store_true",
                   help="Use an OpenAI-compatible server (vLLM / Ollama) instead of the mock.")
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--local-hf-llm", action="store_true",
                   help="Run the answering LLM in-process via transformers (no server).")
    p.add_argument("--hf-llm-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--hf-llm-device", default=None)

    # Segmentation.
    p.add_argument("--baseline-words-per-chunk", type=int, default=200,
                   help="When surprise is OFF, split each document into fixed word windows of "
                        "this size (the no-surprise control; default %(default)s).")

    # Retrieval / offline.
    p.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    p.add_argument("--bootstrap-fraction", type=float, default=0.0,
                   help="Fraction of samples used to build the experience bank (default 0.0 = "
                        "no bank; each sample builds its own graph, so bootstrap is expensive here). "
                        "Set e.g. 0.1 to enable, and cap it with --max-bootstrap-samples.")
    p.add_argument("--max-bootstrap-samples", type=int, default=5,
                   help="Hard cap on bootstrap samples (each builds a full graph). Default %(default)s.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on evaluated samples (default: all).")
    p.add_argument("--max-chars-per-call", type=int, default=DEFAULT_MAX_CHARS_PER_CALL)
    p.add_argument("--no-semantic-retrieval", action="store_true",
                   help="Disable embedding-based cue matching + relevance fallback (revert to the "
                        "lexical token-overlap retrieval). On by default for document QA.")
    p.add_argument("--run-judge", action="store_true",
                   help="Also run the LLM-as-judge (LongBench's primary metric is qa_f1; off by default).")
    p.add_argument("--output", default=None, help="Path to dump full per-sample results as JSON.")

    add_surprise_cli_args(p)
    return p.parse_args()


def make_llm(args: argparse.Namespace) -> LLMClient:
    if args.local_hf_llm:
        print(f"[setup] in-process transformers LLM (model={args.hf_llm_model}) -- no server")
        return TransformersLLMClient(args.hf_llm_model, device=args.hf_llm_device)
    if args.use_real_llm:
        llm = OpenAICompatibleLLMClient(base_url=args.llm_base_url, model=args.llm_model)
        print(f"[setup] real LLM at {llm.base_url} (model={llm.model})")
        return llm
    print("[setup] mock LLM (F1 will be ~0; use --use-real-llm or --local-hf-llm for real numbers)")
    return MockLLMClient()


def build_doc_segmenter(args: argparse.Namespace):
    """A DocumentSurpriseSegmenter iff --surprise-segmentation, else None (the
    fixed-chunk baseline). Imported lazily so torch is only needed with surprise."""
    if not getattr(args, "surprise_segmentation", False):
        return None
    from longbench.doc_segmenter import DocumentSurpriseSegmenter
    return DocumentSurpriseSegmenter(
        args.surprise_model,
        gamma=args.surprise_gamma,
        min_block_size=args.surprise_min_block_size,
        similarity_refinement=not getattr(args, "surprise_no_refinement", False),
        device=getattr(args, "surprise_device", None),
    )


def segment_context(context: str, segmenter, args: argparse.Namespace) -> List[str]:
    if segmenter is not None:
        return segmenter.segment_document(context)
    return fixed_word_chunks(context, words_per_chunk=args.baseline_words_per_chunk)


def answer_one(sample: dict, llm: LLMClient, segmenter, bank, embedder, args: argparse.Namespace) -> Dict[str, Any]:
    start = time.time()
    spans = segment_context(sample["context"], segmenter, args)
    graph = build_ctc_graph_from_document(spans, llm, max_chars_per_call=args.max_chars_per_call)
    if embedder is not None:
        graph.attach_embedder(embedder)  # enables semantic cue-match + relevance fallback
    compiled = build_graph(llm, graph, bank)

    initial_state = new_state(sample["question"], max_iterations=args.max_iterations)
    try:
        final_state = compiled.invoke(initial_state, config={"recursion_limit": 8 * args.max_iterations + 5})
        predicted = final_state.get("final_answer") or ""
        iterations = final_state.get("iteration_count", 0)
    except Exception as exc:  # one bad sample shouldn't abort the run
        predicted, iterations = f"[error: {exc}]", 0

    f1 = score_sample(predicted, sample["answers"], sample["task"])
    correct: Optional[bool] = None
    if args.run_judge:
        gold = sample["answers"][0] if sample["answers"] else ""
        correct = judge_answer(sample["question"], gold, predicted, llm)

    return {
        "id": sample["id"], "task": sample["task"], "question": sample["question"],
        "gold_answers": sample["answers"], "predicted_answer": predicted,
        "n_events": len(spans), "f1": f1, "correct": correct,
        "iterations": iterations, "elapsed_sec": time.time() - start,
    }


def build_bank(bootstrap: List[dict], llm: LLMClient, segmenter, embedder, args: argparse.Namespace):
    if not bootstrap:
        return empty_experience_bank()
    records: List[Dict[str, Any]] = []
    for i, sample in enumerate(bootstrap):
        print(f"[bootstrap] {i + 1}/{len(bootstrap)} ({sample['id']}): building graph + trajectory...")
        spans = segment_context(sample["context"], segmenter, args)
        graph = build_ctc_graph_from_document(spans, llm, max_chars_per_call=args.max_chars_per_call)
        if embedder is not None:
            graph.attach_embedder(embedder)
        records.extend(collect_trajectories([sample["question"]], graph, llm,
                                            max_iterations=args.max_iterations))
    print(f"[bootstrap] scoring {len(records)} trajectory(ies) and distilling experience...")
    bank = construct_experience_banks(records, llm, k_low=evaluator.K_LOW, k_high=evaluator.K_HIGH)
    print(f"[bootstrap] built experience bank: {len(bank.planning_bank)} Planning + "
          f"{len(bank.reflection_bank)} Reflection entries")
    return bank


def main() -> None:
    args = parse_args()
    llm = make_llm(args)

    segmenter = build_doc_segmenter(args)
    if segmenter is not None:
        print(f"[setup] surprise segmentation ON (model={args.surprise_model}, gamma={args.surprise_gamma}): "
              "document events are surprise-bounded spans")
    else:
        print(f"[setup] baseline segmentation (fixed {args.baseline_words_per_chunk}-word chunks)")

    embedder = None
    if not args.no_semantic_retrieval:
        embedder = build_default_embedding_backend()
        print(f"[setup] semantic retrieval ON (embedder={type(embedder).__name__}): "
              "cue matching + relevance fallback bridge question/document vocabulary")

    print(f"[setup] loading LongBench task {args.task!r}"
          + (f" from {args.longbench_path}" if args.longbench_path else " from HuggingFace"))
    samples = load_longbench(args.task, path=args.longbench_path, max_samples=args.max_samples)

    bootstrap, eval_samples = split_bootstrap_eval(samples, bootstrap_fraction=args.bootstrap_fraction)
    if args.max_bootstrap_samples is not None:
        bootstrap = bootstrap[:args.max_bootstrap_samples]
    print(f"[setup] {len(samples)} sample(s): {len(bootstrap)} for bootstrap, "
          f"{len(eval_samples)} for evaluation")

    bank = build_bank(bootstrap, llm, segmenter, embedder, args)

    results: List[Dict[str, Any]] = []
    for i, sample in enumerate(eval_samples):
        r = answer_one(sample, llm, segmenter, bank, embedder, args)
        results.append(r)
        print(f"[eval] {i + 1}/{len(eval_samples)} ({sample['id']}): "
              f"{r['n_events']} events, F1 {r['f1'] * 100:.2f}%, {r['iterations']} iter, "
              f"{r['elapsed_sec']:.1f}s")

    _print_summary(results, args)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "task": args.task,
                       "n_bootstrap": len(bootstrap), "n_eval": len(eval_samples),
                       "results": results}, fh, indent=2)
        print(f"\n[output] wrote full results to {args.output}")


def _print_summary(results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    n = len(results)
    if n == 0:
        print("\n(no samples evaluated)")
        return
    avg_f1 = sum(r["f1"] for r in results) / n
    avg_iter = sum(r["iterations"] for r in results) / n
    avg_events = sum(r["n_events"] for r in results) / n
    total_time = sum(r["elapsed_sec"] for r in results)
    mode = "surprise" if getattr(args, "surprise_segmentation", False) else "baseline"

    print("\n" + "=" * 52)
    print(f"LongBench {args.task}  |  mode={mode}  |  N={n}")
    print("-" * 52)
    print(f"  qa_f1:            {avg_f1 * 100:.2f}%")
    if args.run_judge:
        judged = [r for r in results if r["correct"] is not None]
        if judged:
            jp = 100 * sum(1 for r in judged if r["correct"]) / len(judged)
            print(f"  judge:            {jp:.2f}%")
    print(f"  avg events/doc:   {avg_events:.1f}")
    print(f"  avg iterations:   {avg_iter:.2f}")
    print("=" * 52)
    print(f"Total wall time: {total_time:.1f}s over {n} sample(s)")


if __name__ == "__main__":
    main()
