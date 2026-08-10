"""Calibrate the surprise threshold (gamma) for LongBench documents.

At too-high gamma the surprise module barely fires -- whole documents collapse
into 1-2 events -- so the memory graph has nothing to discriminate on. This
sweeps gamma over a few sample documents and reports events/doc and mean event
size, so you can pick a gamma whose granularity is comparable to the baseline's
fixed chunking (e.g. ~15-30 events/doc for the QA tasks).

Run from the repo root (loads the surprise model once, on --device):

  python -m longbench.calibrate_gamma --task multifieldqa_en \
      --longbench-path data/longbench --surprise-model Qwen/Qwen3-4B-Instruct-2507 \
      --device cuda:0 --n-docs 3 --gammas 1.0 1.5 2.0 2.5 3.0
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from longbench.longbench_data import LONGBENCH_QA_TASKS, load_longbench


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep surprise gamma on LongBench documents.")
    ap.add_argument("--task", required=True, choices=LONGBENCH_QA_TASKS)
    ap.add_argument("--longbench-path", default=None)
    ap.add_argument("--surprise-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default=None, help="cuda:N / cpu (default: auto-detect).")
    ap.add_argument("--min-block-size", type=int, default=64)
    ap.add_argument("--no-refinement", action="store_true")
    ap.add_argument("--n-docs", type=int, default=3, help="How many documents to average over.")
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    args = ap.parse_args()

    from longbench.doc_segmenter import DocumentSurpriseSegmenter

    samples = load_longbench(args.task, path=args.longbench_path, max_samples=args.n_docs)
    contexts = [s["context"] for s in samples if s["context"].strip()]
    print(f"[calibrate] task={args.task}  docs={len(contexts)}  model={args.surprise_model}")

    # Load the model once; segment() rebuilds the tracker each call, so we can
    # mutate the gamma knob between sweeps without reloading weights.
    seg = DocumentSurpriseSegmenter(
        args.surprise_model, gamma=args.gammas[0], min_block_size=args.min_block_size,
        similarity_refinement=not args.no_refinement, device=args.device,
    )

    tok = seg.tokenizer
    doc_tokens = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in contexts]
    print(f"[calibrate] doc length (tokens): "
          f"mean={statistics.mean(doc_tokens):.0f} min={min(doc_tokens)} max={max(doc_tokens)}")

    print(f"\n{'gamma':>6}{'events/doc':>12}{'mean tok/event':>16}{'min':>6}{'max':>6}")
    print("-" * 46)
    for g in args.gammas:
        seg._tracker_kwargs["surprisal_threshold_gamma"] = g
        n_events, ev_tok = [], []
        for c in contexts:
            spans = seg.segment_document(c)
            n_events.append(len(spans))
            ev_tok.extend(len(tok(s, add_special_tokens=False)["input_ids"]) for s in spans)
        mean_ev = statistics.mean(n_events) if n_events else 0
        mean_tok = statistics.mean(ev_tok) if ev_tok else 0
        print(f"{g:>6.1f}{mean_ev:>12.1f}{mean_tok:>16.0f}{min(n_events):>6}{max(n_events):>6}")
    print("-" * 46)
    print("Aim for events/doc comparable to the baseline's fixed chunking "
          "(~doc_words / --baseline-words-per-chunk).")


if __name__ == "__main__":
    main()
