"""Measure what delta-mem's steering actually contributes, with retrieval pinned.

WHY THIS EXISTS
---------------
Every cross-run comparison in this project so far has been confounded: IterRet's
retrieve/reflect loop is LLM-driven and not reproducible across runs, and a
direct check of two 1540-question runs found 1411 of them (91.6%) received
DIFFERENT evidence. So "OSAM run scored X, ablation scored Y" mixes "did
delta-mem help" with "did retrieval happen to do better that day", and no
existing number isolates delta-mem.

This harness removes retrieval from the experiment entirely. Instead of running
IterRet, it REPLAYS the evidence a completed run already used, reconstructed
from that run's `retrieval.final_evidence_ids` plus the cached CTC graphs.
Verified against workmem_phase2_write0_full.jsonl: 19658/19658 ids resolve with
zero misses, and every row's id count matches its n_evidence_retrieved, so the
replay is faithful rather than approximate. The stored order is the POST-filter
order -- i.e. the order actually written into S -- so the delta rule's
order-sensitivity is preserved without needing the embedding model.

Consequences: no vLLM, no sentence-transformers, no graph building, one GPU.
Every arm sees byte-identical evidence, prompts and decoding, so the only
variable left is delta-mem's own contribution.

HOW THE ARMS WORK
-----------------
`delta_scaling` is the alpha/rank multiplier applied to delta-mem's read before
it is added to the backbone's query and attention output (_project_delta_head,
delta_impl.py). Setting it to 0.0 makes delta_q and delta_o exactly zero, so
q_bar == q0 and y_bar == a: the model becomes precisely the frozen backbone
while every other code path -- session handling, prompt construction, Phase 1
writes, ordering -- stays identical. That is a far cleaner control than the
separate no-adapter ablation script, which differs in model loading and session
management too.

    scale 0.0  -> delta-mem neutralised (frozen backbone)
    scale 2.0  -> the shipped configuration (alpha/rank = 16/8)
    scale >2   -> the volume sweep; note the projection weights were TRAINED
                  under 2.0, so higher values push them out of their trained
                  operating range and may degrade for reasons unrelated to
                  whether more memory signal would help.

USAGE
-----
    source env.sh
    python3 -u -m deltamem.workmem.replay_delta_scaling \\
        "$CAIMMS_OUTPUT_DIR/workmem_phase2_write0_full.jsonl" \\
        0.0,2.0 \\
        "$CAIMMS_OUTPUT_DIR/replay_scaling.jsonl"

Then score the arms pairwise with scripts/compare_replay_arms.py.

CAVEAT: if _token_validity_mask is still returning an all-false mask on the
Phase 2 prefill (see docs/HANDOFF.md), arm 2.0 is a PARTIALLY DISABLED
delta-mem, not a healthy one, and this measures the crippled version. Running
this before and after that fix quantifies what the bug costs.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.core.delta_impl import iter_delta_mem_modules
from deltamem.eval.common import attach_delta_adapter_in_place
from deltamem.eval.locomo_protocol import ADVERSARIAL_CATEGORY, score_locomo_prediction
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.workmem.osam_workmem import answer_with_osam, populate_osam_from_evidence
from iterret.ctc_graph import CueTagContentGraph

_ROOT = os.environ.get("CAIMMS_ROOT", "/home/kbasu/arnavbhatt/workmem_test")
MODEL_PATH = os.environ.get("CAIMMS_MODEL_PATH", f"{_ROOT}/models/Qwen3-4B-Instruct-2507")
ADAPTER_DIR = os.environ.get("CAIMMS_ADAPTER_DIR", f"{_ROOT}/models/delta-mem-adapter")
DATA_FILE = os.environ.get(
    "CAIMMS_DATA_FILE", f"{_ROOT}/workmem-vertical/delta-Mem/data/locomo10.json"
)
DEVICE = os.environ.get("CAIMMS_REPLAY_DEVICE", "cuda:0")
MAX_QUESTIONS = int(os.environ["REPLAY_MAX_QUESTIONS"]) if os.environ.get("REPLAY_MAX_QUESTIONS") else None
# answer_with_osam's default max_new_tokens is 2048 (see B2 in the issue
# register -- the main pipeline has this same unbounded default). For a
# diagnostic run the mask/contribution stats are captured right after the
# Phase-2 PREFILL, before decode even starts, so generation length doesn't
# affect what this script is actually checking -- only its wall-clock time.
# Capped by default so a SLURM --time budget is a real number, not a guess.
MAX_NEW_TOKENS = int(os.environ.get("REPLAY_MAX_NEW_TOKENS", "64"))


def set_delta_scaling(model, value: float) -> int:
    """Override the alpha/rank multiplier on every delta-mem module in place.

    Returns the module count so the caller can assert the override actually
    reached them -- a silent no-op here would make every arm identical and the
    whole experiment vacuous.
    """
    count = 0
    for _, module in iter_delta_mem_modules(model):
        module.delta_scaling = float(value)
        count += 1
    return count


def load_source_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            ids = (row.get("retrieval") or {}).get("final_evidence_ids")
            if not ids:
                continue
            rows.append(row)
    return rows


def resolve_evidence(graph: CueTagContentGraph, evidence_ids: list[str]) -> list[str] | None:
    """Turn stored content ids back into the exact strings that were written
    into S. Returns None if ANY id is unresolvable -- a partial replay would
    silently change what the arm sees, which is the confound this harness
    exists to eliminate, so it is better to drop the question than fake it.
    """
    texts = []
    for content_id in evidence_ids:
        node = graph.contents.get(content_id)
        if node is None:
            return None
        texts.append(node.display_text())
    return texts


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit("usage: replay_delta_scaling.py <source_run.jsonl> <scales> <out.jsonl>")

    source_run, scales_arg, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
    scales = [float(s) for s in scales_arg.split(",") if s.strip()]

    print(f"[init] replaying evidence from {source_run}", flush=True)
    rows = load_source_rows(source_run)
    if MAX_QUESTIONS is not None:
        rows = rows[:MAX_QUESTIONS]
    print(f"[init] {len(rows)} answered questions with resolvable evidence", flush=True)
    print(f"[init] arms: {scales}", flush=True)

    with open(DATA_FILE) as handle:
        samples = json.load(handle)

    graph_cache_dir = Path(source_run).parent / "graph_cache"
    graphs: dict[int, CueTagContentGraph] = {}

    def graph_for(sample_idx: int) -> CueTagContentGraph | None:
        if sample_idx not in graphs:
            path = graph_cache_dir / f"sample_{sample_idx}.json"
            if not path.exists():
                print(f"[warn] no cached graph at {path}", flush=True)
                graphs[sample_idx] = None
            else:
                graphs[sample_idx] = CueTagContentGraph.load(str(path))
        return graphs[sample_idx]

    print(f"[init] loading backbone from {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, local_files_only=True
    )
    attach_delta_adapter_in_place(model, Path(ADAPTER_DIR))
    model.eval()
    print("[init] adapter attached.\n", flush=True)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(out_file, "w") as sink:
        for scale in scales:
            n_modules = set_delta_scaling(model, scale)
            if n_modules == 0:
                raise RuntimeError(
                    "set_delta_scaling reached 0 modules -- the adapter is not attached, "
                    "so every arm would be identical and the comparison meaningless."
                )
            print(f"=== arm delta_scaling={scale} ({n_modules} modules) ===", flush=True)

            for position, row in enumerate(rows):
                sample_idx, q_idx = row["sample_idx"], row["q_idx"]
                graph = graph_for(sample_idx)
                if graph is None:
                    continue

                evidence = resolve_evidence(
                    graph, row["retrieval"]["final_evidence_ids"]
                )
                if evidence is None:
                    print(f"[skip {sample_idx}.{q_idx}] unresolvable evidence id", flush=True)
                    continue

                question = samples[sample_idx]["qa"][q_idx]
                if int(question.get("category") or 0) == ADVERSARIAL_CATEGORY:
                    continue
                q_text = question.get("question", "")

                session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=DEVICE)
                session.reset()
                populate_osam_from_evidence(session, evidence)

                prediction = ""
                contribution: dict = {}
                try:
                    out = answer_with_osam(session, q_text, max_new_tokens=MAX_NEW_TOKENS)
                    prediction = out.get("assistant", "") if isinstance(out, dict) else ""
                    if isinstance(out, dict):
                        contribution = out.get("prompt_output_ratio_stats") or {}
                except Exception as exc:  # noqa: BLE001 -- one bad question must not kill the arm
                    print(f"[gen fail {sample_idx}.{q_idx}] {exc}", flush=True)

                score = score_locomo_prediction(question, prediction)
                sink.write(
                    json.dumps(
                        {
                            "delta_scaling": scale,
                            "sample_idx": sample_idx,
                            "q_idx": q_idx,
                            "category": question.get("category"),
                            "question": q_text,
                            "n_evidence": len(evidence),
                            "evidence_ids": row["retrieval"]["final_evidence_ids"],
                            "prediction": prediction,
                            "score": score,
                            "osam_contribution": contribution,
                        }
                    )
                    + "\n"
                )
                sink.flush()
                written += 1

                if position % 50 == 0:
                    print(
                        f"[{scale} | {position}/{len(rows)}] {sample_idx}.{q_idx} "
                        f"score={score:.3f} n_ev={len(evidence)} pred={prediction[:50]!r}",
                        flush=True,
                    )

                del session
                torch.cuda.empty_cache()

            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n[done] {written} rows -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
