#!/usr/bin/env python3
"""Offline A/B for the tag-layer RRF fusion added to rank_tags_by_relevance.

Measures, WITHOUT running the LLM pipeline at all, how much gold LoCoMo
evidence survives round 1 of retrieval under two arms:

  lexical  -- DISABLE_TAG_EMBEDDER_FUSION=1 (the pre-change ranker)
  fused    -- IDF-weighted lexical fused with embedding similarity via RRF

Both arms replay the identical cue-match and traversal, so the ONLY variable
is how the ~350-tag fanout is cut down to MAX_ACTIVE_TAGS. This is the same
measurement that produced the 95.2%-reachable / 28.2%-survives figures the
change was built from.

Two metrics are reported:

  tag survival     -- a gold evidence node has >=1 of its tags in the top-15.
                      This is what the change directly moves.
  content recall   -- the gold evidence node is actually returned by
                      forward_tag_to_content() through the surviving tags and
                      then survives the MAX_NEW_CONTENT_PER_ROUND cut. This is
                      the metric that can actually move the answer score, and
                      it is the more conservative of the two.

IMPORTANT -- the embedding backend decides whether this change does anything
at all. build_default_embedding_backend() silently falls back to a token-count
backend when sentence-transformers is unavailable, and that fallback scores
0.0 on exactly the zero-overlap tags the fusion exists to rescue. The backend
actually in use is printed at the top of every run: if it is not
SentenceTransformerEmbeddingBackend, the result is not a valid test of the
change.

Usage:
    python3 scripts/validate_tag_fusion.py                 # full dataset
    python3 scripts/validate_tag_fusion.py --limit 2       # quick smoke test
    python3 scripts/validate_tag_fusion.py --limit 2 --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# MUST run before `import torch` (transitively, via sentence-transformers
# below) -- torch reads these at import time to size its internal thread
# pool, and by default sizes it to every core the OS reports, not the
# (often much smaller) share actually allotted on a shared login node. Left
# uncapped, encoding ~1000 short strings one at a time turns into heavy
# thread contention that can make the run LOOK hung when it is just very
# slow. Override by exporting these yourself before running this script if
# you know your node's real allotment.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent                      # workmem-vertical/
sys.path.insert(0, str(_REPO / "IterRet"))

from iterret.ctc_graph import CueTagContentGraph              # noqa: E402
from iterret.experience_bank import build_default_embedding_backend  # noqa: E402

# Mirror the production constants rather than re-declaring them, so this can
# never drift from what retrieve_node actually does.
from iterret.nodes import MAX_ACTIVE_CUES, MAX_ACTIVE_TAGS, MAX_NEW_CONTENT_PER_ROUND  # noqa: E402

DATA_FILE = os.environ.get("CAIMMS_DATA_FILE", f"{_REPO}/delta-Mem/data/locomo10.json")
# BUG FIXED: this used to derive from CAIMMS_ROOT, on the wrong assumption
# that it names the workspace (parent of the repo, holding outputs/). env.sh
# actually sets CAIMMS_ROOT to the REPO itself (the directory env.sh lives
# in) -- it is CAIMMS_OUTPUT_DIR / CAIMMS_WORKSPACE that name the workspace.
# Once env.sh is sourced (exactly the documented way to run this), the old
# formula appended "/outputs/graph_cache" onto the repo path and produced a
# doubled, nonexistent ".../workmem-vertical/outputs/graph_cache" -- caught
# on Mahamathi, where env.sh IS sourced (it silently worked on the author's
# Mac only because env.sh was never sourced there, so the fallback -- the
# repo's own parent directory -- happened to be correct by coincidence).
# CAIMMS_OUTPUT_DIR is what every other script in this repo already reads
# for this exact purpose (eval_locomo_iterret_mock.py's own GRAPH_CACHE_DIR,
# score_calculator.py's default, run_pipeline.sh) -- read it the same way,
# rather than re-deriving the workspace root a second, inconsistent way.
_OUTPUT_DIR = os.environ.get("CAIMMS_OUTPUT_DIR", str(_REPO.parent / "outputs"))
GRAPH_CACHE_DIR = os.environ.get("CAIMMS_GRAPH_CACHE", f"{_OUTPUT_DIR}/graph_cache")

CATEGORY_NAMES = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}


def dia_id_map(conversation: dict) -> dict:
    """dia_id -> content id, replaying conversation_to_turns' flatten order
    (all sessions in numeric order, one episodic node per turn, 1-indexed)."""
    session_keys = sorted(
        (k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    mapping, idx = {}, 0
    for key in session_keys:
        for turn in conversation[key]:
            idx += 1
            if "dia_id" in turn:
                mapping[turn["dia_id"]] = f"e{idx}"
    return mapping


def round_one(graph: CueTagContentGraph, query: str) -> tuple[set, set]:
    """Replay exactly what retrieve_node does on its first iteration:
    cue match -> cue_to_tag -> cap tags -> tag_to_content -> cap contents."""
    cues = sorted(graph.match_query_to_cues(query, max_matches=MAX_ACTIVE_CUES))
    tags = graph.forward_cue_to_tag(cues)
    if len(tags) > MAX_ACTIVE_TAGS:
        tags = set(graph.rank_tags_by_relevance(tags, query)[:MAX_ACTIVE_TAGS])
    contents = graph.forward_tag_to_content(cues, tags)
    if len(contents) > MAX_NEW_CONTENT_PER_ROUND:
        contents = set(graph.rank_contents_by_relevance(contents, query)[:MAX_NEW_CONTENT_PER_ROUND])
    return tags, contents


def run_arm(samples, fused: bool, verbose: bool = False) -> dict:
    if fused:
        os.environ.pop("DISABLE_TAG_EMBEDDER_FUSION", None)
    else:
        os.environ["DISABLE_TAG_EMBEDDER_FUSION"] = "1"

    stats = {
        "gold_total": 0, "tag_survived": 0, "content_recalled": 0,
        "reachable": 0, "questions": 0, "q_any_content": 0,
        "by_cat": defaultdict(lambda: {"gold": 0, "tag": 0, "content": 0}),
        "top_tags": {},
    }

    arm_name = "fused" if fused else "lexical"
    t_start = time.monotonic()
    for sample_idx, sample, graph in samples:
        # NOTE: the embedder stays attached in BOTH arms on purpose. Detaching
        # it for the lexical arm would also switch off the CONTENT layer's RRF
        # fusion, so the content-recall delta would conflate two changes. The
        # arm is selected purely by DISABLE_TAG_EMBEDDER_FUSION above, which
        # touches the tag layer and nothing else.
        print(f"  [{arm_name}] sample {sample_idx}: {len(sample['qa'])} questions, "
              f"{len(graph.contents)} content nodes, "
              f"{sum(len(c.tag_set) for c in graph.cues.values())} cue-tag edges "
              f"({time.monotonic() - t_start:.0f}s elapsed)", flush=True)
        mapping = dia_id_map(sample["conversation"])
        for q_idx, qa in enumerate(sample["qa"]):
            if q_idx > 0 and q_idx % 25 == 0:
                print(f"      ...{q_idx}/{len(sample['qa'])} questions "
                      f"({time.monotonic() - t_start:.0f}s elapsed)", flush=True)
            query = qa["question"]
            cat = qa.get("category", 0)
            gold_ids = [mapping.get(e) for e in qa.get("evidence", []) or []]
            gold_ids = [g for g in gold_ids if g and g in graph.contents]
            if not gold_ids:
                continue
            stats["questions"] += 1

            cues = sorted(graph.match_query_to_cues(query, max_matches=MAX_ACTIVE_CUES))
            full_fanout = graph.forward_cue_to_tag(cues)
            tags, contents = round_one(graph, query)
            stats["top_tags"][(sample_idx, q_idx)] = tuple(sorted(tags))

            hit_content = False
            for gid in gold_ids:
                stats["gold_total"] += 1
                stats["by_cat"][cat]["gold"] += 1
                node_tags = set(graph.contents[gid].tags)
                if node_tags & full_fanout:
                    stats["reachable"] += 1
                if node_tags & tags:
                    stats["tag_survived"] += 1
                    stats["by_cat"][cat]["tag"] += 1
                if gid in contents:
                    stats["content_recalled"] += 1
                    stats["by_cat"][cat]["content"] += 1
                    hit_content = True
            if hit_content:
                stats["q_any_content"] += 1
            if verbose and stats["questions"] <= 3:
                print(f"    [{'fused' if fused else 'lexical'}] q{q_idx}: {query[:58]!r}")
                print(f"        top tags: {sorted(tags)[:6]}")

    os.environ.pop("DISABLE_TAG_EMBEDDER_FUSION", None)
    return stats


def pct(num, den):
    return f"{num}/{den} = {100.0 * num / den:.1f}%" if den else "n/a"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N conversations (2 is a good smoke test)")
    ap.add_argument("--verbose", action="store_true", help="print example top-tag sets")
    args = ap.parse_args()

    backend = build_default_embedding_backend()
    name = type(backend).__name__
    print("=" * 72)
    print(f"embedding backend in use : {name}")
    if name != "SentenceTransformerEmbeddingBackend":
        print("  !! WARNING: this is the dependency-free token-count fallback.")
        print("  !! It scores 0.0 on exactly the zero-overlap tags fusion exists to")
        print("  !! rescue, so any delta below is a LOWER BOUND, not the real effect.")
        print("  !! Install sentence-transformers (and the MiniLM weights) to test properly.")
    print(f"data file                : {DATA_FILE}")
    print(f"graph cache              : {GRAPH_CACHE_DIR}")
    print(f"caps replayed            : cues={MAX_ACTIVE_CUES} tags={MAX_ACTIVE_TAGS} "
          f"contents={MAX_NEW_CONTENT_PER_ROUND}")
    print("=" * 72)

    locomo = json.load(open(DATA_FILE))
    if args.limit:
        locomo = locomo[: args.limit]

    samples = []
    for i, sample in enumerate(locomo):
        path = Path(GRAPH_CACHE_DIR) / f"sample_{i}.json"
        if not path.exists():
            print(f"[sample {i}] SKIPPED - no cached graph at {path}")
            continue
        graph = CueTagContentGraph.load(str(path))
        graph.attach_embedder(backend)
        samples.append((i, sample, graph))
    if not samples:
        sys.exit("no cached graphs found - nothing to measure")
    print(f"loaded {len(samples)} conversation graph(s)\n")

    print("running arm: lexical (DISABLE_TAG_EMBEDDER_FUSION=1) ...")
    lex = run_arm(samples, fused=False, verbose=args.verbose)
    print("running arm: fused   (RRF lexical + embedding) ...")
    fus = run_arm(samples, fused=True, verbose=args.verbose)

    g = lex["gold_total"]
    print("\n" + "=" * 72)
    print(f"questions with usable gold evidence : {lex['questions']}")
    print(f"gold evidence nodes                 : {g}")
    print(f"reachable via cue->tag (both arms)  : {pct(lex['reachable'], g)}")
    print("-" * 72)
    print(f"{'metric':<34}{'lexical':>14}{'fused':>14}{'delta':>10}")
    for label, key in (("gold tag survives top-%d" % MAX_ACTIVE_TAGS, "tag_survived"),
                       ("gold content recalled rd.1", "content_recalled")):
        lv, fv = lex[key], fus[key]
        print(f"{label:<34}{100.0*lv/g:>13.1f}%{100.0*fv/g:>13.1f}%{100.0*(fv-lv)/g:>+9.1f}pp")
    lq, fq = lex["q_any_content"], fus["q_any_content"]
    n = lex["questions"]
    print(f"{'questions w/ >=1 gold recalled':<34}{100.0*lq/n:>13.1f}%{100.0*fq/n:>13.1f}%"
          f"{100.0*(fq-lq)/n:>+9.1f}pp")

    changed = sum(1 for k, v in lex["top_tags"].items() if fus["top_tags"].get(k) != v)
    print("-" * 72)
    print(f"top-{MAX_ACTIVE_TAGS} tag set changed by fusion : {pct(changed, len(lex['top_tags']))}")
    print("  (0% would mean the fusion is inert - check the backend warning above)")

    print("-" * 72)
    print(f"{'category':<18}{'gold':>7}{'tag lex':>10}{'tag fus':>10}{'cnt lex':>10}{'cnt fus':>10}")
    for cat in sorted(lex["by_cat"]):
        lc, fc = lex["by_cat"][cat], fus["by_cat"][cat]
        gg = lc["gold"] or 1
        print(f"{CATEGORY_NAMES.get(cat, cat):<18}{lc['gold']:>7}"
              f"{100.0*lc['tag']/gg:>9.1f}%{100.0*fc['tag']/gg:>9.1f}%"
              f"{100.0*lc['content']/gg:>9.1f}%{100.0*fc['content']/gg:>9.1f}%")
    print("=" * 72)
