"""ITERRET offline phase: build the experience bank from a 10% bootstrap split.

Implements the paper's Sec. 2(iii) / Appendix A "Offline bootstrap" exactly:

  "A 10% bootstrap subset of conversations is used to collect unguided
   trajectories; a rubric evaluator scores each Retrieve/Reflect step (query
   specificity, evidence coverage, gap-resolution rate, redundancy) and a
   self-reflection learner distills both high- and low-quality steps into
   'if <situation> then <action>' guidance, stored in a planning bank and a
   reflection bank."

  "The bootstrap phase collects 10 unguided trajectories by default (one
   bootstrap conversation x 10 seed questions)."

Concretely, four stages:

  1. Split      -- conversation 0 bootstraps, 1-9 are held out (bootstrap_split).
  2. Trajectory -- run the REAL online controller over conversation 0's graph
                   with an EMPTY bank ("unguided"), for 10 seed questions.
  3. Score      -- evaluator.score_step on every Planning/Reflection step,
                   classified good / bad / discard against K_LOW=5, K_HIGH=10.
  4. Distill    -- learner.distill_experience turns each kept step into an
                   "IF <situation> THEN <strategy>" entry in its module's bank.

Run it with a vLLM server already up; see scripts/run_offline_bank.slurm.

Why this file exists at all, rather than calling
``iterret.offline_pipeline.construct_experience_banks``: that module does
``from .graph import build_graph`` at import time, and graph.py imports
``langgraph``, which is not installed in this project's environment (the
online path deliberately bypasses LangGraph -- see iterret_bridge.py). So
offline_pipeline cannot even be imported here. Stages 3 and 4 below therefore
call ``evaluator`` and ``learner`` directly, which is exactly what
construct_experience_banks does internally, minus the unusable dependency.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from deltamem.eval.locomo_protocol import ADVERSARIAL_CATEGORY
from deltamem.workmem.bootstrap_split import n_bootstrap_samples, split_description
from deltamem.workmem.iterret_bridge import get_iterret_evidence

from iterret import evaluator, learner
from iterret.ctc_graph import CueTagContentGraph
from iterret.experience_bank import (
    ExperienceBank,
    build_default_embedding_backend,
    empty_experience_bank,
)
from iterret.llm_client import OpenAICompatibleLLMClient
from iterret.memory_builder import DialogueTurn, build_ctc_graph_from_dialogue

_ROOT = os.environ.get("CAIMMS_ROOT", "/home/kbasu/arnavbhatt/workmem_test")
DATA_FILE = os.environ.get(
    "CAIMMS_DATA_FILE", f"{_ROOT}/workmem-vertical/delta-Mem/data/locomo10.json"
)
_OUT_DIR = os.environ.get("CAIMMS_OUTPUT_DIR", f"{_ROOT}/outputs")
BANK_PATH = os.environ.get("CAIMMS_BANK_PATH", f"{_OUT_DIR}/experience_bank.json")
# Full per-step audit trail (score, quality, reason, advice, distilled text).
# The paper's reproducibility appendix promises "the distilled experience bank
# JSON artifacts from the offline phase"; the bank itself only stores the
# surviving entries, so without this there is no record of what was discarded
# or why a given entry exists.
TRACE_PATH = os.environ.get("CAIMMS_BANK_TRACE", f"{_OUT_DIR}/experience_bank_trace.json")

VLLM_BASE_URL = os.environ.get("CAIMMS_VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# n_max = 5 for every run in the paper, offline and online alike.
ITERRET_MAX_ITERATIONS = int(os.environ.get("CAIMMS_MAX_ITERATIONS", "5"))
# "one bootstrap conversation x 10 seed questions".
N_SEED_QUESTIONS = int(os.environ.get("CAIMMS_BOOTSTRAP_SEED_QUESTIONS", "10"))

# Which bank a scored step is distilled into.
#   "native" -- a Planning step feeds the planning bank, a Reflection step the
#               reflection bank. The evaluator already scores each step against
#               its OWN module's rubrics (PLANNING_RUBRICS vs REFLECTION_RUBRICS),
#               and SearchStep.module is typed Literal["Planning","Reflection"],
#               so this is the reading the paper's own data model implies.
#   "both"   -- every kept step is distilled into BOTH banks. This is what
#               iterret.offline_pipeline.construct_experience_banks does, so it
#               is kept reachable for reproducing that path, but it puts
#               planning-rubric-derived advice into the reflection bank, which
#               is cross-contamination rather than a design choice.
MODULE_MODE = os.environ.get("CAIMMS_BANK_MODULE_MODE", "native")


def extract_session_nums(conv_block: dict) -> List[int]:
    import re

    return sorted(
        int(m.group(1))
        for m in (re.fullmatch(r"session_(\d+)", k) for k in conv_block)
        if m
    )


def build_turns(conv_block: dict) -> List[DialogueTurn]:
    """Identical turn construction to the two eval scripts, including the
    "[timestamp] Speaker: text" prefix. It has to be identical: the CTC graph
    cache is SHARED with them, so a graph built here must be byte-compatible
    with one they would have built."""
    turns: List[DialogueTurn] = []
    for session_num in extract_session_nums(conv_block):
        session_ts = conv_block.get(
            f"session_{session_num}_date_time", f"Session {session_num}"
        )
        for dialog in conv_block.get(f"session_{session_num}", []):
            speaker = dialog.get("speaker", "Unknown")
            text = dialog.get("text", "")
            turns.append(
                DialogueTurn(
                    speaker=speaker,
                    text=f"[{session_ts}] {speaker}: {text}",
                    time=str(session_ts),
                )
            )
    return turns


def select_seed_questions(questions: List[dict], n: int) -> List[dict]:
    """Pick ``n`` seed questions, deterministically and category-balanced.

    The paper fixes the COUNT (10) but not the selection rule. Taking the
    first 10 in file order would be a real problem here: LoCoMo's questions
    are not shuffled, and conversation 0's first 10 are dominated by a single
    category, so the rubric evaluator would only ever see one kind of
    retrieval situation and the distilled advice would generalize to one
    quarter of the eval set.

    Round-robin over categories (1,2,3,4 cycling, original order within each)
    gives all four question types representation while staying a pure function
    of the data -- no RNG, no seed to record, byte-identical on every re-run.
    """
    by_cat: Dict[int, List[dict]] = {}
    for q in questions:
        try:
            cat = int(q.get("category", 0))
        except (TypeError, ValueError):
            continue
        if cat == ADVERSARIAL_CATEGORY:
            continue  # excluded from this benchmark everywhere else too
        by_cat.setdefault(cat, []).append(q)

    selected: List[dict] = []
    cats = sorted(by_cat)
    round_idx = 0
    while len(selected) < n and any(len(by_cat[c]) > round_idx for c in cats):
        for cat in cats:
            if len(selected) >= n:
                break
            if len(by_cat[cat]) > round_idx:
                selected.append(by_cat[cat][round_idx])
        round_idx += 1
    return selected


def collect_unguided_trajectory(
    question: str, graph: CueTagContentGraph, llm: OpenAICompatibleLLMClient
) -> Dict[str, Any]:
    """Stage 2. Run the controller with an EMPTY bank.

    "Unguided" is the whole point: the bootstrap pass must not be steered by
    experience, because the experience does not exist yet. empty_experience_bank()
    returns a bank whose .retrieve() short-circuits to [] on an empty bank, so
    both advice strings come back "" and the prompts carry no guidance.

    Runs through get_iterret_evidence -- the same function the online eval
    calls -- so the trajectories scored below are produced by the controller
    that actually runs at inference, not by a parallel reimplementation.
    """
    unguided_bank = empty_experience_bank()
    state: Dict[str, Any] = {}
    diag: Dict[str, Any] = {}
    evidence = get_iterret_evidence(
        question,
        graph,
        unguided_bank,
        llm,
        max_iterations=ITERRET_MAX_ITERATIONS,
        diag=diag,
        state_out=state,
    )
    return {
        "original_query": question,
        "final_evidence": evidence,
        "steps": list(state.get("search_trajectory", [])),
        "rounds": diag.get("rounds", 0),
        "stop_reason": diag.get("stop_reason", "?"),
    }


def distil_bank(
    trajectory_records: List[Dict[str, Any]], llm: OpenAICompatibleLLMClient
) -> tuple[ExperienceBank, List[dict]]:
    """Stages 3 and 4: rubric-score every step, then distill the survivors."""
    bank = ExperienceBank(build_default_embedding_backend())
    trace: List[dict] = []

    for record in trajectory_records:
        original_query = record["original_query"]
        final_evidence = record["final_evidence"]

        for step in record["steps"]:
            if step["decision"] == "answer":
                continue  # rubrics target Planning/Reflection steps only

            score, reason, advice = evaluator.score_step(step, llm)
            quality = evaluator.classify(
                score, k_low=evaluator.K_LOW, k_high=evaluator.K_HIGH
            )
            row = {
                "query": original_query,
                "module": step["module"],
                "iteration": step["iteration"],
                "action_taken": step["action_taken"],
                "found_summary": step["found_summary"],
                "score": score,
                "quality": quality,
                "reason": reason,
                "advice": advice,
                "distilled": [],
            }
            # "distills both high- and low-quality steps" -- good AND bad are
            # both instructive ("IF <situation> THEN <do/avoid>"); only the
            # ambiguous middle band (K_LOW <= score <= K_HIGH) is dropped.
            if quality == "discard":
                trace.append(row)
                continue

            modules = (
                (step["module"],) if MODULE_MODE == "native" else ("Planning", "Reflection")
            )
            for module in modules:
                entry = learner.distill_experience(
                    step,
                    reason,
                    advice,
                    quality,
                    module,
                    llm,
                    original_query=original_query,
                    accumulated_evidence=final_evidence,
                )
                bank.add_experience(
                    entry["condition"], entry["situation"], entry["experience"], module
                )
                row["distilled"].append(
                    {
                        "module": module,
                        "situation": entry["situation"],
                        "experience": entry["experience"],
                    }
                )
            trace.append(row)

    return bank, trace


def main() -> None:
    n_boot = n_bootstrap_samples()
    if n_boot < 1:
        sys.exit(
            "CAIMMS_BOOTSTRAP_SAMPLES is 0, so there is no bootstrap split and no\n"
            "conversation to build a bank from. The paper's protocol is\n"
            "CAIMMS_BOOTSTRAP_SAMPLES=1 (conversation 0 bootstraps, 1-9 held out)."
        )
    if MODULE_MODE not in ("native", "both"):
        sys.exit(f"CAIMMS_BANK_MODULE_MODE must be 'native' or 'both', got {MODULE_MODE!r}")

    print("=" * 62, flush=True)
    print("  ITERRET OFFLINE PHASE -- experience bank construction", flush=True)
    print(f"  split      : {split_description()}", flush=True)
    print(f"  seed qs    : {N_SEED_QUESTIONS} per bootstrap conversation", flush=True)
    print(f"  n_max      : {ITERRET_MAX_ITERATIONS}", flush=True)
    print(f"  module mode: {MODULE_MODE}", flush=True)
    print(f"  bank out   : {BANK_PATH}", flush=True)
    print("=" * 62, flush=True)

    with open(DATA_FILE) as f:
        samples = json.load(f)
    print(f"[init] {len(samples)} conversations loaded from {DATA_FILE}", flush=True)

    llm = OpenAICompatibleLLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME)

    # Shared with both eval scripts. Conversation 0's graph is almost certainly
    # already cached from a previous run, and reusing it saves the ~600 vLLM
    # calls build_ctc_graph_from_dialogue would otherwise spend rebuilding an
    # identical graph.
    graph_cache_dir = Path(_OUT_DIR) / "graph_cache"
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []

    for sample_idx in range(n_boot):
        sample = samples[sample_idx]
        conv_block = sample.get("conversation", {})
        questions = sample.get("qa", [])

        turns = build_turns(conv_block)
        if not turns:
            print(f"[bootstrap {sample_idx}] no turns, skipping.", flush=True)
            continue

        cache_path = graph_cache_dir / f"sample_{sample_idx}.json"
        if cache_path.exists():
            print(f"[bootstrap {sample_idx}] loading cached CTC graph {cache_path}", flush=True)
            graph = CueTagContentGraph.load(str(cache_path))
        else:
            print(
                f"[bootstrap {sample_idx}] building CTC graph ({len(turns)} turns)...",
                flush=True,
            )
            graph = build_ctc_graph_from_dialogue(turns, llm)
            graph.save(str(cache_path))
        print(f"[bootstrap {sample_idx}] graph ready: {len(graph.contents)} content nodes", flush=True)

        # The online eval attaches an embedder to the graph, which turns on
        # semantic cue matching and the RRF-fused content ranker. Attach one
        # here too, or the bootstrap trajectories would be produced by a
        # measurably weaker retriever than the one the advice is meant to guide.
        graph.attach_embedder(build_default_embedding_backend())

        seeds = select_seed_questions(questions, N_SEED_QUESTIONS)
        print(
            f"[bootstrap {sample_idx}] {len(seeds)} seed question(s), categories "
            f"{[q.get('category') for q in seeds]}",
            flush=True,
        )

        for i, q in enumerate(seeds):
            q_text = q.get("question", "")
            try:
                record = collect_unguided_trajectory(q_text, graph, llm)
            except Exception as exc:  # one bad seed shouldn't lose the rest
                print(f"  [seed {i}] FAILED, skipping: {exc}", flush=True)
                continue
            all_records.append(record)
            print(
                f"  [seed {i}] cat={q.get('category')} rounds={record['rounds']} "
                f"stop={record['stop_reason']} steps={len(record['steps'])} "
                f"evidence={len(record['final_evidence'])} q={q_text[:52]!r}",
                flush=True,
            )

    n_steps = sum(len(r["steps"]) for r in all_records)
    print(
        f"\n[offline] {len(all_records)} unguided trajectories, {n_steps} steps total.",
        flush=True,
    )
    if not all_records:
        sys.exit("No trajectories collected -- nothing to distill. Is the vLLM server up?")

    print("[offline] scoring steps and distilling experience...", flush=True)
    bank, trace = distil_bank(all_records, llm)

    Path(BANK_PATH).parent.mkdir(parents=True, exist_ok=True)
    bank.save(BANK_PATH)
    with open(TRACE_PATH, "w") as f:
        json.dump(
            {
                "n_bootstrap_conversations": n_boot,
                "n_seed_questions": N_SEED_QUESTIONS,
                "max_iterations": ITERRET_MAX_ITERATIONS,
                "module_mode": MODULE_MODE,
                "k_low": evaluator.K_LOW,
                "k_high": evaluator.K_HIGH,
                "steps": trace,
            },
            f,
            indent=2,
        )

    kept = sum(1 for row in trace if row["quality"] != "discard")
    good = sum(1 for row in trace if row["quality"] == "good")
    bad = sum(1 for row in trace if row["quality"] == "bad")
    print(f"\n{'=' * 62}", flush=True)
    print(f"  steps scored     : {len(trace)}", flush=True)
    print(f"  kept             : {kept}  (good {good} / bad {bad})", flush=True)
    print(f"  discarded (mid)  : {len(trace) - kept}", flush=True)
    print(f"  planning bank    : {len(bank.planning_bank)} entries", flush=True)
    print(f"  reflection bank  : {len(bank.reflection_bank)} entries", flush=True)
    print(f"  bank  -> {BANK_PATH}", flush=True)
    print(f"  trace -> {TRACE_PATH}", flush=True)
    print("=" * 62, flush=True)

    if not bank.planning_bank and not bank.reflection_bank:
        sys.exit(
            "\nERROR: the bank is EMPTY -- every step landed in the discard band.\n"
            "Running the ablation against this would be identical to the no-bank arm.\n"
            "Check the trace for the score distribution before re-running."
        )


if __name__ == "__main__":
    main()
