
from __future__ import annotations

import json
from typing import Literal

from .ctc_graph import CueTagContentGraph
from .experience_bank import ExperienceBank, planning_condition, reflection_condition
from .json_utils import parse_json_object
from .llm_client import LLMClient
from .state import DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_STUCK_REFLECTS, IterRetState, SearchStep

MAX_ACTIVE_CUES = 40

MAX_ACTIVE_TAGS = 15
MAX_NEW_CONTENT_PER_ROUND = 25

_SITUATION_SYSTEM_PROMPT = """situation_abstraction
Abstract the given condition into a short, general situation description
(no concrete entity names), so it can be matched against past experience.
Reply as JSON: {"situation": str}.
"""

_ACTION_SELECTION_SYSTEM_PROMPT = """action_selection
You control traversal over a Cue-Tag-Content memory graph (MRAgent style).
Given the query, the current active set, and any retrieved Planning
experience advice, choose which traversal action(s) to take this round:
"cue_to_tag", "tag_to_content", and/or "content_to_cue_tag". Also list any
tags that should be excluded based on the experience advice.
In most rounds you should select BOTH "cue_to_tag" AND "tag_to_content"
together: activating tags without then fetching the content behind them
makes no progress and wastes a round. The active_set's tags/cues are
already ordered with the most relevant ones first, so it is normal (and
expected) for later entries to look unrelated -- do not let that stop you
from selecting "tag_to_content" once any clearly relevant tag is present.
Only select "cue_to_tag" alone if there are currently no tags at all yet.
Reply as JSON: {"actions": [str, ...], "exclude_tags": [str, ...]}.
"""

_ROUTING_SYSTEM_PROMPT = """routing_and_reflection
You perform f_route (prune/merge newly retrieved content into evidence)
and f_reflect (judge whether gaps remain and propose a refined query) for
an active memory reconstruction loop (MRAgent + MemR3 style), informed by
retrieved Reflection experience advice.
"new_content" is a list of {"id": str, "text": str} objects -- each "id"
is that item's unique content id. "kept_content_ids" MUST be either the
literal string "ALL", or a list containing only those "id" values for the
items worth keeping as evidence (never the "text" itself, and never an id
that isn't in "new_content").
Reply as JSON: {"kept_content_ids": "ALL" or [str, ...], "resolved_gaps": [str, ...],
"new_gaps": [str, ...], "next_query": str}.
"""

_ANSWER_SYSTEM_PROMPT = """final_answer
Synthesize a final answer to the question using ONLY the provided evidence
bullets. Do not reference gaps, search trajectory, or graph internals.
"""


def abstract_situation(condition: str, llm: LLMClient) -> str:
    """LLM_sigma(c_i): abstracts a condition into a matchable situation (R2-Mem Eq. 11)."""
    raw = llm.chat(_SITUATION_SYSTEM_PROMPT, json.dumps({"condition": condition}))
    return parse_json_object(raw).get("situation") or condition


def retrieve_node(state: IterRetState, graph: CueTagContentGraph, bank: ExperienceBank,
                   llm: LLMClient) -> IterRetState:
    query = state.get("current_refined_query") or state["original_query"]
    active_set = state.setdefault("active_set", {"cues": [], "tags": [], "contents": []})
    visited = set(state.get("visited_content_ids", []))

    if not active_set["cues"]:
        active_set["cues"] = sorted(graph.match_query_to_cues(query, max_matches=MAX_ACTIVE_CUES))

    # Planning experience retrieval (R2-Mem Eq. 12, c_i = q_i)
    condition = planning_condition(query)
    situation = abstract_situation(condition, llm)
    advice_entries = bank.retrieve(condition, situation, module="Planning")
    advice_text = "; ".join(entry["experience"] for entry in advice_entries)

    # Action selection f_select (Eq. 10), Planning-experience-guided
    selection_prompt = json.dumps({
        "query": query,
        "active_set": active_set,
        "planning_advice": advice_text,
    })
    raw = llm.chat(_ACTION_SELECTION_SYSTEM_PROMPT, selection_prompt)
    parsed = parse_json_object(raw)
    actions = parsed.get("actions") or ["cue_to_tag", "tag_to_content"]
    exclude_tags = set(parsed.get("exclude_tags", []))

    # Controlled traversal (Eq. 11), masked against visited_content_ids (MemR3 Eq. 5)
    new_tags = set(active_set["tags"])
    new_contents: set = set()
    if "cue_to_tag" in actions:
        new_tags |= graph.forward_cue_to_tag(active_set["cues"])
        if len(new_tags) > MAX_ACTIVE_TAGS:
            new_tags = set(graph.rank_tags_by_relevance(new_tags, query)[:MAX_ACTIVE_TAGS])
    if "tag_to_content" in actions:
        new_contents |= graph.forward_tag_to_content(
            active_set["cues"], new_tags, exclude_tags=exclude_tags, exclude_content_ids=visited,
        )
        if len(new_contents) > MAX_NEW_CONTENT_PER_ROUND:
            new_contents = set(graph.rank_contents_by_relevance(new_contents, query)[:MAX_NEW_CONTENT_PER_ROUND])
    if "content_to_cue_tag" in actions and active_set["contents"]:
        for cue_id, tag in graph.reverse_content_to_cue_tag(active_set["contents"]):
            if cue_id not in active_set["cues"] and len(active_set["cues"]) < MAX_ACTIVE_CUES:
                active_set["cues"].append(cue_id)
            if len(new_tags) < MAX_ACTIVE_TAGS:
                new_tags.add(tag)

    # Order by relevance to the current query, not alphabetically: the whole point of capping
    # via rank_*_by_relevance above is to put the genuinely relevant items first, and an LLM
    # reading a long list pays the most attention to what's earliest in it -- re-sorting
    # alphabetically here would undo that by burying e.g. "LGBTQ Support Group" behind 50+
    # alphabetically-earlier but irrelevant tags before the model ever reads that far.
    active_set["tags"] = graph.rank_tags_by_relevance(new_tags, query)
    active_set["contents"] = sorted(set(active_set["contents"]) | new_contents)

    state["_scratch_new_retrieval"] = graph.rank_contents_by_relevance(new_contents, query)
    state["active_set"] = active_set
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    trajectory = state.setdefault("search_trajectory", [])
    trajectory.append(SearchStep(
        iteration=state["iteration_count"],
        module="Planning",
        query_used=query,
        action_taken="+".join(actions) + (f" (excluded tags: {sorted(exclude_tags)})" if exclude_tags else ""),
        found_summary=f"{len(new_contents)} new content node(s)",
        decision="retrieve",
    ))
    return state


def reflect_node(state: IterRetState, graph: CueTagContentGraph, bank: ExperienceBank,
                  llm: LLMClient) -> IterRetState:
    new_content_ids = state.get("_scratch_new_retrieval", [])
    new_content_ids = [cid for cid in new_content_ids if cid in graph.contents]
    # The LLM can only return ids for content it was actually shown an id
    # for, so pair each id with its text rather than handing over bare text
    # (which it would otherwise just echo back, never matching a real id).
    new_content_payload = [{"id": cid, "text": graph.contents[cid].display_text()} for cid in new_content_ids]

    original_query = state["original_query"]
    evidence = state.setdefault("accumulated_evidence", [])
    gaps = state.setdefault("information_gaps", [])

    # Reflection experience retrieval (R2-Mem Eq. 12, c_i = [Q + m_i])
    condition = reflection_condition(original_query, evidence)
    situation = abstract_situation(condition, llm)
    advice_entries = bank.retrieve(condition, situation, module="Reflection")
    advice_text = "; ".join(entry["experience"] for entry in advice_entries)

    routing_prompt = json.dumps({
        "original_query": original_query,
        "evidence": evidence,
        "gaps": gaps,
        "new_content": new_content_payload,
        "reflection_advice": advice_text,
    })
    raw = llm.chat(_ROUTING_SYSTEM_PROMPT, routing_prompt)
    parsed = parse_json_object(raw)

    kept = parsed.get("kept_content_ids", "ALL")
    if kept == "ALL":
        kept_ids = new_content_ids
    else:
        kept_ids = [cid for cid in kept if cid in graph.contents]
        if not kept_ids and new_content_ids:
            # The model didn't return ids that match anything it was shown
            # (e.g. it echoed text instead of ids) -- fail open instead of
            # silently discarding everything that was just retrieved.
            kept_ids = new_content_ids
    kept_texts = [graph.contents[cid].display_text() for cid in kept_ids]
    made_progress = bool(kept_texts)
    for text in kept_texts:
        if text not in evidence:
            evidence.append(text)

    resolved = set(parsed.get("resolved_gaps", []))
    remaining_gaps = [g for g in gaps if g not in resolved]
    for new_gap in parsed.get("new_gaps", []):
        if new_gap not in remaining_gaps:
            remaining_gaps.append(new_gap)

    state["information_gaps"] = remaining_gaps
    state["accumulated_evidence"] = evidence
    state["visited_content_ids"] = sorted(set(state.get("visited_content_ids", [])) | set(new_content_ids))
    state["_scratch_new_retrieval"] = []
    state["consecutive_stuck_reflects"] = 0 if made_progress else state.get("consecutive_stuck_reflects", 0) + 1

    next_query = parsed.get("next_query") or ""
    if remaining_gaps and next_query:
        state["current_refined_query"] = f"{original_query} | {next_query}"
    elif not remaining_gaps:
        state["current_refined_query"] = original_query

    trajectory = state.setdefault("search_trajectory", [])
    trajectory.append(SearchStep(
        iteration=state.get("iteration_count", 0),
        module="Reflection",
        query_used=state.get("current_refined_query", original_query),
        action_taken="f_route+gap_update",
        found_summary=f"kept {len(kept_texts)} item(s); {len(remaining_gaps)} gap(s) remain",
        decision="reflect",
    ))
    return state


def route_passthrough_node(state: IterRetState) -> IterRetState:
    """No-op node so routing has its own graph node, mirroring MemR3's
    explicit `router` node (paper Sec. 3.4)."""
    return state


def route_after_reflect(state: IterRetState) -> Literal["retrieve", "answer"]:
    """MemR3 Algorithm 1's router policy: pure function of state, no mutation."""
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    gaps = state.get("information_gaps", [])
    stuck = state.get("consecutive_stuck_reflects", 0)

    if iteration >= max_iterations:
        return "answer"
    if not gaps:
        return "answer"
    if stuck >= DEFAULT_MAX_STUCK_REFLECTS:
        return "answer"
    return "retrieve"


def answer_node(state: IterRetState, llm: LLMClient) -> IterRetState:
    evidence_block = "\n".join(f"- {item}" for item in state.get("accumulated_evidence", []))
    user_prompt = f"Question: {state['original_query']}\nEvidence:\n{evidence_block}"
    state["final_answer"] = llm.chat(_ANSWER_SYSTEM_PROMPT, user_prompt)

    trajectory = state.setdefault("search_trajectory", [])
    trajectory.append(SearchStep(
        iteration=state.get("iteration_count", 0),
        module="Reflection",
        query_used=state["original_query"],
        action_taken="answer",
        found_summary="final answer synthesized",
        decision="answer",
    ))
    return state
