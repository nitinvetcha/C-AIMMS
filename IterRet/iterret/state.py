
from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict


class SearchStep(TypedDict):
    iteration: int
    module: Literal["Planning", "Reflection"]  # R2-Mem step typing (Eq. 6)
    query_used: str
    action_taken: str
    found_summary: str
    decision: str  # "retrieve" | "reflect" | "answer"


class IterRetState(TypedDict, total=False):
    original_query: str  # q
    current_refined_query: str  # q^ret_k = q (+) delta_q_k  (MemR3 Eq. 5)
    accumulated_evidence: List[str]  # E_k / H^(t)
    accumulated_evidence_ids: List[str]  # content ids, index-aligned with accumulated_evidence
    route_diagnostics: List[dict]  # per-round record of how f_route actually decided
    action_parse_failures: int  # rounds where the action-selection reply was unparseable
    information_gaps: List[str]  # G_k
    active_set: Dict[str, List[str]]  # Z^(t): {"cues": [...], "tags": [...], "contents": [...]}
    visited_content_ids: List[str]  # masked retrieval, MemR3 Eq. 5: M \ M_ret_{k-1}
    search_trajectory: List[SearchStep]
    iteration_count: int
    max_iterations: int  # n_max
    consecutive_stuck_reflects: int  # toward n_cap
    _scratch_new_retrieval: List[str]  # content ids retrieved this round, awaiting Reflect's f_route
    final_answer: Optional[str]


DEFAULT_MAX_ITERATIONS = 5  # n_max, MemR3 main config
DEFAULT_MAX_STUCK_REFLECTS = 2  # n_cap


def new_state(original_query: str, *, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> IterRetState:
    return IterRetState(
        original_query=original_query,
        current_refined_query=original_query,
        accumulated_evidence=[],
        # Empty, not a sentinel string: the old "initial: no evidence gathered
        # yet" only cleared if reflect_node's LLM echoed it verbatim in
        # resolved_gaps, which it essentially never did (run 12b: 1476/1540
        # questions ran all 5 rounds), so the "no gaps -> answer" exit was dead.
        information_gaps=[],
        active_set={"cues": [], "tags": [], "contents": []},
        visited_content_ids=[],
        search_trajectory=[],
        iteration_count=0,
        max_iterations=max_iterations,
        consecutive_stuck_reflects=0,
        _scratch_new_retrieval=[],
        final_answer=None,
    )
