"""Build a Cue-Tag-Content graph from a document's event spans.

The document analogue of ``iterret.memory_builder.build_ctc_graph_from_dialogue``:
each event span (from surprise segmentation or fixed chunking) becomes one
episodic content node. The semantic and topic layers are built by *reusing*
``iterret.memory_builder``'s distillation helpers unchanged -- only the
episodic extraction differs, because a document passage has no speakers.
"""

from __future__ import annotations

import json
from typing import List, Sequence

from iterret.ctc_graph import CueTagContentGraph
from iterret.llm_client import LLMClient
from iterret.memory_builder import (
    DEFAULT_MAX_CHARS_PER_CALL,
    _abstract_topics,
    _iter_chunks,
    _safe_chat,
)

_DOC_EVENT_EXTRACTION_SYSTEM_PROMPT = """episode_extraction
You build a Cue-Tag-Content memory graph from a passage of a long document.
Read the passage and produce:
- "tag": a short phrase (<=4 words) naming what the passage is about
  (e.g. "Trial Verdict", "Market Crash", "Method Setup").
- "cues": 5-12 fine-grained cues -- named entities (people, places,
  organizations), dates, numbers, technical terms, or salient key nouns
  explicitly mentioned in the passage. Extract generously: these are the
  surface-form anchors a later query will match on, so more coverage means
  more of the passage is retrievable.
Reply as JSON: {"tag": str, "cues": [str, ...]}.
"""

_DOC_SEMANTIC_EXTRACTION_SYSTEM_PROMPT = """semantic_extraction
You extract stand-alone factual statements from a passage of a long document
(the Cue-Tag-Content semantic layer) -- definitions, attributes, results,
relationships, or claims that a reader might later ask about. For each fact,
give the entity/topic cue it is anchored to (a name, term, or key noun), an
aspect tag (e.g. "Definition", "Result", "Property", "Cause"), and the fact as
one short self-contained sentence.
Reply as JSON: {"semantics": [{"cue": str, "tag": str, "content": str}, ...]}.
Extract several facts per passage where possible; reply {"semantics": []} only
if the passage contains no factual content.
"""


def _extract_doc_event(span_text: str, llm: LLMClient) -> dict:
    parsed = _safe_chat(
        _DOC_EVENT_EXTRACTION_SYSTEM_PROMPT, json.dumps({"text": span_text}), llm,
        on_error="doc event extraction failed, using fallback tag/cues",
    )
    tag = str(parsed.get("tag") or "Passage")
    cues = [str(c) for c in parsed.get("cues") or [] if str(c).strip()]
    if not cues:
        cues = ["Unknown"]
    return {"tag": tag, "cues": cues}


def _extract_doc_semantics(episode_summaries: List[dict], llm: LLMClient, *, max_chars: int) -> List[dict]:
    """Prose-tuned semantic extraction (the dialogue prompt in
    iterret.memory_builder asks for 'personal attributes/preferences', which
    finds little in technical documents)."""
    cleaned: List[dict] = []
    for chunk in _iter_chunks(episode_summaries, text_key="text", max_chars=max_chars):
        full_text = "\n".join(s["text"] for s in chunk)
        parsed = _safe_chat(_DOC_SEMANTIC_EXTRACTION_SYSTEM_PROMPT, full_text, llm,
                            on_error=f"doc semantic extraction skipped a {len(chunk)}-passage chunk")
        for item in parsed.get("semantics") or []:
            if not isinstance(item, dict):
                continue
            cue, tag, content = item.get("cue"), item.get("tag"), item.get("content")
            if cue and tag and content:
                cleaned.append({"cue": str(cue), "tag": str(tag), "content": str(content)})
    return cleaned


def build_ctc_graph_from_document(
    spans: Sequence[str], llm: LLMClient, *,
    max_chars_per_call: int = DEFAULT_MAX_CHARS_PER_CALL,
) -> CueTagContentGraph:
    """Assemble the three-layer CTC graph for one document.

    ``spans`` are the event texts (surprise-bounded events, or fixed chunks for
    the baseline). One episodic node per span; semantic + topic layers ride on
    top via the shared ``iterret.memory_builder`` distillation.
    """
    graph = CueTagContentGraph()
    episode_summaries: List[dict] = []

    # 1. Episodic layer: one (cue, tag, span) set of links per event span.
    for i, span in enumerate(spans):
        content_id = f"e{i + 1}"
        graph.add_content(content_id, span, layer="episodic", time=None)
        extracted = _extract_doc_event(span, llm)
        for cue in extracted["cues"]:
            graph.link(cue, extracted["tag"], content_id)
        episode_summaries.append({"content_id": content_id, "tag": extracted["tag"], "text": span})

    # 2. Semantic layer (prose-tuned; adds distilled fact nodes that give the
    #    retriever more anchors than the raw passages alone).
    for j, semantic in enumerate(_extract_doc_semantics(episode_summaries, llm, max_chars=max_chars_per_call)):
        content_id = f"s{j + 1}"
        graph.add_content(content_id, semantic["content"], layer="semantic")
        graph.link(semantic["cue"], semantic["tag"], content_id)

    # 3. Topic layer (reused unchanged from the dialogue builder).
    for k, topic in enumerate(_abstract_topics(episode_summaries, llm, max_chars=max_chars_per_call)):
        content_id = f"t{k + 1}"
        graph.add_content(content_id, f"Topic: {topic['topic']}", layer="topic",
                          topic_links=topic["episode_ids"])

    return graph
