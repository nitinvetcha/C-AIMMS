
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple

ContentLayer = Literal["episodic", "semantic", "topic"]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> Set[str]:
    return set(_WORD_RE.findall(text.lower()))


@dataclass
class CueNode:
    cue_id: str
    tag_set: Set[str] = field(default_factory=set)


@dataclass
class ContentNode:
    content_id: str
    text: str
    layer: ContentLayer = "episodic"
    time: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    topic_links: List[str] = field(default_factory=list)  # for layer == "topic": episode ids

    def display_text(self) -> str:
        """Text as it should be shown to the LLM: episodic memories carry a
        timestamp (MRAgent Sec. 3.2: "episodic memories are further
        organized along a unified timeline, allowing temporal constraints
        to be imposed during reconstruction") -- without surfacing it here,
        that timeline is stored but never actually reaches the model, so
        temporal questions become unanswerable even when the right episode
        was retrieved.
        """
        return f"[{self.time}] {self.text}" if self.time else self.text


class CueTagContentGraph:
    """The CTC memory graph plus its traversal operators."""

    def __init__(self) -> None:
        self.cues: Dict[str, CueNode] = {}
        self.contents: Dict[str, ContentNode] = {}
        # phi_(c,g)->v : (cue, tag) -> content ids
        self.cue_tag_to_content: Dict[Tuple[str, str], Set[str]] = {}
        # phi_v->(c,g) : content id -> {(cue, tag)}
        self.content_to_cue_tag: Dict[str, Set[Tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def add_cue(self, cue_id: str) -> CueNode:
        return self.cues.setdefault(cue_id, CueNode(cue_id=cue_id))

    def add_content(
        self,
        content_id: str,
        text: str,
        *,
        layer: ContentLayer = "episodic",
        time: Optional[str] = None,
        topic_links: Optional[List[str]] = None,
    ) -> ContentNode:
        node = ContentNode(content_id=content_id, text=text, layer=layer, time=time,
                            topic_links=list(topic_links or []))
        self.contents[content_id] = node
        return node

    def link(self, cue_id: str, tag: str, content_id: str) -> None:
        """Register a Cue-Tag-Content triple (c, g, v) in R."""
        cue = self.add_cue(cue_id)
        cue.tag_set.add(tag)
        content = self.contents[content_id]
        if tag not in content.tags:
            content.tags.append(tag)
        self.cue_tag_to_content.setdefault((cue_id, tag), set()).add(content_id)
        self.content_to_cue_tag.setdefault(content_id, set()).add((cue_id, tag))

    # ------------------------------------------------------------------
    # Traversal operators (Eq. 5, 8-9)
    # ------------------------------------------------------------------
    def cue_to_tags(self, cue_id: str) -> Set[str]:
        """phi_c->g(c): tags activated by a single cue."""
        cue = self.cues.get(cue_id)
        return set(cue.tag_set) if cue else set()

    def forward_cue_to_tag(self, active_cues: Iterable[str]) -> Set[str]:
        """Pi_c->g(C^(t)) = union_{c in C^(t)} phi_c->g(c)  (Eq. 8)."""
        tags: Set[str] = set()
        for cue_id in active_cues:
            tags |= self.cue_to_tags(cue_id)
        return tags

    def forward_tag_to_content(
        self,
        active_cues: Iterable[str],
        active_tags: Iterable[str],
        *,
        exclude_tags: Iterable[str] = (),
        exclude_content_ids: Iterable[str] = (),
    ) -> Set[str]:
        """Pi_(c,g)->v(C^(t), G^(t))  (Eq. 8).

        ``exclude_tags`` is the literal mechanism through which Planning
        experience advice like "avoid this type of tag expansion" prunes a
        branch before it is ever traversed.
        """
        exclude_tags = set(exclude_tags)
        exclude_content_ids = set(exclude_content_ids)
        contents: Set[str] = set()
        for cue_id in active_cues:
            for tag in active_tags:
                if tag in exclude_tags:
                    continue
                contents |= self.cue_tag_to_content.get((cue_id, tag), set())
        return contents - exclude_content_ids

    def reverse_content_to_cue_tag(self, active_contents: Iterable[str]) -> Set[Tuple[str, str]]:
        """Pi_v->(c,g)(V^(t))  (Eq. 9): contents activate new (cue, tag) pairs."""
        pairs: Set[Tuple[str, str]] = set()
        for content_id in active_contents:
            pairs |= self.content_to_cue_tag.get(content_id, set())
        return pairs

    def topic_to_episodes(self, topic_content_id: str) -> List[str]:
        """phi_tau->e: a topic node's constituent episodes."""
        node = self.contents.get(topic_content_id)
        return list(node.topic_links) if node else []

    # ------------------------------------------------------------------
    # Relevance ranking, for truncating an over-budget candidate set
    # (nodes.py's MAX_ACTIVE_TAGS / MAX_NEW_CONTENT_PER_ROUND) without
    # just keeping an arbitrary (e.g. alphabetical-by-id) slice of it.
    # Same dependency-free token-overlap heuristic used throughout this
    # package wherever a real embedding model isn't assumed to be present.
    # ------------------------------------------------------------------
    def rank_tags_by_relevance(self, tags: Iterable[str], query: str) -> List[str]:
        query_tokens = _tokenize(query)
        return sorted(tags, key=lambda tag: (-len(_tokenize(tag) & query_tokens), tag))

    def rank_contents_by_relevance(self, content_ids: Iterable[str], query: str) -> List[str]:
        query_tokens = _tokenize(query)

        def _score(content_id: str) -> int:
            node = self.contents.get(content_id)
            return -len(_tokenize(node.text) & query_tokens) if node else 0

        return sorted(content_ids, key=lambda cid: (_score(cid), cid))

    # ------------------------------------------------------------------
    # Initial cue matching (MRAgent Sec. 4.2: "extracts a set of
    # fine-grained cues and matches them against the stored cue set")
    # ------------------------------------------------------------------
    def match_query_to_cues(self, query: str, *, max_matches: int = 25) -> Set[str]:
        """A cue matches only if *every* one of its tokens appears in the
        query (a strict subset match, not "shares at least one word") --
        with hundreds of LLM-extracted multi-word cues in a real
        conversation, single-token overlap (the old `min_overlap=1`
        behaviour) matches on shared filler words alone (e.g. a cue like
        "group of bowls" matching any query containing "group"), which on
        a ~1000-cue graph fans out to hundreds of tags and contents and
        blows the LLM's context window on the very first retrieval call.
        Results are capped at ``max_matches``, preferring the most
        specific (most-token) matches when more than that many qualify.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return set()
        query_lower = query.lower()
        scored: List[Tuple[int, str]] = []
        for cue_id in self.cues:
            cue_tokens = _tokenize(cue_id)
            if not cue_tokens:
                continue
            if cue_tokens.issubset(query_tokens) or cue_id.lower() in query_lower:
                scored.append((len(cue_tokens), cue_id))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return {cue_id for _, cue_id in scored[:max_matches]}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "cues": {cid: {"tag_set": sorted(c.tag_set)} for cid, c in self.cues.items()},
            "contents": {
                cid: {
                    "text": c.text,
                    "layer": c.layer,
                    "time": c.time,
                    "tags": c.tags,
                    "topic_links": c.topic_links,
                }
                for cid, c in self.contents.items()
            },
            "links": sorted(
                {(cue_id, tag, content_id)
                 for (cue_id, tag), content_ids in self.cue_tag_to_content.items()
                 for content_id in content_ids}
            ),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "CueTagContentGraph":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        graph = cls()
        for cid, content in data.get("contents", {}).items():
            graph.add_content(
                cid,
                content["text"],
                layer=content.get("layer", "episodic"),
                time=content.get("time"),
                topic_links=content.get("topic_links", []),
            )
        for cue_id, tag, content_id in data.get("links", []):
            graph.link(cue_id, tag, content_id)
        return graph
