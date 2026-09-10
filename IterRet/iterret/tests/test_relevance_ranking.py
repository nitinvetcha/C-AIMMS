"""Unit tests for the IDF-weighted relevance ranking added to ctc_graph.py.

Pure Python, no torch/sentence-transformers/GPU dependency -- these test the
scoring logic in isolation, not the real embedder (that needs the cluster;
see the offline validation script instead). Runnable either with pytest or
directly: `python3 -m iterret.tests.test_relevance_ranking`.
"""
from __future__ import annotations

import os
import sys

from iterret.ctc_graph import CueTagContentGraph, _content_tokens, _tokenize


class _FakeEmbedder:
    """Deterministic stand-in for a real embedding backend. Encodes text as
    a bag-of-words count dict and scores cosine similarity over it -- exists
    purely to prove the RRF fusion path and the DISABLE_CONTENT_EMBEDDER_FUSION
    escape hatch both actually run, not to model real semantic behaviour.
    """

    def encode(self, text):
        counts = {}
        for tok in _tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        return counts

    def similarity(self, a, b):
        if not a or not b:
            return 0.0
        dot = sum(a.get(k, 0) * v for k, v in b.items())
        na = sum(v * v for v in a.values()) ** 0.5 or 1.0
        nb = sum(v * v for v in b.values()) ** 0.5 or 1.0
        return dot / (na * nb)


def test_empty_graph_does_not_crash():
    g = CueTagContentGraph()
    assert g.rank_contents_by_relevance([], "anything") == []
    assert g.rank_tags_by_relevance([], "anything") == []
    # _ensure_idf must handle zero content nodes without dividing by zero.
    g._ensure_idf()
    assert g._df is not None
    assert g._idf == {}


def test_single_node_graph():
    g = CueTagContentGraph()
    g.add_content("e1", "Caroline went to a support group")
    order = g.rank_contents_by_relevance(["e1"], "support group")
    assert order == ["e1"]


def test_all_stopword_query_scores_zero_not_crash():
    g = CueTagContentGraph()
    g.add_content("e1", "the group of them")
    g.add_content("e2", "Caroline joined a support group")
    # Query is entirely stopwords/function words -> _content_tokens(query) is
    # empty -> every score must be 0.0, not a crash from an empty sum().
    order = g.rank_contents_by_relevance(["e1", "e2"], "the of to and")
    assert order == ["e1", "e2"]  # falls back to alphabetical tie-break, both score 0.0


def test_add_content_invalidates_cached_idf():
    g = CueTagContentGraph()
    g.add_content("e1", "Caroline went to a support group")
    g._ensure_idf()
    assert "lgbtq" not in g._df
    g.add_content("e2", "Caroline attended an LGBTQ support meeting")
    # Adding a node must invalidate the cache so the new token is counted.
    assert g._df is None
    g._ensure_idf()
    assert g._df["lgbtq"] == 1


def test_idf_weighting_beats_raw_overlap_on_a_real_style_question():
    """Reproduces the measured rank-8-to-rank-4 pathology at small scale.
    "distractor" is built to actually WIN under the old raw-overlap scorer
    (6 shared words -- caroline/did/to/the/go/when, all stopwords except
    "caroline" -- vs gold's 5: caroline/to/group/support/lgbtq). Verified
    against a standalone reimplementation of the pre-fix scorer: old order
    was ['distractor', 'gold']. This test only passes because of the fix,
    not coincidentally.
    """
    g = CueTagContentGraph()
    g.add_content("gold", "Caroline: I went to a LGBTQ support group yesterday")
    g.add_content("distractor", "Caroline: did she go to the store when it opened")
    query = "When did Caroline go to the LGBTQ support group?"

    order = g.rank_contents_by_relevance(["gold", "distractor"], query)
    assert order[0] == "gold", f"expected gold first, got order {order}"


def test_tag_ranking_also_idf_weighted():
    g = CueTagContentGraph()
    g.add_content("e1", "Caroline joined a support group for LGBTQ people")
    tags = ["the support group", "unrelated grocery list", "lgbtq community meetup"]
    order = g.rank_tags_by_relevance(tags, "LGBTQ support group meeting")
    assert order[0] in ("lgbtq community meetup", "the support group")
    assert order[-1] == "unrelated grocery list"


def test_display_text_fix_d4_ranker_sees_timestamp():
    """Before the fix, the lexical scorer read node.text (no timestamp), so
    a query token existing ONLY in the timestamp could never contribute to
    the score, and identical .text on both nodes meant the pre-fix scorer
    could only break the tie alphabetically. IDs are deliberately named so
    alphabetical order favours the WRONG node under the old behaviour
    ("aa_undated" < "zz_dated") -- verified against a standalone
    reimplementation of the pre-fix scorer: old order was
    ['aa_undated', 'zz_dated']. This only passes now because the fix reads
    display_text() and its date tokens genuinely outscore the untimed node.
    """
    g = CueTagContentGraph()
    g.add_content("zz_dated", "Caroline had a picnic with friends", time="7 May 2023")
    g.add_content("aa_undated", "Caroline had a picnic with friends")
    query = "picnic 7 May 2023"
    order = g.rank_contents_by_relevance(["zz_dated", "aa_undated"], query)
    assert order[0] == "zz_dated", f"expected the dated node ranked first, got {order}"


def test_no_embedder_fusion_is_pure_lexical():
    g = CueTagContentGraph()
    g.add_content("a", "apple banana cherry")
    g.add_content("b", "apple banana")
    lexical = g.rank_contents_by_relevance(["a", "b"], "apple banana cherry")
    assert lexical[0] == "a"  # more overlap, no embedder attached at all


def test_disable_fusion_env_flag_bypasses_embedder():
    g = CueTagContentGraph()
    g.add_content("a", "apple banana cherry")
    g.add_content("b", "durian elderberry fig")
    g.attach_embedder(_FakeEmbedder())

    lexical_only = g.rank_contents_by_relevance(["a", "b"], "apple banana cherry")
    assert lexical_only[0] == "a"

    os.environ["DISABLE_CONTENT_EMBEDDER_FUSION"] = "1"
    try:
        with_flag = g.rank_contents_by_relevance(["a", "b"], "apple banana cherry")
        assert with_flag == lexical_only, "flag must reproduce the pure-lexical order"
    finally:
        del os.environ["DISABLE_CONTENT_EMBEDDER_FUSION"]

    # Sanity: with the flag OFF and a real embedder attached, fusion actually
    # runs (doesn't error, still returns a valid permutation) -- this doesn't
    # assert a specific order since RRF fusion behaviour isn't what's under
    # test here, just that the fused path executes without the flag.
    fused = g.rank_contents_by_relevance(["a", "b"], "apple banana cherry")
    assert set(fused) == {"a", "b"}


def test_load_reconstructs_same_idf_as_build():
    """IDF must derive entirely from self.contents so load() (which never
    touches _df/_idf directly) reproduces identical scores -- this is what
    makes the change need no cache migration.
    """
    g1 = CueTagContentGraph()
    g1.add_content("e1", "Caroline went to a LGBTQ support group", time="8 May 2023")
    g1.add_content("e2", "Melanie painted a sunrise last year")
    g1.link("caroline", "support", "e1")

    data = g1.to_dict()
    assert "idf" not in data and "df" not in data  # confirms: no new persisted field

    g2 = CueTagContentGraph.load  # just referencing to ensure it still exists
    g2 = CueTagContentGraph()
    for cid, c in data["contents"].items():
        g2.add_content(cid, c["text"], layer=c.get("layer", "episodic"),
                        time=c.get("time"), topic_links=c.get("topic_links", []))
    for cue_id, tag, content_id in data["links"]:
        g2.link(cue_id, tag, content_id)

    query = "Caroline LGBTQ support group"
    assert (g1.rank_contents_by_relevance(["e1", "e2"], query)
            == g2.rank_contents_by_relevance(["e1", "e2"], query))


def test_rank_functions_return_permutation_not_subset():
    g = CueTagContentGraph()
    g.add_content("a", "one two three")
    g.add_content("b", "four five six")
    g.add_content("c", "seven eight nine")
    ids = ["a", "b", "c"]
    order = g.rank_contents_by_relevance(ids, "one two three four")
    assert sorted(order) == sorted(ids)
    tags = ["alpha beta", "gamma delta", "epsilon zeta"]
    torder = g.rank_tags_by_relevance(tags, "alpha gamma")
    assert sorted(torder) == sorted(tags)


class _SynonymEmbedder:
    """Deterministic stand-in with genuinely NON-lexical similarity.

    _FakeEmbedder above is bag-of-words, so its "semantic" similarity is
    still zero whenever two strings share no token -- which is exactly the
    case the tag-layer fusion exists to fix, so it cannot demonstrate the
    behaviour under test. This one projects tokens into a tiny concept
    space first, so two strings with NO shared token can still score a high
    similarity, the way a real sentence encoder would.
    """

    _CONCEPTS = {
        "counseling": "mentalhealth", "counselor": "mentalhealth",
        "therapy": "mentalhealth", "mental": "mentalhealth",
        "health": "mentalhealth", "psychology": "mentalhealth",
        "motivation": "motive", "career": "motive", "goal": "motive",
        "lgbtq": "queer", "queer": "queer", "trans": "queer",
        "support": "help", "help": "help", "group": "help",
        "community": "help", "circle": "help", "advocacy": "activism",
    }

    def encode(self, text):
        counts = {}
        for tok in _tokenize(text):
            concept = self._CONCEPTS.get(tok)
            if concept:
                counts[concept] = counts.get(concept, 0) + 1
        return counts

    def similarity(self, a, b):
        if not a or not b:
            return 0.0
        dot = sum(a.get(k, 0) * v for k, v in b.items())
        na = sum(v * v for v in a.values()) ** 0.5 or 1.0
        nb = sum(v * v for v in b.values()) ** 0.5 or 1.0
        return dot / (na * nb)


class _CountingEmbedder(_SynonymEmbedder):
    """_SynonymEmbedder that records every encode() call, to prove caching."""

    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return super().encode(text)


def _graph_with_idf_corpus():
    """A graph whose CONTENT nodes give the IDF table something to weight,
    so tag scoring behaves as it does in production (IDF is derived from
    contents, never from the tags themselves)."""
    g = CueTagContentGraph()
    g.add_content("e1", "Caroline joined a LGBTQ support group")
    g.add_content("e2", "Caroline is thinking about a counseling career")
    g.add_content("e3", "Melanie painted a sunrise")
    return g


def test_tag_ranking_without_embedder_is_unchanged():
    """Regression guard: with no embedder the tag ranker must behave exactly
    as it did before fusion existed -- IDF-weighted, alphabetical tie-break."""
    g = _graph_with_idf_corpus()
    tags = ["zzz unrelated", "aaa unrelated", "lgbtq support group"]
    order = g.rank_tags_by_relevance(tags, "LGBTQ support group")
    assert order[0] == "lgbtq support group"
    # the two zero-scoring tags keep the old alphabetical tie-break
    assert order[1:] == ["aaa unrelated", "zzz unrelated"]


def test_tag_fusion_returns_permutation_not_subset():
    g = _graph_with_idf_corpus()
    g.attach_embedder(_SynonymEmbedder())
    tags = ["lgbtq support group", "counseling career", "grocery list"]
    order = g.rank_tags_by_relevance(tags, "mental health goal")
    assert sorted(order) == sorted(tags), f"not a permutation: {order}"


def test_tag_fusion_on_empty_input():
    g = _graph_with_idf_corpus()
    g.attach_embedder(_SynonymEmbedder())
    assert g.rank_tags_by_relevance([], "anything") == []
    assert g.semantic_rank_tags([], "anything") == []


def test_semantic_rescues_zero_lexical_overlap_tag():
    """THE core behaviour this change exists for.

    "zzz counseling motivation" shares NO token with the query, so its
    lexical score is exactly 0.0 -- tied with every distractor -- and its
    id sorts last alphabetically, so the pre-fusion ranker buried it at the
    bottom. Measured on the real corpus, 94% of gold-evidence tags lost at
    the MAX_ACTIVE_TAGS cut were lost exactly this way. With fusion the
    semantic half must pull it to the front.
    """
    g = _graph_with_idf_corpus()
    tags = ["aaa grocery list", "bbb car repair", "ccc weather chat",
            "zzz counseling motivation"]
    query = "mental health career"

    lexical_only = g.rank_tags_by_relevance(tags, query)
    assert lexical_only[-1] == "zzz counseling motivation", (
        f"precondition failed -- expected the gold tag buried last, got {lexical_only}")

    g.attach_embedder(_SynonymEmbedder())
    fused = g.rank_tags_by_relevance(tags, query)
    assert fused[0] == "zzz counseling motivation", (
        f"fusion failed to rescue the zero-overlap tag: {fused}")


def test_fusion_preserves_lexical_precision():
    """RRF must not let a merely-topically-related tag displace an exact
    phrase match. Constructed so the SEMANTIC half ranks the wrong tag
    first: only fusion (not cosine alone) keeps the exact match on top.
    """
    g = _graph_with_idf_corpus()
    g.attach_embedder(_SynonymEmbedder())
    tags = ["lgbtq support group advocacy", "aaa filler", "bbb filler",
            "queer community circle"]
    query = "lgbtq support group"

    # The decoy's concept vector matches the query EXACTLY; the lexical match
    # carries an extra off-concept token ("advocacy"), so cosine alone ranks
    # the decoy first even though the other tag contains every query word.
    semantic_only = g.semantic_rank_tags(tags, query)
    assert semantic_only[0] == "queer community circle", (
        f"precondition failed -- expected cosine to prefer the wrong tag, got {semantic_only}")

    fused = g.rank_tags_by_relevance(tags, query)
    assert fused[0] == "lgbtq support group advocacy", (
        f"fusion lost the exact lexical match: {fused}")


def test_disable_tag_fusion_env_flag_bypasses_embedder():
    g = _graph_with_idf_corpus()
    tags = ["aaa grocery list", "zzz counseling motivation"]
    query = "mental health career"

    lexical_only = g.rank_tags_by_relevance(tags, query)
    g.attach_embedder(_SynonymEmbedder())
    fused = g.rank_tags_by_relevance(tags, query)
    assert fused != lexical_only, "precondition failed -- fusion changed nothing to begin with"

    os.environ["DISABLE_TAG_EMBEDDER_FUSION"] = "1"
    try:
        with_flag = g.rank_tags_by_relevance(tags, query)
        assert with_flag == lexical_only, "flag must reproduce the pure-lexical order"
    finally:
        del os.environ["DISABLE_TAG_EMBEDDER_FUSION"]


def test_content_fusion_flag_does_not_disable_tag_fusion():
    """The two escape hatches must be independent -- otherwise an A/B on one
    layer silently also moves the other, which is exactly the confounding
    that made earlier OSAM measurements uninterpretable."""
    g = _graph_with_idf_corpus()
    g.attach_embedder(_SynonymEmbedder())
    tags = ["aaa grocery list", "zzz counseling motivation"]
    query = "mental health career"
    fused = g.rank_tags_by_relevance(tags, query)

    os.environ["DISABLE_CONTENT_EMBEDDER_FUSION"] = "1"
    try:
        still_fused = g.rank_tags_by_relevance(tags, query)
        assert still_fused == fused, "content flag must not affect the tag layer"
    finally:
        del os.environ["DISABLE_CONTENT_EMBEDDER_FUSION"]


def test_tag_ranking_is_deterministic_across_input_order():
    """nodes.py passes a SET, whose iteration order varies between processes
    (string hash randomisation). The output must not: this pipeline already
    has an open run-to-run reproducibility problem and this ranker must not
    add to it."""
    g = _graph_with_idf_corpus()
    g.attach_embedder(_SynonymEmbedder())
    tags = ["lgbtq support group", "counseling career", "grocery list",
            "car repair", "weather chat"]
    query = "mental health support"

    baseline = g.rank_tags_by_relevance(tags, query)
    assert g.rank_tags_by_relevance(list(reversed(tags)), query) == baseline
    assert g.rank_tags_by_relevance(set(tags), query) == baseline
    assert g.rank_tags_by_relevance(sorted(tags), query) == baseline
    # and stable when called repeatedly (cache warm vs cold)
    assert g.rank_tags_by_relevance(tags, query) == baseline


def test_semantic_rank_tags_falls_back_without_embedder():
    """Must return the lexical order rather than crashing or recursing --
    rank_tags_by_relevance only calls into it when an embedder IS attached,
    so this path exists for direct callers."""
    g = _graph_with_idf_corpus()
    tags = ["lgbtq support group", "aaa unrelated"]
    assert g.semantic_rank_tags(tags, "LGBTQ support group") == \
        g.rank_tags_by_relevance(tags, "LGBTQ support group")


def test_tag_and_query_embeddings_are_cached():
    """Each distinct tag is encoded once; the query is encoded once per
    distinct query even though ranking is called repeatedly."""
    g = _graph_with_idf_corpus()
    emb = _CountingEmbedder()
    g.attach_embedder(emb)
    tags = ["lgbtq support group", "counseling career"]

    g.rank_tags_by_relevance(tags, "mental health")
    g.rank_tags_by_relevance(tags, "mental health")
    g.rank_tags_by_relevance(tags, "mental health")

    for tag in tags:
        assert emb.calls.count(tag) == 1, f"{tag} encoded {emb.calls.count(tag)}x, expected 1"
    assert emb.calls.count("mental health") == 1, (
        f"query encoded {emb.calls.count('mental health')}x, expected 1")


def test_query_memo_updates_when_query_changes():
    """The single-entry query memo must not serve a stale vector when the
    refined query changes between rounds (nodes.py appends to the query
    every reflect round)."""
    g = _graph_with_idf_corpus()
    emb = _CountingEmbedder()
    g.attach_embedder(emb)
    tags = ["zzz counseling motivation", "aaa grocery list"]

    first = g.rank_tags_by_relevance(tags, "mental health career")
    g.rank_tags_by_relevance(tags, "grocery shopping list")
    assert first[0] == "zzz counseling motivation"

    # Re-asking the FIRST query must recompute rather than serve the second
    # query's vector -- and must reproduce the original order exactly.
    again = g.rank_tags_by_relevance(tags, "mental health career")
    assert again == first, f"stale query vector served: {again} != {first}"
    # The memo holds ONE entry by design, so a non-consecutive repeat re-encodes.
    # (Consecutive repeats are the case that matters: see the caching test.)
    assert emb.calls.count("mental health career") == 2, (
        f"expected 2 encodes for a non-consecutive repeat, got "
        f"{emb.calls.count('mental health career')}")


def test_attach_embedder_clears_stale_caches():
    """Vectors from a previous backend are not comparable to a new one's."""
    g = _graph_with_idf_corpus()
    g.attach_embedder(_SynonymEmbedder())
    g.rank_tags_by_relevance(["lgbtq support group"], "queer help")
    assert g._tag_emb, "precondition failed -- nothing was cached"

    g.attach_embedder(_SynonymEmbedder())
    assert g._tag_emb == {}, "tag cache survived re-attach"
    assert g._content_emb == {} and g._cue_emb == {}
    assert g._query_emb is None and g._query_emb_key is None


def test_nodes_cap_path_still_yields_max_active_tags():
    """End-to-end shape check of the exact call nodes.py makes: rank, then
    slice to MAX_ACTIVE_TAGS. Guards against the ranker ever returning
    fewer items than it was given (which would silently shrink the cap)."""
    g = _graph_with_idf_corpus()
    g.attach_embedder(_SynonymEmbedder())
    tags = {f"tag number {i}" for i in range(50)} | {"zzz counseling motivation"}
    ranked = g.rank_tags_by_relevance(tags, "mental health career")
    assert len(ranked) == len(tags)
    assert ranked[0] == "zzz counseling motivation"
    assert len(set(ranked[:15])) == 15


_ALL_TESTS = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]


def _run_all():
    passed, failed = 0, []
    for fn in _ALL_TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            failed.append(fn.__name__)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            failed.append(fn.__name__)
    print(f"\n{passed}/{len(_ALL_TESTS)} passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
