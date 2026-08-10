"""Surprise segmentation for continuous documents.

On dialogue, ``SurpriseEpisodeSegmenter`` snaps surprise boundaries onto whole
*turns*. A LongBench document has no turns, so ``DocumentSurpriseSegmenter``
uses the surprise emitter's raw token spans directly and decodes each back to
text -- this is EM-LLM's *native* mode (the module was designed for continuous
prose, so no turn-snapping is needed or wanted).

It reuses the exact machinery of the dialogue segmenter -- same
``CAIMMSBoundaryEmitter``, Stage-1 Bayesian surprisal + Stage-2 KV-modularity
refinement -- by subclassing ``SurpriseEpisodeSegmenter`` and calling its
shared ``emit_episodes()``. Nothing about the surprise computation changes.
"""

from __future__ import annotations

from typing import List

from iterret.episode_segmenter import SurpriseEpisodeSegmenter


class DocumentSurpriseSegmenter(SurpriseEpisodeSegmenter):
    def segment_document(self, text: str) -> List[str]:
        """Split ``text`` into surprise-bounded event strings. The returned
        spans are contiguous and together cover the whole document."""
        if not text or not text.strip():
            return []
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            return []

        episodes = self.emit_episodes(ids)

        # Build a contiguous, gap-free boundary set from every span endpoint the
        # emitter produced. 0 and len(ids) are added explicitly so the leading
        # head and the trailing tail (which the emitter may not emit as their
        # own spans) are still covered.
        marks = {0, len(ids)}
        for start, end in episodes:
            if 0 < start < len(ids):
                marks.add(start)
            if 0 < end < len(ids):
                marks.add(end)
        bounds = sorted(marks)

        spans: List[str] = []
        for i in range(len(bounds) - 1):
            chunk_ids = ids[bounds[i]:bounds[i + 1]]
            if not chunk_ids:
                continue
            span = self.tokenizer.decode(chunk_ids).strip()
            if span:
                spans.append(span)
        return spans


def fixed_word_chunks(text: str, words_per_chunk: int = 200) -> List[str]:
    """Baseline (no-surprise) document segmentation: fixed-size word windows.

    This is the document analogue of LoCoMo's per-turn episodes -- a
    structure-agnostic control that the surprise segmentation is compared
    against. Word-based (not token-based) so the baseline needs no torch /
    tokenizer, matching how the LoCoMo baseline runs without the surprise deps.
    """
    if not text or not text.strip():
        return []
    words = text.split()
    spans: List[str] = []
    for start in range(0, len(words), words_per_chunk):
        span = " ".join(words[start:start + words_per_chunk]).strip()
        if span:
            spans.append(span)
    return spans
