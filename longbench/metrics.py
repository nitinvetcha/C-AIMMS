"""LongBench scoring. The QA subtasks we target (single-doc and multi-doc QA)
are all scored by LongBench's ``qa_f1_score`` -- a SQuAD-style token-F1 that
takes the max over multiple gold answers. This is a faithful port of the
official LongBench implementation (THUDM/LongBench, metrics.py), kept separate
from ``iterret.metrics.token_f1`` because the normalization differs (LongBench
strips articles and punctuation SQuAD-style).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Sequence, Union


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truths: Union[str, Sequence[str]]) -> float:
    """Max token-F1 of ``prediction`` against any of the gold answers."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    return max((f1_score(prediction, gt) for gt in ground_truths if gt), default=0.0)


# All target QA tasks use qa_f1_score; kept as a dispatch table so other
# LongBench task families (classification, retrieval, summarization) can be
# wired in later without touching the runner.
TASK_METRIC = {task: qa_f1_score for task in (
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
)}


def score_sample(prediction: str, answers: Sequence[str], task: str) -> float:
    metric = TASK_METRIC.get(task, qa_f1_score)
    return metric(prediction, list(answers))
