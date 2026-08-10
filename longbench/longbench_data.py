"""Load LongBench QA tasks into a uniform sample shape.

A LongBench sample is (context, question, answers) -- one long document and one
question over it, unlike LoCoMo where one conversation carries many questions.
We keep only the QA task families that map to IterRet's retrieve-and-answer
paradigm (see ``LONGBENCH_QA_TASKS``); summarization / code / few-shot tasks
don't fit and are intentionally excluded.

Data source: either local JSONL (``--longbench-path``; a file or a directory of
``<task>.jsonl``) for offline clusters, or the HuggingFace ``THUDM/LongBench``
dataset when internet is available.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

# Single-doc QA + multi-doc QA. LongBench scores all of these with qa_f1_score.
LONGBENCH_QA_TASKS: List[str] = [
    "narrativeqa", "qasper", "multifieldqa_en",   # single-document QA
    "hotpotqa", "2wikimqa", "musique",            # multi-document QA (evidence chaining)
]


def _normalize_sample(raw: dict, task: str) -> dict:
    """LongBench field names -> our uniform shape."""
    return {
        "id": raw.get("_id") or raw.get("id") or "",
        "context": raw.get("context", ""),
        "question": raw.get("input", raw.get("question", "")),
        "answers": [str(a) for a in raw.get("answers", []) if a is not None],
        "task": task,
        "all_classes": raw.get("all_classes", []),
        "length": raw.get("length"),
    }


def load_longbench(task: str, *, path: Optional[str] = None,
                   max_samples: Optional[int] = None) -> List[dict]:
    """Return a list of normalized samples for ``task``.

    ``path`` = a ``<task>.jsonl`` file, or a directory containing one; if None,
    fall back to ``datasets.load_dataset('THUDM/LongBench', task)``.
    """
    samples: List[dict] = []
    if path:
        jsonl = path if os.path.isfile(path) else os.path.join(path, f"{task}.jsonl")
        if not os.path.isfile(jsonl):
            raise FileNotFoundError(
                f"LongBench file for task {task!r} not found at {jsonl!r}. Pass --longbench-path "
                f"to a <task>.jsonl file or a directory containing one, or omit it to download "
                f"from HuggingFace (THUDM/LongBench)."
            )
        with open(jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    samples.append(_normalize_sample(json.loads(line), task))
    else:
        samples.extend(_load_from_hub(task))

    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def _load_from_hub(task: str) -> List[dict]:
    """Read ``<task>.jsonl`` out of LongBench's ``data.zip`` on the HuggingFace
    Hub, bypassing ``datasets.load_dataset``.

    ``datasets`` >= 3.0 refuses ``THUDM/LongBench`` because it ships a loading
    script ("Dataset scripts are no longer supported"). Downloading the raw
    ``data.zip`` via ``huggingface_hub`` sidesteps that entirely and doesn't
    depend on the ``datasets`` package at all.
    """
    import zipfile
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset")
    out: List[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.endswith(f"{task}.jsonl")]
        if not members:
            available = sorted(n for n in zf.namelist() if n.endswith(".jsonl"))
            raise FileNotFoundError(
                f"{task}.jsonl not found in LongBench data.zip. Available: {available}")
        with zf.open(members[0]) as fh:
            for raw_line in fh:
                line = raw_line.decode("utf-8").strip()
                if line:
                    out.append(_normalize_sample(json.loads(line), task))
    return out
