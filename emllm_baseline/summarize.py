"""Aggregate EM-LLM baseline results into a model x task qa_f1 table.

Reads <alias>_<task>.json files written by run_baseline.py.

    python -m emllm_baseline.summarize emllm_results
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "emllm_results"
    table = defaultdict(dict)   # model -> {task: qa_f1}
    tasks = set()
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        try:
            d = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        if "qa_f1" not in d or "task" not in d:
            continue
        task = d["task"]
        # filename is "<alias>_<task>.json"; task itself contains underscores
        # (e.g. multifieldqa_en), so strip the known task suffix rather than
        # splitting on the last underscore.
        stem = os.path.basename(path)[:-5]  # drop ".json"
        alias = stem[:-(len(task) + 1)] if stem.endswith("_" + task) else stem
        table[alias][task] = d["qa_f1"]
        tasks.add(task)

    if not table:
        print(f"No result JSONs found in {out_dir!r}.")
        return

    tasks = sorted(tasks)
    w = 13
    print(f"\nEM-LLM backbones -- vanilla LongBench qa_f1 (%)  --  {out_dir}")
    print("=" * (16 + w * (len(tasks) + 1)))
    print(f"{'model':<16}" + "".join(f"{t[:12]:>{w}}" for t in tasks) + f"{'MEAN':>{w}}")
    print("-" * (16 + w * (len(tasks) + 1)))
    for model in sorted(table):
        vals = [table[model].get(t) for t in tasks]
        present = [v for v in vals if v is not None]
        cells = "".join((f"{v:>{w}.2f}" if v is not None else f"{'-':>{w}}") for v in vals)
        mean = sum(present) / len(present) if present else 0.0
        print(f"{model:<16}{cells}{mean:>{w}.2f}")
    print("=" * (16 + w * (len(tasks) + 1)))


if __name__ == "__main__":
    main()
