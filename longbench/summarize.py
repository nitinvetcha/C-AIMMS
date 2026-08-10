"""Aggregate LongBench result JSONs into a baseline-vs-surprise table.

Reads ``<task>_baseline.json`` / ``<task>_surprise.json`` pairs written by
``run_longbench_eval.py`` (via ``run_longbench.sh``) and prints per-task qa_f1
for each mode plus the delta.

    python -m longbench.summarize longbench_results
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, Optional


def _avg_f1(path: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    results = data.get("results", [])
    if not results:
        return None
    return 100.0 * sum(r["f1"] for r in results) / len(results)


def _n(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return len(json.load(fh).get("results", []))
    except (OSError, json.JSONDecodeError):
        return 0


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "longbench_results"
    tasks = sorted({os.path.basename(p).rsplit("_", 1)[0]
                    for p in glob.glob(os.path.join(out_dir, "*_baseline.json"))
                    + glob.glob(os.path.join(out_dir, "*_surprise.json"))})
    if not tasks:
        print(f"No *_baseline.json / *_surprise.json found in {out_dir!r}.")
        return

    print(f"\nLongBench qa_f1 (%)  --  {out_dir}")
    print("=" * 60)
    print(f"{'Task':<20}{'N':>5}{'Baseline':>11}{'Surprise':>11}{'Δ':>10}")
    print("-" * 60)
    base_vals, surp_vals = [], []
    for task in tasks:
        b = _avg_f1(os.path.join(out_dir, f"{task}_baseline.json"))
        s = _avg_f1(os.path.join(out_dir, f"{task}_surprise.json"))
        n = max(_n(os.path.join(out_dir, f"{task}_baseline.json")),
                _n(os.path.join(out_dir, f"{task}_surprise.json")))
        bs = f"{b:.2f}" if b is not None else "  -"
        ss = f"{s:.2f}" if s is not None else "  -"
        ds = f"{s - b:+.2f}" if (b is not None and s is not None) else "  -"
        print(f"{task:<20}{n:>5}{bs:>11}{ss:>11}{ds:>10}")
        if b is not None:
            base_vals.append(b)
        if s is not None:
            surp_vals.append(s)
    print("-" * 60)
    if base_vals and surp_vals:
        mb, ms = sum(base_vals) / len(base_vals), sum(surp_vals) / len(surp_vals)
        print(f"{'MEAN':<20}{'':>5}{mb:>11.2f}{ms:>11.2f}{ms - mb:>+10.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
