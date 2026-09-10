"""Paired comparison of replay_delta_scaling.py arms.

Because every arm saw byte-identical evidence for the same question, the
per-question difference is attributable to delta_scaling alone -- unlike the
cross-run comparisons this project has relied on until now, where retrieval
differed on 91.6% of questions and the delta measured retrieval plus mechanism
together.

Reports, per arm pair: mean paired delta, a bootstrap 95% CI, how many
predictions were byte-identical (the share of questions delta-mem does not
touch at all), and the same breakdown per LoCoMo category.

    source env.sh
    python3 scripts/compare_replay_arms.py "$CAIMMS_OUTPUT_DIR/replay_scaling.jsonl"
    python3 scripts/compare_replay_arms.py <file> --baseline 0.0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

CATEGORY_NAMES = {1: "MULTI_HOP", 2: "TEMPORAL", 3: "OPEN_DOMAIN", 4: "SINGLE_HOP"}
BOOTSTRAP_SAMPLES = 5000


def bootstrap_ci(values: list[float], *, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(BOOTSTRAP_SAMPLES):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="arm to compare the others against (default: the lowest scale present, "
        "which is normally 0.0 = delta-mem neutralised)",
    )
    args = parser.parse_args()

    arms: dict[float, dict[tuple[int, int], dict]] = defaultdict(dict)
    with open(args.path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            arms[row["delta_scaling"]][(row["sample_idx"], row["q_idx"])] = row

    scales = sorted(arms)
    if not scales:
        raise SystemExit("no rows found")
    baseline = args.baseline if args.baseline is not None else scales[0]
    if baseline not in arms:
        raise SystemExit(f"baseline arm {baseline} not in file (have {scales})")

    print(f"arms present: {scales}")
    for scale in scales:
        rows = arms[scale]
        avg = sum(r["score"] for r in rows.values()) / len(rows)
        print(f"  delta_scaling={scale:<6} n={len(rows):<5} mean F1={avg:.4f}")

    base_rows = arms[baseline]
    for scale in scales:
        if scale == baseline:
            continue
        rows = arms[scale]
        common = sorted(set(base_rows) & set(rows))
        if not common:
            print(f"\nno overlap between {baseline} and {scale}")
            continue

        diffs = [rows[k]["score"] - base_rows[k]["score"] for k in common]
        mean = sum(diffs) / len(diffs)
        lo, hi = bootstrap_ci(diffs)
        identical = sum(
            1 for k in common if rows[k]["prediction"] == base_rows[k]["prediction"]
        )
        improved = sum(1 for d in diffs if d > 1e-9)
        worsened = sum(1 for d in diffs if d < -1e-9)

        verdict = "CLEARS ZERO" if (lo > 0 or hi < 0) else "straddles zero -- not distinguishable from noise"
        print(f"\n=== delta_scaling {scale} vs {baseline}  (n={len(common)}) ===")
        print(f"  mean paired delta : {mean:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {verdict}")
        print(
            f"  identical predictions: {identical}/{len(common)} "
            f"({100 * identical / len(common):.1f}%)   improved: {improved}   worsened: {worsened}"
        )

        by_cat: dict[int, list[float]] = defaultdict(list)
        for k in common:
            by_cat[base_rows[k].get("category")].append(rows[k]["score"] - base_rows[k]["score"])
        for cat in sorted(x for x in by_cat if x is not None):
            values = by_cat[cat]
            clo, chi = bootstrap_ci(values, seed=1)
            print(
                f"    [{CATEGORY_NAMES.get(cat, cat):<12}] n={len(values):<4} "
                f"delta={sum(values) / len(values):+.4f}  CI [{clo:+.4f}, {chi:+.4f}]"
            )


if __name__ == "__main__":
    main()
