"""Pre-stage LongBench QA tasks as local JSONL for offline / repeated runs.

Downloads LongBench's ``data.zip`` from HuggingFace (THUDM/LongBench) via
``huggingface_hub`` and extracts one raw ``<task>.jsonl`` per task under an
output directory (default ``data/longbench``). Run this once on a machine with
internet (e.g. a cluster login node), then point the eval at it:

    python -m longbench.fetch_data                         # -> data/longbench/*.jsonl
    python -m longbench.run_longbench_eval --task hotpotqa \
        --longbench-path data/longbench ...

Uses ``huggingface_hub`` (a transformers dependency), NOT ``datasets`` -- newer
``datasets`` refuses THUDM/LongBench's loading script.
"""

from __future__ import annotations

import argparse
import os
import zipfile

from longbench.longbench_data import LONGBENCH_QA_TASKS


def main() -> None:
    ap = argparse.ArgumentParser(description="Download LongBench QA tasks to local JSONL.")
    ap.add_argument("--out-dir", default="data/longbench")
    ap.add_argument("--tasks", nargs="*", default=LONGBENCH_QA_TASKS,
                    help=f"Tasks to fetch (default: all QA tasks {LONGBENCH_QA_TASKS}).")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download  # lazy

    print("[fetch] downloading LongBench data.zip from the Hub ...", flush=True)
    zip_path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset")

    os.makedirs(args.out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for task in args.tasks:
            members = [n for n in zf.namelist() if n.endswith(f"{task}.jsonl")]
            if not members:
                print(f"[fetch] WARNING: {task}.jsonl not in data.zip, skipping")
                continue
            out_path = os.path.join(args.out_dir, f"{task}.jsonl")
            with open(out_path, "wb") as fh:
                fh.write(zf.read(members[0]))
            n = zf.read(members[0]).count(b"\n")
            print(f"[fetch] {task}: ~{n} samples -> {out_path}")
    print("[fetch] done.")


if __name__ == "__main__":
    main()
