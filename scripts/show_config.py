"""Show which settings and code a run uses -- hardcoding status first.

  source env.sh && caimms_activate
  python3 scripts/show_config.py                  # what a run started NOW would use
  OSAM_TIMING_INSTRUCTION=1 python3 scripts/show_config.py   # ...with a flag you plan to set
  python3 scripts/show_config.py RESULTS.jsonl    # what an existing run used (each start/resume)

Needs no GPU. The fingerprint identifies the code: equal fingerprints on the Mac
and the cluster mean the same code is there.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "delta-Mem"), str(HERE.parent / "IterRet")]
try:
    import torch  # noqa: F401
except ImportError:           # e.g. on the Mac: the settings do not need the real model code
    import dryrun_torchstub
    dryrun_torchstub.install()

from deltamem.workmem.run_config import current_config, describe, recorded_configs  # noqa: E402


def main() -> None:
    if len(sys.argv) > 1:
        results = sys.argv[1]
        history = recorded_configs(results)
        if not history:
            sys.exit(f"No recorded config for {results} ({results}.config.jsonl missing): the run was "
                     "started before 2026-09-28. Check its tmux banner or the running process instead.")
        for i, cfg in enumerate(history, 1):
            print(f"=== start {i}/{len(history)} at {cfg.get('started')} ===")
            print(describe(cfg))
        if len({c.get("code_fingerprint") for c in history}) > 1:
            print("!! the code changed between starts of this run")
        return
    print("=== a run started now, in this shell, would use ===")
    print(describe(current_config()))


if __name__ == "__main__":
    main()
