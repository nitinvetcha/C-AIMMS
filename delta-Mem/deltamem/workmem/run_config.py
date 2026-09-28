"""What a run actually uses: every prompt/retrieval switch, and which code.

The eval records this next to its results file (<results>.config.jsonl, one
line per start or resume), prints it into the run log, and refuses to resume
a results file under different settings or different code -- otherwise a
cancelled run resumed after a sync or a changed flag silently mixes two
setups in one file. scripts/show_config.py prints it on demand.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]

# Everything that shapes what the model is given or how it answers. Editing any
# of these changes the fingerprint, so a resume across a code change is caught.
_FINGERPRINT_FILES = [
    "IterRet/iterret/ctc_graph.py", "IterRet/iterret/nodes.py", "IterRet/iterret/state.py",
    "IterRet/iterret/memory_builder.py", "IterRet/iterret/time_resolution.py",
    "delta-Mem/deltamem/workmem/osam_workmem.py", "delta-Mem/deltamem/workmem/iterret_bridge.py",
    "delta-Mem/deltamem/workmem/evidence_filter.py", "delta-Mem/deltamem/workmem/eval_locomo_iterret_mock.py",
    "delta-Mem/deltamem/core/delta_impl.py", "delta-Mem/deltamem/runtime/session.py",
    "delta-Mem/deltamem/eval/locomo_protocol.py",
]

# Informational only: allowed to differ between a run and its resume.
_NOT_GUARDED = {"started", "git_commit", "output_file", "WORKMEM_MAX_QUESTIONS", "VLLM_PORT", "PYTHONNOUSERSITE"}


def code_fingerprint() -> str:
    h = hashlib.md5()
    for rel in _FINGERPRINT_FILES:
        path = _REPO / rel
        h.update(rel.encode())
        h.update(path.read_bytes() if path.exists() else b"<missing>")
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(_REPO), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not sha:
        return "no git (synced copy)"
    return sha + (" + uncommitted changes" if dirty else "")


def current_config(output_file: str | None = None) -> dict:
    """The settings a run started NOW, in this environment, would use."""
    from iterret import ctc_graph, nodes
    from deltamem.workmem import osam_workmem as o
    return {
        "code_fingerprint": code_fingerprint(),
        "ITERRET_RESOLVE_DATES": ctc_graph.RESOLVE_DATES,
        "OSAM_TIMING_INSTRUCTION": o.TIMING_INSTRUCTION_ENABLED,
        "OSAM_NEUTRAL_ANSWER_POLICY": o.NEUTRAL_ANSWER_POLICY_ENABLED,
        "OSAM_PHASE2_PROMPT_WRITE": o.PHASE2_PROMPT_WRITE,
        "OSAM_PHASE1_GRANULARITY": o.PHASE1_WRITE_GRANULARITY,
        "WORKMEM_TEMPORAL_NARROWING": o.TEMPORAL_NARROWING_ENABLED,
        "ITERRET_SEMANTIC_CUES": nodes.SEMANTIC_CUE_SEEDING,
        "ITERRET_FALLBACK_TOPUP_MIN": nodes.FALLBACK_TOPUP_MIN,
        "ITERRET_FALLBACK_TOPUP_ADD": nodes.FALLBACK_TOPUP_ADD,
        "DISABLE_TAG_EMBEDDER_FUSION": bool(os.environ.get("DISABLE_TAG_EMBEDDER_FUSION")),
        "WORKMEM_GRAPH_CACHE_DIR": os.environ.get("WORKMEM_GRAPH_CACHE_DIR") or "<results dir>/graph_cache",
        "git_commit": git_commit(),
        "output_file": output_file or os.environ.get("WORKMEM_OUTPUT_FILE", "-"),
        "WORKMEM_MAX_QUESTIONS": os.environ.get("WORKMEM_MAX_QUESTIONS") or "all",
        "VLLM_PORT": os.environ.get("VLLM_PORT", "8000"),
        "PYTHONNOUSERSITE": bool(os.environ.get("PYTHONNOUSERSITE")),
    }


def describe(cfg: dict) -> str:
    """Human-readable report, hardcoding status first."""
    on = lambda b: "ON " if b else "OFF"
    lines = [
        "  Hardcoding status",
        f"    [{on(cfg['OSAM_TIMING_INSTRUCTION'])}] timing instruction in the prompt (LoCoMo date-format rule)",
        f"    [{on(cfg['ITERRET_RESOLVE_DATES'])}] relative dates resolved in the memory ('yesterday (7 May, 2023)')",
        f"    [{on(not cfg['OSAM_NEUTRAL_ANSWER_POLICY'])}] answer policy 'Every question here has an answer...' "
        f"(OFF = neutral wording)",
        "    [ON ] always, not switchable: 'name a thing' examples ('Pomodoro technique', 'psychology', "
        "'the beach'), yes/no line, no-first-person line, question-type routing, LoCoMo official prompt",
    ]
    if not cfg["OSAM_TIMING_INSTRUCTION"] and not cfg["ITERRET_RESOLVE_DATES"]:
        lines.append("    !! timing instruction AND date resolution both off -- temporal questions get no date help")
    lines.append("  All settings")
    for k, v in cfg.items():
        if k != "started":
            lines.append(f"    {k:28s} {v}")
    return "\n".join(lines)


def _sidecar(output_file: str) -> Path:
    return Path(str(output_file) + ".config.jsonl")


def recorded_configs(output_file: str) -> list[dict]:
    path = _sidecar(output_file)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def record_start(output_file: str, *, resuming: bool) -> dict:
    """Print the config, refuse a resume under different settings or code
    (override: WORKMEM_ALLOW_CONFIG_CHANGE=1), then append it to the sidecar."""
    cfg = current_config(output_file)
    print("[config]\n" + describe(cfg), flush=True)
    history = recorded_configs(output_file)
    if resuming and history:
        prev = history[-1]
        changed = {k: (prev.get(k), v) for k, v in cfg.items()
                   if k not in _NOT_GUARDED and prev.get(k) != v}
        if changed and os.environ.get("WORKMEM_ALLOW_CONFIG_CHANGE") != "1":
            detail = "\n".join(f"    {k}: was {a!r}, now {b!r}" for k, (a, b) in changed.items())
            raise SystemExit(
                f"Refusing to resume {output_file}: its rows were produced with different settings "
                f"or code.\n{detail}\nResume with the original settings/code, write to a new "
                "WORKMEM_OUTPUT_FILE, or set WORKMEM_ALLOW_CONFIG_CHANGE=1 to mix them deliberately."
            )
    elif resuming:
        print("[config] resuming a results file with no recorded config (started before "
              "2026-09-28): cannot verify its earlier rows used these settings.", flush=True)
    with open(_sidecar(output_file), "a") as fh:
        fh.write(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"), **cfg}) + "\n")
    return cfg
