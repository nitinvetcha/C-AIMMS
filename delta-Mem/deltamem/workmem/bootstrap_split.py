"""The bootstrap / held-out conversation split, defined exactly once.

The ITERRET paper's protocol (Sec. 3, "Benchmark and protocol"): "A 10%
bootstrap split builds the experience bank; the remaining conversations
(1,388 questions) are held out for evaluation."

That 1,388 is not an approximation and it is worth stating why, because it
pins the split down completely. LoCoMo's 10 conversations carry 1,540
non-adversarial questions. Dropping conversation 0 leaves:

    multi-hop 250 | temporal 284 | open-domain 83 | single-hop 771 = 1388

which matches the paper's own per-category Ns (Tables 1 and 2) exactly. So
"10% bootstrap" means conversation 0 and only conversation 0, and the
evaluation set is conversations 1-9. Any run that scores the bootstrap
conversation alongside the held-out ones is reporting a number the paper's
protocol does not define -- the bank was distilled from trajectories over
that conversation's own graph.

Why this lives in its own module rather than in each eval script: the split
is used by three places (the bank builder, the OSAM pipeline and the no-OSAM
ablation) and they MUST agree. Three private copies of a shared constant is
precisely how CATEGORY_MAP ended up with three definitions that had already
drifted apart, and how IterRet ended up with two checkouts nothing synced.
"""
from __future__ import annotations

import os

# How many LEADING conversations are reserved for the offline bootstrap.
#
# Default 0 -- i.e. NO held-out split, score all 10 conversations -- because
# that is what every previously-recorded run in this project did, and silently
# changing the denominator under existing result files would make run 11
# (0.3139 over 1540) and a new run incomparable without anyone noticing.
# Set CAIMMS_BOOTSTRAP_SAMPLES=1 to get the paper's protocol.
def n_bootstrap_samples() -> int:
    raw = os.environ.get("CAIMMS_BOOTSTRAP_SAMPLES", "0")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"CAIMMS_BOOTSTRAP_SAMPLES must be an integer, got {raw!r}"
        ) from None
    if n < 0:
        raise ValueError(f"CAIMMS_BOOTSTRAP_SAMPLES must be >= 0, got {n}")
    return n


def is_bootstrap_sample(sample_idx: int) -> bool:
    """True if this conversation is reserved for building the bank."""
    return sample_idx < n_bootstrap_samples()


def is_heldout_sample(sample_idx: int) -> bool:
    """True if this conversation belongs to the evaluation set."""
    return not is_bootstrap_sample(sample_idx)


def split_description() -> str:
    n = n_bootstrap_samples()
    if n == 0:
        return "no held-out split (scoring all conversations, 1540 questions)"
    return (
        f"paper protocol: conversations 0-{n - 1} reserved for the experience "
        f"bank, conversations {n}+ held out for evaluation"
    )
