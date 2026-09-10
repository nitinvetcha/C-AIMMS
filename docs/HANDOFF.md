# C-AIMMS / WORKMEM — Handoff

**Last updated:** 2026-09-10. This file replaces all prior versions (the run-by-run history that used to live here has been trimmed — only conclusions worth not re-deriving are kept, in §7).

---

## 1. What this is

**The project's own spec is `~/Downloads/COLM_Cognitive_AI_Memory_Architecture_Project (1).pdf`** — read this before touching `osam_workmem.py` or `delta_impl.py` again. It formally defines four modules: HETREP (heterogeneous encoding — not built), ADASTORE (adaptive storage/hierarchy — not built), ITERRET (the retrieval graph — built, this is `IterRet/`), and **WORKMEM** (§1.5 of the paper — the OSAM integration this handoff is about).

The underlying memory mechanism is **δ-mem** (arXiv:2605.12357, also in `~/Downloads/`): a PEFT adapter giving a frozen LLM an online-writable associative memory, one **r×r state matrix per attention layer** (r=8 here), updated by a gated delta rule:
```
S_t = λ_t·S_{t-1} + β_t·(v_t - S_{t-1}k_t)·k_tᵀ      (λ_t = 1 - β_t)
```
Read: `r_t = S_{t-1}·q_t`, projected into `Δq_t, Δo_t` which correct the backbone's query and attention output (`q̃ = q⁰ + (α/r)Δq`, `ȳ = a + (α/r)Δo`). Only q/o heads are active in this adapter (`delta_heads: ["q","o"]`); k/v corrections are zero. **The write is not additive — it erases along the new key's direction before writing.**

**The paper's own architectural principle, stated three separate times** (abstract: "keeping the prompt context uncorrupted"; §1.5 prose: "The naive approach — injecting E_T as additional prompt tokens — consumes token budget, suffers from retrieval noise... Instead, C-AIMMS routes memory influence through δ-mem"; eq. 10 caption: "the prompt context q_T carries no retrieved content"): **retrieved evidence should influence generation only through S, never by also being placed in the prompt.** See §6 item 1 — the current pipeline does not follow this.

**Two-phase OSAM, per the paper (§1.5, "Interaction with ITERRET"):** Phase 1 populates S from the retrieved evidence set E_T, *offline, before generation begins*. Phase 2 writes *incrementally from the model's own newly generated tokens* during response generation. This is a spec about **what gets written when** — not about write granularity (message_mean vs token), which is a separate, orthogonal axis the paper studies (§3.5 of the arXiv paper) as three complete alternative training configurations (TSW/SSW/MSW), not a mid-inference switch.

Benchmark: **LoCoMo**, 1540 scored questions (adversarial excluded — its correct answer is a refusal, which rides the same bias hurting every other category). Four categories: multi-hop, temporal, open-domain, single-hop. Metric: token-level F1, Porter-stemmed.

---

## 2. Where things live

| | |
|---|---|
| **Local Mac** | `~/Documents/C-AIMMS/` — a workspace, NOT a repo. Contains `workmem-vertical/` (the repo, branch `workmem-vertical`, remote `nitinvetcha/C-AIMMS`), plus `models/` and `outputs/` deliberately outside git. IterRet is vendored inside the repo; there is no sibling copy any more. |
| **resiliente-2003** | `ashwinkm@10.24.32.171`, `~/C-AIMMS/`. 2× RTX 4090, no scheduler. **Primary run environment.** Same layout as the Mac. |
| **Mahamathi** | `ssh arnavbhatt@10.16.63.23` (the node identifies itself as `mahamathi2` once logged in -- that hostname is not reachable directly from outside, use the IP to connect), `/home/kbasu/arnavbhatt/workmem_test/`. SLURM, 8× A100. Same layout. Login node has no GPU access -- anything needing a GPU must go through `sbatch`, never run directly. Secondary — see §9. |
| **Papers** | `~/Downloads/COLM_Cognitive_AI_Memory_Architecture_Project (1).pdf` (this project's spec), `~/Downloads/2605.12357v1.pdf` (δ-mem method paper), `~/Downloads/Version_2.pdf`. Not in the repo. |

Key files (repo root = `workmem-vertical/`):
```
docs/HANDOFF.md                 this file
docs/ENVIRONMENT.md             environment notes
scripts/                        slurm + guardian + score_calculator.py
delta-Mem/deltamem/
  core/delta_impl.py            OSAM math (DeltaMemAttention.forward) — verified against paper eq. 1-11, see §6
                                 _token_validity_mask carries the permanent causal-mask fix (§5 item 3)
  runtime/session.py            DeltaMemChatSession: generate_reply, _ingest_full_ids
  eval/locomo_protocol.py       scoring + shared category constants
  workmem/
    eval_locomo_iterret_mock.py MAIN EVAL LOOP
    osam_workmem.py             populate_osam_from_evidence, answer_with_osam, all prompts
    iterret_bridge.py           wraps IterRet's loop, now carries retrieval diagnostics
    ab_write_granularity.py     paired A/B harness (granularity — result in §7)
IterRet/iterret/                VENDORED into this branch (see §9)
  ctc_graph.py, nodes.py        retrieval graph + ranking (out of scope for §5 — that's delta-mem only)
                                 rank_tags_by_relevance now fuses lexical + MiniLM-embedding RRF (§4 run 12b, §7)
scripts/validate_tag_fusion.py  offline A/B for the tag-layer fusion, no GPU/LLM needed -- see its own docstring
```

---

## 3. How to run

Identical on every machine — `env.sh` resolves all paths from its own location.

```bash
# from the repo root
source env.sh

# archive the old checkpoint FIRST -- a full run does NOT clear it (see §8 trap 1)
cd "$CAIMMS_OUTPUT_DIR" && mv workmem_iterret_full.jsonl "workmem_iterret_full_$(date +%Y%m%d_%H%M%S).jsonl"

tmux new -s caimms
bash scripts/run_pipeline.sh              # full, ~30-35h. Add --smoke for 152q/~1h
```

No-GPU static check before committing to a run (5s, no model or weights needed):
```bash
source env.sh && cd scripts && PYTHONPATH=".:$PYTHONPATH" python3 dryrun_pipeline.py
```

Score: `python3 scripts/score_calculator.py "$CAIMMS_OUTPUT_DIR/workmem_iterret_full.jsonl"`

Sync the repo to a cluster (run from the Mac; `--delete` keeps the remote honest,
and `models/`+`outputs/` live outside the repo so they are never touched):
```bash
rsync -av --delete --exclude='__pycache__' --exclude='.git' \
  ~/Documents/C-AIMMS/workmem-vertical/ ashwinkm@10.24.32.171:~/C-AIMMS/workmem-vertical/
```

Pull results back (rename on arrival — never overwrite a prior run's file):
```bash
rsync -avP ashwinkm@10.24.32.171:~/C-AIMMS/outputs/workmem_iterret_full.jsonl \
  ~/Documents/C-AIMMS/outputs/workmem_runNN_$(date +%Y-%m-%d).jsonl
```

**Env knobs**, all default to current pipeline behavior:

| var | default | effect |
|---|---|---|
| `CAIMMS_WORKSPACE` | parent of repo | where `models/` and `outputs/` live |
| `OSAM_PHASE1_GRANULARITY` | `message_mean` | measured not to matter (§7) |
| `OSAM_PHASE2_PROMPT_WRITE` | `1` | `0` = prefill reads S without overwriting it — closest knob to the paper's Phase 2 spec (§5 item 2) |
| `WORKMEM_TEMPORAL_NARROWING` | `0` | `1` restores temporal evidence narrowing (was hurting the category, §7) |
| `OSAM_TEMPORAL_QUERY_PATTERN` | built-in regex | override the timing-question detector |
| `VLLM_PORT` | `8000` | override if the port is taken |

---

## 4. Results

All numbers recomputed directly from the `.jsonl` files, adversarial excluded.

| run | date | overall | multi-hop | temporal | open-dom | single-hop |
|---|---|---|---|---|---|---|
| run 6 | Jul 7 | 0.2756 | 0.2192 | 0.3015 | 0.0960 | 0.3051 |
| run 8 | Aug 12 | 0.2935 | 0.2412 | 0.2697 | 0.2014 | 0.3306 |
| run 9 (+IDF ranking) | Aug 13 | 0.3016 | 0.2582 | 0.2430 | 0.2183 | 0.3479 |
| run 10 (resiliente port) | Aug 16 | 0.2906 | 0.2311 | 0.2343 | 0.2090 | 0.3413 |
| run 11 (Aug-19 prompt/retrieval fixes) | Aug 19 | 0.3139 | 0.2422 | 0.3233 | 0.1349 | 0.3547 |
| run 12a (mask fix alone) | Sep 7 | 0.3404 | 0.2648 | 0.3672 | 0.1668 | 0.3754 |
| **run 12b** (mask fix + tag-layer RRF fusion) | Sep 7 | **0.4214** | **0.3614** | **0.4035** | **0.1732** | **0.4766** |

**Run 11 narrative (superseded as the top result by run 12b, but the analysis below still holds for anyone reading run 11's own file).** Temporal broke its four-run losing streak decisively (+0.080 vs run 9, matched-pair: 113 improved / 41 regressed / net +25.78 on 321 questions) — driven by two prompt changes: the timing instruction moved to immediately after the question (restoring run 6's structure), and its enumerated relative-expression list ("yesterday, last week...") removed (was priming the model to emit exactly those phrases). Open-domain regressed sharply (−0.083) — root-caused to the new yes/no instruction ("start with Yes/No, then add a short phrase") combined with `score_locomo_prediction`'s pre-existing category-3 behavior of truncating gold at the first `;`, so any elaboration is precision-poison against a one-word target. Measured: 0/49 yes/no-gated questions elaborated in run 9 across all categories; 23/23 did on open-domain alone in run 11, scoring 0.145 mean F1. A second, smaller new issue: the "name a thing" instruction has no case for count questions (`"How many dogs?"` → `"Scout"`, a dog's name, not a number) — F1 on "how many" questions fell 0.190→0.122. Neither has been fixed yet.

**Run 12: two mechanistically unrelated fixes, deliberately measured separately rather than bundled.** One lives in attention/OSAM (`delta_impl.py`'s forward pass), the other in IterRet's retrieval graph (`ctc_graph.py`) — conflating them would have made neither number trustworthy, so run 12a holds retrieval fixed and isolates the mask fix, and run 12b adds the tag-fusion change on top of that same mask-fixed baseline.

- **Mask fix alone (run 12a vs run 11): +0.0265 absolute (+8.4% relative).** Fixes the incremental-forward-pass causal-mask bug described in §5 item 3 — `mean_base_o_norm` was `0.0` on every question before this, meaning OSAM's correction was mathematically inert on every prior full run including run 11. Fixed permanently in `delta-Mem/deltamem/core/delta_impl.py`'s `_token_validity_mask` (2026-09-10 — originally landed as a reversible runtime patch, since folded into the vendored file directly and the patch module deleted). Confirmed live: `mean_delta_o_ratio` averages 0.0257 across run 12a's 1523/1540 instrumented rows (range [0.0225, 0.0278]), vs the field not existing at all in run 11 (predates the `osam_contribution` instrumentation, §5 item 3). Retrieval diagnostics (`fail_open_all` rate, `stop_reason` distribution) are statistically indistinguishable between run 11 and run 12a — expected, since this fix never touches retrieval, and it confirms run 12a's IterRet side never picked up the tag-fusion patch below (clean isolation, not contamination).
- **Tag-layer RRF fusion, on top of the mask fix (run 12b vs run 12a): +0.0810 absolute (+23.8% relative) — the larger of the two effects.** Fixes `IterRet/iterret/ctc_graph.py`'s `rank_tags_by_relevance`. Measured over all 1986 LoCoMo questions against the cached CTC graphs: the cue→tag hop reaches a tag carrying gold evidence 95.2% of the time (cue matching itself is NOT the retrieval bottleneck — see the corrected §6 entry), but the fanout averages ~350 tags and gets cut to `MAX_ACTIVE_TAGS` (15) in 99.4% of rounds. Under the old lexical-only scorer, only 28.2% of gold-evidence tags survived that cut, and 94% of the losses scored an exact lexical `0.0` — an alphabetical tie-break among ~350 equally-uninformative candidates, not a ranking failure on merit. The fix adds `semantic_rank_tags` (embedding cosine, mirroring the content layer's existing `semantic_rank_contents`) fused with the lexical score via reciprocal rank fusion — every zero-lexical-score tag is first collapsed to one shared "uninformative" rank before fusion, otherwise alphabetical noise gets fed into RRF as a real vote (a bug caught and fixed mid-implementation). Falls back to byte-identical old behavior with no embedder attached or `DISABLE_TAG_EMBEDDER_FUSION=1` set (verified across all 1986 questions). Confirmed genuinely active in run 12b via three diagnostics moving together vs run 12a: `fail_open_all` 45.0%→36.3%, `stop_reason: graph_exhausted` 20.6%→9.4%, zero-evidence questions 17→2. This is the first measurement against the real MiniLM embedder (`sentence-transformers/all-MiniLM-L6-v2`) — an earlier offline-only measurement using this project's dependency-free `KeywordOverlapEmbeddingBackend` fallback showed only a few points of recall improvement, undershooting the real effect by roughly an order of magnitude.

Open-domain remains the clear laggard through both fixes (0.1349→0.1668→0.1732): the mask fix's relative gain here (+23.6%) is proportionally its best category, but tag fusion adds almost nothing further (+3.8%). Whatever is wrong with open-domain is not primarily an attention-correctness or retrieval-ranking problem — see §7's existing open-domain findings, none of which either fix touches.

---

## 5. Delta-Mem issues, prioritized (paper-grounded)

Full writeup with code citations and the mechanism given in the session that produced this handoff — summarized here so it isn't re-derived.

1. **Retrieved evidence is placed in the prompt, which the paper explicitly says not to do.** `populate_osam_from_evidence` sets `session.messages` to the evidence list; nothing clears it before `generate_reply` tokenizes `self.messages` for Phase 2, so evidence text is in the backbone's attended context on every question. This directly contradicts the paper's stated principle (three places, quoted in §1).

    **Tested 2026-08-20, then reverted — result is real, keep it in mind.** A toggle was built (write evidence into S in Phase 1 as normal, then wipe the text/KV context before Phase 2 without touching S) and paired A/B'd on resiliente-2003, n=23: removing evidence from the prompt cost **-0.2585 F1** (95% CI **[-0.4307, -0.0830]**, entirely negative, no_evidence worse on 14/23, identical on 0/23). Matches the δ-mem paper's own no-context ablation (46-51%→8.05%) in direction and rough size — **confirms the risk this item warned about was real**, on this adapter, as currently trained. Per explicit instruction, the toggle, its session method, and its A/B harness have all been removed from the codebase — this bullet is what's left of the experiment. Don't re-implement it casually; if revisited, it needs OSAM's standalone signal to be stronger first (see item 4's new sub-finding).
2. **Phase 2 writes the instruction+question block into S, which the paper's Phase 2 does not describe.** The paper's Phase 2 is "incrementally from the model's own newly generated tokens" — i.e. the answer being produced, not a ~250-330 token instruction/question prefill. `OSAM_PHASE2_PROMPT_WRITE=0` makes the prefill read-only, matching the paper's actual spec more closely than the current default. Not yet run. (This knob is unrelated to item 1's reverted one and is still in the code.)

3. **`online_gain=0.05` / `last_delta_o_ratio` — now logged, kept in the code.** Every result row carries `osam_contribution` (mean/max delta_o-vs-base_o ratio per question, via `collect_delta_mem_output_ratio_stats`, snapshotted in `generate_reply` right after the Phase-2 prefill). This is the instrument that produced item 1's measurement above.

    **Bug found while using it — FIXED 2026-09-07, see §4 run 12a/12b.** On the incremental-ingest code path (Phase 1's evidence KV cache still attached, only the new instruction+question suffix gets a fresh forward pass — i.e. every full run through run 11), `osam_contribution` reported exactly `0.0` for base_o_norm, delta_o_norm, and the ratio: not a small real number, a broken reading. Root cause confirmed (was "likely cause, not yet pinned" as of the prior update): under this project's transformers version, the causal mask on an incremental forward pass against an existing KV cache was denying every token attention to its own position. **Fixed permanently in `delta_impl.py`'s own `_token_validity_mask` (2026-09-10).** Since this project never batches (`batch_size=1` always), the padding-detection heuristic the 4-D branch exists for has nothing to legitimately detect, so it is skipped entirely for `batch_size==1`; `batch_size>1` raises `NotImplementedError` rather than silently reusing a heuristic already shown wrong, so introducing batching later fails loudly. Confirmed on a pinned 30-question test: `mean_base_o_norm` 0.0000→19.9964, `mean_delta_o_ratio` 0.0000→0.0257. This began as a reversible runtime monkey-patch (`workmem/mask_fix.py`) while the diagnosis was still being confirmed; that module, its `DELTAMEM_APPLY_MASK_FIX`/`DELTAMEM_DEBUG_MASK` toggles, and `scripts/run_mask_fix_test.slurm` have all been **deleted** — the fix is unconditional and there is deliberately no way to run the broken behaviour any more. **Consequence to know:** reproducing the pre-fix state for a future regression check now requires reverting the `delta_impl.py` edit itself. **Isolated effect, measured properly (mask fix alone, holding retrieval fixed): +0.0265 F1 absolute (+8.4% relative), run 11→run 12a, §4.** Separately, an n=300 pinned-evidence sweep of `delta_scaling` 0.0 vs 2.0 with the mask already fixed in both arms found no confirmed benefit from OSAM's correction itself (paired Δ ≈ −0.0062, 95% CI [−0.0283, +0.0154], crosses zero) — so the mask fix's score gain comes from the mechanism now being *computable* and not corrupting the forward pass, not from δ-mem's own correction being large or beneficial once it IS computable. Don't conflate these two findings.

    **What "inert" actually meant, traced through the tensors (2026-09-10).** The broken mask made `token_mask` all-`False`, and `_token_state_reads` multiplies `reads` by it — so `reads` became the exact zero tensor. `delta_q/k/v/o` are `F.linear(reads, weight)` with **no bias term** (`delta_o_proj` is a raw `nn.Parameter`, not an `nn.Linear`), so they were exactly zero, not merely small. Per `q̃ = q⁰ + (α/r)Δq` and `ȳ = a + (α/r)Δo`, the backbone's query and attention output were therefore **bit-identical to running with no adapter at all**. This was not a logging artefact: the correction genuinely contributed nothing to generation on every full run through run 11. Phase 1 was NOT affected — `eval_locomo_iterret_mock.py` builds a fresh `DeltaMemChatSession` per question, so evidence ingestion is the session's first forward pass (`past_key_values=None`, not incremental), meaning **writes into S were always correct**; only the Phase-2 read-back was dead.
4. **Two new prompt-side issues from run 11** (see §4): the yes/no instruction needs a category-3-aware exception or should drop "then add a short phrase"; the named-thing instruction needs a fourth branch for `^how\s+many\b` questions that asks for a count, not a name.

---

## 6. Other open issues

- **`match_query_to_cues` is NOT the retrieval ceiling — measured 2026-09-07, corrects the prior entry here.** Over all 1986 LoCoMo questions against the cached CTC graphs, cue matching reaches a tag carrying gold evidence 95.2% of the time. The real ceiling is one hop later: `rank_tags_by_relevance`'s cut to `MAX_ACTIVE_TAGS` (15, out of a ~350-tag mean fanout, hit in 99.4% of rounds) — see §4 run 12b and §7 for the full measurement and the fix (tag-layer RRF fusion, now landed). `MAX_ACTIVE_TAGS` itself is still a flat, non-adaptive constant regardless of fanout size -- raising it, or splitting the cut into a cheap-then-careful two-stage filter, is unexplored.
- **Evidence strings duplicate timestamp/speaker** — `memory_builder` prepends `Speaker:`, `display_text()` prepends `[ts]` again on top of turn text that already has both.
- **Topic layer is dead code** — `_abstract_topics` spends LLM calls building nodes that are never `.link()`'d, so they're unreachable. Deleting the call is a pure speed win.
- **`information_gaps` seeds with a literal sentinel string** that only clears if the LLM echoes it verbatim, likely making the early-exit route dead.
- **`evidence_filter` silently no-ops if MiniLM is unavailable** — `KeywordOverlapEmbeddingBackend.encode` returns a dict, `_cosine` assumes float lists, `zip` + `TypeError` gets swallowed by a bare `except`. `run_pipeline.sh` preflights against this; `eval_locomo_ablation.py` and `ab_write_granularity.py` do not. Not fixed.
- **Nothing is committed** — five+ weeks of work across two repos, uncommitted.

---

## 7. Durable findings — don't re-derive these

- **Granularity genuinely doesn't matter, measured properly.** Paired A/B, resiliente-2003, n=380: `message_mean` vs `token` → +0.0005 F1, 95% CI [−0.0106, +0.0107], 63.4% identical predictions (confirms the arms took genuinely different code paths, not a repeat of an earlier `role="system"` bug that made Phase 1 silently fall back to token-only writes for every run through run 10). See §5 item 3 for why this is unsurprising given the adapter's training.
- **Removing evidence from the prompt costs a lot, measured once.** Paired A/B, resiliente-2003, n=23: -0.2585 F1, 95% CI entirely negative [−0.4307, −0.0830]. See §5 item 1. The mechanism that produced this has been reverted out of the code; this number is what's left and shouldn't need re-deriving unless the adapter itself changes.
- **Abstention is not calibration.** r = −0.03 between abstaining and whether the gold evidence is actually present — the model refuses on ~27% of questions whose evidence it has and attempts ~70% of those it doesn't. Abstaining is never correct on cats 1-4 (0/1540 golds is a refusal) and scores 0.028 mean F1 vs 0.397 for answering.
- **Temporal length-band effect**: gold answers average 3.35 words; only the 3-5 word band can hold a date, and it scores 0.42-0.52 F1 in every run measured. The category's score tracks the population of that band far more than any change in the model's actual date-finding ability.
- **Retrieval scorer comparison** (offline, gold LoCoMo evidence): raw token overlap R@12=0.469; +stopwords+IDF → 0.624; BM25 adds nothing beyond that (default b=0.75 is *worse* than plain IDF on short turns); cosine alone is worst (R@1 0.186). 98.3% of gold turns missed at k=12 still share a content word with the query — it's a ranking problem, not a reachability one.
- **Tag-layer cap, not cue matching, is retrieval's real ceiling (measured 2026-09-07).** 95.2% of gold evidence is reachable via the cue→tag hop; only 28.2% survived the old lexical-only `MAX_ACTIVE_TAGS` cut, and 94% of what was lost scored an exact lexical `0.0` (an alphabetical tie-break among ~350 equally-uninformative candidates, not a ranking failure on merit). Oracle ceiling with a perfect tag ranker: 93.7% content recall — ~66 points of headroom existed at this one step alone. See §4 run 12b for the fix (tag-layer RRF fusion) and its measured in-production effect.
- **Mask-fix and tag-fusion contributions are mechanistically separable and were measured that way, not conflated.** Holding one fixed while varying the other (§4 run 11 / 12a / 12b) showed retrieval diagnostics (`fail_open_all`, `stop_reason`, zero-evidence question count) move only between 12a and 12b, not between 11 and 12a -- confirming the mask fix's score gain comes entirely from the attention/OSAM side and the tag-fusion gain entirely from retrieval, with no cross-contamination between the two arms.
- **Capacity**: state is r×r=64 scalars per layer; ~12 evidence writes already exceed the 8 independent directions available. Not the only channel though — see §5 item 1, the evidence is also (currently, wrongly per spec) in the KV cache.
- Open-domain is structurally hardest — the paper's own no-OSAM baseline gets 0.1894/10.77 there. Date arithmetic is unreliable at 4B regardless of prompting. At least one LoCoMo gold has a typo (`"Yesteammates..."`) that zeroes a correct answer.

---

## 8. Traps

1. **Full run does not clear its checkpoint** (only `--smoke` does). Archive `workmem_iterret_full.jsonl` before every full run or you'll get old scores back in 2 minutes.
2. **Keep `outputs/graph_cache/`** — unaffected by prompt/ranking changes, costs ~600 vLLM calls/conversation to rebuild.
3. **Generation is greedy** (`do_sample=False`), not the official protocol's `temp=0.4/top_p=0.9/top_k=10`. `max_new_tokens` defaults to 2048, not the protocol's 50 (`OFFICIAL_MAX_NEW_TOKENS` imported, unused) — harmless today since answers are terse, but a footgun.
4. **`hasattr(graph, "nodes")` is always False** — "Graph ready: N nodes" has never printed a real number. Cosmetic.
5. **RESOLVED 2026-09-07 — `osam_contribution` used to show `0.0`/`0.0`/`0.0` on every row of every full run through run 11.** Fixed, see §5 item 3 and §4 run 12a. Kept here so anyone reading an old run's file (run 11 and earlier) knows why its `osam_contribution` field is either absent or all-zero, and doesn't mistake that for "OSAM contributes nothing" on those files specifically.
6. **`run_full_pipeline.slurm` writes its output one directory too deep, found 2026-09-07 — not yet fixed.** The eval script's `OUTPUT_FILE` fallback derives from `CAIMMS_ROOT`, treating it as the workspace; `env.sh` actually sets `CAIMMS_ROOT` to the repo itself. `run_pipeline.sh` (the local, non-SLURM runner) already works around this correctly with an explicit `export WORKMEM_OUTPUT_FILE=...`; `run_full_pipeline.slurm` never does, so on Mahamathi the real output lands at `<repo>/outputs/workmem_iterret_full.jsonl` (inside the repo) instead of `<workspace>/outputs/workmem_iterret_full.jsonl`. Consequence: `guardian.sh`'s own row-counting checks the *workspace* path and can never see a SLURM-submitted job's real progress -- once such a job finishes or is killed, guardian will think 0/1540 is done and resubmit forever. Workaround used this session: check/score the repo-internal path directly; kill the guardian `tmux` session manually once the real file hits 1540 rather than trusting it to stop itself. Proper fix: add the same explicit `export WORKMEM_OUTPUT_FILE="${OUTPUT_DIR}/workmem_iterret_full.jsonl"` line to `run_full_pipeline.slurm` that `run_pipeline.sh` already has -- not yet applied, deliberately held off mid-run to avoid orphaning an in-flight job's checkpoint.

---

## 9. Ownership and admin — unfinished

- **`IterRet/` is VENDORED and there is now exactly ONE copy** (2026-08-20). It lives at `workmem-vertical/IterRet/`, tracked by this branch. The standalone sibling checkout was **deleted** — two copies of the same package with nothing syncing them is precisely how this project already produced three divergent `CATEGORY_MAP` definitions, and the risk is now removed rather than documented.
  - `docs/patches/iterret-workmem-modifications.patch` preserves the exact diff vs Pragnya's upstream `IterRet` branch (`949d4a3`), so the reconciliation task is still actionable without the standalone checkout. `episode_segmenter.py` and `tests/` are new files, not in the patch — take them from the vendored tree.
  - **Still Pragnya's vertical, still unreconciled.** Vendoring made the branch self-contained; it did not resolve ownership. The upstream `IterRet` branch is untouched.
- **The repo is now self-sufficient — the separate `ashwinGPU` deployment bundle is gone** (2026-08-20). `env.sh` at the repo root derives every path from two roots (`CAIMMS_ROOT` = the repo, `CAIMMS_WORKSPACE` = its parent, holding `models/` and `outputs/`), so the identical tree runs on the Mac, Mahamathi and resiliente-2003 with no per-machine patching. The four `deltamem/workmem/*.py` files that used to exist in two versions (repo copy with hardcoded Mahamathi paths, bundle copy with env-var paths) are now the single env-var version. Deployment scripts (`run_pipeline.sh`, `setup_env.sh`, `download_assets.sh`, `dryrun_pipeline.py`, …) live in `scripts/`.
- **Mahamathi job kills**: `cn6`/`a100` jobs get `CANCELLED by 0` (root) after 40min-1.5h. Ruled out: QOS, preemption, self-cancellation. Leading hypothesis: oversubscribed-node contention. **Admins never asked.** Largely moot while resiliente-2003 is primary.
- **`pydantic` version untested**: `setup.md` says `2.9.2` is critical for vLLM/FastAPI; `requirements_exact.txt` pins `2.13.4`, which is what both boxes actually ran successfully. Which one actually matters has never been isolated.
- **A second session/agent worked against the same Mahamathi checkout concurrently, observed 2026-09-07.** It ran an independent production run isolating the causal-mask fix (`workmem_iterret_mask_fixed.jsonl`, run 12a above) and, inspecting the shared local `outputs/` folder, flagged a file from a different session (this one's tag-fusion run, run 12b) as unrecognized -- it had no visibility into the other session's work. Nothing enforces isolation between concurrent sessions sharing one `workmem-vertical` checkout on a cluster; a sync from either side can silently overwrite the other's in-progress uncommitted edits (e.g. `IterRet/iterret/ctc_graph.py`, still uncommitted as of this update). Worth committing work more frequently, or coordinating explicitly, before this causes a real lost-work incident.
