# Delta-Mem + Iter-Ret — Full LoCoMo Run (29 September 2026)

This document describes the latest full evaluation of the C-AIMMS pipeline, which combines **IterRet** (iterative retrieval over a memory graph) with **δ-mem / OSAM** (an online associative memory inside the attention layers of a frozen Qwen3-4B), on the LoCoMo long-conversation memory benchmark.

---

## 1. Headline result

| | Overall | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---|---|---|---|---|
| **This run** | **0.5134** | **0.4113** | **0.6046** | **0.1657** | **0.5525** |
| Questions | 1540 | 282 | 321 | 96 | 841 |

**0.5134 token-F1 over all 1540 scored LoCoMo questions** — the highest full-benchmark score this pipeline has produced.

### Compared with the previous best full run (run 12b, 7 September)

| | Run 12b | This run | Change | 95% CI (paired bootstrap) | Better / worse questions |
|---|---|---|---|---|---|
| **Overall** | 0.4214 | **0.5134** | **+0.092** | [+0.074, +0.111] | 529 / 263 |
| Temporal | 0.4035 | **0.6046** | **+0.201** | [+0.159, +0.243] | 186 / 44 |
| Single-hop | 0.4766 | **0.5525** | +0.076 | [+0.050, +0.102] | 237 / 140 |
| Multi-hop | 0.3614 | **0.4113** | +0.050 | [+0.016, +0.086] | 90 / 61 |
| Open-domain | 0.1732 | 0.1657 | −0.007 | [−0.062, +0.044] | 16 / 18 |

The comparison is paired: both runs answered the same 1540 questions, and the change is measured question by question. The overall gain is significant (sign test p ≈ 2 × 10⁻²¹).

For reference, the `bm2772/del-mem` fork of this codebase reported **0.4969** on its own machine and graph cache (multi-hop 0.448, temporal 0.433, open-domain 0.143, single-hop 0.576).

### Every conversation improved

LoCoMo has 10 long conversations; this run beats run 12b on all of them.

| Conversation | Questions | Run 12b | This run | Change | Temporal: run 12b → this run |
|---|---|---|---|---|---|
| 0 | 152 | 0.425 | 0.511 | +0.086 | 0.466 → 0.782 |
| 1 | 81 | 0.434 | 0.517 | +0.084 | 0.567 → 0.739 |
| 2 | 152 | 0.471 | 0.587 | +0.117 | 0.402 → 0.730 |
| 3 | 199 | 0.378 | 0.479 | +0.101 | 0.300 → 0.562 |
| 4 | 178 | 0.453 | 0.529 | +0.076 | 0.394 → 0.521 |
| 5 | 123 | 0.402 | 0.511 | +0.109 | 0.378 → 0.423 |
| 6 | 150 | 0.424 | 0.520 | +0.096 | 0.298 → 0.457 |
| 7 | 191 | 0.393 | 0.502 | +0.110 | 0.400 → 0.633 |
| 8 | 156 | 0.428 | 0.483 | +0.055 | 0.395 → 0.576 |
| 9 | 158 | 0.424 | 0.508 | +0.084 | 0.481 → 0.591 |

---

## 2. How the pipeline works

The backbone is a frozen **Qwen3-4B-Instruct-2507** with a **δ-mem adapter**: one small 8×8 memory matrix `S` per attention layer, updated online by a gated delta rule. Reading `S` produces small corrections to each layer's queries and attention outputs.

### Step 1 — Build a memory graph for each conversation (once)

1. **Every conversation turn becomes a memory node**, stored with its session's date:
   `[8 May, 2023] Caroline: I went to a LGBTQ support group yesterday (7 May, 2023) and it was so powerful.`
2. **Relative dates are resolved when the node is created** (section 3).
3. **An LLM tags every turn** with *cues* (people, things, attributes mentioned) and a short *tag* describing the episode (e.g. "LGBTQ Support Group"). Cues, tags and turns form a Cue → Tag → Turn graph.
4. **The graph is saved** and reused by later runs.

| Graph statistics (all 10 conversations) | |
|---|---|
| Turn (episodic) nodes | 5,882 |
| Cues | 12,740 |
| Tags | 4,865 |
| Turns carrying a resolved date | 498 |

### Step 2 — Retrieve evidence for each question (IterRet)

Up to five rounds, each:

1. **Seed cues** by embedding similarity (MiniLM) between the question and every cue.
2. **Cue → Tag:** follow links to candidate tags and rank them by fusing word-overlap relevance (IDF-weighted) with embedding similarity via reciprocal rank fusion; keep the top 15.
3. **Tag → Turn:** fetch the turns behind those tags; if a round finds few, **top up** with up to 10 of the nearest unseen turns by embedding.
4. **Route and reflect:** an LLM keeps the useful turns as evidence, records what is still missing, and writes a refined query for the next round.

On average about **35 evidence items** are gathered per question. They are then ordered by relevance.

### Step 3 — Answer with OSAM live (two phases)

1. **Phase 1 — write the evidence into memory.** A fresh session resets `S`, then writes each evidence item into it as one message (message-mean granularity).
2. **Phase 2 — read and answer.** The answer prompt (section 4) is read *without* being written into `S`, so the memory holds only the evidence when the first answer token is produced. The model then generates greedily, and its own answer tokens are written into `S` as they are produced — the paper's Phase 2.

OSAM's correction was active on every question: its output averaged **2.8%** of the attention output (`mean_delta_o_ratio` 0.0282).

---

## 3. Relative-date resolution

LoCoMo stamps each conversation session with one timestamp, e.g. `1:56 pm on 8 May, 2023`. A turn that says *"I went yesterday"* therefore carries its date only implicitly — answering *"When did she go?"* requires date arithmetic, which a 4B model does unreliably.

The resolver (`IterRet/iterret/time_resolution.py`) does that arithmetic **once, inside the memory**, using only each turn's own timestamp. It keeps the original words and appends the date they mean:

| Said in a session dated 8 May 2023 (unless noted) | Stored as |
|---|---|
| yesterday | `yesterday (7 May, 2023)` |
| three days ago | `three days ago (5 May, 2023)` |
| last week *(session: 9 June 2023)* | `last week (the week before 9 June, 2023)` |
| last Friday *(session: 15 July 2023)* | `last Friday (the Friday before 15 July, 2023)` |
| next month | `next month (June 2023)` |
| five years ago | `five years ago (2018)` |
| on the 17th *(session: 19 August 2023, past tense)* | `on the 17th (17 August, 2023)` |
| last summer *(session: 11 August 2023)* | `last summer (the summer of 2022)` |

- "On Friday" and "the 17th" resolve to the one before or after the session depending on the sentence's tense ("I went on Friday" vs "we will play on Saturday").
- Timestamps are shortened to their date (`1:56 pm on 8 May, 2023` → `8 May, 2023`).
- Dates are written in the same style as the dataset's own timestamps.
- Nothing reads questions or answers, so the same step applies to any dataset whose memories carry timestamps.

Because the date is now written in the evidence, the prompt no longer needs a special timing instruction: the model copies the date.

**Measured on its own** (same 300 questions, only this change and the removal of the timing instruction): overall 0.469 → **0.554** (+0.084, 95% CI [+0.052, +0.117]); temporal 0.502 → **0.785** (+0.283, 95% CI [+0.219, +0.349]; 65 questions better, 7 worse). The share of temporal questions whose exact answer appears in the retrieved evidence rose from 28/89 to 75/89, and the 47 questions whose answer became findable only through resolution went from 0.571 to 0.957.

In the full run, temporal answers now almost always state a real date:

| Temporal answers (321) | Run 12b | This run |
|---|---|---|
| Contain a year | 203 | **243** |
| Only a relative expression ("last week") | 25 | **5** |

Refusal-like answers across all questions fell from 46 to **31**.

---

## 4. The answer prompt

Evidence items come first, one per message. The final message is built from a few instruction lines, LoCoMo's official question prompt, and the question.

**Timing question** (e.g. "When did …", "How long …", "What year …") — no timing instruction; the question is the last thing the model reads:
```
Do not use first-person pronouns (I, me, my, we, our). Refer to people by name.
Every question here has an answer. If the evidence states it, reply using those exact words. If the evidence only implies it, work out the most likely answer from the evidence and general knowledge, and give that. Commit to one specific answer even when you are not certain -- a specific answer you are unsure of is always better than a vague one.

Based on the above conversations, write a short answer for the following question in a few words. Do not write complete and lengthy sentences. Answer with exact words from the conversations whenever possible.

Question: When did Caroline go to the LGBTQ support group?
```

**Yes/no question** adds, after the second line:
```
This is a yes/no question. Start with Yes or No, then add a short phrase naming the detail that settles it.
```

**Any other question** adds, after the second line:
```
Answer by naming the specific thing being asked for (e.g. 'Pomodoro technique', 'psychology', 'the beach') as a short phrase, stated directly before any elaboration. This question asks which thing, not whether something is true, so the answer must name a thing. Do not give only the supporting reasoning without ever stating the answer itself.
```

The question type is decided from the question's wording only (never its answer or category label).

---

## 5. How the score is computed

- **Token F1 per answer**, against LoCoMo's gold answer. Both are lower-cased and stripped of punctuation, commas and the words *a, an, the, and*. (The scorer applies Porter stemming only when `nltk` is installed; it is not part of the pinned environment, so this run compared words as they are — as did run 12b, so the comparisons above are like for like.) Precision is the share of the answer's words found in the gold answer, recall the share of the gold answer's words found in the answer, and F1 their harmonic mean.
  Example: gold `7 May 2023`, answer `yesterday (7 May, 2023)` → precision 3/4, recall 3/3, **F1 0.86**.
- **Multi-hop:** the gold answer is split at commas; each part is matched against the best part of the answer, and the part scores are averaged.
- **Open-domain:** the gold answer is cut at its first `;` before scoring.
- **Temporal and single-hop:** plain F1.
- **Adversarial questions (446) are excluded** — their correct answer is a refusal. That leaves 1540 scored questions.
- **Overall score** = mean F1 over all 1540 questions (not an average of the four category scores), so single-hop, 55% of the questions, carries the most weight.

---

## 6. Run configuration

| | |
|---|---|
| Machine | resiliente-2003, 2 × RTX 4090 (vLLM on GPU 0, the δ-mem model on GPU 1) |
| Backbone | Qwen3-4B-Instruct-2507, bf16, served by vLLM 0.8.5 for graph building and retrieval |
| Memory adapter | δ-mem, rank 8, q/o heads |
| Code | branch `workmem-vertical`, commit `bc6c74f` |
| Graphs | rebuilt from scratch for this run |
| Decoding | greedy |
| Results file | `outputs/workmem_rebuilt_dates_full.jsonl` (1540 rows) |

Settings (all at their defaults):

| Setting | Value | Meaning |
|---|---|---|
| `ITERRET_RESOLVE_DATES` | 1 | relative dates resolved in the memory |
| `OSAM_TIMING_INSTRUCTION` | 0 | no timing instruction in the prompt |
| `OSAM_PHASE2_PROMPT_WRITE` | 0 | question and instructions are read, not written into OSAM |
| `OSAM_PHASE1_GRANULARITY` | message_mean | one memory write per evidence item |
| `OSAM_NEUTRAL_ANSWER_POLICY` | 0 | answer-policy line as shown in section 4 |
| `ITERRET_SEMANTIC_CUES` | 1 | cues seeded by embedding similarity |
| `ITERRET_FALLBACK_TOPUP_MIN` / `_ADD` | 8 / 10 | top up rounds with fewer than 8 hits by up to 10 turns |
| `WORKMEM_TEMPORAL_NARROWING` | 0 | no temporal evidence narrowing |

---

## 7. Reproducing this run

On resiliente-2003, from the repository root, in bash:

```bash
source env.sh && caimms_activate
python3 scripts/show_config.py        # confirm the settings above
```

To rebuild every graph from scratch, move any existing cache out of the way first:

```bash
mkdir -p "$CAIMMS_OUTPUT_DIR/archive_graphs" && mv "$CAIMMS_OUTPUT_DIR/graph_cache" "$CAIMMS_OUTPUT_DIR/archive_graphs/"
```

Then, inside tmux:

```bash
PYTHONNOUSERSITE=1 VLLM_PORT=8002 WORKMEM_OUTPUT_FILE=$CAIMMS_OUTPUT_DIR/workmem_rebuilt_dates_full.jsonl bash scripts/run_pipeline.sh
```

A cancelled run resumes from where it stopped when the same command is run again. Score it with:

```bash
python3 scripts/score_calculator.py "$CAIMMS_OUTPUT_DIR/workmem_rebuilt_dates_full.jsonl"
```
