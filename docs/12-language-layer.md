# 12. The language layer

The model is not the system. This document covers the part that decides what a
question means and what an answer is allowed to say, and it is deliberately
model-free: every component here is deterministic, costs microseconds, and can
be tested without weights. That is what lets the SLM be added later without
being trusted — see §12.6 for what is still missing.

Code: `vehiclegpt/nlu/`. Run it with `make nlu`.

## 12.1 Where the model sits, and what it is not allowed to do

The design keeps three things out of the model's reach, because each has a
failure mode a driver cannot see:

| The model must not | Because | Enforced by |
|---|---|---|
| compute a number | a plausible wrong figure is unfalsifiable at the wheel | `evidence.Pack` — the only source of values |
| choose the query | a wrong intent returns a confident answer to a different question | `guard.verify_numbers` + handler dispatch |
| be the only reader of text | injection arrives through voice, service notes, fleet uploads | `guard.sanitise` |

So the model's job, when it arrives, is to turn an evidence pack into prose —
and its output is checked against the pack afterwards by a machine, not by a
reviewer.

## 12.2 Time expressions

`nlu/timeparse.py` resolves "yesterday", "the last two weeks", "between 11 and
12 March" and similar to a half-open window. Three decisions change answers:

1. **"Now" is a parameter, not the wall clock.** A parked vehicle's clock runs
   months past its newest sample, so a query anchored to wall-clock time returns
   nothing and looks broken. The caller passes the newest sample instant.
2. **Relative phrases end now.** "the last two weeks" means two weeks up to the
   latest data, not two calendar weeks.
3. **Nothing parsed is a valid answer.** `parse` returns `None`; it never
   silently defaults to "all time", which is how a query becomes a full-table
   scan on a 2 GB device.

The distinction between "last week" (the previous calendar week) and "the last
week" (seven days to now) hangs on a single article, and the code tests for it
with a negative lookbehind. That is fragile in exactly the way it should be:
the failure is visible — a one-week window in the wrong place — rather than a
quietly wrong aggregate.

Two bugs found while building it, both now tests:

- "the day before yesterday" contains "yesterday", so the more specific phrase
  must be tested first or it is indistinguishable.
- A range ("between A and B") must be tried before single dates, or it collapses
  to one day.

## 12.3 Routing, and what the holdout says

`nlu/intent.py` scores five intents rather than requiring all of an intent's
keywords to appear as substrings. Each intent has strong terms (3 points) and
weak terms (1 point); the highest total wins; below a floor of 3 the answer is
`unknown`.

Two rules do the real work:

- **Abstain rather than guess.** A routing error is invisible; an escalation is
  not.
- **Time words are stripped before scoring.** "yesterday" says *when*, not
  *what*; without the strip, "what happened yesterday" would route on the word
  "happened".

Surface forms are handled by equivalence classes rather than by patching
phrases into the term lists — "the car is thirstier", "it burns more power per
distance" and "consumption" all canonicalise to one token before scoring. This
is also where a subtle ordering bug lived: canonicalisation has to run *before*
time-word removal, because "it does not last as long" is a range complaint and
stripping "last" first destroys the phrase.

### The numbers, including the bad one

| Corpus | Phrases | Accuracy | Negatives wrongly claimed |
|---|---|---|---|
| design (used while writing the vocabulary) | 25 | 96% | 0 / 10 |
| holdout v1 (written before tuning, measured after) | 17 | 94% | 0 / 10 |
| holdout v2 (written after tuning, never used to tune) | 17 | **65%** | 0 / 10 |

The 65% is the honest generalisation figure and the most useful result in this
document. Two properties are worth more than the headline:

- **Every miss is an abstention, never a wrong intent.** The router fails
  visibly. On the holdout, `0/17` misses were misroutes.
- **The gap is the argument for task 3.3.** 96% on the set its author wrote
  against, 65% on fresh phrasing, means lexical routing does not generalise and
  hand-tuning is close to its ceiling. That is the empirical case for a small
  trained encoder (3.3) or the SLM router (3.5) — not a preference for models,
  a measured gap in coverage.

The failures are informative: "the guess-o-meter has got pessimistic", "I'm
charging way more often than I used to", "any moments the car asked me to take
the wheel". All mean exactly what a supported phrase means. A keyword list
cannot reach them; an embedding can.

`tests/test_nlu_intent.py` pins the relationship, so if either corpus is ever
tuned against, a test fails.

## 12.4 The evidence pack, and the grounding gate

The gate for Phase 3 is *numeric grounding rate = 1.000*. It is measured, not
asserted:

```
$ make nlu
grounding 28/28 = 100.0% across 5 answers
```

The mechanism: `evidence.Pack` registers computed values with a name, a unit, a
provenance string and **the format they will be printed in**; `render` is the
only thing that substitutes slots; `guard.verify_numbers` re-reads the finished
sentence and requires every numeric token to be re-producible from the pack.

Keeping the format in the fact matters. If the verifier only knew the value, it
would have to accept "249", "249.4" and "249.44" as all correct, and the check
would degrade into tolerance. Because the declared format is known, rounding is
checked at the precision actually permitted.

### First measurement: 89.3%, and three real defects

Run against the existing handlers, the gate failed at **25/28**. Each failure
was a genuinely untraceable number:

| Answer | Number | Cause |
|---|---|---|
| efficiency drop | `+24` | percentage computed inside the f-string, never registered |
| before warning | `55` | a **hard-coded literal** in the prose, not a computed threshold |
| city vs highway | `1.39` | ratio computed at the call site, printed, not stored |

The `55` is the instructive one. It was a constant in a sentence describing a
computed condition — the sentence would have kept saying 55 if the threshold had
changed. It is now `queries.TEMP_RISE_C`, printed from the constant and
registered in `data`.

All three were fixed by registering the derived value, not by loosening the
check. Every answer's text is byte-identical; only its traceability changed.

### What this gate does not prove

It proves every number appears in the pack. It does **not** prove each number is
attributed to the specific fact that produced it — the check is membership, not
per-sentence provenance. A sentence that swapped two registered values would
pass. Tightening that means rendering from named slots with no free-text
formatting, which is the remaining half of task 3.7.

It also only covers the five brief questions. The verifier is general, but the
corpus it runs on is not yet.

## 12.5 Inbound guard

`sanitise` normalises the question (NFKC, strip zero-width and bidi characters,
collapse whitespace, cap length) and reports instruction patterns: overrides,
prompt probing, role hijack, chat-template tokens, SQL fragments.

Zero-width stripping is not cosmetic. `\u200b` inside a word is invisible in a
transcript and meaningful to a tokeniser — the current implementation catches
exactly that, with a test that embeds it in "drop".

Severity is graded because the responses differ: an override is **blocked** (the
question is refused and the refusal is shown), while a brace or a build-id-shaped
string only warns. Refusing to answer is a safe failure; silently answering a
hijacked question is not.

What it does not do: stop a paraphrase of an attack that uses none of the
patterns. That is a job for the model-side guardrail and the red-team phase, and
a regex list must never be presented as a defence on its own.

## 12.6 What is still missing

Phase 3 splits cleanly and only half of it is buildable here, because HuggingFace
is unreachable from the build sandbox (the same block that stopped dense
retrieval in §11).

| Task | Status |
|---|---|
| 3.1 rule pre-pass (time, units, DTC) | ✅ time and slots done; DTC regex pending data |
| 3.2 embedding kNN bootstrap | ⛔ needs an embedding model |
| 3.3 fine-tune encoder for intent + slots | ⛔ needs weights and a GPU |
| 3.4 quantise to int8, verify delta | ⛔ blocked on 3.3 |
| 3.5 SLM router (GBNF grammar) | ⛔ no weights |
| 3.6 evidence pack with provenance | ✅ |
| 3.7 template renderer | 🔶 renderer and slots exist; handlers still format inline |
| 3.8 SLM narrator | ⛔ no weights |
| 3.9 guardrails | ✅ scope, sanitiser, numeric verifier |
| 3.10 eval harness | ✅ routing and grounding measured |
| 3.11 latency harness | ⛔ blocked on 3.5 |

The blocking items are the ones that need a ~200 MB download. Everything that
constrains the model — the pack, the renderer, the verifier, the harness — is
in place *before* the model arrives, which is the order that makes it possible
to say no to a fluent wrong answer.

**Gate G3 status: not met.** Numeric grounding is 1.000 on the measured corpus,
but intent accuracy is 65% on a clean holdout against a 92% target, and P0
latency is measured separately in `docs/10`. The gap is coverage, not
correctness — the router abstains rather than misroutes — and closing it is
tasks 3.3 or 3.5, both of which need weights.

## 12.7 Running it

```bash
make nlu                                    # routing score + grounding measurement
python3 tools/phase0/nlu.py --db /tmp/vehiclegpt.db --ask "why is my range down"
python3 tools/phase0/nlu.py --db /tmp/vehiclegpt.db --eval holdout_v2
python3 tools/phase0/nlu.py --db /tmp/vehiclegpt.db --grounding --json
```

`--ask` returns 0 when the answer is grounded, 1 when the question abstains, 2
when it is refused, and 3 when an answer is produced but a number in it cannot
be traced. That last exit code is the one worth wiring into CI: it means the
substitution mechanism has a hole.
