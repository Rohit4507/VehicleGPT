# 11 — The deterministic analytics layer

**Status.** Everything in this document is measured by running the code in this
repository, on the machine named in each figure. Where a number is derived
rather than observed it is tagged `[D]`. Nothing here is a measurement of a
finished product; it is a measurement of the components that exist.

---

## 11.1 Why this layer exists at all

[Doc 10](10-tierc-latency-strategy.md) established that on the constrained tier
the model cannot meet the 2-second target on P1, P2 or P3 `[D]`. Rather than
treat that as a defect to be argued around, the architecture treats it as a
constraint to be routed around: **questions that can be answered by a filter or
an aggregate are answered by a filter or an aggregate.**

That is not a compromise. For "show all ADAS disengagements from yesterday" a
language model adds latency and a chance of stating the wrong count, and adds
nothing else. The deterministic path is exact, auditable and reproducible.

Of the five questions in the brief, four are filter/aggregate/sequence questions
and one ("why has my range reduced?") is a decomposition. All five are handled
without decoding a token.

---

## 11.2 The pipeline

```
simulated 14-day program  (sim/program.py, seeded)
        |
        v
1 Hz store  (vehiclegpt/store/db.py, SQLite, WAL)
        |
        +--> segment  : speed-regime segmentation, V x I energy integration
        +--> daily    : distance, energy, consumption, ambient, HVAC, SOC
        |
        v
SQL views  (vehiclegpt/analytics/views.sql)
        |
        v
query handlers  (vehiclegpt/analytics/queries.py)  ->  Answer(text, data, evidence)
```

The store builds in **7.0 s** for 14 days, 426 373 samples and 281 segments,
and occupies 9.3 MB. Samples are decimated per signal on write, so this is
**22.0 B/sample** against the 65.4 B/sample the same data cost before the
schema work. Doc 06 projected 103 MB for 28 days `[D]`; at the driving duty
cycle this store actually sees that is 18.7 MB, but the projection's own
workload (15 signals at 1 Hz, 24/7) would cost ~799 MB at this density — see
doc 06 §7 for why the two are not comparable and which one is being claimed.

---

### How samples get decimated

The storage key is `(signal_id, ts_s)` — whole seconds — so the reduction has to
happen on write. Each signal gets a policy from `vehiclegpt/store/decimate.py`:

| Policy | Applies to | Rule |
|---|---|---|
| `LAST` | ordinary continuous signals | last sample in each second |
| `ON_CHANGE` | `slow`-tier signals with a deadband | keep a value only when it moves beyond the deadband, **plus a keyframe every 60 s** |
| `ANY` | flags (declared range 0–1) | max per second, then collapse to transitions |

Two of these rules exist because the obvious version was wrong in a way that
row counts did not show:

- **Bucketing by the input rate is not decimation.** The first version bucketed
  at `1e9 / rate_hz` — 100 ms at 10 Hz — and looked correct because
  `INSERT OR REPLACE` on a one-second key silently swallowed the duplicates. The
  fingerprint is `rows / distinct_seconds == 1.0` for every signal, which is
  worth asserting directly rather than reading the total.
- **Point-sampling a flag deletes the event.** Taking the last value in a second
  turns a 0→1→0 pulse into 0. Flags reduce by max-per-second, then collapse to
  transitions, so an ADAS disengagement that lasted 0.1 s survives.
- **On-change without keyframes destroys duration.** A 15-minute coolant-pump
  stall collapsed to its two endpoints, and the Q4 stall detector — which needs
  a reading still at zero 60 s later — reported that no stall had occurred. The
  keyframe bounds every plateau at 60 s, which costs +0.5 % of rows and is
  covered by an end-to-end test that fails if it is removed.

## 11.3 What the data actually contains

The program is generated, not recorded, and it contains three events by
construction so that the questions have real answers:

| When | What | Purpose |
|---|---|---|
| 11 Mar 17:35 | coolant pump stops for good | precursor for Q4 |
| 11 Mar 17:50 | battery overheating warning raised | the thing Q4 asks about |
| 12 Mar | ambient falls 12 C → −4 C | the cold snap behind Q1 and Q2 |
| 15 Mar 08:03–08:10 | three ADAS disengagements | Q3, "yesterday" |

---

## 11.4 The five answers, verbatim

Produced by `make ask`, which rebuilds the store and runs every question. The
slowest of the five took **2.3 ms** on this machine.

> **Why has my driving range reduced over the last two weeks?**
> Consumption over the last 7 days averaged 213 Wh/km, up from 173 Wh/km the
> week before (+23%). Mean ambient temperature fell from 16.3 C to 2.7 C. HVAC
> draw rose from 1323 W to 4047 W, which is 66 Wh/km (31% of consumption)
> against 22 Wh/km (13%) before. Heating alone would explain more than the
> observed increase; part of it is offset by lower average speeds in the cold.

> **When did fuel efficiency show the largest drop?**
> Consumption jumped on Thu 12 Mar: 240 Wh/km against 193 Wh/km the day before,
> an increase of 47 Wh/km (+24%). Ambient temperature went from 12.0 C to
> −4.0 C over the same step.

> **Show all ADAS disengagement events from yesterday.**
> 3 ADAS disengagement events on Sun 15 Mar at 08:03:20, 08:07:10, 08:10:10 UTC.

> **What happened before the battery overheating warning?**
> Before the battery overheating warning: the coolant pump stopped for good at
> 17:35 UTC, 15 minutes earlier; pack temperature passed 55 C at 17:48 UTC; the
> overheating warning was raised at 17:50 UTC.

> **Compare energy consumption between city and highway driving.**
> City driving averaged 263 Wh/km over 97.6 km at a mean 24 kph. Motorway
> driving averaged 189 Wh/km over 345.4 km at a mean 105 kph. City driving is
> the more expensive of the two here, by a factor of 1.39.
> note: energy is attributed per segment, so acceleration across a regime
> boundary counts toward the segment it started in; a per-second attribution
> gives a lower city/highway factor

**The factor is methodology-sensitive and the answer says so.** Attributing each
second's energy to the regime in force during that second gives city 243 Wh/km
and motorway 205 Wh/km — a factor of **1.18**, not 1.39. Both totals reconcile
(443.0 km, ~205 Wh/km overall), so the disagreement is purely about which regime
owns the seconds spent accelerating across a boundary. The direction of the
answer is robust; the magnitude is not, and a reviewer should be told which
convention produced the number rather than left to assume.

Every one of these is a single SQL query plus arithmetic. The numbers can be
checked by hand against `segment` and `daily`.

### On the 2.3 ms figure

That is SQL execution plus rule-based intent classification. The 110 ms P0
budget in `tools/paths.py` covers the whole path — parsing, retrieval, response
assembly, UI hand-off — none of which this demo includes. Treat 2.3 ms as a
lower bound on the database work, not as evidence that P0 runs in 2.3 ms.

---

## 11.5 Three defects the tests caught

Recorded because they are the kind that produce confident, wrong answers.

**Segmentation integrated across the overnight gap.** The first version split
the timeline into drives *after* segmenting by speed regime, so a segment could
span nine parked hours and integrate distance at the last known speed. It
produced a 415 km commute and −1 652 Wh/km consumption. The fix is ordering:
split into drives first, then segment within each drive, and refuse to integrate
across any gap longer than one sample interval.

**The pump stall detector fired on traffic lights.** A coolant pump legitimately
drops to zero when the car is stationary, so the first `>5 % → 0` transition is
usually a red light. The detector now requires the zero to persist for 60 s.
That also keeps the answer honest: it is found by looking at the signal, not by
reading the `fault.coolant_pump_failure` event the simulator injected.

**The HVAC attribution was a hardcoded fraction.** The first version of Q1
multiplied the total increase by 0.55 and called it "the HVAC share", which is
indistinguishable from making the number up. It now computes mean HVAC draw in
watts × hours driven ÷ kilometres in both windows and reports the difference.
When that difference exceeds the total change, the answer says so instead of
forcing the residual to be positive.

A related rule, enforced by a test: **`event` rows whose kind begins `fault.`
are harness metadata and are never used as evidence.** Building an answer from
them would be reading the answer key.

---

## 11.6 The query DSL: what the model is allowed to say

The five handlers above take no window argument - they look at "the last two
weeks" because that is the question the brief asks. That is fine for a demo and
wrong for a product: the user will say "the last month", and the language model
in Phase 3 has to be able to express it. So the handlers no longer contain their
own SQL. They are renderers over `vehiclegpt/analytics/dsl.py`.

```
free text ──▶ classify()  ─┐
                           ├─▶  Query(kind, window, params)  ──▶  compile  ──▶  SQL + bindings
model JSON ─▶ dsl.parse() ─┘
```

A query is a small JSON object:

```json
{"kind": "regime_compare", "start_day": "2026-03-02", "end_day": "2026-03-15"}
```

`kind` is one of five shapes (`consumption_trend`, `efficiency_drop`,
`adas_events`, `precursor`, `regime_compare`). Anything else - an unknown key, a
date that is not a date, a window longer than two years - is rejected with an
error naming the problem.

**The property that makes this safe to hand to a model:** the compiled SQL is a
function of `kind` alone. Every value is a bound parameter, so there is no code
path in which a value reaches the query text. Injection is not filtered out; it
has nowhere to enter. That is asserted, not asserted-to:

- `test_sql_text_is_identical_across_wildly_different_inputs` compiles each kind
  against wildly different windows and params and compares the SQL byte for byte.
- `test_hostile_dates_never_approach_the_sql` feeds eight payloads
  (`'; DROP TABLE sample; --`, `UNION SELECT`, NUL bytes) through validation.
- `test_a_query_built_without_parse_cannot_reach_the_database` bypasses the
  validator by constructing `Query` directly - the compiler still refuses.
- The suite is mutation-checked: rewriting the compiler to interpolate
  `start_day` fails three tests immediately.

Trying it by hand:

```
$ python3 tools/phase0/ask.py --db /tmp/vg.db --dsl \
    '{"kind":"adas_events","start_day":"2026-03-01'\''; DROP TABLE sample; --","end_day":"2026-03-05"}'
invalid query: start_day is not an ISO date: "2026-03-01'; DROP TABLE sample; --"
```

Exit code 2, and `sample` still has all 426 373 rows.

**Why a closed set of five kinds rather than a general SQL dialect.** A general
compiler is a much larger surface to test, and its extra power would only ever
be exercised by the model - the deterministic handlers need exactly these five.
The five are the brief's five questions; a new kind is a deliberate code change
with its own SQL constant, not something a model can invent at runtime.

## 11.7 What this layer does not do

- **It does not parse free text.** `classify()` is a keyword matcher over five
  known shapes. A paraphrase it does not recognise escalates rather than
  guessing. Replacing it with a small classifier is a Phase 3 decision — but the
  DSL is now the target that classifier emits, so the two can be developed
  separately.
- **The DSL covers four of the five questions directly.** `precursor` returns
  warning events and their event-level predecessors, and on this dataset that is
  *empty by design*: the only predecessors recorded are the harness's own
  `fault.*` rows, which are excluded. The real antecedent - the coolant pump
  stopping - is found in the sample series by `_sustained_stall`, not by SQL.
  Time-series reasoning is a typed accessor, not a query shape, and pretending
  otherwise would put a scan of millions of rows behind a single DSL token.
- **It does not explain causality.** It reports a decomposition and labels it an
  attribution. "HVAC accounts for 66 Wh/km of 213" is arithmetic; "the cold
  caused the range loss" is an inference and belongs to the model layer.
- **It has never seen a real vehicle.** The thresholds that split city from
  motorway (5 and 80 kph) and the stall window (60 s) are tuned to the
  simulator. They need re-tuning against recorded data.
