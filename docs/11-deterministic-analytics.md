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

## 11.7 Trends and anomalies

Two questions the five answers do not cover, because they are asked about
*arbitrary* series rather than the five fixed shapes: "when did this level
change?" and "which readings broke away from it?" Both live behind
`tools/phase0/trends.py`.

### Change points

`pelt()` is an exact PELT segmentation for a piecewise-constant mean, about
sixty lines and no dependencies. Writing it rather than importing `ruptures` is
a decision that has to be justified, so it is: on 300 random piecewise series it
agrees with `ruptures.Pelt(model="l2")` on the breakpoints **300/300**, and that
comparison runs as a test whenever the library is installed.

`cusum()` answers the online version. Two things about it were wrong in the
first implementation and are worth recording:

- **The reference statistics must not include the change.** Standardising with
  the median of the whole series makes a half-low, half-high series look like a
  sustained downward excursion for its entire first half, and the detector fires
  *before* the change it was looking for. Centre and scale now come from a
  baseline window (a quarter of the series, never more than half).
- **The false-alarm rate is a measured number, not a claim.** CUSUM is a
  sequential test; it will cry wolf eventually. `false_alarm_rate()` simulates
  it: h=4 gives 4.24 alarms per 500 quiet samples, h=5 gives 1.85, h=6 gives
  0.92. The default is 6, and on 200 simulated series with a 6-sigma step it
  raised an alarm before the change 12.5 % of the time while detecting 198/200
  with a median lag of zero.

### Anomaly scores

The score is MAD-based (median absolute deviation, scaled by 1.4826) evaluated
on residuals, for two reasons that are each a failure mode of the naive version:

- **A spike must not hide itself.** With mean and standard deviation, a single
  large excursion inflates the very scale used to judge it; the classic z-score
  saturates near sqrt(n) and, with two spikes, drops below the threshold
  entirely. The test asserts that the robust score stays above the classic one
  at 6, 60 and 6000 units.
- **The local level must be a line, not a median.** A centred rolling median is
  biased at the ends of a trending series, so a battery warming smoothly
  produces a large residual at every edge reading - a detector that fires
  whenever the car is driven normally. The level is therefore a robust linear
  fit (median of pairwise slopes): a ramp fits exactly and has zero residual,
  and a spike cannot move a median-of-slopes fit.

### The lesson the tool exists to demonstrate

Consumption per kilometre is only comparable between days that drove a
comparable distance. On the shipped store the last two days are 7.9 km cold
trips at 321 Wh/km:

| | change point it reports |
|---|---|
| all days | **14 March**, 193.7 → 321.4 Wh/km |
| days ≥ 10 km (default) | **12 March**, 191.2 → 250.1 Wh/km |

The first answer is confident and wrong: nothing degraded on 14 March, the
driving pattern changed. The second matches the physical cause - ambient fell
from 12 C to -4 C on 12 March - and agrees with the independent day-over-day
result in 11.4. So the tool excludes short days by default and prints how many
it excluded, because a filter that silently changes the answer is worse than no
filter.

Run against `battery.cell_temp_max` over the overheat day it finds the level
climbing 56 → 77 C in 2 C steps between 17:48 and 17:56, and flags the peak at
17:56:59 (24 C above its local level) and the following morning's reset.

### What it does not do

- **PELT assumes piecewise constant.** A continuous ramp is approximated by
  steps, which is why the overheat run reports a change every ~90 s while the
  temperature rises smoothly. That is the model, not a bug, but it should not be
  read as nine distinct faults.
- **Statistical significance is not practical significance.** On a smooth
  signal a 0.1 C wobble scores above eleven sigma: arithmetically right,
  operationally useless. `min_delta` sets an absolute floor and the default is
  zero - the domain threshold belongs to the caller, not to this module.
- **Signals are scored one at a time.** A point that is unremarkable in every
  margin but rare in combination is missed, and the test measures that gap
  against IsolationForest rather than claiming the robust score is better.

## 11.8 Episode cards and retrieval

Retrieval searches cards, not samples and not events. A sample is too small to be
an answer ("coolant pump = 0 at 17:35:04") and an event row is too small in a
different way: one overheating incident is four rows in the store and one memory
for the driver. A card is the middle - a bounded stretch of behaviour with a
narrative, the signals it touched, a severity, and provenance pointers back to
the rows that justify it. They are built in the same pass as segments and
rollups, so `make store` produces them.

```
event ─┐
       ├─▶ episodes.distill ─▶ card ─▶ card_fts (FTS5, porter)
daily ─┘                              │
                          retrieval.search(text) ─▶ bm25 ranking
```

Four things went wrong on the way, and each is now a test:

- **Context that is merely co-located is not context.** The first version
  scanned every "interesting" signal for every episode, so the ADAS card cited
  cell imbalance - a sentence that reads fine and means nothing. Signals are now
  mapped to event families, and an ADAS episode has no battery signals at all.
- **A stall is not an outlier.** The coolant pump toggles on and off for minutes
  before it fails, so its pre-stall swings are already extreme and no outlier
  score can pick out the moment it stopped *for good*. Persistence is the
  signature. The pump therefore gets `changepoint.sustained_below` - the same
  primitive Q4's stall detector uses, now shared rather than duplicated in two
  places.
- **A duration is not a proof time.** `sustained_below` returned the moment the
  stall became *provable* (one minute), so the card read "stayed off for 1
  minutes" for a twenty-minute stall. It now returns the whole run, and a third
  field saying whether the run had ended before the data did - because "off for
  22 minutes" is a fact and "off when the recording stopped" is only a bound.
- **Stopwords were doing real damage.** With three cards, BM25's inverse document
  frequency has nothing to work with, so "when did the coolant pump fail" matched
  the consumption card on the words "when" and "the". A small stopword list fixed
  it; IDF does the same job once the corpus is large, and the two compose.

**The query is data, not syntax** - the same principle as the DSL. FTS5 has its
own language (`NEAR`, `AND`, `*`, `:`, quoted phrases), so passing user text into
`MATCH` lets that language through. Every token is quoted and joined with OR, and
a token of only stopwords expands to nothing rather than to everything.

### On not reporting recall@8

Gate G2 asks for recall@8 ≥ 85 % on a 100-query set. **No such number is
reported here.** With three cards on the shipped store, recall@8 is 1.0 for any
query that matches anything, which measures the corpus rather than the retriever.
What is tested instead is that the mechanism ranks correctly and that hostile
input is inert; the recall figure needs a richer scenario corpus and real
queries, and it stays open rather than being filled in with a number that would
not mean what it looks like.

The dense half of hybrid retrieval is likewise **not implemented**: it needs
`bge-small-en-v1.5` and HuggingFace is unreachable from the build sandbox.
`reciprocal_rank_fusion` is written and tested against synthetic rankings so the
dense side is a plug-in, but nothing in this repository is measured on an index
that does not exist.

## 11.9 What moved first

"What happened before the battery overheating warning?" is a precedence
question, and precedence is where analytics most easily lies. Two instruments
answer different halves of it: **onset** (where a series departs from a settled
state) dates an event; **lagged correlation** measures how alike two series are
once slid against each other. `vehiclegpt/analytics/leadlag.py` runs both, and
`make precursors` prints them for the warning event.

### The correlation is right; reading it is not

`cross_correlation` is O(1) per lag: the cross term comes from one
`np.correlate(y, x, "full")` call and the moments from prefix sums. The index
mapping is easy to invert — the first attempt used `(n-1)-L` and returned `-5`
for a known `+5` lead — so the test suite pins it against a naive double loop
(`4.4e-16` agreement) and against a pure shift, which must give exactly `1.0`.

The physics of the data, not the arithmetic, is the problem. Every onboard
signal is a slow ramp, so pairwise correlation between two *independent* AR(1)
series at φ=0.98 over 600 samples averages `|r| = 0.33` and peaks at `0.57`.
Gating on a p-value does not fix this: the circular-shift null was already
correctly calibrated, at 5–10% false positives. The coefficient itself is
overstated because 600 smooth samples contain about 6 independent ones.

So every row travels with `n_eff = n(1-φ)/(1+φ)` and `reliable`, defined as
`n_eff ≥ 10`. On the 2026-03-11 event the aligned series are so smooth that
`n_eff` collapses to **1.2** and all eighteen signals are unreliable: their
correlations run 0.86–1.00 and mean nothing. The table still reports them, and
the CLI marks each one `smooth, corr not usable`. Reporting the coefficient
without the sample size is the error to avoid; the statistic was never wrong.

### Precedence needed three fixes to say anything

`changepoint.segments()[1]["start"]` — the first change in the window — is not
an onset. It fired at the window's edge, so almost every signal "led" by the
same amount and the ranking was arbitrary. Three corrections, each caught by a
test:

1. **A change in the opening seconds is the window starting.** Every signal
   begins somewhere; without a skip the drive start wins for all eighteen.
2. **Anchor the window to the drive.** Between drives there is no data, so a
   window straddling take-off makes "the car started" the largest change in
   every signal at once.
3. **A precursor is a departure from a settled state.** The preceding segment
   must have lasted minutes. This is what separates *this signal changed* from
   *this signal has been changing all along*.

The results are then honest but unflattering. On the warning event, the coolant
inlet and `cell_temp_min` turn at 17:36:46 — before the 17:50:12 warning — while
speed, rpm, pedal and brake all turn together at 17:50:00 when the car stops.
`crowded` is set: **11 signals departed within seconds of each other**, which is
the signature of one shared driver, not of one cause among them. Precedence
cannot separate them and the CLI says so instead of printing a ranking.

### The cause is in the group that cannot be dated

The coolant pump is not in that list at all. It duty-cycles before it fails, so
it never holds a steady state and has **no baseline to depart from**. It is
returned separately as `unsettled` rather than dropped, because "not dated" is
not "not involved" — it is the actual cause. A duty-cycled fault needs a
different instrument, and `episodes.py` has one: `sustained_below` dates it at
17:35, 22 minutes before the warning. Both mechanisms are needed; neither
subsumes the other, and the tool points at the one that applies.

## 11.10 What this layer does not do

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
  motorway (5 and 80 kph), the stall window (60 s), and the onset parameters
  (300 s of quiet, 1.0 min-delta) are tuned to the simulator. They need
  re-tuning against recorded data.
- **It does not establish causation, only ordering.** A signal that departs
  first is a candidate antecedent, and on this dataset the candidates arrive in
  a crowd. Separating them needs intervention — the `fault.*` injections the
  generator already records — or a domain model, not a better correlation.
- **Precedence costs about 20 s per query** on 18 signals: PELT over 1500
  samples each, plus a 64-shift surrogate test per pair. That is far beyond the
  2 s budget, so it belongs in build-time analysis, not in the request path. If
  it must answer live, the candidates need pre-ranking and the shifts need
  cutting.
