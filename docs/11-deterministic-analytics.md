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

The store builds in **4.4 s** for 14 days, 686 400 samples and 305 segments,
and occupies 45 MB. Doc 06 projected 103 MB for 28 days `[D]`; this stores 26
signals uncompressed at 1 Hz rather than the projected mix, so the agreement is
ballpark, not exact.

---

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
> City driving averaged 214 Wh/km over 97.5 km at a mean 24 kph. Motorway
> driving averaged 188 Wh/km over 345.4 km at a mean 105 kph. City driving is
> the more expensive of the two here, by a factor of 1.14.

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

## 11.6 What this layer does not do

- **It does not parse free text.** `classify()` is a keyword matcher over five
  known shapes. A paraphrase it does not recognise escalates rather than
  guessing. Replacing it with a small classifier is a Phase 3 decision.
- **It does not explain causality.** It reports a decomposition and labels it an
  attribution. "HVAC accounts for 66 Wh/km of 213" is arithmetic; "the cold
  caused the range loss" is an inference and belongs to the model layer.
- **It has never seen a real vehicle.** The thresholds that split city from
  motorway (5 and 80 kph) and the stall window (60 s) are tuned to the
  simulator. They need re-tuning against recorded data.
