# VehicleGPT

**Natural-language vehicle information using an on-device SLM/LLM.**

Entry for *Tata Elxsi TelePort Season 4 — Case Study 5*.

> **Status: Phase −1 — research and design review.** No production code yet.
> This repository currently contains the requirements analysis, research
> findings, technology comparison, proposed architecture, roadmap, hardware
> feasibility model, risk register and the open decisions awaiting sign-off.
> Implementation starts only after review.

---

## The one-paragraph pitch

> A vehicle generates enormous volumes of numbers, and almost none of it is
> usable by a human at the moment they need it. VehicleGPT turns weeks of
> CAN/SOME-IP/VHAL telemetry, diagnostic trouble codes, ADAS events and logs
> into something you can *ask a question about*, entirely offline, on a
> processor that costs less than a tank of fuel — and fast enough to answer the
> common questions before the driver finishes reading them.
>
> The trick is not putting a big model on a small chip. It is putting a **small
> model in the right place**: a deterministic analysis engine does the
> arithmetic, a compact typed query language does the retrieval, and the
> language model does the one thing only it can do — turn a verified evidence
> pack into a sentence a driver or a service engineer understands. **Every
> number in every answer is substituted from the evidence pack, never
> generated.** That single design rule is what makes an on-device assistant
> trustworthy enough to put in a car.

---

## The four findings that shape the design

### Evidence status — read this first

**No latency figure in this repository is an end-to-end measurement of
VehicleGPT, because VehicleGPT has not been built.** We have measured the
*hardware capability* it depends on and derived everything else, transparently
enough that a reviewer can check the arithmetic.

| Tag | Meaning |
|---|---|
| `[M]` | Measured by a cited source, on named hardware, with a stated methodology |
| `[D]` | Derived arithmetic from `[M]` anchors — reproducible via `tools/feasibility_model.py` |
| `[T]` | Target / not measured. Must be measured in Phase 0 or gate G4 |

Full per-claim audit: [doc 06 §11, the claims ledger](docs/06-hardware-feasibility.md#11-claims-ledger).

---

**1. Latency, not memory, is the binding constraint — and it is prefill-bound.**

Arm's benchmark of Qwen3-0.6B Q4_K_M on a physical Raspberry Pi 5 measures
**3.64 s time-to-first-token for a 512-token prompt** and 25.3 tok/s decode
`[M]` ([source](https://developer.arm.com/ai/models/hugging-face/Arm/qwen3-0-6b-base-q4-k-m-llamacpp-raspberrypi5/qwen3-0.6b-base-q4_k_m-gguf)).
Reconstructing that from the two rates reproduces Arm's 8.70 s end-to-end figure
to within 10 ms, so the numbers are internally consistent.

The consequence is a *design constraint, not a benchmark result*: any
architecture that stuffs raw telemetry or retrieved chunks into the prompt cannot
meet a 2-second budget on edge hardware, no matter how small the model. The
prompt has to stay under ~150 tokens, which means **the retrieval and analysis
layers must be good enough that the model never needs to see raw data.**

**2. The ~2 GB / 2×Cortex-A5x envelope cannot run a generative step per query.**

Cortex-A53 single-thread performance is ~4.5× lower than Cortex-A76
(PassMark 236 vs 1050) `[M]`. Extrapolating the Pi 5 numbers `[D]` puts decode at
~5.7 tok/s and prefill at ~16.5 tok/s in that envelope — a 140-token prompt alone
would take 8 seconds. The case study's demonstration targets (Pi 4/5, Graviton,
Cuttlefish) are all substantially faster than its *design* envelope. We therefore
design **three tiers with graceful degradation**, not one.

The sharper version: on the design envelope, **only a sub-200M model has any
chance of fitting an LLM step inside 2 s at all.** On the canonical P1 path
(140 in / 28 out — see below):

| Model | Prefill `[D]` | Decode `[D]` | P1 total `[D]` | vs. 2 000 ms |
|---|---|---|---|---|
| SmolLM2-135M | 76.0 tok/s | 26.2 tok/s | 1 857 ms | under target |
| Qwen3-0.6B | 16.5 tok/s | 5.7 tok/s | 8 073 ms | over |
| Qwen3-1.7B | 6.1 tok/s | 2.1 tok/s | 21 692 ms | over |

**That table is an extrapolation on an extrapolation** — parameters→throughput,
then Pi 5→Cortex-A53. It is `[D]`, it is not a benchmark, and it must be
re-measured in Phase 0 before anyone quotes it externally.

**3. The 2-second target is met per *path*, not per system.**

The brief says "under 2 seconds for common queries". A single yes/no answer is
not honest, because it depends on which path answers the query. Canonical paths
are defined once in [`tools/paths.py`](tools/paths.py):

| Path | What it does | Spec | Tier A | Tier B (Pi 5) | Tier C (2 GB / 2× A5x) |
|---|---|---|---|---|---|
| **P0** | classifier → DSL → SQL → template, no LLM | 0 in / 0 out | 110 ms `[D]` | 110 ms `[D]` | 110 ms `[D]` |
| **P1** | SLM intent-only (GBNF JSON) + template | 140 in / 28 out | `[T]` | 1 579 ms `[D]` | 8 073 ms `[D]` |
| **P2** | SLM short narration, streamed | 260 in / 70 out | `[T]` | 4 058 ms `[D]` | 22 714 ms `[D]` |
| **P3** | deep explain, multi-hop | 420 in / 160 out | `[T]` | 8 731 ms `[D]` | 48 220 ms `[D]` |

What we are entitled to claim:

* **P0 meets the target on every tier `[D]`.** It is 110 ms of pure software —
  query parse, SQL, template — so the figure is tier-independent. It answers all
  five of the brief's example queries.
* **On Tier B, P1 is *projected* to meet the target at 1 579 ms `[D]`.** The
  throughput underneath is measured on real ARM; our end-to-end path is not.
  **This remains a target, not a demonstration, until Phase 4 gate G4 measures
  it.** What Arm measured is 25.26 tok/s and a 3 636 ms TTFT — not our pipeline.
* **On Tier C, no current Qwen3 model can be in the per-query path `[D]`.**

**4. Storage is a solved problem; the interesting problem is semantic
compression.**

28 days of tiered telemetry (1 Hz hot signals, on-change slow signals,
event-window bursts, 1-minute rollups, embedded episode cards) models out at
**~103 MB on disk and ~64 MB steady-state RAM** `[D]`. Weeks of history is cheap.
Turning it into ~5,000 retrievable, human-meaningful *episodes* is the real
engineering.

---

## The data plane, working today (Phase 0)

The documents above are the plan. This part is running code, and it is the only
part of the repository whose claims are backed by execution rather than
arithmetic.

```bash
make demo          # simulated drive -> real CAN frames -> decoded signals
make check-data    # the data-plane test suite (60 tests)
```

What `make demo` actually does, with nothing stubbed out between the steps:

1. **Generates a drive** from a physical model — road load
   (`F = ½ρC_dAv² + C_rr·mg`), a lumped-parameter pack thermal model, and
   coulomb counting — deterministically from a seed.
2. **Packs it into real CAN frames.** [`tesla_can.dbc`](vehiclegpt/dbc/tesla_can.dbc)
   is the genuine production DBC from [commaai/opendbc](https://github.com/commaai/opendbc)
   (MIT, 44 messages). Frames are encoded at their real bit offsets.
3. **Sends them over a real `can.Bus`**.
4. **Decodes and normalises** them back onto canonical signal paths.
5. **Compares against ground truth** and re-detects the injected faults *from
   the signal stream alone*.

```
frames on bus            13000
frames decoded           13000
undecodable frames       0
canonical samples        32500  (25 distinct signals)
out-of-bounds values     0
wall time                4.2s (3,100 frames/s)

round-trip fidelity: worst error 0.50 LSB across all 25 signals

fault signatures recovered from the bus (not from the scenario file)
  coolant pump ran at 18% then stalled at t=300s
  pack crossed 60 C at t=1131s, 831s later
  PASS the precursor precedes the warning
```

**What is real and what is simulated.** The bus, the DBCs and the frame format
are real. The driving is modelled, not recorded. The vendor extension
([`vehiclegpt_vendor_ext.dbc`](vehiclegpt/dbc/vehiclegpt_vendor_ext.dbc)) is
synthetic because Tesla's pack CAN is proprietary and carries no HV battery
signals — that gap is stated rather than papered over. See
[`vehiclegpt/dbc/README.md`](vehiclegpt/dbc/README.md) for the per-file table.

**Why this matters for the case study.** "What happened before the battery
overheating warning?" is only answerable if the answer is known. Scripted
scenarios with injected faults give ground truth; a public dataset or a CARLA
run does not, because you would still be scripting the fault. That ground truth
is what the retrieval and analytics phases get scored against.

| Component | File | Purpose |
|---|---|---|
| Canonical signal registry | [`vehiclegpt/schema/signals.yaml`](vehiclegpt/schema/signals.yaml) | **Single source of truth**: 26 signals, units, physical bounds, rate, retention tier, deadband, DBC mapping |
| Registry loader | [`vehiclegpt/schema/registry.py`](vehiclegpt/schema/registry.py) | Strict validation; cross-checks every source against the DBCs |
| Adapter contract | [`vehiclegpt/ingest/base.py`](vehiclegpt/ingest/base.py) | `RawRecord` → `Sample`; **read-only by construction** |
| CAN adapter | [`vehiclegpt/ingest/can_source.py`](vehiclegpt/ingest/can_source.py) | SocketCAN and in-process bus; identical decode path either way |
| Scenario DSL | [`sim/scenario.py`](sim/scenario.py), [`sim/scenarios/`](sim/scenarios) | Typed, seeded, version-controlled drives with fault injection |
| Generator | [`sim/generator.py`](sim/generator.py) | Physics-lite vehicle model producing ground truth |
| Emitter | [`sim/emitter.py`](sim/emitter.py) | DBC-encodes truth onto a bus |

Two design decisions worth flagging, both forced by contact with the real DBC:

- **The canonical schema must describe the bus, not an ideal model.** The registry
  originally declared `chassis.brake_pedal` as a 0–100 % value; the real DBC
  carries `DI_brakePedal` as a **single bit** with the value table
  `Applied`/`Not_applied`. A continuous pedal percentage is not available there,
  so the signal is `chassis.brake_active` and the gap is documented. A test now
  asserts every declared range is representable in its DBC field.
- **Read-only is enforced by the type system, not by discipline.** The
  `VehicleDataSource` protocol exposes only `open`, `stream` and `close`. There
  is no method through which a transmission could be expressed, so the
  freedom-from-interference argument in [doc 04 §1](docs/04-architecture.md) is
  structural. [`tests/test_readonly.py`](tests/test_readonly.py) fails if anyone
  adds one.

---

## The deterministic layer (Phase 2)

Four of the five questions in the brief are filter, aggregate or sequence
questions. They are answered by SQL, not by a language model — which is both
faster and exactly right: a model adds latency and a chance of printing the
wrong count, and nothing else.

```bash
make ask      # build 14 days of data, then answer all five questions
make emulate  # why a laptop pinned to 2 cores is still not a Tier C device
```

```
Q: What happened before the battery overheating warning?
   3 questions answered deterministically; slowest 2.3 ms (target 2000 ms)

   Before the battery overheating warning: the coolant pump stopped for good
   at 17:35 UTC, 15 minutes earlier; pack temperature passed 55 C at 17:48;
   the overheating warning was raised at 17:50 UTC.
```

| Layer | File | What it does |
|---|---|---|
| 14-day program | [`sim/program.py`](sim/program.py) | Seeded calendar of drives containing a cold snap, a pump failure and ADAS disengagements |
| Store | [`vehiclegpt/store/db.py`](vehiclegpt/store/db.py) | SQLite, 1 Hz, WAL; `segment` and `daily` derived and rebuilt, never patched |
| Views | [`vehiclegpt/analytics/views.sql`](vehiclegpt/analytics/views.sql) | Distance-weighted rolling windows, regime aggregates, event sequences |
| Handlers | [`vehiclegpt/analytics/queries.py`](vehiclegpt/analytics/queries.py) | Five answerers plus rule-based intent routing |

Two rules are enforced by tests rather than trusted to discipline. `fault.*`
events are simulator metadata and are **never** used as evidence — the pump
stall in Q4 is found by noticing the signal stopped, which is what a real
vehicle would give you. And consumption is always `SUM(Wh)/SUM(km)`, never the
average of per-sample Wh/km, which would weight a stationary second the same as
a motorway one.

---

## Emulating the target hardware honestly

`make emulate` measures the host and reports the correction to Tier C. On the
development machine it reports **3.5× the bandwidth** of a Raspberry Pi 5:

| | Host (2 cores pinned) | Tier C anchor |
|---|---|---|
| Single-stream bandwidth | 36.7 GB/s | 10.47 GB/s `[D]` |
| Implied decode, 0.6B Q4 | 88.5 tok/s `[D]` | 25.26 tok/s `[M]` |

This is the trap `taskset` and `cpulimit` set. They reproduce **core count** and
**CPU time**, but LLM decode is bandwidth-bound and they do not touch memory
bandwidth. A two-core laptop still has the laptop's memory controller, so it
will decode several times faster than a two-core Cortex-A53. Reporting that as
"Tier C" would repeat the extrapolation error this project exists to fix, in a
more convincing costume.

So the tool never claims emulation. `--calibrate` gives the ratio; the harness
stamps every result with the machine it ran on; and the correction to Tier C
stays tagged `[D]`.

---

## Documents

| # | Document | What it answers |
|---|---|---|
| 01 | [Requirements analysis](docs/01-requirements-analysis.md) | What is actually being asked for; functional/non-functional decomposition; what "done" means; evaluation mapping |
| 02 | [Research findings](docs/02-research-findings.md) | Models, runtimes, simulators, datasets, protocols, tooling — with sources and evidence tiers |
| 03 | [Technology comparison](docs/03-technology-comparison.md) | Head-to-head tables and the reasoning behind each pick |
| 04 | [Architecture](docs/04-architecture.md) | The proposed system, data flow, safety isolation, and four rejected alternatives |
| 05 | [Roadmap](docs/05-roadmap.md) | Ten phases, deliverables, gates, what is automated vs. human |
| 06 | [Hardware feasibility](docs/06-hardware-feasibility.md) | Latency budgets, memory envelopes, storage math, risk of extrapolation |
| 07 | [Risk register](docs/07-risk-register.md) | 16 risks with likelihood, impact, mitigation, owner |
| 08 | [Decisions required](docs/08-decisions-required.md) | Ten decisions with a recommendation and a default for each |
| 10 | [Tier C latency strategy](docs/10-tierc-latency-strategy.md) | Where the 8 s actually goes (61 % is decode, not prefill), the two thresholds a model must clear, and the routing change that answers the Tier C objection |
| 11 | [Deterministic analytics](docs/11-deterministic-analytics.md) | The layer that answers four of the five brief questions with SQL in under 3 ms, and the three defects its tests caught |

### Verifying the numbers

Every latency and storage figure in this repository is generated, not typed.
Three commands, no dependencies, runs on a Pi or a 2 GB container:

```bash
make check       # doc linter + feasibility model + 15 stdlib tests (no deps)
make check-data  # everything above, plus the 60-test data-plane suite
make model       # the full feasibility report, with per-figure status tags
make specs       # the canonical P0-P3 spec table
```

`make check` is deliberately dependency-free so CI can verify the documented
numbers on a bare Python install. The data plane needs `cantools`,
`python-can`, `numpy` and `PyYAML`, and runs in a separate CI job.

To replace the Tier C projection with a measurement, run the bake-off on
hardware that can hold the weights:

```bash
python3 tools/phase0/bakeoff.py --required-only          # what a model must do
python3 tools/phase0/bakeoff.py --models ~/models/*.gguf --eval
```

It has deliberately **not** been run: the development sandbox cannot reach
Hugging Face. See [doc 10](docs/10-tierc-latency-strategy.md) for what the
result would and would not settle.

| File | Role |
|---|---|
| [`tools/paths.py`](tools/paths.py) | **Single source of truth** for P0/P1/P2/P3 token budgets. Nothing else may define them |
| [`tools/feasibility_model.py`](tools/feasibility_model.py) | Turns cited `[M]` anchors into `[D]` latency and storage figures |
| [`tools/check_docs.py`](tools/check_docs.py) | Fails CI if a document drifts from `paths.py`, quotes a Tier C latency without a projection marker, or claims a measurement the project has not made |
| [`tests/test_consistency.py`](tests/test_consistency.py) | Asserts no path latency is ever tagged `[M]`, P0 is tier-independent, and Tier C meets the target only on P0 |

The linter exists because an earlier draft of these documents defined P1 as
`140 in / 28 out` in one file and `100 in / 10 out` in another — a 15 % swing in a
headline number — and printed a PassMark-derived extrapolation in a table that
read like a benchmark. Prose drifts; the checks make it fail instead.

---

## Proposed stack at a glance

*Everything below is free, open-source and self-hosted. Nothing calls a cloud
API. Nothing requires a licence purchase.*

| Layer | Choice | Why |
|---|---|---|
| Model | **Qwen3-0.6B** (Q4_K_M, Apache-2.0), **Qwen3-1.7B** on larger tiers | Best instruction-following per MB at this size; Apache-2.0; the Pi 5 numbers above are for exactly this model |
| Intent router | Fine-tuned **MiniLM-class encoder** (~30 M, ONNX int8) | 5–15 ms instead of 1.5 s. The router is a classification problem, not a generation problem |
| Runtime | **llama.cpp** (GGUF) | Only runtime with credible ARM-CPU story, grammar-constrained decoding, prefix caching, and a measured Pi 5 baseline |
| Hot store | **SQLite** (WAL) + zstd-packed columnar blocks | Embedded, ACID, single file, nothing to administer on a vehicle |
| Analytics store | **DuckDB** over Parquet *(Phase 3, if benchmarks justify it)* | Kept optional; SQLite must be proven insufficient first |
| Retrieval | **FTS5 (BM25) + sqlite-vec, fused with RRF** | Dense-only retrieval fails on DTC codes and numeric queries, which is most of this domain |
| Query language | Custom **typed DSL → parameterised SQL** | The model never emits SQL. This is a security property, not a preference |
| Data sim | **vcan / SocketCAN + real DBCs** (opendbc), **nuScenes CAN expansion**, synthetic fault injection | Real frames, real DBCs, real dynamics — plus deterministic fault scenarios nobody publishes |
| Serving | **FastAPI** on loopback + local web UI | Previewable in this environment; no cloud dependency |
| Red teaming | **garak** + **PyRIT** + **promptfoo** + a custom vehicle-domain probe set | The custom probe set is the novel contribution |
| Interpretability | **TransformerLens** on Qwen small models | Feasible on 6 GB VRAM; a real research question is available (see doc 05) |

---

## Objectives mapping

| Your objective | Where it is served |
|---|---|
| 1. A genuinely working VehicleGPT | Phases 0–4; the deterministic path (P0) alone already answers every example query in the brief |
| 2. AI/ML engineering experience | Quantisation, LoRA fine-tuning, grammar-constrained decoding, hybrid retrieval, eval harness design |
| 3. AI red teaming | Phase 5. **Novel angle: indirect prompt injection through telemetry and log content** — a threat model that only exists for this class of system |
| 4. Mechanistic interpretability | Phase 5. **Novel angle: activation patching on the evidence span to measure whether a 0.6B model grounds answers in retrieved data or in parametric priors.** Directly measures hallucination |
| 5. Reproducible, presentable | Phase 0 gates, seeded synthetic scenarios, CI on CPU, one-command demo, published eval numbers |

---

## Constraints check

| Constraint | Status |
|---|---|
| ₹0 budget, no paid APIs | ✅ Every component is free and open-source; no API keys anywhere in the design |
| RTX 3050 6 GB laptop | ✅ Sufficient for QLoRA on ≤1.7B, TransformerLens on ≤0.6B, and all evaluation. Analysis in `docs/06` |
| CPU-only support | ✅ Tier B/C paths are CPU-only by construction; `llama.cpp` is the only inference dependency |
| Offline, no cloud | ✅ Enforced architecturally (no outbound network client in the runtime image) |
| <2 s common queries | ⚠️ **Path-specific, and not yet measured.** P0 meets it on every tier `[D]` (110 ms, pure software). P1 is *projected* to meet it on Tier B only (1 579 ms `[D]`, target until gate G4). On Tier C no Qwen3 model fits. See [doc 06 §5.3](docs/06-hardware-feasibility.md) |
| ~2 GB RAM / 2× A5x | ⚠️ Model peak alone is 946 MB `[M]`; modelled system headroom is ~954 MB `[D]`. Only the Tier C profile fits, and only with a hard context ceiling |
| Weeks of searchable history | ✅ ~103 MB modelled `[D]` for 28 days; gate G1 requires measured within 25 % |
| Safety-critical isolation | ✅ Read-only by construction; no actuation interface exists in the codebase |

---

## Licence

MIT (see [LICENSE](LICENSE)). Third-party datasets retain their own licences —
notably nuScenes is **CC BY-NC-SA 4.0** (non-commercial), which is fine for a
competition but must not ship in a commercial product. See doc 02 §6.
