# 01 — Requirements Analysis

> Source: *Case Study 5 — TBU Vehicle Information using On-Device SLM or LLM
> (VehicleGPT)*, Tata Elxsi TelePort Season 4. Three pages; no annexures,
> appendices or hidden requirements. Full text was extracted and is quoted
> verbatim where it matters.

---

## 1. What the case study actually asks for

Stripped of formatting, the brief contains **one scenario, one challenge
statement, five key points, five constraints, eight expected outputs and six
evaluation criteria.** It is a systems-architecture brief that happens to
contain an LLM. That framing matters: the LLM is roughly 15% of the work, and
teams that treat it as 80% will fail on the data plane.

### 1.1 The scenario (verbatim)

> One scenario is to have a local SLM/LLM inside the vehicle which has access to
> various data and based on user or service engineer queries can provide the
> relevant data.

Two distinct user roles are named — **driver** and **service engineer** — and
they need different answers to the same question. A driver asking "why has my
range dropped?" needs a plain-language cause; an engineer needs the DTC, the
freeze-frame, and the correlated signals. This is a *first-class requirement*,
not a UI detail, because it changes the answer length, the vocabulary, and the
safety policy.

### 1.2 The five example queries (verbatim)

| # | Query | What it actually requires | Naïve-RAG verdict |
|---|---|---|---|
| Q1 | "Why has my driving range reduced over the last two weeks?" | 14-day trend on range/energy + change-point detection + attribution across candidate causes (temperature, speed profile, HVAC, tyre pressure, battery health) | ❌ Cannot be answered by retrieving text chunks. Needs arithmetic over a time series and a causal *ranking* |
| Q2 | "When did fuel efficiency show the largest drop?" | Windowed efficiency series + argmax of negative delta + surrounding context | ❌ Same. This is `argmax` over a computed series |
| Q3 | "Show all ADAS disengagement events from yesterday." | Event retrieval with a time filter and a category filter | ✅ The only "pure RAG" query of the five |
| Q4 | "What happened before the battery overheating warning?" | Causal-precedence search: find warning, look back in a window, surface correlated precursor signals | ⚠️ Hybrid — needs event index + windowed signal join |
| Q5 | "Compare energy consumption between city and highway driving." | Segmented aggregation over an inferred driving-mode label | ❌ Needs a derived feature (driving-mode classifier) and a grouped aggregate |

**Four of five example queries are analytical, not retrieval.** This is the
single most important observation in this document. A vector-database-over-log-
chunks architecture — which is what most teams will build — fails 80% of the
brief's own examples. Doc 04 proposes a hybrid that treats retrieval as the
*fallback*, not the default.

### 1.3 The challenge statement (verbatim)

> Design a practical AI-on-Edge solution that runs a Small Language Model (SLM)
> or Large Language Model (LLM) directly inside the vehicle and continuously
> consumes vehicle data streams, events, and logs, making them searchable
> through natural language queries.
>
> The system should act as an intelligent vehicle assistant capable of
> understanding user questions, retrieving relevant vehicle information,
> identifying trends and anomalies, and presenting understandable explanations
> without requiring cloud connectivity.

Five verbs, five subsystems: **understand**, **retrieve**, **identify trends**,
**identify anomalies**, **explain**. Only the first and last belong to the LLM.

### 1.4 The five key points (verbatim, condensed)

1. How driving scenarios and error-condition sequences will be **recreated**
   (CARLA or other simulator, or simulated/recorded data).
2. How vehicle information will be **collected and stored** from: CAN/CAN-FD,
   SOME/IP services, Android Vehicle HAL, QNX services/logs, ADAS, camera
   feeds, diagnostic logs.
3. How real-time and historical data will be **retained** for analysis.

Note the second bullet is a *breadth* test. No individual can plausibly build
production-grade adapters for seven automotive interfaces in a competition
timeframe. The correct response is an **adapter abstraction with one or two
genuinely deep implementations and the rest honestly stubbed** — and to say so
plainly rather than claiming seven working integrations. Doc 04 §5 makes this
explicit.

### 1.5 The five constraints (verbatim, condensed)

| # | Constraint | How it is treated |
|---|---|---|
| C1 | Must operate without cloud dependency | Architectural, not a setting. No outbound HTTP client ships in the runtime image. Verified in CI. |
| C2 | Model must run on ~2 GB RAM, ~2 ARM A5x cores | **This is the hard one.** See §4 below and doc 06. Addressed by tiering, not by wishful thinking. |
| C3 | Response time under 2 s for common queries | Achievable; "common" must be *defined*, and it is defined in doc 04 §7 as the P0/P1 paths |
| C4 | Historical data searchable for at least several weeks | Modelled at 28 days / ~103 MB. Comfortable. |
| C5 | Safety-critical functions isolated from the AI system | Read-only by construction; no actuation code path exists. Doc 04 §6 |

### 1.6 The eight expected outputs (verbatim)

High-level system architecture · vehicle data collection strategy · SLM/LLM
selection rationale · search and retrieval workflow · historical data storage
approach · edge deployment architecture · user interaction design · engineering
justification for model, OS, architecture and protocol choices.

**This is a documentation-heavy brief.** Roughly half the score is the
*reasoning*, not the artefact. The repo is therefore built so that the design
documents are generated alongside the code and stay in sync, not written the
night before submission.

### 1.7 The six evaluation criteria

| Criterion | Sub-points | Our answer lives in |
|---|---|---|
| System Architecture | Hardware + software completeness, logical integration of simulation and user interface | doc 04 |
| AI Capability | NL understanding; **accuracy of responses and summaries** | doc 04 §3, eval harness in Phase 3 |
| Data Integration | Support for multiple automotive protocols and sources | doc 04 §5 (adapter matrix) |
| Search & Analytics | Find historical events, trends, anomalies; **explain causes** | doc 04 §4 |
| Edge Performance | Memory footprint, query latency, **scalability from reference board → vehicle HPC** | doc 06 |
| Privacy & Security | Local-only processing; protection of vehicle and driver data | doc 04 §6 |
| Innovation & Practicality | Ease of production deployment in SDVs; value for drivers, engineers, fleets, validation teams | doc 04 §9 |

Two sub-points deserve emphasis because they are easy to under-serve:

* **"Accuracy of responses and summaries"** — we need a *measured* number, not a
  demo. The eval harness (§3.3) produces a numeric grounding rate.
* **"Scalability from reference board / Pi / VM to vehicle HPC"** — this is
  asking for a *tier ladder*, which is exactly what doc 06 provides.

---

## 2. Functional decomposition

The system decomposes into nine capabilities. Each is independently testable.

| ID | Capability | Input | Output | Owner |
|---|---|---|---|---|
| F1 | **Ingest** | CAN frames, SOME/IP events, VHAL property changes, log lines, replay files | Raw records | Data plane |
| F2 | **Normalise** | Raw records + signal registry | Canonical `(ts, vehicle, signal_path, value, quality)` | Data plane |
| F3 | **Persist** | Canonical records | Tiered store (ring buffer, hot series, events, rollups) | Data plane |
| F4 | **Distil** | Persisted series + events | Episode cards (text summaries of interesting moments) | Analytics |
| F5 | **Analyse** | Series | Trends, change points, anomalies, driving-mode segments | Analytics |
| F6 | **Understand** | NL query | Intent + slots (+ resolved time range, entity, role) | AI plane (router) |
| F7 | **Retrieve** | Intent + slots | Evidence pack: rows, events, cards, provenance | AI plane (retrieval) |
| F8 | **Explain** | Evidence pack + role | Natural-language answer with inline provenance | AI plane (render) |
| F9 | **Guard** | Query + answer | Scope/refusal decision, injection sanitisation, numeric validation | AI plane (guardrails) |

F1–F5 are classical engineering. F6 and F8 are ML. F7 is the hard hybrid. F9 is
where the red-teaming objective lives.

---

## 3. Non-functional requirements

### 3.1 Latency

The brief says "under 2 seconds for common queries." We decompose this, because
a single number is unactionable:

These are **design targets**, not measurements. Canonical path definitions live
in [`tools/paths.py`](../tools/paths.py).

| Class | Budget (target) | Path | Rationale | Status of current evidence |
|---|---|---|---|---|
| Instant | ≤ 250 ms | P0: classifier → DSL → SQL → template | Feels instantaneous; no LLM in the loop | 110 ms `[D]` — pure software, tier-independent |
| Common | ≤ 2 000 ms | P1: SLM intent (GBNF JSON) + templated answer | Meets the brief literally | projected 1 579 ms on Tier B `[D]`; **not demonstrated** |
| Explanatory | ≤ 6 000 ms, **streamed** | P2/P3: SLM narration over evidence pack | First token < 2 s; the user is reading while it writes | projected 4 058 ms on Tier B `[D]` |
| Deep explain | unbounded, with progress | P3 + multi-hop | Explicitly invoked by the user, so latency is expected | unmeasured `[T]` |

**The "Common ≤ 2 000 ms" row is the one that matters for scoring, and it is
currently a projection.** It becomes a measured claim only at Phase 4 gate G4,
on real ARM hardware, for our actual pipeline.

Streaming is not a workaround — it is the correct UX. A driver who has the
number in ~110 ms and prose in 3 s has a *better* experience than one who waits
2 s for both.

### 3.2 Memory

| Component | Model target | Notes |
|---|---|---|
| SQLite page cache + hot ring buffer | ≤ 80 MB | Modelled at 64 MB |
| Embedding model (MiniLM int8, ONNX) | ≤ 30 MB | Loaded lazily, evictable |
| SLM (Qwen3-0.6B Q4_K_M) | 414 MB weights, **946 MB measured peak** | The measured peak includes KV cache at 512/128 |
| Python interpreter + FastAPI | ≤ 90 MB | The real reason to prefer a slim base image |
| **Total, Tier C target** | **≤ 2 000 MB** | Tight but reachable — see doc 06 §5 |

### 3.3 Accuracy

Because "accuracy" is scored, it must be *measured*. Three programmatic
metrics, no LLM-as-judge in the pass/fail path:

| Metric | Definition | Target |
|---|---|---|
| **Intent accuracy** | Correct intent+slot JSON vs. human-labelled set (≥ 300 queries, covering all 5 example queries × role variants) | ≥ 92 % |
| **Numeric grounding rate** | Fraction of numeric tokens in the answer that exactly match an evidence-pack value | **100 %** (hard requirement — see doc 04 §3.3) |
| **Retrieval recall@8** | Relevant episode/event appears in the top 8 for event-search queries | ≥ 85 % |
| Abstention rate on out-of-scope | Must say "I don't have that data" rather than invent | ≥ 95 % |

The numeric grounding rate is enforced by construction (numbers are substituted,
never generated) so it is really a *regression test*, not a KPI. That is the
point.

### 3.4 Privacy & security

| Requirement | Mechanism |
|---|---|
| Local-only processing | No network client in the runtime; egress blocked in CI; verified by `pytest` that greps the dependency tree for HTTP clients |
| Driver data protection | PII-by-default: location traces are coarsened at ingest; VIN and driver IDs are replaced with salted hashes before persistence |
| Untrusted input | Telemetry text (DTC descriptions, log lines, OTA payloads) is treated as **untrusted user input**, not trusted context — see doc 07 R5 |
| Safety isolation | No actuation API. The word "set" does not appear in any vehicle interface |

---

## 4. Requirement conflicts and how they resolve

Three genuine conflicts exist in the brief. Naming them is worth marks;
papering over them is not.

### Conflict 1 — "~2 GB RAM, ~2 A5x cores" vs. "the model must run on it"

A 0.6B model at Q4 needs ~946 MB peak in practice. That fits in 2 GB *alone*,
but the brief also requires storage, retrieval, a user interface and weeks of
history. **Resolution:** the 2 GB envelope is treated as a *total system*
budget, and the deployment is tiered. Tier C runs a 0.6B model but restricts
generation to the P0/P1 paths. This is documented, modelled and measured —
not assumed.

### Conflict 2 — "embedded automotive hardware" vs. the permitted demo targets

The brief permits demonstration on "raspberry pi 4/5 with AI HAT, ARM
virtualized environment like cuttlefish, qemu, Amazon AWS Graviton or
equivalent." Every one of those is **substantially faster than 2× Cortex-A53**.
The design envelope and the demo platform are therefore mismatched, and any
latency figure measured on a Pi 5 will look better than the design target
delivers. **Resolution:** demo on Tier B, *measure* and report the A53/A55
extrapolation explicitly, and never quote a Pi 5 number as a Tier C number.

### Conflict 3 — "camera feeds" vs. "2 GB RAM"

Running any vision model alongside a 0.6B LLM in 2 GB is not credible.
**Resolution:** camera feeds enter the system as **ADAS *events* and metadata**
(perception results), not as pixels. The LLM reasons about "pedestrian detected,
AEB engaged" — never about an image. This is also the right production design:
an SDV already runs perception in a dedicated, safety-rated domain.

---

## 5. Scope boundary — what this project is and is not

**In scope (v1):**

* Single vehicle, single ego vehicle.
* Text in, text out. Voice is a Phase 6 stretch (see doc 05).
* CAN/CAN-FD (real or `vcan`), replay sources, diagnostic logs, ADAS event
  feed. SOME/IP and VHAL as *adapters with a documented integration point*.
* Driver and service-engineer personas.
* English.

**Out of scope (explicitly, and stated in the submission):**

* Vehicle control of any kind. The system is read-only.
* Real-vehicle deployment. Demo is simulator/recorded-data based.
* Multi-vehicle fleet analytics (architecturally anticipated, not built).
* Functional-safety certification. We *design for* isolation and demonstrate it;
  we do not claim an ASIL rating.
* ASR/vision on-device.

---

## 6. Definition of done (for the competition submission)

A runnable repository plus a demo that satisfies all of the following:

1. `make demo` starts the stack offline on a clean machine and opens a local UI.
2. All **five example queries from the brief** return correct, grounded answers,
   with the numbers traceable to source rows in the UI.
3. A replay of **four weeks** of telemetry answers "why did range drop over the
   last two weeks?" correctly against a known injected ground truth.
4. **At least one simulated fault sequence** (battery overheating) is
   reproducible from a fixed seed and answerable via Q4.
5. Measured latency p50/p95 **per query class per hardware tier**, with the
   measurement script in the repo.
6. Measured peak RSS per tier.
7. An eval report with intent accuracy, numeric grounding rate, recall@8, and
   abstention rate — generated by CI, not hand-written.
8. A red-team report with ≥ 100 vehicle-domain probes and a pass rate.
9. One interpretability experiment with a figure and a written finding.
10. The eight expected outputs from §1.6, each as a section in the design docs.

Items 1–7 are the product. Items 8–9 are your differentiators. Item 10 is half
the score.
