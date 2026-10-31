# 07 — Risk Register

> Scored L (likelihood) × I (impact), each 1–5. **Score ≥ 15 needs a mitigation
> agreed before Phase 1 starts.** Every risk has an *early-warning signal* — the
> thing you will actually notice — because a risk you only detect at the end is a
> post-mortem, not a risk.

---

## Summary

| ID | Risk | L | I | Score | Status |
|---|---|---|---|---|---|
| R1 | 2-second latency unachievable on the design envelope | 5 | 5 | **25** | 🔴 Mitigated by tiering (doc 06) |
| R2 | Extrapolated Tier C numbers are wrong | 4 | 4 | **16** | 🔴 Open — needs Phase 0 measurement |
| R3 | 2 GB memory envelope exceeded | 3 | 5 | **15** | 🟠 Partially mitigated |
| R4 | Scope creep across 4 research tracks | 5 | 4 | **20** | 🔴 Mitigated by phase gates |
| R5 | Prompt injection via telemetry content | 4 | 4 | **16** | 🟠 Designed for, unproven |
| R6 | Answer grounds in parametric priors, not evidence | 4 | 5 | **20** | 🟠 Mitigated by number substitution |
| R7 | No real fault-sequence data exists | 5 | 3 | **15** | 🟢 Accepted (synthetic + ground truth) |
| R8 | Hallucinated safety advice reaches a user | 2 | 5 | **10** | 🟢 Hard policy refusal |
| R9 | TransformerLens doesn't support Qwen3 | 3 | 2 | 6 | 🟢 Fallback model identified |
| R10 | Development hardware hides edge problems | 4 | 4 | **16** | 🟠 Mitigated by tier configs |
| R11 | sqlite-vec is pre-1.0 | 2 | 3 | 6 | 🟢 Wrapper interface |
| R12 | ARM container build friction | 3 | 3 | 9 | 🟢 Emulation fallback |
| R13 | Demo fails live in front of judges | 2 | 5 | **10** | 🟠 Pre-recorded fallback |
| R14 | Dataset licence problem | 2 | 4 | 8 | 🟢 Disclosed; links and the 0.6B model are Apache-2.0 |
| R15 | Evaluation is self-serving / not credible | 3 | 4 | **12** | 🟠 Independent ground truth from injection |
| R16 | Solo-developer time collapse | 4 | 5 | **20** | 🔴 Mitigated by the two-phase fallback plan |

---

## R1 — The 2-second target is unachievable on the design envelope

**Score 25 — the defining risk of the project.**

| | |
|---|---|
| **Likelihood** | 5 — arithmetic, not opinion. Tier C prefill extrapolates `[D]` to ~16.5 tok/s; the canonical 140-token P1 prompt alone is 8.5 s |
| **Impact** | 5 — it is an explicit constraint in the brief |
| **Early warning** | Phase 0 shows < 20 tok/s decode on the constrained profile (today's projected figure is 5.7 tok/s `[D]`) |
| **Mitigation** | Three-tier design. P0 (deterministic) projects to ~110 ms `[D]` on every tier — it is pure software, so the figure is tier-independent — and covers all five example queries. P1 is **projected** `[D]` to fit on Tier B — a target until gate G4, not a demonstration. Tier C routes the LLM behind an explicit "explain" action with a progress indicator |
| **Residual** | **LLM-generated answers on Tier C are projected `[D]` to miss "under 2 s" by ~4×.** This must be stated, not hidden |
| **Why this is still the right answer** | The alternative is to demo on a Pi 5, quietly omit the design envelope, and hope nobody does the arithmetic. A reviewer who checks will find it. Owning the limitation and showing the tier table is strictly more credible |
| **Escape hatch** | SmolLM2-135M projects to 1 857 ms `[D]` for P1 on Tier C (doc 06 §6.3) — inside the budget by ~143 ms, on a three-deep extrapolation. Phase 0 task 0.5 decides |

---

## R2 — The Tier C extrapolation is wrong

**Score 16.**

| | |
|---|---|
| **Likelihood** | 4 — the PassMark-ratio method is defensible but unvalidated for quantised GEMM on NEON |
| **Impact** | 4 — every Tier C claim in the submission rests on it |
| **Early warning** | Measured Tier B under `taskset -c 0,1` + `cpulimit` differs from prediction by > 2× |
| **Mitigation** | Phase 0 task 0.7 (core-pinned, duty-capped measurement). Label every Tier C figure `[D]` and publish the derivation. Never present a Pi 5 number as a Tier C number |
| **Residual** | If no ARM hardware is available at all, Tier C is *modelled only*. Say so in the submission in one sentence |

---

## R3 — The 2 GB memory envelope is exceeded

**Score 15.** Detail in doc 06 §6.

| | |
|---|---|
| **Likelihood** | 3 — measured model peak is 946 MB of 2 048 MB |
| **Impact** | 5 — a hard constraint |
| **Early warning** | Peak RSS > 1 400 MB in Phase 4 measurement |
| **Triggers that break it** | KV cache growth from retained chat history; loading 1.7B; DuckDB without `memory_limit`; Python allocator fragmentation over long uptime |
| **Mitigation** | Hard context ceiling (`--ctx-size 1024`); conversation history summarised, never retained verbatim; DuckDB behind an explicit memory cap or not at all; supervisor restart policy for long uptime; `systemd MemoryMax=` as a backstop that fails loudly instead of OOM-killing the vehicle |
| **Residual** | The 1 000 MB of headroom is comfortable but not lavish. It disappears the moment someone enables a 4 K context |

---

## R4 — Scope creep across four research tracks

**Score 20 — the most likely way this project fails.**

| | |
|---|---|
| **Likelihood** | 5 — the brief asks for a lot; you have four personal objectives on top |
| **Impact** | 4 — an unfinished product scores worse than a finished modest one |
| **Early warning** | Three consecutive weeks with no gate passed |
| **The actual trap** | Seven protocols × three simulators × red teaming × interpretability × a polished UI is roughly four projects. Attempting all four yields four half-demos |
| **Mitigation** | **Phase gates are hard stops.** G2 (analytics + retrieval working without an LLM) is the real product milestone. Everything after it is enhancement. Phase 6 items are explicitly ordered by value-per-hour and are designed to be cut from the bottom |
| **Fallback plan (decide now)** | If time runs short, do P0/P1/P2 and a trimmed P3. You still pass G1 and G2, demo all five queries, and have a complete honest submission — with the red-team and interp tracks reduced to a single experiment each |

---

## R5 — Indirect prompt injection via telemetry content

**Score 16.** *The novel threat, and one of your research objectives.*

| | |
|---|---|
| **Likelihood** | 4 — any attacker who can write to a log, an OTA manifest, a DTC description field, or a paired-device name controls text that reaches the model's context |
| **Impact** | 4 — the assistant is read-only, so the worst case is misleading answers and data disclosure, not actuation. Still serious in a vehicle |
| **Attack surface (six channels)** | DTC description strings · log line content · OTA/firmware metadata · Bluetooth device names · user query · retrieved episode cards (secondary injection) |
| **Early warning** | A Phase 5 probe family shows any non-zero bypass rate for the DTC or log channels |
| **Mitigation** | (1) Telemetry free-text never enters the prompt verbatim — cards are generated from *numeric* facts by our own templates. (2) An imperative-pattern sanitiser runs before assembly. (3) The scope layer refuses instructions-as-data. (4) Every detected attempt is logged and counted as a metric |
| **Residual** | Some bypasses will succeed. **That is the finding.** Publish the per-family bypass table |
| **Why it's valuable** | Generic scanners test the *user* channel. This system's dominant surface is the *data* channel, which garak and PyRIT do not model |

---

## R6 — The answer grounds in parametric priors, not in the evidence

**Score 20.**

| | |
|---|---|
| **Likelihood** | 4 — this is *the* characteristic failure of small models given retrieval context. Qwen3-0.6B scores 44.6 on MMLU-Redux; it has strong prior opinions and weak instruction adherence |
| **Impact** | 5 — an assistant that confidently states a false number is worse than no assistant |
| **Early warning** | Numeric grounding rate drifts below 1.0; or a manual audit finds a plausible-but-wrong figure |
| **Mitigation** | **Structural, not statistical:** numbers are *substituted* from the evidence pack, never generated (doc 04 §3.2). A model emitting a literal number is rejected and the template fallback used. `pytest` asserts grounding = 1.000, including against a deliberately misbehaving stub model |
| **Residual** | The model can still be wrong about *qualitative* claims ("this is normal") while being right about numbers. Mitigated by persona policy and by never asserting normality |
| **Research payoff** | This is exactly what Phase 5 interpretability measures: activation patching on the evidence span tells us *when* the model ignored the evidence. If the signal is reliable, it becomes a runtime hallucination detector |

---

## R7 — No real fault-sequence data exists

**Score 15.**

| | |
|---|---|
| **Likelihood** | 5 — certain. "Battery overheating warning" sequences are OEM-confidential |
| **Impact** | 3 — the brief explicitly permits "simulated/recorded data" |
| **Mitigation** | Synthetic injection onto *real* recorded dynamics (nuScenes). The scenario DSL is seeded and version-controlled, so scenarios are reproducible and testable. **Ground truth is known by construction**, which makes evaluation possible at all — a genuine advantage over real data |
| **Residual** | A reviewer may ask "is this realistic?" Answer: the *base dynamics* are real (Motional's Renault Zoe fleet); only the *fault overlay* is synthetic, and it is derived from physical models (thermal rise rates, BMS thresholds) rather than invented |
| **Do not** | Present synthetic data as recorded. That is the one thing that would be indefensible |

---

## R8 — Hallucinated safety advice

**Score 10, but impact is severe.**

| | |
|---|---|
| **Likelihood** | 2 — small models are eager to please, and "is it safe to drive?" invites an answer |
| **Impact** | 5 — a wrong "yes" has a physical consequence |
| **Mitigation** | Hard policy: this system never makes safety determinations. It reports the DTC, the severity class from the J2012 table, and the manufacturer-recommended action; then defers to a service centre. Enforced in the scope layer, not the prompt. A red-team family tests it explicitly |
| **Residual** | None acceptable; treated as a blocking defect if any probe passes |

---

## R9 — TransformerLens doesn't support Qwen3

**Score 6.**

| | |
|---|---|
| **Likelihood** | 3 — my GitHub issue search was inconclusive (doc 02 §11) |
| **Impact** | 2 — the research track has a clean fallback |
| **Mitigation** | Phase 0 task 0.5 tests both. Fallback: **Qwen2.5-0.5B-Instruct as an "interpretable twin"** — same family, GQA architecture, and a published precedent ([arXiv:2605.08853](https://arxiv.org/html/2605.08853) reports its layer-4 head-7 factual-recall bottleneck). We validate that behavioural findings transfer |
| **Secondary fallback** | `nnsight` if a specific architecture is missing from TransformerLens |

---

## R10 — Development hardware hides every edge problem

**Score 16.**

| | |
|---|---|
| **Likelihood** | 4 — the RTX 3050 is ~14× the Pi 5 and the Pi 5 is ~4.5× the design envelope. On Tier A, *everything* is fast |
| **Impact** | 4 — you will unknowingly build an architecture that cannot ship |
| **Early warning** | You have not run anything under `taskset`/`cpulimit` in two weeks |
| **Mitigation** | Tier configs from day one; the latency harness reports per tier, not just Tier A; the constrained profile is the one in the demo script; CI asserts a latency budget on the constrained profile |
| **Note** | This is the most common failure mode for on-device ML projects, and it is entirely avoidable with discipline |

---

## R11 — sqlite-vec is pre-1.0

**Score 6.**

| Likelihood 2 (API churn is real but the library is stable in practice) · Impact 3 (would require swapping a dependency mid-project).

**Mitigation:** wrap it behind a 3-method interface (`add`, `search`, `delete`) so LanceDB or FAISS drops in without touching call sites. Version-pin exactly.

---

## R12 — ARM container build friction

**Score 9.** Multi-arch builds, missing wheels for ARM64 (`onnxruntime` and `llama-cpp-python` both need source builds on some ARM configs), and slow compile times under emulation.

**Mitigation:** pin to versions with published `aarch64` wheels; build `llama.cpp` from source (it's clean on ARM); cross-compile or build natively on the target; keep the runtime image minimal. Budget a day.

---

## R13 — The demo fails live

**Score 10.**

| | |
|---|---|
| **Likelihood** | 2 — with a deterministic, seeded demo this is low |
| **Impact** | 5 — judges remember the crash, not the architecture |
| **Mitigation** | Everything runs from a fixed seed and pinned model SHA. No network at demo time. **A pre-recorded video of the full flow is mandatory**, presented as a backup, not a substitute. A `make demo` that has been run 20 times on a clean checkout |
| **Anti-pattern to avoid** | Live-typing novel queries into the model in front of judges. Demo the seeded scenarios; offer to try a novel query as a bonus |

---

## R14 — Dataset licence problem

**Score 8.**

nuScenes is **CC BY-NC-SA 4.0** — non-commercial. Fine for a competition; a
blocker for any commercial use. comma2k19 and opendbc are permissive (MIT).

**Mitigation:** disclose clearly in the submission and in `DATA.md`. Keep the
architecture dataset-agnostic so a commercial deployment can swap in proprietary
data. Note that the *model* (Qwen3) and every *tool* are Apache-2.0/MIT, so the
licence constraint is confined to the demo dataset.

---

## R15 — The evaluation is self-serving

**Score 12.**

| | |
|---|---|
| **Likelihood** | 3 — it is trivially easy to write eval queries your system happens to pass |
| **Impact** | 4 — a reviewer who spots this discounts everything else |
| **Mitigation** | (1) The 300-query intent set is generated from templates with **held-out phrasing patterns** — the model is fine-tuned on one phrasing family and evaluated on another. (2) Ground truth for the five brief queries comes from **injected faults**, not from our own outputs. (3) Publish the eval set in-repo so it can be attacked. (4) Include failure cases in the report |
| **Residual** | The query set is small. Say so: n=300 is a sanity check, not a benchmark |

---

## R16 — Solo-developer time collapse

**Score 20.**

| | |
|---|---|
| **Likelihood** | 4 |
| **Impact** | 5 |
| **Early warning** | Two consecutive weeks without a gate passed |
| **Mitigation** | The phase gates exist precisely for this. **The fallback is pre-agreed** (R4). Also: P0/P1/P2 deliver a complete, honest, demonstrable submission without any of the research tracks — so the floor is high even if the ceiling is not reached |
| **Structural advice** | Build the demo script *at G2*, not at the end. If the demo works at G2, every later phase is upside |

---

## The five risks I'd worry about, in order

1. **R4 / R16 — scope and time.** Everything else is an engineering problem; this is a project-management problem, and it is the one that actually kills projects.
2. **R1 — latency on Tier C.** Unavoidable, but survivable if owned openly.
3. **R6 — grounding.** Structurally solved, but only if the substitution mechanism is implemented rigorously and tested adversarially. One shortcut here and the whole thesis collapses.
4. **R2 — Tier C extrapolation.** Fixable with one day of measurement. Do it in week 1.
5. **R10 — building for the wrong machine.** Silent, systematic, and the reason most on-device projects ship something that doesn't fit.
