# 05 — Development Roadmap

> Ten phases. Each has a **gate**: a measurable result that must be achieved
> before spending time on the next one. If a gate fails, we change the design —
> we do not push on and hope.

Legend for who does what:

* 🤖 **Automated** — I can do this end-to-end in the sandbox, no weights or GPU needed.
* 🖥️ **[RUN-BY-YOU]** — needs your laptop's GPU, a model download, or real hardware. I write the script, the expected output contract, and the analysis; you run it and paste back the numbers.
* 👤 **Human decision** — needs your judgement or approval (see doc 08).

---

## Phase 0 — Feasibility gates (1 week)

**Why first:** every architectural choice rests on a handful of measurements
nobody has published. Get them before writing product code. This phase is cheap
and it can invalidate the design — which is exactly why it goes first.

| # | Task | Who | Gate |
|---|---|---|---|
| 0.1 | Build `llama.cpp` with CUDA + CPU backends on your machine | 🖥️ | `llama-bench` runs |
| 0.2 | Download Qwen3-0.6B/1.7B Q4_K_M GGUFs (HuggingFace is blocked in my sandbox — verified) | 🖥️ | Files verify by SHA |
| 0.3 | `llama-bench` sweep: 0.6B & 1.7B × {CPU, GPU} × prompt {128, 256, 512} × gen 64. **Record pp/s and tg/s.** | 🖥️ | Numbers replace the [T] cells in doc 06 |
| 0.4 | Measure peak RSS for the same matrix | 🖥️ | Confirms or refutes the 946 MB anchor |
| 0.5 | Does TransformerLens load Qwen3-0.6B? If not, does it load Qwen2.5-0.5B? | 🖥️ | **Gate: at least one loads** — determines the Phase 5 plan |
| 0.6 | Check `sqlite-vec` + `FTS5` build on your platform | 🤖 | `pytest` green |
| 0.7 | Confirm Tier C proxy: run 0.3 under `taskset -c 0,1` + `cpulimit -l 25` and compare | 🖥️ | A calibration factor for the A53 extrapolation |

**Gate G0:** prefill and decode throughput recorded for at least one model on at
least one tier (Phase 0 output — this is the first point at which any latency
figure in these documents becomes `[M]` rather than `[D]` or `[T]`), and we know
whether TransformerLens works.

**If G0 fails:** if 0.6B decode is below ~15 tok/s even on your laptop CPU, P1
is dead on Tier B and everything collapses to the P0 deterministic path. That is
still a defensible project — it just becomes a different one, and we would know
in week 1 rather than week 8.

---

## Phase 1 — Data plane (2 weeks)

The part that makes the rest possible, and the part most projects under-build.

| # | Task | Who |
|---|---|---|
| 1.1 | Canonical signal registry (YAML): ~60 signals, units, physical bounds, sample rates, deadbands | 🤖 |
| 1.2 | `VehicleDataSource` protocol + `ReplaySource` (nuScenes CAN expansion loader) | 🤖 (loader) / 🖥️ (745 MB download) |
| 1.3 | `CanSource`: real DBC (opendbc) → `cantools` encode → `vcan` → `python-can` decode → canonical | 🤖 |
| 1.4 | `LogSource`: DTC/`slog2`-style parser + J2012 decoder + severity map | 🤖 |
| 1.5 | Tiered store: L0 ring (RAM), L1 packed zstd blocks, L1b deadband, L1c burst windows | 🤖 |
| 1.6 | L2 rollups (1-min, 1-hour) + retention policy | 🤖 |
| 1.7 | Normalisation tests: resampling 2 Hz / 100 Hz / 950 Hz onto a common grid | 🤖 |
| 1.8 | Storage benchmark: measured MB/day vs. the model in doc 06 | 🤖 |

**Gate G1:** replay 28 simulated days through the real `vcan` path, decode them,
and show measured on-disk footprint within **25 %** of the doc 06 model
(~103 MB). Ingest sustains the replay without dropping records.

**Notable risk addressed here:** the nuScenes channel-rate spread (2 Hz /
100 Hz / 950 Hz) *is* the hard part of this phase. Resampling and alignment is
where real telemetry work lives.

---

## Phase 2 — Analytics & retrieval, no LLM (2 weeks)

**Deliberately model-free.** If the analysis layer is good, most queries never
need the LLM. Building it first also means the eval harness has something to
measure against.

| # | Task | Who |
|---|---|---|
| 2.1 | Windowed aggregation engine + typed accessors for the whitelisted views | 🤖 |
| 2.2 | Change-point detection (CUSUM + PELT via `ruptures`) | 🤖 |
| 2.3 | Robust anomaly scoring (median/MAD z-score on residuals; IsolationForest as a baseline comparison) | 🤖 |
| 2.4 | Derived `driving_mode` label (rules → GMM k=3) | 🤖 |
| 2.5 | Lead-lag / precedence search for "what happened before X" | 🤖 |
| 2.6 | Episode distillation: triggers → cards | 🤖 |
| 2.7 | Hybrid retrieval: FTS5 BM25 + `sqlite-vec` dense + RRF fusion | 🤖 |
| 2.8 | BGE-small embedding pipeline, int8 quantised | 🖥️ (model download) / 🤖 (code) |
| 2.9 | DSL schema + validator + SQL compiler, with an adversarial test suite | 🤖 |

**Gate G2:** all five brief queries, expressed as **hand-written DSL**, return
correct answers against a scenario with known ground truth. Retrieval recall@8
≥ 85 % on a 100-query event-search set.

**Where this phase actually stands:**

| Task | State |
|---|---|
| 2.1 windowed accessors | ✅ four typed views + one compiled statement per window |
| 2.9 DSL schema, validator, compiler | ✅ five kinds, adversarial suite, mutation-checked |
| 2.2/2.3 change-point and anomaly scoring | ✅ PELT (agrees with `ruptures` 300/300), CUSUM with a measured false-alarm rate, MAD scoring on a robust local linear fit |
| 2.4 `driving_mode` derivation | 🔶 speed thresholds only, no GMM |
| 2.5 lead-lag / precedence | 🔶 event-level only, and empty on this data by design |
| 2.6 episode cards | ✅ distilled into `card` + FTS5 at build time; adas/overheat/trend cards |
| 2.7 lexical retrieval (FTS5/BM25) | ✅ card search with quoted-token safety and a stopword list |
| 2.7 dense retrieval + RRF, 2.8 BGE | ⛔ **blocked**: HuggingFace is unreachable from the sandbox, so no embedding model can be downloaded. RRF is implemented and tested so the dense side plugs in. |

The DSL half of G2 is met: all five questions run through the compiled path and
return byte-identical answers to the pre-refactor handlers. The recall half is
untouched, and its dense-retrieval component cannot be measured here.

Episode cards and lexical retrieval (2.6/2.7) are in `make episodes`. No
recall@8 figure is claimed: with three cards it would measure the corpus, not the
retriever. See doc 11 §11.8.

Trends and anomalies (2.2/2.3) are in `tools/phase0/trends.py`. The early
finding is that the *filter matters more than the detector*: on all days the
change point is 14 March, on days of comparable length it is 12 March, and only
the second matches the weather. See doc 11 §11.7.

**This gate is the real milestone.** At G2 you have a working vehicle-analytics
product that happens to have no AI in it. Everything after is enhancement.

---

## Phase 3 — The language layer (2–3 weeks)

| # | Task | Who |
|---|---|---|
| 3.1 | Rule pre-pass (DTC regex, time-expression parser, unit parser) | 🤖 |
| 3.2 | Embedding intent kNN bootstrap (works day 1, zero training data) | 🤖 |
| 3.3 | Fine-tune MiniLM-class encoder for 12-way intent + slot tagging on a synthetic query set (500+ paraphrases generated from templates, hand-corrected) | 🖥️ |
| 3.4 | Quantise encoder to ONNX int8, verify accuracy delta ≤ 2 pts | 🖥️ |
| 3.5 | SLM router: GBNF grammar + prompt template + prefix-cache strategy | 🤖 |
| 3.6 | **Evidence pack builder** with provenance IDs | 🤖 |
| 3.7 | **Template renderer with named-slot substitution** | 🤖 |
| 3.8 | **SLM narrator** (P2/P3) with the same slot contract | 🤖 |
| 3.9 | Guardrails: scope, injection sanitiser, **numeric verifier** | 🤖 |
| 3.10 | Eval harness: intent accuracy, grounding rate, recall@8, abstention | 🤖 (harness) / 🖥️ (runs) |
| 3.11 | Latency harness: p50/p95 per path per tier | 🖥️ |

**Gate G3:** intent accuracy ≥ 92 %, **numeric grounding rate = 1.000**, and
P0 latency ≤ 250 ms p95. If grounding is not exactly 1.0, the substitution
mechanism has a hole and we stop and fix it.

---

## Phase 4 — Edge deployment (1–2 weeks)

| # | Task | Who |
|---|---|---|
| 4.1 | Three tier configs; one image, `config/tier_{a,b,c}.yaml` | 🤖 |
| 4.2 | `systemd` hardening units + cgroup limits | 🤖 |
| 4.3 | ARM64 container build (multi-arch), CI matrix | 🤖 (definitions) / 🖥️ (build+run) |
| 4.4 | Deploy to Tier B (Pi 5 or Graviton free tier) | 🖥️ |
| 4.5 | Measure p50/p95 latency and peak RSS **on real ARM** | 🖥️ |
| 4.6 | Compare measured Tier B against doc 06's [D] predictions; publish the delta | 🤖 |
| 4.7 | Tier C proxy measurement (core-pinned, frequency-capped) | 🖥️ |

**Gate G4:** Tier B measured P1 latency ≤ 2 000 ms p95, peak RSS ≤ 1 500 MB.
**This gate is the first point at which a P1 latency figure may be presented as a
measurement rather than a target.** Until it passes, every P1 number in the repo
stays tagged `[D]`.
The extrapolation in doc 06 is either confirmed or corrected — **either outcome is
a valid result**, provided it is reported.

---

## Phase 5 — Research tracks: red teaming & interpretability (2 weeks)

Your objectives 3 and 4. Both are designed to produce a *finding*, not just a
tool run — that is the difference between a portfolio line and a conversation.

### 5.1 Red teaming

| # | Task | Who |
|---|---|---|
| 5.1.1 | Stand up `llama-server` behind the OpenAI-compatible facade | 🤖 |
| 5.1.2 | `garak` broad sweep; triage hits | 🖥️ |
| 5.1.3 | `PyRIT` multi-turn campaigns (Crescendo, Skeleton Key) against the full assistant | 🖥️ |
| 5.1.4 | `promptfoo` redteam run wired into CI as a gate | 🤖 |
| 5.1.5 | **Build the vehicle-domain probe set (150+ probes)** across six families: indirect injection via DTC description; via log line; via OTA manifest; via paired-device name; PII extraction via time+location; hallucinated safety advice; role-confusion (driver claims engineer) | 🤖 |
| 5.1.6 | Measure defence-in-depth: which probes are stopped by the sanitiser, which by the scope layer, which by the numeric verifier, and **which get through** | 🤖 |
| 5.1.7 | Write-up with a per-family bypass table | 🤖 |

**The finding:** generic scanners test the *user* channel. This system's dominant
attack surface is the *data* channel, and we can quantify how much of it is
defended. Expect some probes to succeed — publishing that is the contribution.

### 5.2 Mechanistic interpretability

| # | Task | Who |
|---|---|---|
| 5.2.1 | Load the model in TransformerLens (Qwen3 if supported, else Qwen2.5-0.5B as an interpretable twin — decided at G0) | 🖥️ |
| 5.2.2 | Build paired prompts: identical question, correct evidence vs. shuffled/absent evidence | 🤖 |
| 5.2.3 | **Activation patching on the evidence span** → causal attribution of retrieved tokens to the output | 🖥️ |
| 5.2.4 | Compare attribution on questions where the model is *right* vs. *wrong*, with evidence present vs. absent | 🖥️ |
| 5.2.5 | Head-level analysis: which heads move evidence into the answer position; do they match the layer-4/head-7 bottleneck reported for Qwen2.5-0.5B ([arXiv:2605.08853](https://arxiv.org/html/2605.08853))? | 🖥️ |
| 5.2.6 | Optional: train a small SAE on one layer; look for a "numeric/time" feature | 🖥️ (overnight) |

**The research question:** *does a 0.6B model answer from the retrieved evidence
or from parametric priors — and can we see the difference in its activations?*
If attribution to the evidence span is low precisely when the model is wrong,
that gives a **mechanistic hallucination detector**: patching-based confidence
without a second model. That is a genuinely useful result, and it is achievable
on 6 GB.

---

## Phase 6 — Demo polish & stretch features (1 week)

Ordered by value per hour. Cut from the bottom if time is short.

| # | Task | Value | Who |
|---|---|---|---|
| 6.1 | Provenance UI: click a number, see the source rows and window | **Very high** — makes the whole thesis visible | 🤖 |
| 6.2 | Fault-scenario gallery: 3 seeded scenarios, replayable, with expected answers | **Very high** — the demo script | 🤖 |
| 6.3 | Live streaming timeline in the UI while replay runs | High — visual dynamism | 🤖 |
| 6.4 | Two personas (driver / engineer) switching answer depth and vocabulary | High — brief explicitly names both | 🤖 |
| 6.5 | Voice input via `whisper.cpp` tiny, behind a flag | Medium — +75 MB, +0.5–2 s | 🖥️ |
| 6.6 | CARLA closed-loop: drive, then ask | Medium-high demo value, high setup cost | 🖥️ |
| 6.7 | AAOS Cuttlefish VHAL bridge | Medium — proves protocol breadth | 🖥️ |
| 6.8 | `vsomeip` bridge | Medium | 🤖 |
| 6.9 | Report/export: one-click PDF of an answer with provenance | Medium | 🤖 |

---

## Phase 7 — Packaging & submission (1 week)

| # | Task | Who |
|---|---|---|
| 7.1 | `make demo` one-command start, offline, on a clean machine | 🤖 |
| 7.2 | Reproducibility: pinned deps, model SHA manifest, seeded scenarios, `Dockerfile` + `Makefile` | 🤖 |
| 7.3 | CI: lint, unit, boundary tests, eval report, red-team gate (CPU-only, no model download) | 🤖 |
| 7.4 | Generate the eight expected outputs from the live numbers — **not hand-written** | 🤖 |
| 7.5 | Architecture diagram (published version of §2), data-flow diagram, latency chart | 🤖 |
| 7.6 | Demo video script (3–5 min) + recorded run | 👤 record / 🤖 script |
| 7.7 | `RESULTS.md` with every measured number, each tagged [M]/[D], with hardware named | 🤖 |

---

## Timeline

Assuming ~15 h/week of focused effort:

```
Week   1    2    3    4    5    6    7    8    9   10   11   12
P0    ███
P1        ████████
P2                ████████
P3                        ████████████
P4                                    ████████
P5                                            ████████
P6                                                    ████
P7                                                        ████
                G1   G2        G3   G4        G5     
```

**Critical path:** P0 → P1 → P2 → P3 → G4. Everything else has float.
**If you have half the time:** do P0, P1, P2, and a trimmed P3 (skip the encoder
fine-tune, keep embedding-kNN + rules). You still hit gates G1 and G2 and can
demo all five queries via templates. That is a complete, honest project.

---

## 9. What is automated vs. what needs you

### 9.1 Fully automated (I do it)

| Area | Detail |
|---|---|
| Schema & registry design | Signal registry, units, bounds, deadband policy |
| Store implementation | Packed blocks, rollups, retention, FTS5/vec |
| Analytics | Change-point, anomaly, driving-mode, lead-lag |
| Retrieval | BM25, dense, RRF, evidence pack |
| DSL + compiler | Including the adversarial test suite |
| Prompt engineering | Templates, GBNF grammars, slot contracts |
| Guardrails | Sanitiser, numeric verifier, scope policy |
| Eval harness | Query sets, metrics, reports |
| Synthetic query generation | 500+ paraphrases from templates |
| Red-team probe authoring | The 150-probe vehicle-domain set |
| CI/CD, packaging, docs | Including these design documents |
| Analysis of your measurements | Turn raw `llama-bench` output into the tables |

### 9.2 Needs your machine (I script, you run)

| Task | Why | Est. wall time |
|---|---|---|
| Build `llama.cpp` + `llama-cpp-python` with CUDA | Needs your GPU/toolchain | 20–40 min |
| Download GGUFs (~1.5 GB total) | HuggingFace blocked here (verified) | 10–20 min |
| `llama-bench` sweep | Needs the GPU | 15 min |
| LoRA/encoder fine-tune | Needs the GPU | 30–90 min |
| nuScenes `can_bus.zip` (745 MB) | Needs network + disk | 20 min |
| TransformerLens experiments | Needs the GPU | 1–3 h per sweep |
| garak / PyRIT runs | Need the running model | 30–120 min |
| Real ARM deployment (if available) | Needs the board | 1 day |

### 9.3 Needs your judgement (a decision, not a task)

| # | Question | Why only you |
|---|---|---|
| 1 | Do you own a Raspberry Pi, or an ARM machine capable of free-tier Graviton? | Determines whether Tier B is measured or modelled. Biggest single unknown |
| 2 | Time budget — is this 4 weeks or 12? | Determines how far down Phase 6 we go |
| 3 | Is this portfolio/learning or competition-winning? | Determines how much effort goes into the write-up vs. the research tracks |
| 4 | Do you want the red-team and interp tracks to be substantial, or present? | They're objectives 3–4, but they're also the first thing to cut under time pressure |
| 5 | Accept publishing measured limitations? | The honest per-tier latency table is our strongest differentiator *and* it publicly shows Tier C missing 2 s on P2 |
| 6 | Persona scope: driver only, or both driver + engineer? | Both is better but roughly +30 % UI work |
| 7 | Dataset licence posture — is non-commercial-only acceptable? | nuScenes is CC BY-NC-SA. Fine for a competition, not for a product |
| 8 | Do we attempt the Cuttlefish/CARLA stretches? | High demo value, high setup cost, high schedule risk |
| 9 | Repository language for the write-up — English only? | Competition language |
| 10 | Anything in doc 08 you want to overrule | The whole point of the review |
