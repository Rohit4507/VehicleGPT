# 02 — Research Findings

> Everything below was checked during this research pass. Sources are linked
> inline. Where I could not verify something, it is listed in §11 rather than
> guessed at.

## 0. Evidence tiers

Every performance number in this repository carries a tag. Please hold me to it.

| Tag | Meaning | Trust |
|---|---|---|
| **[M]** | **Measured** — published by a named source, on named hardware, with the configuration stated | High. Quotable. |
| **[D]** | **Derived** — arithmetic performed by `tools/feasibility_model.py` on top of [M] numbers | Medium. Reproducible; assumption-sensitive. |
| **[R]** | **Reported** — a vendor/blog/community claim without a stated methodology | Low. Used only for direction, never in a design calculation. |
| **[T]** | **Target** — unknown; must be measured in Phase 0 on our own hardware | None yet. Never quoted as fact. |

The distinction matters because this space is full of confident blog posts with
no methodology. You will notice several popular claims below that turned out to
be wrong or misleading when checked.

---

## 1. The small-model landscape

### 1.1 Candidates considered

| Model | Params | Licence | Q4 size | Context | Verdict |
|---|---|---|---|---|---|
| **Qwen3-0.6B** | 0.6 B | Apache-2.0 | **414 MB [M]** | 32 K | **Primary.** Smallest model with a *published, methodology-stated* Pi 5 benchmark |
| **Qwen3-1.7B** | 1.7 B | Apache-2.0 | ~1.1 GB [R] | 32 K | **Secondary** for Tier A/B. Same tokenizer family → swap without prompt changes |
| Qwen3-4B | 4 B | Apache-2.0 | ~2.6 GB [R] | 32 K | Too large for the memory envelope; useful only as a quality *ceiling* in evals |
| Gemma 3 270M / 1B | 0.27 / 1 B | Gemma Terms | ~0.2 / 0.5 GB | 32 K | Gemma licence is **not OSI-approved** — usage restrictions. Avoided for an "open-source" claim |
| Llama 3.2 1B / 3B | 1 / 3 B | Llama Community | ~0.8 / 2 GB | 128 K | Licence has acceptable-use and 700 M-MAU clauses. Fine for a competition, awkward for the "open" narrative |
| SmolLM2-135M/360M | 0.135 / 0.36 B | Apache-2.0 | ~0.1 / 0.25 GB | 8 K | **Interesting for Tier C.** A 135M model is projected `[D]` to be the only one that fits path P1 inside 2 s on A53. Candidate for a Phase-0 bake-off |
| Phi-4-mini | 3.8 B | MIT | ~2.7 GB | 128 K | Strong quality, MIT, but 3× our memory envelope. Out of scope for the edge claim |
| BitNet b1.58 2B-4T | 2 B (1.58-bit) | MIT | 1.2 GB | 4 K | Fascinating (300 MB RSS reported [R]) but 7 tok/s on a Pi 5 [R] and requires a Microsoft `llama.cpp` fork. **Rejected on tooling risk**, noted as a research curiosity |

### 1.2 Why Qwen3

1. **Apache-2.0.** Clean for both the competition and any future commercialisation.
   Gemma and Llama both carry usage restrictions that would need a caveat in the
   submission.
2. **A real, methodologically-stated edge benchmark exists at exactly our target
   size.** Arm's AI Portal publishes Qwen3-0.6B-Base Q4_K_M on a physical
   Raspberry Pi 5 via `llama-bench`: **25.26 tok/s decode, 3635.67 ms TTFT for a
   512-token prompt, 946.12 MB peak memory, MMLU 51.75 % (5-shot)**
   ([Arm AI Portal](https://developer.arm.com/ai/models/hugging-face/Arm/qwen3-0-6b-base-q4-k-m-llamacpp-raspberrypi5/qwen3-0.6b-base-q4_k_m-gguf)).
   This is the single most valuable datapoint found in the entire research pass,
   because it converts "can we hit 2 s?" from an opinion into arithmetic.
3. **Consistent family ladder.** 0.6B / 1.7B / 4B share a tokenizer and chat
   template. We can benchmark quality-vs-latency by swapping one GGUF file, with
   no prompt engineering changes.
4. **Well-benchmarked in the literature.** Qwen3-0.6B scores 44.6 MMLU-Redux and
   Qwen3-1.7B scores 64.4, against Gemma-3-1B-IT's 33.3 and Phi-4-mini's 67.9
   ([Qwen3 Technical Report, arXiv:2505.09388](https://arxiv.org/pdf/2505.09388)).
   Note the cliff: 0.6B → 1.7B more than doubles the score. **The router must not
   rely on 0.6B for anything requiring world knowledge.**

### 1.3 The thinking-mode trap

Qwen3 models default to a `thinking` mode that emits long chain-of-thought before
answering. On a 25 tok/s device this is catastrophic for latency — a 400-token
reasoning trace costs 16 seconds. Two mitigations, both required:

* Disable thinking via the chat template (`enable_thinking=False`).
* Constrain the router's output with a **GBNF grammar** so it can only emit a
  valid JSON object. This caps output length *structurally* rather than by
  prompt instruction, which small models ignore unreliably.

### 1.4 What no model gives us

A 0.6B model cannot do multi-step arithmetic over 4 weeks of telemetry. It will
hallucinate the arithmetic and get the trend direction wrong. Every aggregation,
difference, ranking and comparison must be computed in code. This is not a
limitation we are working around; it is the core design principle (doc 04 §3).

---

## 2. Inference runtimes

| Runtime | ARM CPU | Grammar-constrained output | Prefix caching | Measured Pi 5 data | Packaging cost | Verdict |
|---|---|---|---|---|---|---|
| **llama.cpp** | ✅ excellent, KleidiAI-tuned kernels | ✅ GBNF | ✅ `--prompt-cache`, slot reuse | ✅ the Arm numbers above | Zero — single binary + GGUF | **Chosen** |
| ONNX Runtime GenAI | ✅ | ⚠️ limited | ⚠️ | ✗ | Needs ONNX export per model/quant | Strong on x86 (Microsoft measured a 1.2–1.6× throughput edge over llama.cpp on Windows, [ISE blog](https://devblogs.microsoft.com/ise/running-rag-onnxruntime-genai/)) but no ARM-LLM story as strong |
| MLC-LLM | ✅ (TVM-compiled) | ✗ | ⚠️ | ✗ | Per-target compilation, TVM toolchain | Best for browser/mobile; too much build complexity for a 10-week project |
| ExecuTorch | ⚠️ 1.0 GA Oct 2025 | ✗ | ✗ | ✗ | `.pte` export pipeline | Right answer for a *phone* product; wrong answer for a Linux ARM SBC |
| Ollama | ✅ | ⚠️ | ✅ | Indirect (wraps llama.cpp) | Very low | Great for prototyping, but it is a wrapper we would have to strip for a 2 GB target. Use for dev convenience only, ship llama.cpp |

**Decision:** `llama.cpp` for the runtime, with `llama-cpp-python` bindings for
development and the `llama-server` binary (OpenAI-compatible) for the deployed
service. This also happens to be the interface garak and promptfoo can attack
directly, which serves objective 3.

> **Note on the OpenAI-compatible endpoint:** it is a *local* HTTP interface on
> loopback. It does not create a cloud dependency. Worth stating explicitly in
> the submission, because the phrase invites the wrong assumption.

---

## 3. Edge hardware

### 3.1 The AI HAT+ / Hailo-10H surprise

The obvious move is "Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H, 40 TOPS, 8 GB)". I
checked it, and **it is slower than the Pi 5's own CPU for the models we care
about**:

| Model | AI HAT+ 2 (Hailo-10H) | Pi 5 CPU (no HAT) |
|---|---|---|
| Qwen2 1.5B | 8.25 tok/s | **12.47 tok/s** |
| Qwen2.5-Instruct 1.5B | 6.86 tok/s | **11.41 tok/s** |
| DeepSeek-R1-Distill-Qwen 1.5B | 6.85 tok/s | **10.99 tok/s** |
| Llama 3.2 3B | 2.66 tok/s | **5.55 tok/s** |

Source: [computingforgeeks comparison](https://computingforgeeks.com/edge-llm-jetson-orin-vs-ai-hat-2-vs-hailo-10h/),
cross-checked against [Hailo's model zoo](https://github.com/hailo-ai/hailo_model_zoo)
and a second independent write-up reporting 9.45 tok/s for Qwen2-1.5B with 289 ms
TTFT ([awesomeagents.ai](https://awesomeagents.ai/hardware/hailo-10h/)).

Caveats that make the HAT's *time-to-first-token* genuinely better (289–730 ms
vs. ~3 600 ms on CPU for a 512-token prompt) — but its **context window is
capped at 2 048 tokens** and it only supports a fixed model list.

**Conclusion:** at ₹0 budget this is moot (the HAT costs ~$130). More usefully,
it tells us **not to architect around an NPU**. The design must be CPU-first,
which is also what survives into a real vehicle HPC.

### 3.2 CPU class matters more than anything else

| Core | Class | Single-thread (PassMark) | vs. A76 |
|---|---|---|---|
| Cortex-A76 (Pi 5, 2.4 GHz) | Out-of-order, high IPC | 1050 | 1.00× |
| Cortex-A55 (1.8 GHz) | In-order, improved | 311 | 0.30× (**3.4× slower**) |
| Cortex-A53 (1.8 GHz) | In-order, 8-stage | 236 | 0.22× (**4.5× slower**) |

Source: [PassMark CPU comparison](https://www.cpubenchmark.net/compare/5743vs4128vs6777/ARM-Cortex-A76-4-2400-MHz-vs-ARM-Cortex-A53-4-1800-MHz-vs-ARM-Cortex-A55-6-1800-MHz).
Also note Cortex-A53 is ~2.3 DMIPS/MHz vs. A72's ~4.7
([cross-check](https://armbasedsolutions.com/info-detail/features-and-applications-of-arm-cortex-a53-processors)).

**This is the finding that reshapes the whole design.** The case study's design
envelope ("~2 ARM A5X cores") is 4.5× slower than the hardware everyone will
demo on. Any latency result measured on a Pi 5 must be divided by ~4.5 before it
can be claimed for the design target. Doc 06 does exactly that, and it is why the
architecture has a no-LLM path.

### 3.3 Development GPU: RTX 3050 6 GB Laptop

| Spec | Value |
|---|---|
| CUDA cores | 2 560 |
| Memory | 6 GB GDDR6, **96-bit** bus |
| Bandwidth | **144 GB/s** |
| TGP | 35–80 W |

Source: [Notebookcheck](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3050-6GB-Laptop-GPU-GPU-Benchmarks-and-Specs.692276.0.html),
[cross-check](https://www.itechguides.com/nvidia-geforce-rtx-3050-6gb-laptop-gpu-specs-performance-and-buying-guide/).

Capability assessment for our workloads:

| Task | Fits in 6 GB? | Notes |
|---|---|---|
| Inference, Qwen3-0.6B Q4/Q8 | ✅ easily | <1 GB; expect hundreds of tok/s |
| Inference, Qwen3-1.7B Q4 | ✅ | ~1.1 GB + KV |
| Inference, Qwen3-4B Q4 | ✅ | ~2.6 GB, tight with long context |
| LoRA fine-tune, 0.6B–1.7B (QLoRA 4-bit) | ✅ | Comfortable |
| Full fine-tune, 1.7B | ❌ | Needs ~28 GB for Adam states |
| TransformerLens on Qwen3-0.6B | ✅ | Activations for 1–2 K tokens at batch 1 |
| SAE training on 0.6B | ⚠️ | Feasible but slow; plan overnight runs |
| CARLA rendering | ⚠️ | Runs at low quality; 144 GB/s and 6 GB is the floor |
| CUDA compute capability | 8.6 (Ampere) | Supported by every tool listed |

---

## 4. Simulators and data sources

### 4.1 The blunt conclusion

**No open dataset contains a "battery overheating warning" sequence.** Fault
sequences of the kind the brief asks about are OEM-confidential. We must either
simulate them or inject them synthetically onto real recorded data. Synthesis is
the only reproducible option, and it is also the *better* option because it gives
us ground truth to score against.

### 4.2 Simulator comparison

| Option | Realism of vehicle internals | Cost | Outputs real CAN frames? | Verdict |
|---|---|---|---|---|
| **CARLA** | High physics (Unreal Engine), gives speed/throttle/brake/steer/gear/RPM/lights per tick | Heavy — GPU, ~20 GB install, 14.5 k★ | No — you must build it | **Optional Phase 6.** Great for a live closed-loop demo, wrong as a data dependency |
| **SUMO** | Traffic-level only, no vehicle internals | Very light, CPU-only, 4.2 k★ | No | Useful for one thing: generating realistic city vs. highway speed profiles. Cheap win |
| **AWSIM / Autoware** | Good, ROS2-native | Heavy | Partial | Overkill for this project |
| **MetaDrive** | Mid-fidelity, Python-native | Light | No | Viable CARLA alternative if CARLA won't run |
| **custom scenario DSL + fault injection** | Exactly what we define | Trivial | **Yes** (we emit frames onto `vcan`) | **Chosen.** Deterministic, seeded, unit-testable, diffable in git |
| **nuScenes CAN expansion** | **Real vehicle, real physics** | 745–781 MB download | JSON, not frames | **Chosen as the realism backbone** |
| **comma2k19 / openpilot segments** | Real vehicle, real CAN | Large | Real CAN, needs the right DBC | Good supplementary realism |

### 4.3 nuScenes CAN bus expansion — the key dataset

* Real CAN data from Motional's **Renault Zoe** autonomous test fleet, ~979 scenes.
* Channels: `vehicle_monitor` (2 Hz, speed/yaw/brake/throttle), `ms_imu`
  (100 Hz), `zoe_veh_info` (100 Hz, 4 wheel speeds + steering),
  `zoesensors` (~950 Hz raw pedals), plus battery, torque, odometry, solar.
  ([nuscenes-devkit README](https://github.com/nutonomy/nuscenes-devkit),
  [a practitioner's teardown of the channel rates](https://github.com/dlwogud/nuscenes-driving-safety-pipeline))
* This channel-rate spread (2 Hz → 950 Hz) is *itself* the realistic engineering
  problem: you must resample and align it. That is exactly capability F2 and F3.
* **Licence: CC BY-NC-SA 4.0 — non-commercial only**
  ([nuScenes paper, arXiv:1903.11027](https://arxiv.org/pdf/1903.11027)).
  Fine for a competition; must be flagged for production.
* A 745 MB mirror exists on Zenodo under CC-BY-4.0
  ([zenodo record](https://zenodo.org/records/15667420)) — verify before relying
  on that mirror's licence claim.

### 4.4 Real DBC files — the thing that makes the CAN demo genuine

Most "CAN" demos are fake: a CSV of decoded signals pretending to be a bus. We
can do better. **opendbc** (MIT, 3.5 k★) ships **120 real production DBC files**
covering hundreds of vehicle models from Audi, BMW, Ford, Hyundai, Toyota and more
([commaai/opendbc](https://github.com/commaai/opendbc) — verified the file count
via the GitHub trees API during this research).

That means we can:

1. Take a real DBC.
2. Encode realistic physical values into real CAN frames with `cantools`.
3. Push them onto a **Linux `vcan` virtual bus** via `python-can`.
4. Decode them back with the *same* DBC inside our ingest adapter.

Which is, byte for byte, the same code path a real vehicle would exercise. The
only thing missing is the physical transceiver. **This is the single highest-
credibility engineering trick available to us at zero cost**, and it should be
front and centre in the demo.

---

## 5. Automotive protocols and middleware

| Interface | Open-source option | Licence | Feasibility at ₹0 | Plan |
|---|---|---|---|---|
| **CAN / CAN-FD** | `vcan` (kernel), `python-can` (LGPL-3.0, 1.6 k★), `cantools` (MIT, 2.3 k★), opendbc DBCs (MIT) | all permissive | **Full.** Runs on any Linux, no hardware | **Deeply implement** |
| **SOME/IP** | `COVESA/vsomeip` (MPL-2.0, 1.5 k★) | MPL-2.0 | Buildable on Linux; needs Boost; heavy for 2 GB | **Adapter + bridge**; run vsomeip on the x86 reference stack, consume over a local socket |
| **Android Automotive VHAL** | AOSP + **Cuttlefish** (AAOS virtual device, KVM) | Apache-2.0 | Works on x86 with KVM; needs ~8 GB and a long build | **Adapter, Phase 6 stretch.** VHAL exposes `VehiclePropertyId`s via `CarPropertyManager` ([AOSP VHAL docs](https://source.android.com/docs/automotive/vhal)) |
| **QNX services/logs** | ✗ proprietary | — | **Not obtainable** | **Log-format adapter only**, fed synthetic `slog2`-style logs. Say so plainly |
| **ADAS events** | Our own event schema; AOSP VHAL ADAS properties are standardised as of AAOS 14 ([release notes](https://source.android.com/docs/automotive/start/releases/u_udc_release)) | — | Full, synthetic | **Implement** |
| **Camera feeds** | Out of scope for the LLM | — | — | **Metadata only** (see doc 01 §4, conflict 3) |
| **Diagnostic logs / DTCs** | SAE J2012 DTC semantics; open DTC description lists exist | — | Full | **Implement**, with a curated DTC dictionary |

**Design consequence:** every one of these sits behind one interface,
`VehicleDataSource`. Adding a protocol must never touch the AI plane. Doc 04 §5.

---

## 6. Storage and retrieval

### 6.1 Vector store

| Store | Form | Footprint | Hybrid search | Verdict |
|---|---|---|---|---|
| **sqlite-vec** | SQLite extension (C) | **< 20 MB**, brute-force KNN | via FTS5 | **Chosen.** Lives in the same file as the telemetry; one file, one backup, one integrity check |
| LanceDB | Embedded, Rust, Lance columnar | Larger, disk-based IVF-PQ | ✅ native FTS | Better above ~1 M vectors. We will have ~10 k. Overkill |
| ChromaDB | Embedded/server | ~120 MB base | limited | Prototyping only |
| FAISS | Library | small | ✗ (no metadata) | Fastest raw KNN, but we need SQL joins and metadata filters — the wrong shape |
| Qdrant embedded | Rust, mmap | configurable | ✅ | Excellent, but a second storage engine to run on a vehicle |

sqlite-vec is MIT/Apache-2.0 dual-licensed, version 0.1.**9** on PyPI as of this
research (verified via the PyPI JSON API). Note: it is still pre-1.0, which is a
minor risk (doc 07 R11).

**Brute-force KNN is correct at our scale.** With ~10 k embeddings of 384 dims
(int8), a full scan is ~3.7 M multiply-adds ≈ a few milliseconds. An HNSW index
would add build complexity and memory for no measurable gain. Reported latency
figures for sqlite-vec (~17 ms at 1 M vectors, [LatticeDB comparison](https://docs.latticedb.org/comparisons/vs-vector-databases))
scale down linearly: at 10 k vectors, sub-millisecond.

### 6.2 Embedding model

| Model | Params | Dims | MTEB retrieval (EN v2) | Licence | Verdict |
|---|---|---|---|---|---|
| **bge-small-en-v1.5** | 33.4 M | 384 | **53.86** | MIT | **Chosen.** Best quality per MB |
| all-MiniLM-L6-v2 | 22.7 M | 384 | 42.92 | Apache-2.0 | Faster, ~11 points worse. Good baseline |

Source: [MTEB-based ranking](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/).
bge-small is 384-dim / 512-token input; int8-quantised it is ~33 MB in memory.
Sub-10 ms per short text on CPU.

> **Domain caveat:** both models are trained on natural language. Without
> fine-tuning, neither will match `"P0AFA"` to `"hybrid battery voltage low"`.
> That is precisely why retrieval must be **hybrid** with BM25, and why the
> episode cards (F4) are written in *deliberately redundant, human-readable*
> language — the distillation step is where the embedding model's weakness is
> compensated for.

### 6.3 Time-series storage

| Option | Strengths | Weakness on a vehicle | Verdict |
|---|---|---|---|
| **SQLite (WAL)** | Single file, ACID, zero admin, everywhere, `FTS5` built in | Row-oriented → wide tables are bulky; no native compression | **Chosen as the system of record** |
| DuckDB + Parquet | 10–30× compression, superb analytical scans, `ASOF JOIN` | Single-writer, weak at unbatched high-frequency ingest ([benchmarks](https://github.com/bherbruck/silodb) show ~56× slower single-row writes than SQLite) | **Optional Phase 3**, exactly as the *cold analytical* tier |
| TimescaleDB | Real TSDB features, retention policies | Full Postgres server — wrong for a gateway | Rejected |
| InfluxDB | Purpose-built | Heavy JVM/Go service, memory-hungry | Rejected |

**Chosen pattern:** SQLite as the system of record + **zstd-compressed packed
columnar BLOBs** for 1 Hz series (one blob per signal per minute, decoded with
NumPy). This gives columnar-ish storage inside a row store, keeps writes
sequential, and needs no second engine. If Phase 3 benchmarks show query latency
or footprint problems, Parquet + DuckDB is added *behind the same interface*.

Storage arithmetic — 28 days, 15 hot signals @ 1 Hz, 30 slow (on-change),
25 burst signals in 60 s event windows @ 10 Hz, 1-minute rollups, ~5 000
episode cards → **~103 MB on disk, ~64 MB steady-state RAM**
(`python3 tools/feasibility_model.py`). Weeks of history is not the hard part.

---

## 7. Red-teaming tooling

| Tool | Licence | Shape | Best for | Verdict |
|---|---|---|---|---|
| **garak** (NVIDIA) | Apache-2.0, 9.8 k★ | CLI scanner, probe library, JSONL reports | Broad first-pass sweep against the local `llama-server` endpoint | **Chosen** |
| **PyRIT** (Microsoft) | MIT, 4.7 k★ | Python orchestration framework | Multi-turn attacks (Crescendo, TAP, Skeleton Key) — the right shape for a *conversational* assistant | **Chosen** |
| **promptfoo** | MIT, 26.6 k★ | YAML + CLI, CI gating, OWASP/MITRE mapping | Regression-gating the red-team suite in CI | **Chosen** |
| Giskard | Apache-2.0 core | Scanner + platform | RAG-specific scans (retrieval vs. generation separation) | Optional |
| DeepTeam | Apache-2.0 | Python `red_team()` | If we standardise on DeepEval | Optional |

Star counts and licences verified via the GitHub API during this research.
Tool shapes from [a practitioner comparison](https://aminrj.com/posts/attack-patterns-red-teaming/)
and [Braintrust's 2026 round-up](https://www.braintrust.dev/articles/best-llm-red-teaming-tools-2026).

**The gap none of these tools fill:** every one of them assumes the attack
surface is *user input*. In a vehicle assistant, the more dangerous surface is
**the data**: a DTC description field, a log line, an OTA manifest, a Bluetooth
device name, or the caller-ID of a paired phone can all contain adversarial text
that lands in the model's context. Generic scanners will not find those. Phase 5
builds a **vehicle-domain probe set** for exactly this (doc 05 §6).

---

## 8. Mechanistic interpretability tooling

| Tool | Licence | Status | Fit |
|---|---|---|---|
| **TransformerLens** | MIT, 4.0 k★ | Actively maintained (pushed this week) | **Chosen.** `HookedTransformer`, activation caching, ablation and patching, direct logit attribution |
| SAELens | (fork lineage unresolved — see §11) | Active | Optional. Training an SAE on a 0.6B model on 6 GB is possible but expensive; only if the primary experiment needs features |
| nnsight / NDIF | Apache-2.0 | Active | Viable if a specific Qwen3 architecture isn't in TransformerLens |

**Crucially, there is a strong precedent for interpreting models at exactly our
scale.** *"Architecture, Not Scale: Circuit Localization in Large Language
Models"* analyses six model sizes from 160 M to 7 B with TransformerLens and
finds that **grouped-query attention produces more concentrated, more
mechanistically stable circuits** than multi-head attention
([arXiv:2605.08853](https://arxiv.org/html/2605.08853)). It reports concrete
finding for Qwen2.5-0.5B: the top factual-recall head is at layer 4, head 7, and
ablating layers 0 and 4 together leaves 12.7 % residual accuracy — a partial
backup pathway.

That paper gives us (a) a validated toolchain, (b) a published baseline to
replicate or contradict, and (c) confidence that a 0.6B model has a real,
tractable circuit to study.

**Risk:** TransformerLens support for the *Qwen3* architecture specifically is
not confirmed (my GitHub issue search returned no results — see §11). Mitigation:
run the interpretability study on **Qwen2.5-0.5B-Instruct** as an "interpretable
twin" and validate that our behavioural findings transfer, or use nnsight. Doc 05
Phase 5 has the fallback.

---

## 9. Safety isolation — the standards picture

The brief requires safety-critical functions to be isolated from the AI system.
This is the ISO 26262 **"freedom from interference"** requirement, met in
industry by hypervisor partitioning:

* QNX Hypervisor 2.0 for Safety is TÜV-certified to **ASIL D**, explicitly for
  partitioning safety-critical from non-safety-critical guests
  ([EENews](https://www.eenewseurope.com/en/qnx-hypervisor-certified-for-asil-d/)).
  QNX Hypervisor 8.0 for Safety went GA in March 2026.
* The mechanism is **spatial isolation** (MMU-enforced memory regions) plus
  **temporal isolation** (core pinning, bounded scheduling), with all
  inter-partition communication explicit and mediated
  ([analysis](https://www.rtos.club/qnx/qnx-hypervisor-mixed-criticality-systems-on-one-soc/)).

**What we can honestly claim:** we cannot certify anything. What we *can* do —
and what is architecturally stronger than a hypervisor for this specific
threat — is make **VehicleGPT read-only by construction**: there is no code path
from the AI plane to any actuation interface, and this is verifiable by
inspection and by a CI test that fails if an actuation client is ever added. A
hypervisor stops a compromised guest from *interfering*; a read-only design stops
it from *causing* anything in the first place. We propose both: the read-only
guarantee now, and hypervisor partitioning documented as the production
deployment model.

---

## 10. Deployment targets

| Target | Access at ₹0 | ARM cores | Notes |
|---|---|---|---|
| **Local x86 laptop + RTX 3050** | ✅ own hardware | n/a | Tier A. Development and demo |
| **Raspberry Pi 5 8 GB** | ~₹8 500 — **not free** | 4× A76 | Tier B reference. Only if you already own one |
| **AWS Graviton `t4g.small`** | Free tier (750 h/month for 12 months, then paid) | 2× Neoverse-N1 | Free-tier limits change; verify at signup. N1 is roughly A76-class |
| **QEMU ARM VM** | ✅ free | emulated | **Not representative.** QEMU system emulation is 5–50× slower than native and cannot be used for latency claims |
| **Android Cuttlefish (AAOS)** | ✅ free | x86 host, KVM | Good for the VHAL story, not for ARM latency |
| **2 GB / 2× A53 container with CPU quota** | ✅ free | n/a | **Proxy for Tier C**: constrain cores + memory + frequency to emulate. Imperfect but honest if labelled |

**The honest position:** we can measure Tier A natively, and Tier B if a Pi 5 or
Graviton instance is available. Tier C can only be *modelled* (§0 [D]) unless
real A53/A55 hardware is obtained. The submission must say so, and doc 06 §7
gives the exact script to close the gap.

---

## 11. What I could not verify (do not treat as fact)

| # | Item | Status |
|---|---|---|
| 1 | **TransformerLens support for Qwen3** | GitHub issue search returned no results, which is ambiguous (may be handled without an issue). **Must be checked empirically in Phase 0.** |
| 2 | **SAELens canonical repository** | My API queries for `decodingtrust/SAELens` and `saprmarks/SAELens` did not resolve cleanly. Confirm before depending on it. |
| 3 | **Any latency number on real Cortex-A53/A55** | None found. All Tier C figures are [D] extrapolations from PassMark ratios. **This is the single biggest unvalidated assumption in the design.** |
| 4 | **Qwen3-1.7B Q4 memory peak on ARM** | No published `llama-bench` figure found. ~1.1 GB is a size estimate, not a measurement. |
| 5 | **DuckDB memory behaviour at `memory_limit=64MB` on ARM** | DuckDB works on ARM64, but its footprint under a hard 64 MB cap is unmeasured. One more reason it stays optional. |
| 6 | **Cuttlefish AAOS build time / RAM** | Not attempted. Estimate 8 GB RAM and 1–3 hours; unverified. |
| 7 | **Whether nuScenes' Zenodo mirror really is CC-BY-4.0** | The Zenodo record claims CC-BY-4.0 while the canonical nuScenes licence is CC BY-NC-SA 4.0. **Assume the stricter one.** |
| 8 | **HuggingFace reachability from this sandbox** | **Blocked** (verified: TLS handshake fails). All model downloads must happen on your machine. This shapes doc 05 §2. |

---

## 12. Sandbox constraints discovered during this pass

Worth recording because they shape the roadmap:

| Constraint | Detail | Consequence |
|---|---|---|
| No GPU in this environment | 2 vCPU, 3 GB RAM, no `nvidia-smi` | Nothing here can run a model. All Phase 0 measurements happen on your laptop |
| `huggingface.co` unreachable | Verified `SSL_ERROR_SYSCALL`; DNS resolves, connection refused | Model weights cannot be fetched here. GGUF files must be downloaded by you |
| No Docker, no Ollama | Verified absent | Container-based workflows must be authored here, run by you |
| GitHub, PyPI, npm reachable | Verified | Code, dependencies, tests and docs *can* be built and validated here |

**Practical consequence for the roadmap:** this environment is a **code, test and
documentation factory**, not a runtime. Everything that needs weights, a GPU or a
real device is a "you-run-it" step with a script and an expected-output contract.
Doc 05 marks those steps explicitly (`[RUN-BY-YOU]`).
