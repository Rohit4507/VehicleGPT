# 06 — Hardware Feasibility Analysis

> Every number here is produced by `python3 tools/feasibility_model.py`.
> If a figure looks wrong, change the script's inputs and re-run — don't take my
> word for it.

---

## 1. Method and honesty policy

Three rules, applied without exception:

1. **A number is either measured by someone with a stated methodology, or
   explicitly labelled as derived.** No silent extrapolation.
2. **Derived numbers show their arithmetic** and live in code, not prose.
3. **An unmeasured cell stays unmeasured.** Where I don't know, it says `[T]`.

| Tag | Meaning | Where it comes from |
|---|---|---|
| **[M]** | Measured | A cited source, naming the hardware, the model, the quantisation, and the tool |
| **[D]** | Derived | `tools/feasibility_model.py`, reproducible |
| **[R]** | Reported | Vendor/blog claim with no methodology — used for direction only |
| **[T]** | Target | Unknown; measured in Phase 0 |

---

## 2. The anchor: one real benchmark, on the right hardware

The case study permits demonstration on Raspberry Pi 4/5, Cuttlefish, QEMU, or
AWS Graviton. That means Raspberry Pi 5-class performance is a **legitimate
reference point**. Fortunately, Arm measured exactly the model we want to use:

> **Qwen3-0.6B-Base, Q4_K_M GGUF, `llama-bench`, physical Raspberry Pi 5
> (Cortex-A76, arm64, 4 threads), prompt length 512, generation 128, 5 runs**
>
> | Metric | Value |
> |---|---|
> | Time to first token | **3 635.67 ms** |
> | Decode throughput | **25.26 tok/s** |
> | Peak memory | **946.12 MB** |
> | End-to-end latency p50 | 8 702.57 ms |
> | Weights file size | 414.3 MB |
> | MMLU (5-shot) | 51.75 % |
>
> — [Arm AI Portal](https://developer.arm.com/ai/models/hugging-face/Arm/qwen3-0-6b-base-q4-k-m-llamacpp-raspberrypi5/qwen3-0.6b-base-q4_k_m-gguf) **[M]**

This one table determines the entire latency argument. Note what it implies:

```
prefill throughput ≈ 512 tokens / (3.636 s − ~0.15 s fixed overhead) ≈ 147 tok/s
```

**Prefill runs at ~147 tok/s while decode runs at 25 tok/s** — prefill is ~5.8×
faster per token. But a prompt carries far more tokens than an answer.

Reconstructing the measurement from those two rates:

```
prefill  512 / 147     = 3.49 s
decode   128 / 25.26   = 5.07 s
overhead               = 0.15 s
                       ─────────
total                  = 8.71 s   vs. 8.70 s measured  ✓
```

The model reproduces Arm's end-to-end figure to within 10 ms, which is a strong
sign the two rates are correct and internally consistent.

So for this shape, **41 % of the wait is reading and 59 % is writing** — not the
95/5 split you would see on a GPU. That changes the optimisation target, and it
is worth being precise about *why* prefill is the binding constraint even though
it is the smaller share here:

* Prefill is **fixed by the design**, not by the user. A 512-token prompt costs
  3.5 s whether the answer is 10 tokens or 200.
* Fake-turn answers are short. At the 28-token output shape we actually want
  (P1), prefill and decode are roughly **equal**: 340 ms and 1 108 ms on Tier B.
  Shrinking the prompt is the only lever that moves both.
* Prefill **scales with everything retrieval hands you.** Decode scales with what
  you ask for, and we can cap that with a grammar. Prompt size is the unbounded
  variable.

This drives design principles P1 and P2: **the model must never read raw data**,
because reading is the part we cannot shrink by asking for less.

---

## 3. Cross-check: is the measurement trustworthy?

Before extrapolating from one benchmark, sanity-check it physically. CPU decode
of a dense transformer is memory-bandwidth bound — every token streams the whole
weight matrix once. So:

```
effective bandwidth = weights × decode rate
                     = 414.3 MB × 25.26 tok/s
                     = 10.47 GB/s
```

The Pi 5 uses LPDDR4X-4267 — 4 channels × 16 bit × 4267 MT/s ≈ **17.1 GB/s
theoretical**. Our implied 10.47 GB/s is **61 % of theoretical**, which is a
textbook-plausible achievable fraction for a CPU workload.

**Conclusion: the Arm benchmark is physically coherent.** The bandwidth model
also lets us project other model sizes, with clear caveats:

| Model | Projected decode on Pi 5 [D] | Projected prefill on Pi 5 [D] |
|---|---|---|
| SmolLM2-135M | ~116 tok/s | ~676 tok/s |
| **Qwen3-0.6B** | **25.3 tok/s (measured, matches exactly)** | **147 tok/s (measured)** |
| Qwen3-1.7B | ~9.3 tok/s | ~54 tok/s |
| Phi-4-mini 3.8B | ~3.9 tok/s | ~22 tok/s |

The 0.6B row reproduces the measurement exactly, which is the validation. The
other rows are **upper bounds** — attention and sampling overhead bite harder on
smaller models. Treat them as `[D]`, not `[M]`.

**A useful consequence:** a 135 M model is not just "smaller and worse". It is
~4× faster at decode *and* ~4.6× faster at prefill. On the constrained tier that
is the difference between having an LLM and not having one (§6.3).

---

## 4. The three tiers, with numbers

| | **Tier A** | **Tier B** | **Tier C** |
|---|---|---|---|
| Hardware | x86 laptop + RTX 3050 6 GB | Raspberry Pi 5 8 GB (4× A76) | ~2 GB, 2× Cortex-A53/A55 |
| Case-study role | Development, demo | Permitted reference platform | **The actual design envelope** |
| Prefill | [T] unmeasured | **146.9 tok/s [M]** | **16.5 tok/s [D]** |
| Decode | [T] unmeasured | **25.3 tok/s [M]** | **5.7 tok/s [D]** |
| Model peak | [T] unmeasured | **946 MB [M]** | 946 MB (memory doesn't scale with speed) |
| Measured on | your laptop (Phase 0) | physical Pi 5 (Arm) | **nobody has measured this** |

### How the Tier C figures were derived — and why they might be wrong

```
PassMark single-thread:  A76 @2.4 GHz = 1050    A53 @1.8 GHz = 236
ratio = 236 / 1050 = 0.225                       (≈ 4.45× slower per core)

prefill (compute-bound)  = 146.9 × 0.225 × (2 cores / 4) = 16.5 tok/s
decode  (bandwidth-bound) = 25.3  × 0.225                 =  5.7 tok/s
```

Two assumptions to attack:

1. **PassMark single-thread is a proxy for LLM throughput.** It is a decent one
   (it captures IPC × clock, which is what dominates quantised GEMM), but llama.cpp
   uses NEON intrinsics whose throughput ratio between A76 and A53 may differ
   from scalar integer code. A76 has 2× 128-bit NEON; A53 has 128-bit NEON with
   different issue characteristics. **Unvalidated.**
2. **Decode scales only with the ST ratio, not core count.** Defensible — decode
   is memory-bound and multi-core scaling is poor — but the *actual board's
   memory bandwidth* matters more than its core. An A53 board with LPDDR4-3200
   (theoretical ~12.8 GB/s, maybe 7 achievable) would give a different answer
   than one with LPDDR3. **We do not know what board this envelope describes.**

**This is the weakest number in the whole document, and it is load-bearing.** See
§9 for how to fix it in one day.

---

## 5. Latency by query path, per tier

### 5.1 The canonical path specification

**One definition, used everywhere.** The normative numbers live in
`tools/paths.py`; every table in this repository is generated to match it, and
`tools/check_docs.py` fails CI if a document drifts.

| Path | What it does | Prompt in | Output out | Cached prefix | Re-prefilled | Pipeline | LLM in loop? |
|---|---|---|---|---|---|---|---|
| **P0** | Deterministic: classifier → DSL → SQL → template | 0 | 0 | 0 | 0 | 110 ms | **no** |
| **P1** | SLM intent-only (GBNF-constrained JSON) + template answer | 140 | 28 | 90 | 50 | 130 ms | yes |
| **P2** | SLM short narration over a pre-computed evidence pack | 260 | 70 | 90 | 170 | 130 ms | yes |
| **P3** | SLM long explanation / multi-hop (deep-explain mode) | 420 | 160 | 90 | 330 | 150 ms | yes |

"Pipeline" is everything that is not the model: query parse, DSL compile, SQL,
retrieval, evidence assembly, guardrails. "Cached prefix" is the system prompt
plus few-shot header, prefill-cached once at start-up and reused.

### 5.2 Latency by tier — and how much we actually know

**No figure in this table is an end-to-end measurement of VehicleGPT**, because
VehicleGPT has not been built. The best case is `[D]`: our path shape applied to
somebody else's measured throughput on similar hardware.

| Path | Tier A (RTX 3050) | Tier B (Pi 5) | Tier C (2 GB / 2× A5x) |
|---|---|---|---|
| **P0** | 110 ms `[D]` | 110 ms `[D]` | 110 ms `[D]` |
| **P1** | not measured `[T]` | 1 579 ms `[D]` | 8 073 ms `[D]` |
| **P2** | not measured `[T]` | 4 058 ms `[D]` | 22 714 ms `[D]` |
| **P3** | not measured `[T]` | 8 731 ms `[D]` | 48 220 ms `[D]` |

| Tier | What the figure rests on |
|---|---|
| **A** | Nothing. Nobody has run this model on this GPU. `[T]` until Phase 0 |
| **B** | An `[M]` throughput anchor (Arm's `llama-bench` on a physical Pi 5) combined with **our** token counts and **our** assumed 130 ms pipeline cost. The throughput is measured; the end-to-end result is derived |
| **C** | The `[D]` throughput extrapolation from §4 (§4), then the same derivation as B. **Two extrapolations deep. Never read this row as a benchmark** |

### 5.3 The 2-second verdict — path-specific

The brief says "under 2 seconds for common queries". A single yes/no answer is
not honest, because the answer depends on the path. Here is the qualified claim:

| Tier | P0 (no LLM) | P1 (intent only) | P2 / P3 |
|---|---|---|---|
| **A** | meets target `[D]` | **not measured** `[T]` — do not claim | **not measured** `[T]` |
| **B** | meets target `[D]` | **projected to meet** `[D]` — target until gate G4 | projected to exceed `[D]` |
| **C** | meets target `[D]` | projected to exceed `[D]` | projected to exceed `[D]` |

**Three statements we are entitled to make, and their limits:**

1. **P0 meets the target on every tier.** The figure is 110 ms of pure software
   (parse, SQL, template) and does not depend on unmeasured hardware — which is
   precisely why it is the designed primary path. It answers all five of the
   brief's example queries (doc 04 §4).
2. **On Tier B, P1 is *projected* to meet the target at 1 579 ms.** The
   underlying throughput is measured on real ARM; the end-to-end path is not.
   **This is a target, not a demonstration, until Phase 4 gate G4 measures it.**
   We must not describe it as demonstrated on ARM hardware — what is measured on
   ARM hardware is 25.26 tok/s decode and 3 636 ms TTFT for a 512-token prompt,
   not our pipeline.
3. **On Tier C, no current Qwen3 model can be in the per-query path.** The
   extrapolation is 4× short of the target for P1 (§4 explains why that gap is
   hardware, not prompt engineering).

### 5.4 What to do about Tier C

In order of preference:

1. **Ship P0 as the product, P2 as an explicit "explain" action** with a visible
   progress indicator. Users tolerate a slow answer they asked for; they do not
   tolerate a slow answer they did not.
2. **Try a sub-200M model for P1** — see §6.3. This is the single most valuable
   experiment in Phase 0.
3. **Accept and report it.** See §9 and the claims ledger in §11.

### Why streaming changes the verdict

P2 completing in 4 s is not the same as P2 *feeling* like 4 s. With SSE streaming:

* t ≈ 150 ms — the number is on screen (P0 facts, computed by SQL)
* t ≈ 900 ms — first prose token arrives
* t ≈ 4 000 ms — prose completes, user has been reading since 900 ms

**Perceived latency is time-to-first-meaningful-content, not completion time.**
The UI must be built around this, and it is (doc 04 §7).

---

## 6. Memory envelope

### 6.1 Tier C budget, line by line

| Component | Resident MB | Notes |
|---|---|---|
| Qwen3-0.6B Q4_K_M weights | 414.3 | [M] file size |
| KV cache @ 1 024 ctx | ~240 | Derived from the 946 MB measured peak minus weights minus ~280 MB runtime overhead. **Estimate** |
| llama.cpp runtime overhead | ~290 | [D] from the same subtraction |
| **LLM subtotal** | **~946** | **[M] measured peak** |
| SQLite page cache | 32 | `PRAGMA cache_size = -32000` |
| L0 raw ring buffer (15 min, 15 signals, 50 Hz) | 2.6 | [D] modelled |
| Embedding model, bge-small int8 | 33 | 384 dims, 33.4 M params |
| Python + FastAPI resident | ~80 | Realistic for CPython 3.11 + uvicorn |
| Remaining headroom | **954** | The real question: is it enough? |

**Verdict: it fits, with ~950 MB spare** — *provided* the KV cache is capped and
context stays short. The moment someone runs P3 with a 2 K-token context, the KV
cache grows and the budget breaks. **Context length must be a configured ceiling,
not a hope.**

### 6.2 What breaks the envelope

| Risk | Effect |
|---|---|
| Long conversations with retained history | KV cache grows linearly; 4 K ctx ≈ +1 GB |
| Loading Qwen3-1.7B instead | +700 MB of weights → over budget |
| DuckDB added without `memory_limit` | Allocates aggressively; needs an explicit cap |
| Python's allocator fragmentation over days of uptime | 10–30 % growth; mitigated by a supervisor restart policy |

### 6.3 The sub-200 M escape hatch

If Tier C must have an LLM, SmolLM2-135M (Apache-2.0) at Q4 is ~90 MB of weights
with perhaps ~100 MB of runtime and a small KV cache — **under 250 MB total**,
leaving ~1.75 GB for everything else.

Memory is not the interesting question though; latency is. Scaling every
candidate model onto Tier C and evaluating the canonical P1 path
(140 in / 28 out, 50 tokens re-prefilled after the 90-token prefix cache, 130 ms
pipeline):

| Model | Prefill (Tier C) `[D]` | Decode (Tier C) `[D]` | P1 total `[D]` | vs. 2 000 ms target |
|---|---|---|---|---|
| **SmolLM2-135M** | 76.0 tok/s | 26.2 tok/s | **1 857 ms** | **under target** `[D]` |
| Qwen3-0.6B | 16.5 tok/s | 5.7 tok/s | 8 073 ms | over `[D]` |
| Qwen3-1.7B | 6.1 tok/s | 2.1 tok/s | 21 692 ms | over `[D]` |
| Phi-4-mini 3.8B | 2.5 tok/s | 0.9 tok/s | 51 781 ms | over `[D]` |

**Every number in that table is a projection on a projection** — parameters to
throughput is one extrapolation, Pi 5 to Cortex-A53 is a second. It is `[D]` and
must never be presented as a benchmark. Reproduce it with
`python3 tools/feasibility_model.py`.

The conclusion is not "the LLM doesn't fit on Tier C". It is **"only a sub-200M
model has a chance, and only for intent extraction."** That reframes the Tier C
question from an architecture decision to a model-choice experiment.

Quality is the open question. A 135 M model will be materially worse at slot
filling, and silent slot errors are worse than slow answers. So this is a
**bake-off, not a decision** — measure intent accuracy on the 300-query set
before committing.

---

## 7. Storage: four weeks of history

Modelled by `StoragePlan` in `tools/feasibility_model.py` with these assumptions
(all stated so they can be challenged):

| Layer | What | Rate | Retention |
|---|---|---|---|
| L0 | Raw ring buffer | 15 signals @ 50 Hz | 15 min, **RAM only** |
| L1 | Hot series | 15 signals @ 1 Hz | 28 days |
| L1b | Slow signals | on-change, deadband | 28 days |
| L1c | Burst signals (IMU, wheel speeds) | 25 signals @ 10 Hz | 60 s windows around 25 events/day |
| L2 | Rollups (min/max/mean/std/count) | 1-minute buckets | 28 days |
| L3 | Episode cards + int8 embeddings + FTS5 | 180 cards/day | 28 days |

| Layer | Size |
|---|---|
| L0 raw ring (RAM) | **2.57 MB** |
| L1 hot 1 Hz | 41.53 MB |
| L1b slow on-change | 2.77 MB |
| L1c burst event windows | 40.05 MB |
| L2 rollups | 3.46 MB |
| L3 cards (text + FTS5) | 2.46 MB |
| L3 embeddings (int8) | 1.85 MB |
| SQLite overhead (12 %) | 11.05 MB |
| **Total on disk** | **103.17 MB** |
| **Total steady-state RAM** | **64.42 MB** |

**Conclusion: storage is a non-problem.** 28 days costs ~103 MB — under 1 % of a
32 GB eMMC. Even a 12-week horizon is ~300 MB.

The assumption doing the heavy lifting is `hot_bytes_per_value = 1.2 B` — int16
quantisation plus delta encoding plus zstd. That is optimistic for noisy signals
and pessimistic for slow ones. **Phase 1 gate G1 requires the measured footprint
to land within 25 % of this model**, so the assumption gets tested early, not
assumed forever.

### Why the naive approach fails

Naively storing all 60 signals at full 50 Hz for 28 days:
`60 × 50 × 86 400 × 28 × 4 B ≈ 29 GB`. **~281× more than the tiered design.**

And that is the *charitable* case — it assumes every signal is only 50 Hz. Real
dual-bus capture at a few thousand frames/second (8-byte payloads) is
`3 000 × 8 × 86 400 × 28 ≈ 58 GB` before any decoding. Both numbers are far
outside the storage of a typical automotive eMMC.
The tiering is not an optimisation; it is the difference between possible and
impossible.

---

## 8. Development hardware: RTX 3050 6 GB

| Task | Fits? | Headroom |
|---|---|---|
| Qwen3-0.6B inference (Q4/Q8) | ✅ | Huge — expect hundreds of tok/s |
| Qwen3-1.7B inference (Q4) | ✅ | Comfortable |
| Qwen3-4B inference (Q4) | ✅ | Tight at long context |
| QLoRA fine-tune, 0.6B–1.7B | ✅ | Comfortable |
| MiniLM encoder full fine-tune | ✅ | Trivial |
| TransformerLens, 0.6B, 1–2 K ctx, batch 1 | ✅ | Fits |
| SAE training on 0.6B | ⚠️ | Possible, slow. Overnight runs |
| CARLA rendering | ⚠️ | Runs at low quality; 96-bit bus is the limit |
| Full fine-tune of 1.7B | ❌ | Needs ~28 GB for Adam states. Use QLoRA |

Specs: 2 560 CUDA cores, 6 GB GDDR6, **96-bit bus / 144 GB/s**, 35–80 W TGP,
compute capability 8.6
([Notebookcheck](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3050-6GB-Laptop-GPU-GPU-Benchmarks-and-Specs.692276.0.html)).

**The 96-bit bus is the constraint that matters**, not the core count. At
144 GB/s, a 414 MB model reloads in ~2.9 ms, giving a theoretical decode ceiling
of ~350 tok/s for 0.6B — 14× the Pi 5. Plenty for development.

**Note the asymmetry this creates:** on Tier A, everything is fast enough that
you will *not* feel the latency problems that define Tier C. **Test on the
constrained tier or you will design for the wrong machine.** This is the most
common way this kind of project goes wrong.

---

## 9. The extrapolation risk, and how to close it in one day

### 9.1 The honest statement

**No latency figure exists anywhere in this document for real Cortex-A53/A55
hardware.** Every Tier C number is `[D]`, obtained by scaling a Cortex-A76
measurement by a PassMark single-thread ratio. If that ratio is wrong by 2×, the
Tier C verdict changes from "P1 is 8 s" to "P1 is 4 s" — still failing, but the
margin matters for the write-up.

### 9.2 Three ways to close the gap, cheapest first

| Method | Cost | Fidelity | Recommendation |
|---|---|---|---|
| **`taskset` + `cpulimit` on real ARM** (pin to 2 cores, cap frequency and duty cycle to emulate an A53) | ₹0 if you have any ARM board | Medium — emulates the *scheduling* constraint, not the IPC gap | ✅ **Do this in Phase 0 (task 0.7)** |
| **AWS Graviton free tier** (`t4g.small`, 2× Neoverse-N1) | ₹0 for 12 months | High for a 2-core ARM target; N1 is A76-class, so *still not A53* | ✅ If available — and it is a *legitimate demo platform* per the brief |
| **Buy/borrow an A53 board** (e.g. a used Radxa/Pine64, or any automotive dev board) | ₹2 000–6 000 | Highest | ❌ Out of budget, but note it |
| **QEMU ARM emulation** | ₹0 | **Poor — 5–50× slower than native, unusable for latency claims** | ❌ Do not use. Say so explicitly in the submission |

### 9.3 The measurement protocol (Phase 0)

```bash
# 1. Build with the CPU backend only, no GPU, to match the reference
cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-bench llama-server

# 2. Sweep the matrix. Save the raw TSV — it goes in RESULTS.md verbatim.
for model in qwen3-0.6b-q4_k_m qwen3-1.7b-q4_k_m; do
  for threads in 1 2 4; do
    taskset -c 0-$((threads-1)) ./build/bin/llama-bench \
      -m models/$model.gguf -t $threads -p 128,256,512 -n 64 -r 5
  done
done

# 3. Peak RSS, measured independently of llama.cpp's own accounting
/usr/bin/time -v ./build/bin/llama-server -m models/qwen3-0.6b-q4_k_m.gguf \
  --ctx-size 1024 -t 2 2>&1 | grep "Maximum resident set size"

# 4. Tier C proxy: 2 cores, throttled to ~28% duty to emulate the IPC gap
sudo cpulimit --limit 28 -- \
  taskset -c 0,1 ./build/bin/llama-bench -m models/qwen3-0.6b-q4_k_m.gguf -t 2 -p 128 -n 64
```

**Expected-output contract** (so you know if a run is broken before analysing it):

| Metric | Plausible range on Pi 5 | Suspicious if |
|---|---|---|
| 0.6B decode, 4 threads | 20–30 tok/s | < 10 → using a debug build, or thermal throttling |
| 0.6B prefill, 4 threads | 120–180 tok/s | < 80 → wrong quantisation or no NEON |
| Peak RSS, ctx 1024 | 800–1 050 MB | > 1 400 MB → context is larger than configured |
| 1.7B decode, 4 threads | 7–11 tok/s | > 15 → you are probably measuring 0.6B |

Paste the raw output back and I will rebuild the tables in this document from
real numbers and mark them `[M]`.

---

## 10. Implications for the design (the point of this document)

| Finding | Design consequence |
|---|---|
| Prefill cost is set by the design, not the user | Prompts stay ≤ 260 tokens. Evidence packs, not retrieved chunks |
| Prefill runs at 147 tok/s `[M]` on Pi 5, 16.5 `[D]` on A53 | **Retrieval must produce *answers*, not *material for the model to read*** |
| Tier C decode is 5.7 tok/s `[D]` | Tier C runs P0 only; LLM becomes an explicit "explain" action |
| 135 M model is ~4.5× faster in prefill and ~4.6× in decode | Phase 0 must bake off SmolLM2-135M for Tier C — it projects to 1 857 ms `[D]` |
| Storage is 103 MB for 28 days | Spend the engineering budget on distillation quality, not compression |
| LLM peak is 946 MB of a 2 GB budget | Context length is a hard configured ceiling; conversation history must be summarised, not retained |
| 96-bit bus on the dev GPU | Development hardware hides every latency problem. Test on the constrained tier |
| No real A53 measurement exists | Tier C is the weakest claim in the submission. Either measure it or label it loudly |

**The single sentence to take away:** *this project is a retrieval and analysis
engineering problem that happens to contain an LLM, because on the target
hardware the LLM is too slow to read anything you give it.*

---

## 11. Claims ledger

Every headline claim in this repository, with what it actually rests on. Use this
table in the submission; a reviewer can check any row in seconds.

| # | Claim | Figure | Status | Rests on | Becomes `[M]` at |
|---|---|---|---|---|---|
| 1 | Qwen3-0.6B Q4_K_M runs on a Pi 5 at ~25 tok/s | 25.26 tok/s | **`[M]`** | Arm `llama-bench`, physical Pi 5, 4 threads | already measured |
| 2 | ...with 946 MB peak memory | 946.12 MB | **`[M]`** | same benchmark, prompt 512 / gen 128 | already measured |
| 3 | Prefill is ~147 tok/s on Pi 5 | 146.9 tok/s | **`[D]`** | 512 tokens ÷ (3 636 ms TTFT − 150 ms overhead). Reconstructs the 8.70 s end-to-end figure to within 10 ms | Phase 0 task 0.3 |
| 4 | Cortex-A53 is ~4.5× slower per core than A76 | ratio 0.225 | **`[M]`** | PassMark single-thread ratings, 236 vs 1050 | already measured |
| 5 | ...so Tier C prefill is ~16.5 tok/s | 16.5 tok/s | **`[D]`** | Claim 3 × claim 4 × core ratio. **Two assumptions, unvalidated** | Phase 0 task 0.7 |
| 6 | P0 answers in ~110 ms on every tier | 110 ms | **`[D]`** | Pure software; no model, no hardware dependency | Phase 3 gate G3 |
| 7 | P1 on Tier B is ~1 579 ms | 1 579 ms | **`[D]`** | Claim 3 + canonical P1 spec + assumed 130 ms pipeline. **Not demonstrated on ARM** | Phase 4 gate G4 |
| 8 | P1 on Tier C is ~8 073 ms | 8 073 ms | **`[D]`** | Claims 5 and 7 stacked. **Weakest latency claim in the repo** | Phase 0 task 0.7 |
| 9 | SmolLM2-135M fits P1 on Tier C | 1 857 ms | **`[D]`** | Bandwidth model → params → throughput → Tier C. **Three extrapolations deep** | Phase 0 task 0.4 |
| 10 | 28 days of history costs ~103 MB | 103.17 MB | **`[D]`** | `StoragePlan` assumptions, `hot_bytes_per_value = 1.2 B` unverified | Phase 1 gate G1 |
| 11 | Tier A latency | — | **`[T]`** | Nothing. Unmeasured | Phase 0 task 0.3 |

**Status totals: 3 measured, 7 derived, 1 unmeasured.** No claim is at `[M]` for
*VehicleGPT's own performance*, because VehicleGPT does not exist yet.

**The honest one-liner for the submission:**

> *"We have not measured our own system. We have measured the hardware capability
> it depends on, and derived everything else transparently enough that a reviewer
> can check the arithmetic."*
