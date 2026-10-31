# 03 — Technology Comparison

> Fifteen decisions, each with the alternatives that were seriously considered,
> the reasoning, and what would make us change our mind. If a decision has no
> falsifier, it isn't a decision — it's a habit.

Legend: ✅ chosen · ⚪ viable alternative · ❌ rejected

---

## D1 — Overall architectural pattern

| Pattern | What it means here | Verdict |
|---|---|---|
| **Deterministic core + LLM at the edges** | SQL/analytics does the maths; the model classifies intent and writes prose over verified facts | ✅ **Chosen** |
| LLM-as-orchestrator (agent with tools) | Model decides which tool to call, loops until done | ❌ Each hop costs 0.5–8 s on ARM. A 3-hop agent blows the latency budget by 10×, and 0.6B models are unreliable planners |
| Pure RAG over log chunks | Embed everything, retrieve top-k, stuff into prompt | ❌ Fails 4 of the brief's 5 own example queries (doc 01 §1.2). Also blows the prefill budget |
| Fine-tuned end-to-end NL→answer model | Teach a small model the domain | ❌ Cannot do arithmetic. Would confidently invent trends. Unfixable by data volume |

**Why:** the brief's queries are analytical (trend, argmax, compare, correlate),
not lookup. A 0.6B model cannot reliably compute `argmax` over a 2 000-point
series, and if it appears to, it is guessing. Computing in SQL is both correct
and ~1 000× cheaper.

**Falsifier:** if ≥ 60 % of a realistic query workload turns out to be genuine
free-text lookup, revisit toward a RAG-heavier mix.

---

## D2 — Language model

| Model | Licence | Edge evidence | Verdict |
|---|---|---|---|
| **Qwen3-0.6B** (Q4_K_M) | Apache-2.0 | **Published Pi 5 `llama-bench`: 25.26 tok/s, 3 636 ms TTFT, 946 MB peak** | ✅ **Primary** |
| Qwen3-1.7B | Apache-2.0 | Same tokenizer, no published edge bench | ✅ **Secondary (Tier A/B)** |
| Qwen3-4B | Apache-2.0 | — | ⚪ Quality ceiling in evals only |
| SmolLM2-135M/360M | Apache-2.0 | — | ⚪ **Phase-0 bake-off candidate for Tier C** |
| Gemma 3 1B | Gemma Terms (non-OSI) | — | ❌ Licence friction vs. Qwen's clean Apache-2.0, for no measured edge gain |
| Llama 3.2 1B/3B | Llama Community | — | ❌ Licence clauses + no edge-bench advantage |
| Phi-4-mini 3.8B | MIT | — | ❌ 3× the memory envelope |
| BitNet b1.58 2B | MIT | 7 tok/s on Pi 5 [R] | ❌ Requires a vendor fork of llama.cpp; tooling risk outweighs the novelty |

**Why Qwen3-0.6B:** it is the only model in the target size class with a
*methodologically stated* benchmark on the exact hardware class the brief
permits. That single fact converts the entire latency argument from opinion to
arithmetic, which is worth more in a technical review than a few MMLU points.

**Falsifier:** if Phase 0 shows Qwen3-0.6B's intent accuracy is < 90 % after
fine-tuning, move to 1.7B and reduce the answer-length budget.

---

## D3 — Inference runtime

| Runtime | Verdict | Reason |
|---|---|---|
| **llama.cpp** (`llama-server` + `llama-cpp-python`) | ✅ | Only runtime with all four of: measured ARM-CPU baseline, GBNF grammar-constrained output, prefix caching, and zero packaging overhead |
| ONNX Runtime GenAI | ⚪ | Genuinely faster on x86-Windows (1.2–1.6× [M] per Microsoft). Revisit if we ever target Windows HPC. No ARM-LLM story |
| MLC-LLM | ❌ | Every target needs its own compilation. Weeks of toolchain work for unproven gains here |
| ExecuTorch | ❌ | Correct for phones, not for Linux ARM SBCs. 1.0 GA is only very recent |
| Ollama | ❌ (ship) / ✅ (dev) | It *is* llama.cpp underneath plus a daemon. Use for quick experiments, strip for deployment |
| vLLM / TGI | ❌ | GPU-server software. Irrelevant to a 2 GB envelope |

**Why grammar-constrained decoding is the deciding factor:** it lets us
*structurally* cap output to a valid JSON object. Small models ignore
"output only JSON" instructions with dispiriting regularity; a GBNF grammar makes
the invalid string literally unreachable. That is the difference between a demo
and a system.

---

## D4 — Intent understanding: how does a query become a query?

This is the crux decision, because it determines the latency profile.

| Approach | Latency on A53/Tier C | Accuracy risk | Verdict |
|---|---|---|---|
| **Fine-tuned encoder classifier (MiniLM ~30 M, ONNX int8)** | **5–15 ms** `[T]` | Slot filling needs a second head or a small seq2seq | ✅ **Primary router** |
| **SLM + GBNF grammar → JSON** | 0.8 s (Tier C est.) – 1.6 s [D Tier B] | Handles novel phrasings; may invent slot values | ✅ **Secondary / fallback** |
| Regex + keyword rules | < 1 ms | Brittle, but excellent for DTC codes, part numbers, "yesterday", "last 2 weeks" | ✅ **Pre-pass**, always |
| Zero-shot embedding similarity to intent exemplars | ~2 ms | Surprisingly decent for 5–10 intents | ✅ **Bootstrap** — works on day 1, before any training data exists |
| Full agentic planner | 5–30 s | Unreliable at 0.6B | ❌ |

**Recommended cascade (cheap → expensive, stop at first confident answer):**

```
1. Rule pre-pass           → regex for DTC codes, time expressions, units   (~1 ms)
2. Embedding intent kNN    → nearest exemplar, threshold on margin          (~3 ms)
3. Encoder classifier      → fine-tuned, 12-way intent + slot tagging       (~12 ms)
4. SLM + GBNF              → only if steps 1–3 disagree or are low-conf    (0.8–1.6 s)
5. Clarifying question     → if still ambiguous, ASK. Do not guess.
```

Step 5 matters more than it looks. A vehicle assistant that asks
"Which vehicle period — last week or last month?" is *better* than one that
answers the wrong question confidently.

**Why not just use the SLM for everything:** it costs 100× more latency for a
task that is genuinely a classification problem. Using a generative model to
classify intent is like using a lorry to deliver a letter.

---

## D5 — Query execution: how does intent become data?

| Approach | Safety | Latency | Verdict |
|---|---|---|---|
| **Model emits a typed DSL JSON; a deterministic compiler emits parameterised SQL against a whitelist of views** | Model output is validated against a schema; cannot express a dangerous query | ~2 ms | ✅ **Chosen** |
| Model emits raw SQL, executed against SQLite | Full DB read/write reachable. Prompt injection → data exfiltration or corruption | — | ❌ **Never.** Non-negotiable |
| Model calls Python tool functions | Safe-ish, but each tool is a code path that must be secured | — | ⚪ Acceptable, but coarser-grained than a DSL |
| Hand-written query templates per intent | Safest, fastest | Cannot express "compare X between A and B over window W" generically | ⚪ Used as the compile target |

**The DSL is deliberately tiny.** Roughly:

```json
{"op": "compare" | "trend" | "argmax_delta" | "list_events" | "precede",
 "signals": ["energy.per_km"], "group_by": "driving_mode",
 "window": {"start": "...", "end": "..."},
 "filters": {"severity": [">=warn"]}}
```

The compiler maps this onto a fixed set of pre-vetted SQL views. Unknown
`signals` or `op` values are rejected. **The model never sees a table name.**

---

## D6 — Storage engine

| Option | Verdict | Reason |
|---|---|---|
| **SQLite (WAL) as system of record + zstd-packed columnar BLOBs** | ✅ | One file, ACID, zero admin, `FTS5` included, runs everywhere including a Pi. Packed blobs give columnar-style density inside a row store |
| **DuckDB + Parquet as optional cold/analytics tier** | ⚪ (Phase 3) | 10–30× better compression and analytical scan speed, but single-writer and ~56× slower at unbatched single-row writes ([benchmark](https://github.com/bherbruck/silodb)) — the exact shape a vehicle produces |
| TimescaleDB / InfluxDB | ❌ | A full server process on a vehicle ECU is the wrong shape and the wrong memory profile |
| Plain CSV/Parquet files | ❌ | No indexing, no transactions, no concurrent access |
| Raw CAN log files (`.blf`, `.asc`) | ⚪ | Keep as an **archive format** for export/ingest, not as the query store |

**Why both:** they solve different problems. SQLite is the *system of record* —
small, transactional, always-on. DuckDB is an *analytical accelerator* over the
same data. Keeping DuckDB optional means we can prove SQLite is insufficient
before adding a second engine to a memory-constrained device. **Simplicity is a
feature you only get to keep if you defend it.**

---

## D7 — Vector store

| Option | Vectors we'll hold | Verdict |
|---|---|---|
| **sqlite-vec** (brute-force KNN inside SQLite) | ~10 k | ✅ < 20 MB, same file as the data, SQL joins with filters, trivially backed up |
| LanceDB | ~10 k | ⚪ Better above ~1 M vectors. At 10 k it's operational overhead for nothing |
| FAISS | ~10 k | ❌ No native metadata filtering — we always filter by time range and severity |
| Chroma | ~10 k | ❌ 120 MB base footprint is 6 % of the whole Tier C budget |
| Qdrant embedded | ~10 k | ⚪ Excellent engine; a second process on a vehicle is a second thing to crash |

**The scale argument is decisive:** at 10 k vectors of 384 dims, brute force is
~3.7 M MACs ≈ single-digit milliseconds. Building an ANN index to avoid
milliseconds we don't have is premature optimisation with a memory cost.

**Falsifier:** if episode volume exceeds ~200 k cards over the retention horizon
(≈ 3 years at current rates), re-evaluate.

---

## D8 — Retrieval strategy

| Strategy | Weakness in *this* domain |
|---|---|
| Dense only | Cannot match `"P0AFA"` → `"hybrid battery voltage low"` without domain fine-tuning; ignores numeric range queries entirely |
| BM25 only | Misses paraphrase ("why is my car drinking more fuel" ≠ "fuel consumption increased") |
| **Hybrid: BM25 + dense, fused with Reciprocal Rank Fusion** | — ✅ **Chosen** |
| Hybrid + cross-encoder reranker | +80–300 ms `[D]` for a marginal gain at k=20; against Tier C's projected ~1.9 s P1 budget that is 4–16 % of the whole thing `[D]`. ⚪ **Optional on Tier A only** |

**Why RRF specifically:** it needs no score calibration between BM25 and cosine
similarity, which is essential when the two retrievers have wildly different
score distributions. It is one line of arithmetic and it is robust.

**And critically:** retrieval is not the primary path. For analytical queries the
DSL compiler produces SQL directly; retrieval is used for *event* queries and
*evidence gathering around a numeric finding*. This inverts the usual design.

---

## D9 — Vehicle simulation & data strategy

| Option | Verdict | Reason |
|---|---|---|
| **Layered: real recorded data + synthetic fault injection + optional CARLA** | ✅ **Chosen** | Realism where it's free, synthetic where reality isn't available, heavyweight sim only where it adds demo value |
| CARLA only | ❌ | GPU-heavy; produces no CAN frames; and no CARLA scenario contains a "battery overheating warning" either — you'd still be scripting it |
| Recorded data only | ❌ | No fault sequences exist in public datasets. Fail on key point #1 of the brief |
| Synthetic only | ❌ | Reviewers will (rightly) ask whether it's realistic. The nuScenes backbone answers that |
| SUMO | ⚪ | Cheap and useful for one narrow thing: realistic city vs. highway speed profiles for Q5 |

**The scenario DSL is the actual deliverable.** It makes fault sequences
deterministic, seedable, version-controlled, diffable and testable — properties a
CARLA scenario does not have. That's what "how will error conditions be
recreated" should mean.

---

## D10 — CAN integration approach

| Option | Verdict | Reason |
|---|---|---|
| **`vcan` + `python-can` + real opendbc DBCs + `cantools`** | ✅ | Exercises the *identical* decode path a real vehicle would. Free. Runs in CI. Highest credibility per rupee |
| SocketCAN on real hardware + USB-CAN adapter | ⚪ | Ideal, costs ~₹2 000. Out of budget; the code path is the same |
| CSV of decoded signals | ❌ | Fake. Anyone who has worked on a vehicle will spot it instantly |
| `can-utils` only (candump logs) | ⚪ | Useful for capture/replay realism; less good than encoding from physical values |

**Why this matters for scoring:** "Data Integration — support for multiple
automotive protocols" is an explicit criterion. Demonstrating real DBC decoding
of real frame layouts on a real (virtual) bus, with the OEM's actual message
definitions, is a fundamentally different claim from "we have a CSV".

---

## D11 — SOME/IP and VHAL

| Option | Verdict |
|---|---|
| **Adapter interface + `vsomeip` bridge on the x86 reference stack** | ✅ Phase 5, documented integration point |
| Full `vsomeip` on the 2 GB target | ❌ Boost dependency and footprint make it a poor fit for the constrained tier |
| **AAOS Cuttlefish for the VHAL story** | ⚪ Phase 6 stretch; high demo value, high setup cost |
| Claim full support for both without building them | ❌ Dishonest, and easily caught in Q&A |

**Position:** two protocols deeply implemented (CAN, replay/log), two with a
production-shaped adapter and a working bridge (SOME/IP, VHAL), three documented
with an integration point (QNX logs, camera metadata, ADAS feed). **Say which is
which, in the submission.** Reviewers reward calibrated honesty and punish
overclaiming.

---

## D12 — Language / runtime / packaging

| Option | Verdict | Reason |
|---|---|---|
| **Python 3.11 + `uv` + FastAPI + SQLite** | ✅ | Fast iteration; every library we need exists; the ML ecosystem is Python |
| C++ core for ingest/store | ⚪ Phase 3 if RSS or ingest throughput demands it | The likely hot spot if we ever exceed budget |
| Rust | ❌ | Compile times and a thinner ML ecosystem for no measured benefit at this stage |
| Node/TypeScript backend | ❌ | Wrong ecosystem for embeddings, ONNX and llama.cpp bindings |

**Note on Python's memory cost:** a bare CPython + FastAPI process is ~60–90 MB.
That is **3–4 % of the Tier C envelope**, and it is entirely avoidable later by
moving the ingest hot path to C++. Flagged as a risk (doc 07 R4), not ignored.

---

## D13 — Frontend / user interaction

| Option | Verdict | Reason |
|---|---|---|
| **Local web UI, served and proxied by the same FastAPI process** | ✅ | Zero install, works in a browser, streamable, previewable in this environment, easy to record for the submission video |
| Native desktop (Qt/Tauri) | ❌ | Packaging burden, no benefit for a demo |
| CLI | ⚪ | Keep as the primary *test* interface and for scripted demos |
| Voice-first | ⚪ Phase 6 stretch | Adds `whisper.cpp` (~75 MB tiny) plus ~0.5–2 s ASR latency — a large fraction of the budget for a nice-to-have |

**Design detail that matters:** the UI must show **provenance inline**. Click a
number, see the source rows and the window. This directly serves the "accuracy of
responses" criterion — it makes correctness *visible* rather than asserted.

---

## D14 — Evaluation approach

| Option | Verdict |
|---|---|
| **Programmatic metrics: numeric grounding rate, intent accuracy, recall@8, abstention rate** | ✅ Primary. Deterministic, CI-gatable, no judge bias |
| LLM-as-judge (local 1.7B) | ⚪ Secondary signal for prose quality only. Never in the pass/fail path — a judge that shares the answerer's weaknesses is not a judge |
| Human evaluation | ⚪ Spot-checks only (n≈30). Too slow for regression |
| Nothing ("it looks good") | ❌ |

**The numeric grounding rate is the killer metric.** Because numbers are
*substituted* rather than generated, it should be exactly 100 %, and the test
therefore catches any future refactor that accidentally lets the model write a
number. A metric that is trivially 100 % is a *guardrail*, and guardrails are
what you want in safety-adjacent software.

---

## D15 — Safety isolation mechanism

| Mechanism | Verdict | Reason |
|---|---|---|
| **Read-only by construction (no actuation code path exists) + process/cgroup isolation** | ✅ Now | Verifiable, demonstrable, and arguably stronger than a hypervisor for this threat |
| **Hypervisor partitioning (QNX Hypervisor for Safety, ASIL-D)** | ✅ Documented as the production model | The real automotive answer; out of reach at ₹0 and unnecessary for a demo |
| Separate physical ECU | ⚪ Production alternative | Architecturally cleanest, costliest |
| "We promise the AI won't touch safety functions" | ❌ | Not an architecture |

**The key argument:** a hypervisor prevents a compromised guest from
*interfering* with a safety partition. Removing the actuation path means a
compromised AI component cannot *cause* anything at all. Both together are the
production answer; only the second is available to us, and it is the more
fundamental of the two.

---

## Summary table

| # | Decision | Choice | Revisit if |
|---|---|---|---|
| D1 | Pattern | Deterministic core, LLM at edges | >60 % of workload is free-text lookup |
| D2 | Model | Qwen3-0.6B (+1.7B on big tiers) | Intent accuracy < 90 % after fine-tuning |
| D3 | Runtime | llama.cpp | We ever target Windows HPC |
| D4 | Intent | Rules → embedding kNN → encoder → SLM+GBNF | Encoder accuracy plateaus below 85 % |
| D5 | Execution | Typed DSL → parameterised SQL | Never |
| D6 | Storage | SQLite + packed blobs (+ DuckDB optional) | Query p95 > 150 ms |
| D7 | Vectors | sqlite-vec | > 200 k vectors |
| D8 | Retrieval | Hybrid BM25 + dense, RRF | Recall@8 < 85 % |
| D9 | Data | Real recorded + synthetic faults + optional CARLA | — |
| D10 | CAN | vcan + real DBCs | Hardware budget appears |
| D11 | Protocols | 2 deep, 2 bridged, 3 documented | — |
| D12 | Language | Python 3.11 | RSS exceeds envelope |
| D13 | UI | Local web, inline provenance | — |
| D14 | Eval | Programmatic metrics | Never |
| D15 | Isolation | Read-only + cgroups; hypervisor documented | — |
