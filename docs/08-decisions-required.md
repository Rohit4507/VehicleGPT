# 08 — Decisions Requiring Your Approval

> Ten decisions. Each has a **recommendation**, the **alternative**, what it
> costs, and a **default** that applies if you say nothing.
>
> If you want to move fast, read D1, D2, D3 and D9 and reply
> *"go with all recommendations."* Everything else is reversible later.

---

## D1 — Hardware for the Tier B reference demo

**This is the single most consequential decision, because it determines whether
your headline latency number is measured or modelled.**

| Option | Cost | What you'd measure |
|---|---|---|
| **A. Raspberry Pi 5 8 GB** (if you already own one) | ~₹8 500 new | Real ARM latency, real thermal behaviour, real memory pressure. The honest number |
| **B. AWS Graviton `t4g.small`** — free tier, 750 h/month for 12 months | ₹0 | Real 2-core ARM. Neoverse-N1 is roughly A76-class. **Explicitly permitted by the brief.** Best ₹0 option |
| **C. Neither — model it** | ₹0 | Tier A measured; Tier B and C modelled `[D]` only |

**Recommendation: B if you don't already own a Pi; A if you do; C only if A and B
both fail.**

Rationale: the brief lists Graviton as an acceptable demonstration platform. It is
free, it is real ARM, and 2 vCPU maps neatly onto the composition of the
constrained envelope — which makes the extrapolation to Tier C more defensible
than a 4-core A76 measurement would be.

**Cost of getting this wrong:** if you go with C, your latency table is entirely
derived and a reviewer can fairly ask whether the system was ever run on ARM.
That's a weak position for the "Edge Performance" criterion.

**Default:** C (modelled), clearly labelled.

> ⚠️ Free-tier terms change. Verify 750 h/month and instance eligibility at
> signup before planning around it.

---

## D2 — How hard do we lean on the deterministic (no-LLM) path?

The finding in doc 06 is that P0 can answer all five example queries in ~110 ms
on *every* tier, and the LLM cannot beat that on the constrained envelope.

| Option | Consequence |
|---|---|
| **A. P0 is the product; the LLM is an enhancement** | Every query answered fast and correctly on every tier. The LLM adds narrative on demand. Risk: a judge asks "where is the AI?" — answered by showing the SLM router, the personae, and P2/P3 |
| **B. LLM-first, P0 as fallback** | Feels more "AI-native". Fails the 2-second constraint on Tier C for most queries. Highest risk to the headline constraint |
| **C. Split honestly by persona** — driver gets P0/P1, engineer gets P2/P3 | Same as A but framed better. Matches the brief's two named roles |

**Recommendation: C.** It is option A with a narrative that makes the design
choice look deliberate rather than like a retreat — because it *is* deliberate.

**Cost:** ~2 days of extra UI work for persona switching.

**Default:** C.

---

## D3 — Scope of the protocol adapters

Doc 02 §5 and doc 04 §5 propose: **4 deep, 2 bridged, 2 documented.**

| Option | Work | Credibility |
|---|---|---|
| **A. The proposed mix** (CAN + replay + logs + ADAS deep; SOME/IP + VHAL bridged; QNX + camera documented) | Baseline | High — claims match reality, and the honesty itself scores |
| **B. Add Cuttlefish VHAL** (real AAOS property changes) | +1–2 weeks, ~8 GB RAM, long AOSP build | Higher on "Data Integration". Significant schedule risk |
| **C. Add `vsomeip` bridge properly** | +2–3 days | Good value for the cost |
| **D. Attempt all seven** | Not achievable | Low — reviewers detect overclaiming |

**Recommendation: A, with C folded into Phase 5 if time allows.** Treat B as a
Phase 6 stretch and decide at G4 based on remaining time.

**The line to write in the submission:** *"Two protocols are implemented end to
end, two have working bridges, two are integrated at the format level, and one
is deliberately metadata-only for safety and memory reasons."* That sentence is
worth more than seven half-working integrations.

**Default:** A.

---

## D4 — Which model for Tier C: 0.6B, or a 135M bake-off?

| Option | Evidence | Risk |
|---|---|---|
| **A. Qwen3-0.6B everywhere (one model, three configs)** | Only model with a published Pi 5 benchmark [M]. Simplest story. Known quality | Fails 2 s on Tier C for P1 (8.1 s [D]) |
| **B. Bake off Qwen3-0.6B vs SmolLM2-135M vs SmolLM2-360M in Phase 0** | 135M projects to ~1.8 s for P1 on Tier C [D] | +3 days. Quality at slot-filling may be unusable. Two models to maintain |
| **C. Qwen3-1.7B for Tier A/B, 0.6B for Tier C** | Best quality on the dev box | Slight prompt drift between tiers |

**Recommendation: B, then choose.** It is three days that could rescue the
Tier C constraint entirely — the highest expected value of any single experiment
in the plan. If 135M's intent accuracy lands within ~5 points of 0.6B's, the
latency argument changes completely.

**If you skip it:** go with C — 1.7B on Tier A for the demo's quality, 0.6B on
Tier B/C. One extra config file.

**Default:** B (run the bake-off), falling back to C.

---

## D5 — Do we publish the honest per-tier latency table?

| Option | Consequence |
|---|---|
| **A. Publish it, including "Tier C is projected to miss 2 s on LLM paths `[D]`"** | Shows we understood the constraint better than anyone who didn't check. Matching our claim to the brief becomes an explicit, defensible finding |
| **B. Report only Tier B numbers, stay quiet about Tier C** | Riskier than it looks. Anyone who does the PassMark arithmetic finds the gap, and then every other number is suspect |
| **C. Report only Tier A ("we hit 0.4 s!")** | Reads as evasive for a brief that says "embedded hardware, ~2 GB, ~2× A5x" |

**Recommendation: A, unambiguously.** This is the decision that most determines
how the submission is received. A team that says *"here is exactly where our
architecture fails and here is the arithmetic"* is more credible on everything
else than a team that reports its best number.

**Default:** A.

---

## D6 — Persona scope

| Option | Work |
|---|---|
| **A. Both driver and engineer** | Baseline + ~30 % UI |
| **B. Driver only** | −30 % UI, but ignores an explicit part of the brief ("user or service engineer queries") |

**Recommendation: A.** The brief names both roles; answering only one is a
visible omission. The difference is mostly answer length, vocabulary, and whether
DTC codes and raw signal names appear in the output — a rendering-layer concern,
not a new subsystem.

**Default:** A.

---

## D7 — Research track depth: red teaming vs. interpretability

You have both as objectives. They compete for the same Phase 5 fortnight.

| Option | Red teaming | Interpretability |
|---|---|---|
| **A. Balanced** (recommended) | garak + PyRIT + promptfoo CI gate + 150-probe vehicle-domain set + bypass table | Activation patching on the evidence span + head-level analysis + one figure + written finding |
| **B. Red-team heavy** | + adversarial training, + a defence evaluation | One patching experiment only |
| **C. Interp heavy** | Scanner run + probe set only | + SAE training, + circuit tracing, + a paper-style write-up |

**Recommendation: A.** Both objectives get a *finding*, not just a tool run.
Red teaming's vehicle-domain probe set is the more original contribution (nobody
else tests injection through DTC fields); interpretability's groundedness
question is the more publishable one. Equal weight is the right call at this
scope, and both fit comfortably in two weeks if the probe authoring is automated
(it is — I can generate the probe set).

**Default:** A.

---

## D8 — Dataset licence posture

nuScenes is **CC BY-NC-SA 4.0** — non-commercial, share-alike.

| Option | Consequence |
|---|---|
| **A. Use it, disclose clearly** | Best realism for ₹0. Competition use is fine. Must be flagged in `DATA.md` and the submission |
| **B. Avoid non-commercial data entirely** | Use comma2k19 (MIT) + synthetic. Less realistic dynamics, cleaner licensing story |
| **C. Use both** — nuScenes for realism, comma2k19 as the permissive fallback | Slightly more loader work |

**Recommendation: A**, with a one-line disclosure. Competition entries are
non-commercial by nature, and the licence explicitly permits non-commercial
research use.

**Default:** A.

---

## D9 — Time budget, and the pre-agreed cut line

**This determines everything about how the other nine decisions play out.**

| Option | Plan |
|---|---|
| **A. ~4 weeks part-time** | P0, P1, P2, trimmed P3 (skip encoder fine-tune; embedding-kNN + rules). Gates G1, G2 only. Demo all five queries via templates. Red team and interp reduced to one experiment each |
| **B. ~8 weeks** | P0–P5. Complete submission. Phase 6 partially cut |
| **C. ~12 weeks** | Everything including CARLA/VHAL stretches |

**Recommendation: pick B, and pre-agree that if you slip one gate you drop to A.**
Writing the cut line down *now* is what prevents the R16 failure mode.

**The floor is high:** option A is still a complete, honest, demonstrable project
with a real evaluation and a real finding. That is worth knowing.

**Default:** B.

---

## D10 — Repository and writing conventions

Low-stakes, but decide once.

- Language: English only?
- Commit convention: Conventional Commits + PR-per-phase?
- Do we want a `RESULTS.md` that is generated from the eval harness (recommended)
  or written by hand?
- Should the design docs stay in this `docs/` structure (recommended) or become a
  single long `DESIGN.md` for submission?
- Any Tata Elxsi template or page limit for the submission document?

**Default:** English; Conventional Commits; generated `RESULTS.md`; keep `docs/`
and generate a combined PDF at Phase 7.

---

## What I will do next, if you say nothing

Defaults in force, I proceed with **Phase 0 only** — the feasibility gates — because
those are the measurements that can invalidate the design, and they are the
cheapest thing to get wrong. Specifically:

1. 🤖 Write the Phase 0 harness scripts, the `llama-bench` sweep runner, the RSS
   measurement, and the expected-output contracts.
2. 🤖 Scaffold the repository (packaging, CI, lint, boundary tests, `make` targets).
3. 🤖 Build the scenario DSL and the synthetic fault generator — no weights needed,
   fully testable here.
4. 🤖 Build the CAN path (`vcan` + real opendbc DBC + `cantools`) and prove it
   works end to end in this sandbox. This is free and high-value.
5. 🖥️ Hand you the Phase 0 runbook to execute on your laptop.

**I will not touch:** model selection beyond a temporary default, the LLM prompts,
the interpretability experiments, or anything that assumes Tier B hardware exists.

---

## One-paragraph summary for a busy reader

> The brief is 15 % LLM and 85 % data engineering. Four of its five example
> queries are analytical, so a RAG chatbot fails them. The 2-second constraint is
> really a *prefill* constraint, and on the design envelope (2 GB, 2× A5x) no
> generative step per query can fit — so the design makes the deterministic path
> primary and the model a narrator over a verified evidence pack, with every
> number substituted rather than generated. Pick a free ARM target if you can
> (D1); let the no-LLM path be the product (D2); publish the honest per-tier
> latency table (D5); and pre-agree the 4-week cut line (D9). Everything else is
> reversible.
