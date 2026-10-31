#!/usr/bin/env python3
"""
VehicleGPT feasibility model.

Purpose
-------
Every latency / storage claim in docs/06-hardware-feasibility.md is produced by
this script. It is deliberately explicit about *what is measured* and *what is
derived*, so nobody has to trust prose.

Evidence tiers used throughout:
  [M] measured  - published by a cited source, on the stated hardware
  [D] derived   - arithmetic performed *here* on top of [M] numbers
  [T] target    - not known yet; must be measured in Phase 0 on real hardware

Run:
    python tools/feasibility_model.py            # full report
    python tools/feasibility_model.py --json     # machine-readable

No third-party dependencies. Stdlib only, so it runs on an RTX 3050 laptop, a
Raspberry Pi, or a 2 GB container.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from paths import BY_KEY, CACHED_PREFIX_TOKENS, PATHS, TARGET_LATENCY_MS, PathSpec

# Evidence status attached to every computed figure.
STATUS_MEASURED = "[M]"    # measured by a cited source, on stated hardware
STATUS_DERIVED = "[D]"     # arithmetic from [M] anchors
STATUS_TARGET = "[T]"      # not measured; must be measured in Phase 0 / gate G4


# ---------------------------------------------------------------------------
# 1. Anchor measurements  ([M] - cited, see docs/02-research-findings.md)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Anchor:
    """A single published measurement."""
    value: float
    unit: str
    hardware: str
    note: str
    source: str


ANCHORS: dict[str, Anchor] = {
    # Arm AI Portal, llama-bench on a physical Raspberry Pi 5 (Cortex-A76,
    # arm64, 4 threads), Qwen3-0.6B-Base Q4_K_M, prompt 512 / gen 128.
    "rpi5_qwen3_0_6b_ttft_s": Anchor(
        3.63567, "s", "RPi5 A76 4T", "TTFT for 512-token prompt",
        "https://developer.arm.com/ai/models/hugging-face/Arm/"
        "qwen3-0-6b-base-q4-k-m-llamacpp-raspberrypi5/qwen3-0.6b-base-q4_k_m-gguf"),
    "rpi5_qwen3_0_6b_decode_tps": Anchor(
        25.26, "tok/s", "RPi5 A76 4T", "decode throughput",
        "same Arm AI Portal benchmark"),
    "rpi5_qwen3_0_6b_peak_mem_mb": Anchor(
        946.12, "MB", "RPi5 A76 4T", "peak RSS incl. KV cache",
        "same Arm AI Portal benchmark"),
    "rpi5_qwen3_0_6b_weights_mb": Anchor(
        414.3, "MB", "any", "Q4_K_M GGUF file size",
        "same Arm AI Portal benchmark"),

    # PassMark single-thread ratings (cpubenchmark.net comparison page).
    "passmark_a76_4c_2400_single": Anchor(
        1050, "score", "Cortex-A76 4c @2.4GHz", "single-thread rating",
        "https://www.cpubenchmark.net/compare/5743vs4128vs6777/"
        "ARM-Cortex-A76-4-2400-MHz-vs-ARM-Cortex-A53-4-1800-MHz-"
        "vs-ARM-Cortex-A55-6-1800-MHz"),
    "passmark_a53_4c_1800_single": Anchor(
        236, "score", "Cortex-A53 4c @1.8GHz", "single-thread rating",
        "same PassMark comparison"),
    "passmark_a55_6c_1800_single": Anchor(
        311, "score", "Cortex-A55 6c @1.8GHz", "single-thread rating",
        "same PassMark comparison"),

    # llama.cpp CPU decode on a modern x86 core, Phi-4-mini class (3.8B) Q4.
    # Used only as a sanity cross-check, not as a design input.
    "x86_38b_cpu_decode_tps": Anchor(
        20.0, "tok/s", "modern x86 CPU", "Q4 3.8B", "vendor-reported range 15-25"),
}


# ---------------------------------------------------------------------------
# 2. Hardware tiers
# ---------------------------------------------------------------------------

@dataclass
class Tier:
    """A deployment target the case study asks us to cover."""
    key: str
    name: str
    ram_mb: int
    cores: int
    core_class: str
    # measured llama.cpp throughput on this exact class, or None
    measured_prefill_tps: float | None
    measured_decode_tps: float | None
    measured_peak_mb: float | None
    provenance: str
    latency_status: str = STATUS_TARGET


def derive_tiers() -> list[Tier]:
    """Build the three tiers, scaling published A76 numbers by PassMark ST ratio."""
    a76_st = ANCHORS["passmark_a76_4c_2400_single"].value
    a53_st = ANCHORS["passmark_a53_4c_1800_single"].value
    a55_st = ANCHORS["passmark_a55_6c_1800_single"].value

    ttft = ANCHORS["rpi5_qwen3_0_6b_ttft_s"].value
    decode = ANCHORS["rpi5_qwen3_0_6b_decode_tps"].value
    peak = ANCHORS["rpi5_qwen3_0_6b_peak_mem_mb"].value

    # Prefill throughput implied by the Arm TTFT measurement. Subtract a
    # nominal 150 ms of fixed overhead (tokenise + graph alloc) first.
    prefill_a76 = 512 / max(ttft - 0.15, 0.001)

    def scale(anchor_st: float, cores: int, compute_bound: bool) -> float:
        """Scale a 4-thread A76 measurement to another core class.

        Single-thread PassMark ratio carries the IPC+clock effect. Prefill is
        compute-bound so it also scales ~with core count (capped at 4, the
        reference thread count). Decode is memory-bandwidth-bound, so it does
        NOT scale with extra cores; only the ST ratio applies.
        """
        st = anchor_st / a76_st
        if compute_bound:
            st *= min(cores, 4) / 4.0
        return st

    a53_factor_c, a53_factor_m = scale(a53_st, 2, True), scale(a53_st, 2, False)
    a55_factor_c, a55_factor_m = scale(a55_st, 2, True), scale(a55_st, 2, False)

    return [
        Tier(
            key="A",
            name="x86 dev box / RTX 3050 6 GB",
            ram_mb=16384, cores=8, core_class="x86-64 (Zen4/P-core class)",
            measured_prefill_tps=None, measured_decode_tps=None,
            measured_peak_mb=None,
            provenance=(
                "[T] not yet measured. GPU-resident 0.6B decode is expected to be "
                "bandwidth-bound: 414 MB of Q4 weights / ~144 GB/s GDDR6 => several "
                "hundred tok/s ceiling. Phase 0 measures it."
            ),
        ),
        Tier(
            key="B",
            name="Raspberry Pi 5 8 GB (Cortex-A76 x4)",
            ram_mb=8192, cores=4, core_class="Cortex-A76",
            measured_prefill_tps=round(prefill_a76, 1),
            measured_decode_tps=decode,
            measured_peak_mb=peak,
            provenance="[M] Arm AI Portal llama-bench, Qwen3-0.6B Q4_K_M.",
            latency_status=STATUS_DERIVED,
        ),
        Tier(
            key="C",
            name="~2 GB / 2x Cortex-A5x (case-study envelope)",
            ram_mb=2048, cores=2, core_class="Cortex-A53/A55",
            measured_prefill_tps=round(prefill_a76 * a53_factor_c, 1),
            measured_decode_tps=round(decode * a53_factor_m, 1),
            measured_peak_mb=peak,  # memory is not scaled, only speed
            provenance=(
                f"[D] RPi5 numbers scaled by PassMark ST ratio "
                f"{a53_st}/{a76_st}={a53_st/a76_st:.3f} and core ratio 2/4 for "
                f"prefill. A55 variant: prefill "
                f"{prefill_a76*a55_factor_c:.1f} tok/s, decode "
                f"{decode*a55_factor_m:.1f} tok/s. UNVALIDATED - Phase 0 must "
                f"confirm on an ARM VM or board before this is quoted publicly."
            ),
            latency_status=STATUS_DERIVED,
        ),
    ]


# ---------------------------------------------------------------------------
# 3. Latency budget
# ---------------------------------------------------------------------------
# Path definitions live in tools/paths.py (single source of truth). This
# section only turns them into wall-clock times per hardware tier, and labels
# every result with how much we actually know about it.

def latency_ms(
    path: PathSpec, tier: Tier, prefix_cache: bool = True
) -> tuple[float | None, str]:
    """Wall-clock latency for a path on a tier, plus its evidence status.

    Important: NO figure returned here is an end-to-end measurement of our
    system, because our system has not been built. The best case is [D]: our
    path shape applied to somebody else's measured throughput on similar
    hardware. Tier A is [T] because nobody has measured this model on this GPU.
    """
    if not path.needs_llm:
        # Pure software: parse + SQL + template. No model involved, so the
        # estimate does not depend on unmeasured hardware. Still [D].
        return path.pipeline_ms, STATUS_DERIVED

    if tier.measured_prefill_tps is None or tier.measured_decode_tps is None:
        return None, STATUS_TARGET

    prefill_tokens = (
        path.uncached_prefill_tokens if prefix_cache else path.prompt_tokens
    )
    prefill_s = prefill_tokens / tier.measured_prefill_tps
    decode_s = path.output_tokens / tier.measured_decode_tps
    return path.pipeline_ms + (prefill_s + decode_s) * 1000.0, tier.latency_status


# ---------------------------------------------------------------------------
# 4. Storage model
# ---------------------------------------------------------------------------

@dataclass
class StoragePlan:
    horizon_days: int = 28
    hot_signals: int = 15          # sampled at 1 Hz for the whole horizon
    slow_signals: int = 30         # stored on-change, deadband compressed
    burst_signals: int = 25        # IMU / wheel speeds / currents, event windows only
    burst_window_s: int = 60       # context window persisted per event
    burst_rate_hz: float = 10.0    # decimated on persist; full rate stays in L0 only
    burst_events_per_day: int = 25
    hot_bytes_per_value: float = 1.2  # int16 + delta + zstd, empirically ~1-2 B
    rollup_bytes_per_value: float = 1.2
    raw_ring_minutes: int = 15
    raw_rate_hz: float = 50.0
    cards_per_day: int = 180
    card_bytes: int = 320
    embedding_dim: int = 384
    embedding_bytes: int = 1       # int8 quantised


def storage_mb(plan: StoragePlan) -> dict[str, float]:
    d = plan.horizon_days
    out: dict[str, float] = {}

    # L0: raw ring buffer, RAM only, newest N minutes at full rate
    out["L0_raw_ring_MB"] = (
        plan.hot_signals * plan.raw_rate_hz * 60 * plan.raw_ring_minutes * 4 / 2**20)

    # L1: 1 Hz hot signals
    out["L1_hot_1Hz_MB"] = (
        plan.hot_signals * 86400 * d * plan.hot_bytes_per_value / 2**20)

    # L1b: slow signals, deadbanded - assume ~1 sample / 30 s sustained
    out["L1b_slow_onchange_MB"] = (
        plan.slow_signals * (86400 / 30) * d * plan.hot_bytes_per_value / 2**20)

    # L1c: burst signals, only kept in event context windows
    out["L1c_burst_eventwindows_MB"] = (
        plan.burst_signals * plan.burst_window_s * plan.burst_rate_hz
        * plan.burst_events_per_day * d * 4 / 2**20)

    # L2: 1-minute rollups (mean/min/max/std/count)
    n_min = 60 * 24 * d
    out["L2_rollups_MB"] = (
        5 * plan.hot_signals * n_min * plan.rollup_bytes_per_value / 2**20)

    # L3: episode cards (text + int8 embeddings + FTS5 index ~1.6x text)
    n_cards = plan.cards_per_day * d
    out["L3_cards_text_MB"] = n_cards * plan.card_bytes * 1.6 / 2**20
    out["L3_embeddings_MB"] = (
        n_cards * plan.embedding_dim * plan.embedding_bytes / 2**20)

    # SQLite overhead (page slack, indexes, WAL)
    disk = sum(v for k, v in out.items() if not k.startswith("L0"))
    out["overhead_MB"] = disk * 0.12

    out["TOTAL_DISK_MB"] = disk + out["overhead_MB"]
    out["TOTAL_RAM_STEADY_MB"] = (
        out["L0_raw_ring_MB"] + out["L3_embeddings_MB"] + 60)  # + interpreter
    return {k: round(v, 2) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 4b. Bandwidth cross-check
# ---------------------------------------------------------------------------
# CPU decode of a dense transformer is memory-bandwidth bound: every generated
# token must stream the whole weight matrix once. So
#       effective_bandwidth ≈ weights_bytes × tokens_per_second
# If that number lands somewhere plausible for the memory technology, the
# measured decode figure is trustworthy and we can project other model sizes.

MODEL_SIZES_Q4_MB = {
    "SmolLM2-135M": 90.0,     # [R] approximate GGUF Q4 size
    "Qwen3-0.6B": 414.3,      # [M] exact, Arm AI Portal
    "Qwen3-1.7B": 1120.0,     # [R] approximate
    "Phi-4-mini-3.8B": 2700.0,  # [R] approximate
}


def bandwidth_crosscheck() -> dict:
    w = ANCHORS["rpi5_qwen3_0_6b_weights_mb"].value
    tps = ANCHORS["rpi5_qwen3_0_6b_decode_tps"].value
    eff_gbs = (w * 1e6) * tps / 1e9
    # Raspberry Pi 5 uses LPDDR4X-4267: 4 channels x 16-bit x 4267 MT/s
    # ≈ 17.1 GB/s theoretical, typically 55-65% achievable for a single core.
    proj = {
        name: round(eff_gbs * 1e9 / (mb * 1e6), 1)
        for name, mb in MODEL_SIZES_Q4_MB.items()
    }
    # Prefill is compute-bound and scales ~inversely with parameter count.
    base_prefill = 512 / (ANCHORS["rpi5_qwen3_0_6b_ttft_s"].value - 0.15)
    prefill_proj = {
        name: round(base_prefill * (414.3 / mb), 1)
        for name, mb in MODEL_SIZES_Q4_MB.items()
    }
    return {
        "projected_prefill_tps_on_pi5": prefill_proj,
        "implied_effective_bandwidth_GBs": round(eff_gbs, 2),
        "pi5_lpddr4x_4267_theoretical_GBs": 17.1,
        "achievable_fraction": round(eff_gbs / 17.1, 3),
        "projected_decode_tps_on_pi5": proj,
        "caveat": (
            "Projections assume pure bandwidth-boundedness. Real decode adds "
            "attention and sampling overhead, so smaller models will come in "
            "below these ceilings. Treat as upper bounds, [D] not [M]."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------

def model_size_sweep(tier: Tier, path: PathSpec) -> dict[str, dict]:
    """Scale every candidate model onto a tier and evaluate one query path.

    Used to answer the Tier C question: is there ANY model that fits P1 in 2 s?
    Always [D] - it stacks two extrapolations (params -> throughput, Pi 5 ->
    constrained tier) on top of one [M] anchor.
    """
    bc = bandwidth_crosscheck()
    ref_pf = bc["projected_prefill_tps_on_pi5"]["Qwen3-0.6B"]
    ref_dc = bc["projected_decode_tps_on_pi5"]["Qwen3-0.6B"]
    pf_factor = (tier.measured_prefill_tps or 0) / ref_pf
    dc_factor = (tier.measured_decode_tps or 0) / ref_dc

    out: dict[str, dict] = {}
    for name in MODEL_SIZES_Q4_MB:
        pf = bc["projected_prefill_tps_on_pi5"][name] * pf_factor
        dc = bc["projected_decode_tps_on_pi5"][name] * dc_factor
        total = (path.pipeline_ms
                 + path.uncached_prefill_tokens / pf * 1000
                 + path.output_tokens / dc * 1000)
        out[name] = {
            "prefill_tps": round(pf, 1),
            "decode_tps": round(dc, 1),
            "latency_ms": round(total),
            "within_target": total <= TARGET_LATENCY_MS,
            "status": STATUS_DERIVED,
        }
    return out


def decision_boundary(tier: Tier, path: PathSpec) -> dict:
    """Where the time actually goes, and what a model must do to meet target.

    Written because "Tier C takes 8 s" is not actionable: it does not say which
    half to attack. Splitting prefill from decode shows that on a constrained
    tier the *decode* side dominates, so shrinking the prompt cannot rescue the
    path - only a smaller or faster-decoding model can.

    The two "necessary conditions" are the useful output: they are thresholds a
    candidate model must clear, independent of every other assumption. A model
    that misses either one cannot meet the target on this tier no matter how
    good the rest of the system is.
    """
    if not path.needs_llm:
        return {"needs_llm": False, "pipeline_ms": path.pipeline_ms,
                "status": STATUS_DERIVED}

    if tier.measured_prefill_tps is None or tier.measured_decode_tps is None:
        return {"needs_llm": True, "status": STATUS_TARGET}

    prefill_ms = path.uncached_prefill_tokens / tier.measured_prefill_tps * 1000.0
    decode_ms = path.output_tokens / tier.measured_decode_tps * 1000.0
    total_ms = path.pipeline_ms + prefill_ms + decode_ms
    llm_budget_ms = TARGET_LATENCY_MS - path.pipeline_ms

    return {
        "needs_llm": True,
        "path": path.key,
        "prefill_ms": round(prefill_ms),
        "decode_ms": round(decode_ms),
        "pipeline_ms": path.pipeline_ms,
        "total_ms": round(total_ms),
        "prefill_share": round(prefill_ms / total_ms, 3),
        "decode_share": round(decode_ms / total_ms, 3),
        "llm_budget_ms": round(llm_budget_ms),
        "min_prefill_tps_if_decode_free": round(
            path.uncached_prefill_tokens / (llm_budget_ms / 1000.0), 1),
        "min_decode_tps_if_prefill_free": round(
            path.output_tokens / (llm_budget_ms / 1000.0), 1),
        "status": tier.latency_status,
    }


def report() -> dict:
    tiers = derive_tiers()
    plan = StoragePlan()

    latency: dict[str, dict[str, dict]] = {}
    for t in tiers:
        latency[t.key] = {}
        for p in PATHS:
            for cache in ((True,) if not p.needs_llm else (False, True)):
                tag = p.key + ("+cache" if cache else "")
                v, status = latency_ms(p, t, prefix_cache=cache)
                latency[t.key][tag] = {
                    "ms": None if v is None else round(v),
                    "status": status,
                }

    verdicts = {}
    for t in tiers:
        within, over, unknown = [], [], []
        for p in PATHS:
            cell = latency[t.key].get(p.key + "+cache")
            if cell["ms"] is None:
                unknown.append(p.key)
            elif cell["ms"] <= TARGET_LATENCY_MS:
                within.append({"path": p.key, "status": cell["status"]})
            else:
                over.append({"path": p.key, "status": cell["status"]})
        verdicts[t.key] = {"within_target": within, "over_target": over,
                           "not_measured": unknown}

    tc_tier = next(t for t in tiers if t.key == "C")
    return {
        "paths": {p.key: {"spec": p.spec, "prompt": p.prompt_tokens,
                          "output": p.output_tokens,
                          "pipeline_ms": p.pipeline_ms,
                          "reprefill": p.uncached_prefill_tokens,
                          "needs_llm": p.needs_llm} for p in PATHS},
        "cached_prefix_tokens": CACHED_PREFIX_TOKENS,
        "target_latency_ms": TARGET_LATENCY_MS,
        "anchors": {k: asdict(v) for k, v in ANCHORS.items()},
        "tiers": [asdict(t) for t in tiers],
        "latency": latency,
        "verdicts": verdicts,
        "storage": storage_mb(plan),
        "storage_plan": asdict(plan),
        "bandwidth_crosscheck": bandwidth_crosscheck(),
        "tier_c_model_sweep": model_size_sweep(tc_tier, BY_KEY["P1"]),
        "tier_c_decision_boundary": decision_boundary(tc_tier, BY_KEY["P1"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--specs", action="store_true",
                    help="print the canonical path spec table (markdown)")
    args = ap.parse_args()

    if args.specs:
        from paths import spec_table
        print(spec_table())
        return

    r = report()
    if args.json:
        print(json.dumps(r, indent=2))
        return

    W = 78
    print("=" * W)
    print("VehicleGPT feasibility model".center(W))
    print("=" * W)

    print("\n-- Canonical query paths (tools/paths.py) " + "-" * (W - 43))
    print(f"    target = {r['target_latency_ms']} ms | "
          f"cached prefix = {r['cached_prefix_tokens']} tok")
    print(f"    {'path':<5} {'spec':>20} {'re-prefill':>11} {'pipeline':>9}  LLM")
    for k, v in r["paths"].items():
        print(f"    {k:<5} {v['spec']:>20} {v['reprefill']:>9} tok "
              f"{v['pipeline_ms']:>7.0f} ms  {'yes' if v['needs_llm'] else 'no'}")

    print("\n-- Deployment tiers " + "-" * (W - 19))
    for t in r["tiers"]:
        print(f"\n  Tier {t['key']}: {t['name']}  [{t['latency_status']}]")
        print(f"    RAM envelope : {t['ram_mb']} MB   cores: {t['cores']} "
              f"({t['core_class']})")
        if t["measured_prefill_tps"] is None:
            print("    prefill/decode: [T] unmeasured on this hardware")
        else:
            print(f"    prefill      : {t['measured_prefill_tps']} tok/s")
            print(f"    decode       : {t['measured_decode_tps']} tok/s")
            print(f"    peak model   : {t['measured_peak_mb']} MB")
        print(f"    provenance   : {t['provenance']}")

    print("\n-- Latency by query path (ms, prefix-cached) " + "-" * (W - 48))
    print("    no figure below is an end-to-end measurement of our system;")
    print("    see the status column. [D]=derived from [M] anchors, [T]=target.")
    print(f"\n    {'path':>5} " + "".join(f"{'Tier ' + t['key']:>15}"
                                          for t in r["tiers"]))
    for p in PATHS:
        row = f"    {p.key:>5} "
        for t in r["tiers"]:
            cell = r["latency"][t["key"]][p.key + "+cache"]
            txt = "n/a" if cell["ms"] is None else str(cell["ms"])
            row += f"{(txt + ' ' + cell['status']):>15}"
        print(row)

    print(f"\n-- {r['target_latency_ms']} ms verdict " + "-" * (W - 22))
    for k, v in r["verdicts"].items():
        parts = [f"{x['path']} {x['status']}" for x in v["within_target"]]
        print(f"    Tier {k}: within target -> {', '.join(parts) or 'none'}")
        parts = [f"{x['path']} {x['status']}" for x in v["over_target"]]
        print(f"    {'':8} over target   -> {', '.join(parts) or 'none'}")
        if v["not_measured"]:
            print(f"    {'':8} unmeasured    -> {', '.join(v['not_measured'])}")

    print("\n-- Storage, "
          f"{r['storage_plan']['horizon_days']}-day horizon " + "-" * (W - 26))
    for k, v in r["storage"].items():
        print(f"    {k:28s} {v:>10.2f} MB")
    print("\n    Note: L0 lives in RAM; everything else on disk.")

    bc = r["bandwidth_crosscheck"]
    print("\n-- Bandwidth cross-check " + "-" * (W - 26))
    print(f"    Implied effective bandwidth : {bc['implied_effective_bandwidth_GBs']} GB/s")
    print(f"    Pi 5 LPDDR4X-4267 (theoretical): "
          f"{bc['pi5_lpddr4x_4267_theoretical_GBs']} GB/s")
    print(f"    Achievable fraction         : {bc['achievable_fraction']*100:.1f}%")
    print("\n    Projected decode ceiling on a Pi 5 ([D], upper bounds):")
    for k, v in bc["projected_decode_tps_on_pi5"].items():
        print(f"      {k:22s} {v:>7} tok/s")
    print("\n    Projected prefill ([D], scales ~1/params):")
    for k, v in bc["projected_prefill_tps_on_pi5"].items():
        print(f"      {k:22s} {v:>7} tok/s")

    db = r["tier_c_decision_boundary"]
    print("\n-- Where the Tier C time goes " + "-" * (W - 31))
    if db.get("needs_llm") and "total_ms" in db:
        print(f"    path {db['path']} on Tier C, best available model")
        print(f"      prefill  {db['prefill_ms']:>6} ms  {db['prefill_share']:>6.1%}")
        print(f"      decode   {db['decode_ms']:>6} ms  {db['decode_share']:>6.1%}")
        print(f"      pipeline {db['pipeline_ms']:>6} ms")
        print(f"      TOTAL    {db['total_ms']:>6} ms   {db['status']}")
        print("\n    A candidate model must clear BOTH of these to meet the target")
        print("    on this tier, regardless of anything else in the system:")
        print(f"      prefill >= {db['min_prefill_tps_if_decode_free']:>5} tok/s "
              "(even if decode were free)")
        print(f"      decode  >= {db['min_decode_tps_if_prefill_free']:>5} tok/s "
              "(even if prefill were free)")
        print("\n    Decode dominates, so trimming the prompt cannot rescue this")
        print("    path - only a faster-decoding model, or a path that skips the")
        print("    model, can. See docs/10-tierc-latency-strategy.md.")
    else:
        print("    [T] unavailable until Tier C throughput is measured")

    p1 = r["paths"]["P1"]
    print(f"\n-- Tier C model sweep, path P1 ({p1['spec']}) " + "-" * (W - 40))
    print("    Pure extrapolation on extrapolation. [D] only, never a benchmark.")
    print(f"      {'model':22s} {'prefill':>9} {'decode':>8} {'P1 total':>10}  verdict")
    for name, v in r["tier_c_model_sweep"].items():
        print(f"      {name:22s} {v['prefill_tps']:>7.1f} t/s "
              f"{v['decode_tps']:>6.1f} t/s {v['latency_ms']:>8} ms  "
              f"{'under target' if v['within_target'] else 'over target'}")
    print("\n      Only a sub-200M model has a chance here. Phase 0 task 0.5 decides.")

    print(f"\n    {bc['caveat']}")


if __name__ == "__main__":
    main()
