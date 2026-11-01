#!/usr/bin/env python3
"""Model bake-off harness: throughput, end-to-end latency, and accuracy.

Why this exists
---------------
The feasibility model projects Tier C latency by scaling a published Raspberry
Pi 5 measurement by a PassMark single-thread ratio. That is an extrapolation,
and no amount of care makes it a benchmark. This harness replaces it with a
measurement - on whatever machine you run it on.

What it measures, and what it therefore proves
----------------------------------------------
``llama-bench`` reports *tokens per second* for prompt processing and for
generation. Those are [M] measurements of the machine it ran on. The end-to-end
latency for each query path is then assembled from them plus the assumed
pipeline cost, so it stays [D]: arithmetic over measurements, not a measurement
of VehicleGPT, which does not exist yet.

The distinction matters. A run on an RTX 3050 laptop does not tell you Tier C
performance; it tells you laptop performance. The harness stamps every result
with the machine it ran on for exactly that reason.

Usage
-----
    # discover what this machine actually does
    python3 tools/phase0/bakeoff.py --models ~/models/SmolLM2-135M*.gguf

    # benchmark against the canonical query paths, emit JSON for CI
    python3 tools/phase0/bakeoff.py --models ~/models/*.gguf --json > bakeoff.json

    # score answer quality as well as speed
    python3 tools/phase0/bakeoff.py --models ~/models/*.gguf --eval

    # prove the harness works without downloading anything
    python3 tools/phase0/bakeoff.py --backend mock --allow-mock

Requires ``llama-bench`` and ``llama-cli`` on PATH (or pass --llama-bench /
--llama-cli). Get them from https://github.com/ggml-org/llama.cpp.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from paths import BY_KEY, CACHED_PREFIX_TOKENS, PATHS, TARGET_LATENCY_MS, PathSpec  # noqa: E402

STATUS_MEASURED = "[M]"
STATUS_DERIVED = "[D]"
STATUS_MOCK = "[MOCK]"


# ---------------------------------------------------------------------------
# Machine identity - every number is meaningless without it
# ---------------------------------------------------------------------------

def machine_info() -> dict:
    cpu = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def is_arm() -> bool:
    return platform.machine().lower() in {"aarch64", "arm64", "armv7l", "armv8l"}


# ---------------------------------------------------------------------------
# llama-bench parsing
# ---------------------------------------------------------------------------

class BenchParseError(RuntimeError):
    pass


def parse_llama_bench_json(text: str) -> dict[str, float]:
    """Extract prompt-processing and text-generation throughput.

    ``llama-bench -o json`` returns an array with one object per test. Prompt
    tests are named ``pp<N>`` and report *prompt* tokens/s in ``avg_ts``;
    generation tests are ``tg<N>`` and report *generated* tokens/s. Mixing the
    two up is the single easiest way to produce a wrong latency figure, so both
    are returned separately and never averaged.
    """
    text = text.strip()
    if not text:
        raise BenchParseError("llama-bench produced no output")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchParseError(
            f"llama-bench output was not JSON ({exc}). Run it with -o json."
        ) from exc

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise BenchParseError(f"unexpected llama-bench payload: {type(data)}")

    pp, tg = [], []
    for entry in data:
        test = str(entry.get("test", ""))
        avg_ts = entry.get("avg_ts")
        if avg_ts is None:
            continue
        if test.startswith("pp"):
            pp.append(float(avg_ts))
        elif test.startswith("tg"):
            tg.append(float(avg_ts))

    if not pp or not tg:
        raise BenchParseError(
            f"expected both a pp* and a tg* entry, got pp={pp} tg={tg}"
        )
    return {"prefill_tps": media(pp), "decode_tps": media(tg)}


def media(values: list[float]) -> float:
    """Median. llama-bench already reports a mean per test; with -r > 1 we get
    one row per repetition, and the median is the robust summary."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


# ---------------------------------------------------------------------------
# Results and scoring
# ---------------------------------------------------------------------------

@dataclass
class Throughput:
    model: str
    prefill_tps: float
    decode_tps: float
    status: str = STATUS_MEASURED
    backend: str = "llama-bench"
    params_b: float | None = None
    gguf_mb: float | None = None
    n_threads: int | None = None


@dataclass
class PathRow:
    model: str
    path: str
    spec: str
    prefill_ms: float
    decode_ms: float
    pipeline_ms: float
    total_ms: float
    target_ms: int
    verdict: str
    status: str


@dataclass
class ModelReport:
    model: str
    throughput: Throughput
    rows: list[PathRow] = field(default_factory=list)
    accuracy: dict | None = None

    @property
    def meets_target_everywhere(self) -> bool:
        return all(r.verdict == "within" for r in self.rows if r.verdict != "n/a")


def score_path(spec: PathSpec, tp: Throughput,
               target_ms: int = TARGET_LATENCY_MS) -> PathRow:
    """Assemble end-to-end latency for one path from measured throughput."""
    if not spec.needs_llm:
        return PathRow(
            model=tp.model, path=spec.key, spec=spec.spec,
            prefill_ms=0.0, decode_ms=0.0, pipeline_ms=spec.pipeline_ms,
            total_ms=spec.pipeline_ms, target_ms=target_ms,
            verdict="within" if spec.pipeline_ms <= target_ms else "over",
            status=STATUS_DERIVED,
        )

    if tp.prefill_tps <= 0 or tp.decode_tps <= 0:
        raise ValueError(
            f"{tp.model!r}: needs positive throughput to score "
            f"(prefill={tp.prefill_tps}, decode={tp.decode_tps})")
    prefill_ms = spec.uncached_prefill_tokens / tp.prefill_tps * 1000.0
    decode_ms = spec.output_tokens / tp.decode_tps * 1000.0
    total = spec.pipeline_ms + prefill_ms + decode_ms
    return PathRow(
        model=tp.model, path=spec.key, spec=spec.spec,
        prefill_ms=round(prefill_ms, 1), decode_ms=round(decode_ms, 1),
        pipeline_ms=spec.pipeline_ms, total_ms=round(total, 1),
        target_ms=target_ms,
        verdict="within" if total <= target_ms else "over",
        status=STATUS_DERIVED,
    )


def required_rates(spec: PathSpec, target_ms: int = TARGET_LATENCY_MS) -> dict:
    """Thresholds a candidate model must clear to meet the target on this path.

    Depends only on the path shape and the target, so it is the one part of a
    bake-off that is valid before any model has been benchmarked.
    """
    budget = target_ms - spec.pipeline_ms
    return {
        "path": spec.key,
        "spec": spec.spec,
        "llm_budget_ms": round(budget, 1),
        "min_prefill_tps": round(spec.uncached_prefill_tokens / (budget / 1000.0), 1),
        "min_decode_tps": round(spec.output_tokens / (budget / 1000.0), 1),
    }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class Backend:
    name = "base"

    def measure(self, gguf: Path, prompt_tokens: int, gen_tokens: int,
                reps: int, threads: int | None) -> Throughput:
        raise NotImplementedError

    def generate(self, gguf: Path, prompt: str, n_predict: int) -> str:
        raise NotImplementedError


class LlamaBenchBackend(Backend):
    """Real measurements via llama-bench."""

    name = "llama-bench"

    def __init__(self, bench_bin: str = "llama-bench", cli_bin: str = "llama-cli"):
        self.bench_bin = bench_bin
        self.cli_bin = cli_bin

    def _binary(self, name: str, path: str) -> str:
        found = shutil.which(path)
        if not found:
            raise FileNotFoundError(
                f"{name} not found ({path!r}). Build llama.cpp and put it on PATH, "
                "or pass --llama-bench/--llama-cli. "
                "See https://github.com/ggml-org/llama.cpp"
            )
        return found

    def measure(self, gguf, prompt_tokens, gen_tokens, reps, threads) -> Throughput:
        bench = self._binary("llama-bench", self.bench_bin)
        cmd = [bench, "-m", str(gguf), "-p", str(prompt_tokens),
               "-n", str(gen_tokens), "-r", str(reps), "-o", "json"]
        if threads:
            cmd += ["-t", str(threads)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise BenchParseError(
                f"llama-bench failed ({proc.returncode}): {proc.stderr[-800:]}")
        rates = parse_llama_bench_json(proc.stdout)
        return Throughput(
            model=Path(gguf).stem,
            prefill_tps=round(rates["prefill_tps"], 2),
            decode_tps=round(rates["decode_tps"], 2),
            backend=self.name,
            gguf_mb=round(Path(gguf).stat().st_size / 1e6, 1),
            n_threads=threads,
        )

    def generate(self, gguf, prompt, n_predict) -> str:
        cli = self._binary("llama-cli", self.cli_bin)
        cmd = [cli, "-m", str(gguf), "-p", prompt, "-n", str(n_predict),
               "--temp", "0", "--no-display-prompt", "-no-cnv"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.stdout


class MockBackend(Backend):
    """Deterministic stand-in used to test the harness itself.

    A harness that has never been run is not evidence that the harness works.
    This backend lets the scoring, parsing and verdict logic be tested without
    downloading a model, and refuses to be mistaken for a measurement: every
    figure it produces carries ``[MOCK]``.
    """

    name = "mock"

    # (prefill_tps, decode_tps) on some arbitrary reference machine.
    RATES = {
        "SmolLM2-135M": (620.0, 240.0),
        "Qwen3-0.6B": (150.0, 55.0),
        "Qwen2.5-0.5B": (180.0, 65.0),
        "SmolLM2-360M": (260.0, 95.0),
    }
    DEFAULT = (150.0, 55.0)

    def measure(self, gguf, prompt_tokens, gen_tokens, reps, threads) -> Throughput:
        name = Path(gguf).stem or "unknown"
        key = next((k for k in self.RATES if k in name), None)
        pf, dc = self.RATES.get(key, self.DEFAULT)
        # Mild size-dependence so shorter prompts measure faster, as they would.
        scale = (prompt_tokens / 140.0) ** 0.15
        return Throughput(
            model=name, prefill_tps=round(pf / scale, 2), decode_tps=round(dc, 2),
            status=STATUS_MOCK, backend=self.name, gguf_mb=None, n_threads=threads,
        )

    def generate(self, gguf, prompt, n_predict) -> str:
        return "[mock response] the coolant pump stalled before the temperature rose"


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def load_eval_cases(path: Path | None = None) -> list[dict]:
    import yaml

    path = path or REPO / "tools" / "phase0" / "eval_queries.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["cases"]


def build_prompt(case: dict) -> str:
    """Grounded-summarisation prompt: the evidence is supplied, so this measures
    reasoning over given facts rather than recall."""
    return (
        "You are a vehicle assistant. Answer using ONLY the evidence below.\n"
        "Be brief and concrete.\n\n"
        f"EVIDENCE:\n{case['evidence'].strip()}\n\n"
        f"QUESTION: {case['query']}\n"
        "ANSWER:"
    )


def score_answer(text: str, case: dict) -> dict:
    """Keyword scoring. Crude, but reproducible and inspectable - which is what
    a Phase 0 screening decision needs. Anything finer belongs in Phase 3."""
    low = " ".join(text.lower().split())
    found = [alt for group in case.get("must_contain", [])
             for alt in ([group] if isinstance(group, str) else group)
             if alt.lower() in low]
    hits = sum(
        1 for group in case.get("must_contain", [])
        if any((a.lower() in low)
               for a in ([group] if isinstance(group, str) else group))
    )
    banned = [b for b in case.get("must_not_contain", []) if b.lower() in low]
    total = len(case.get("must_contain", [])) or 1
    return {
        "id": case["id"],
        "passed": hits == total and not banned,
        "hits": hits,
        "total": total,
        "matched": found[:6],
        "banned_hit": banned,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_bakeoff(models: list[Path], backend: Backend, reps: int,
                threads: int | None, do_eval: bool) -> dict:
    p1 = BY_KEY["P1"]
    reports: list[ModelReport] = []

    for gguf in models:
        tp = backend.measure(gguf, p1.prompt_tokens, p1.output_tokens, reps, threads)
        rep = ModelReport(model=tp.model, throughput=tp)
        rep.rows = [score_path(spec, tp) for spec in PATHS]
        if do_eval:
            cases = load_eval_cases()
            results = []
            for case in cases:
                out = backend.generate(gguf, build_prompt(case), 60)
                results.append(score_answer(out, case))
            rep.accuracy = {
                "cases": len(results),
                "passed": sum(r["passed"] for r in results),
                "detail": results,
            }
        reports.append(rep)

    return {
        "machine": machine_info(),
        "is_arm": is_arm(),
        "backend": backend.name,
        "cached_prefix_tokens": CACHED_PREFIX_TOKENS,
        "target_ms": TARGET_LATENCY_MS,
        "required_rates": [required_rates(spec) for spec in PATHS if spec.needs_llm],
        "models": [
            {
                "model": r.model,
                "throughput": asdict(r.throughput),
                "rows": [asdict(row) for row in r.rows],
                "accuracy": r.accuracy,
            }
            for r in reports
        ],
    }


def print_report(result: dict, allow_mock: bool) -> None:
    W = 92
    mock = result["backend"] == "mock"
    print("=" * W)
    print("VehicleGPT model bake-off".center(W))
    print("=" * W)
    if mock:
        banner = "MOCK BACKEND - THESE ARE SYNTHETIC NUMBERS, NOT MEASUREMENTS"
        print(f"\n{'!' * W}\n{banner.center(W)}\n{'!' * W}")

    m = result["machine"]
    print(f"\nmachine   {m['processor']} | {m['machine']} | {m['cpu_count']} threads")
    print(f"platform  {m['platform']}")
    if not result["is_arm"]:
        print("\nNOTE: this is not an ARM machine, so nothing here measures Tier C.")
        print("      Tier C figures must still be projected, or measured on ARM.")

    print("\n-- throughput required to meet the target, per path " + "-" * (W - 55))
    for r in result["required_rates"]:
        print(f"    {r['path']:<4} {r['spec']:>18}  budget {r['llm_budget_ms']:>6.0f} ms"
              f"  -> prefill >= {r['min_prefill_tps']:>6.1f} t/s, "
              f"decode >= {r['min_decode_tps']:>5.1f} t/s")

    for model in result["models"]:
        tp = model["throughput"]
        print(f"\n-- {model['model']} " + "-" * max(W - len(model["model"]) - 5, 4))
        print(f"    prefill {tp['prefill_tps']:>8.1f} tok/s   "
              f"decode {tp['decode_tps']:>7.1f} tok/s   "
              f"backend {tp['backend']}   status {tp['status']}")
        if tp.get("gguf_mb"):
            print(f"    model file {tp['gguf_mb']:>7.1f} MB   threads {tp['n_threads']}")
        print(f"\n    {'path':<5} {'spec':>18} {'prefill':>9} {'decode':>9} "
              f"{'total':>9}  {'verdict':<8}")
        for row in model["rows"]:
            mark = "within" if row["verdict"] == "within" else "OVER"
            print(f"    {row['path']:<5} {row['spec']:>18} "
                  f"{row['prefill_ms']:>8.0f}ms {row['decode_ms']:>8.0f}ms "
                  f"{row['total_ms']:>8.0f}ms  {mark:<8}")
        if model.get("accuracy"):
            acc = model["accuracy"]
            print(f"\n    accuracy {acc['passed']}/{acc['cases']} cases passed")
            for case in acc["detail"]:
                flag = "PASS" if case["passed"] else "FAIL"
                print(f"      {flag} {case['id']:<28} "
                      f"{case['hits']}/{case['total']} facts")
                if case["banned_hit"]:
                    print(f"           hallucinated: {case['banned_hit']}")

    print(f"\n{'-' * W}")
    print("Evidence: throughput is [M] on the machine above. End-to-end totals are")
    print("[D] - arithmetic over that throughput plus the assumed pipeline cost.")
    print("Neither is a measurement of VehicleGPT, which does not exist yet.")
    if mock and not allow_mock:
        print("\n(Mock backend used; pass --allow-mock to silence this.)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", type=Path, default=[],
                    help="GGUF files to benchmark")
    ap.add_argument("--backend", choices=["llama-bench", "mock"], default="llama-bench")
    ap.add_argument("--llama-bench", default="llama-bench")
    ap.add_argument("--llama-cli", default="llama-cli")
    ap.add_argument("--reps", type=int, default=3, help="llama-bench repetitions")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--eval", action="store_true", help="also score answer quality")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--allow-mock", action="store_true",
                    help="permit the mock backend without a warning")
    ap.add_argument("--required-only", action="store_true",
                    help="print only the throughput targets and exit")
    args = ap.parse_args()

    if args.required_only:
        print(json.dumps([required_rates(s) for s in PATHS if s.needs_llm], indent=2))
        return 0

    if args.backend == "mock":
        backend: Backend = MockBackend()
        if not args.models:
            args.models = [Path("SmolLM2-135M.gguf"), Path("Qwen3-0.6B.gguf"),
                           Path("Qwen2.5-0.5B.gguf")]
        if not (args.allow_mock or args.json):
            print("Using the MOCK backend: numbers will be synthetic.\n", file=sys.stderr)
    else:
        backend = LlamaBenchBackend(args.llama_bench, args.llama_cli)
        if not args.models:
            ap.error("--models is required with the llama-bench backend")

    missing = [m for m in args.models if not Path(m).exists()]
    if missing and args.backend != "mock":
        ap.error(f"model file(s) not found: {missing}")

    try:
        result = run_bakeoff(args.models, backend, args.reps, args.threads, args.eval)
    except (BenchParseError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result, args.allow_mock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
