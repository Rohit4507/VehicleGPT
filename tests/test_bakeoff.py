"""Bake-off harness tests.

The harness is the thing that replaces a PassMark extrapolation with a
measurement, so it has to be right before anyone trusts its output. These tests
run it against captured llama-bench output and a deterministic mock backend, so
the parsing, scoring and verdict logic are exercised without downloading a
model.

The lesson that motivated this file: a linter that had never been fed the fault
it was meant to catch silently passed a deliberate 100/10 drift. A benchmark
harness that has never produced a number is in the same position.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml", reason="PyYAML not installed")

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "phase0"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import bakeoff  # noqa: E402
from paths import BY_KEY, PATHS, TARGET_LATENCY_MS  # noqa: E402

# A faithful reduction of `llama-bench -o json` output: one row per repetition,
# pp tests reporting prompt tokens/s and tg tests reporting generated tokens/s.
LLAMA_BENCH_JSON = """[
  {"build_commit":"b6000","build_number":6000,"cuda":false,"gpu_info":"",
   "model_filename":"SmolLM2-135M-Instruct-Q4_K_M.gguf","model_size":103156000,
   "model_n_params":135000000,"n_batch":2048,"n_threads":4,"test":"pp140",
   "n_prompt":140,"n_gen":0,"avg_ns":225806451,"stddev_ns":1200000,
   "avg_ts":619.99,"stddev_ts":3.1},
  {"test":"pp140","n_prompt":140,"n_gen":0,"avg_ts":615.0,"stddev_ts":2.0},
  {"test":"tg28","n_prompt":0,"n_gen":28,"avg_ts":239.5,"stddev_ts":1.4},
  {"test":"tg28","n_prompt":0,"n_gen":28,"avg_ts":241.0,"stddev_ts":1.9}
]"""


class TestParsing:
    def test_extracts_both_rates(self):
        rates = bakeoff.parse_llama_bench_json(LLAMA_BENCH_JSON)
        assert rates["prefill_tps"] == pytest.approx(617.495)  # median of 619.99, 615
        assert rates["decode_tps"] == pytest.approx(240.25)   # median of 239.5, 241.0

    def test_never_confuses_prefill_with_decode(self):
        """Mixing these up understates latency by an order of magnitude."""
        rates = bakeoff.parse_llama_bench_json(LLAMA_BENCH_JSON)
        assert rates["prefill_tps"] > rates["decode_tps"]
        assert rates["prefill_tps"] != rates["decode_tps"]

    def test_rejects_non_json(self):
        with pytest.raises(bakeoff.BenchParseError, match="not JSON"):
            bakeoff.parse_llama_bench_json("| model | test | t/s |")

    def test_rejects_empty(self):
        with pytest.raises(bakeoff.BenchParseError, match="no output"):
            bakeoff.parse_llama_bench_json("   ")

    def test_rejects_missing_generation_row(self):
        pp_only = '[{"test":"pp140","avg_ts":600.0}]'
        with pytest.raises(bakeoff.BenchParseError, match="both a pp"):
            bakeoff.parse_llama_bench_json(pp_only)

    def test_accepts_single_object(self):
        rates = bakeoff.parse_llama_bench_json(
            '[{"test":"pp140","avg_ts":600.0},{"test":"tg28","avg_ts":240.0}]')
        assert rates == {"prefill_tps": 600.0, "decode_tps": 240.0}

    def test_median_is_robust_to_a_slow_repeat(self):
        text = ('[{"test":"pp140","avg_ts":600},{"test":"pp140","avg_ts":610},'
                '{"test":"pp140","avg_ts":90},'
                '{"test":"tg28","avg_ts":240},{"test":"tg28","avg_ts":242},'
                '{"test":"tg28","avg_ts":30}]')
        rates = bakeoff.parse_llama_bench_json(text)
        assert rates["prefill_tps"] == 600
        assert rates["decode_tps"] == 240


class TestScoring:
    def tp(self, prefill=600.0, decode=240.0, status="[M]"):
        return bakeoff.Throughput(model="m", prefill_tps=prefill,
                                  decode_tps=decode, status=status)

    def test_arithmetic_matches_hand_calculation(self):
        """P1 = 140 in (90 cached, 50 re-prefilled) / 28 out, 130 ms pipeline."""
        row = bakeoff.score_path(BY_KEY["P1"], self.tp())
        assert row.prefill_ms == pytest.approx(50 / 600 * 1000, abs=0.1)
        assert row.decode_ms == pytest.approx(28 / 240 * 1000, abs=0.1)
        assert row.total_ms == pytest.approx(130 + 83.3 + 116.7, abs=0.2)

    def test_p0_ignores_model_speed_entirely(self):
        slow = bakeoff.score_path(BY_KEY["P0"], self.tp(prefill=1.0, decode=0.2))
        fast = bakeoff.score_path(BY_KEY["P0"], self.tp(prefill=9999, decode=9999))
        assert slow.total_ms == fast.total_ms == BY_KEY["P0"].pipeline_ms
        assert slow.prefill_ms == slow.decode_ms == 0.0

    def test_verdict_boundary_is_inclusive(self):
        """Equal to the target counts as meeting it; one ms over does not."""
        spec = BY_KEY["P1"]
        budget = TARGET_LATENCY_MS - spec.pipeline_ms
        # Solve for a decode rate that lands exactly on the target.
        prefill_ms = spec.uncached_prefill_tokens / 1000.0 * 1000  # 1000 t/s
        decode_s = (budget - prefill_ms) / 1000
        exact_tps = spec.output_tokens / decode_s
        at = bakeoff.score_path(spec, self.tp(prefill=1000.0, decode=exact_tps))
        over = bakeoff.score_path(spec, self.tp(prefill=1000.0, decode=exact_tps * 0.98))
        assert at.verdict == "within"
        assert over.verdict == "over"

    def test_faster_model_never_slower(self):
        fast = bakeoff.score_path(BY_KEY["P2"], self.tp(1200, 500))
        slow = bakeoff.score_path(BY_KEY["P2"], self.tp(150, 55))
        assert fast.total_ms < slow.total_ms

    def test_zero_throughput_is_rejected_not_scored(self):
        """A broken measurement must not read as 'meets target'."""
        with pytest.raises(ValueError, match="positive throughput"):
            bakeoff.score_path(BY_KEY["P1"], self.tp(prefill=0.0))
        with pytest.raises(ValueError, match="positive throughput"):
            bakeoff.score_path(BY_KEY["P1"], self.tp(decode=0.0))


class TestRequiredRates:
    def test_matches_the_feasibility_model(self):
        """The harness and the feasibility model must agree on the thresholds.

        Both are derived from paths.py; if they drift, one of them is lying.
        """
        from feasibility_model import decision_boundary, derive_tiers

        tier_c = next(t for t in derive_tiers() if t.key == "C")
        db = decision_boundary(tier_c, BY_KEY["P1"])
        req = bakeoff.required_rates(BY_KEY["P1"])
        assert req["min_prefill_tps"] == pytest.approx(
            db["min_prefill_tps_if_decode_free"], abs=0.05)
        assert req["min_decode_tps"] == pytest.approx(
            db["min_decode_tps_if_prefill_free"], abs=0.05)

    def test_p0_is_not_an_llm_path(self):
        assert BY_KEY["P0"].needs_llm is False

    def test_monotonic_across_paths(self):
        """Bigger paths need more throughput, or paths.py is inconsistent."""
        p1, p2, p3 = (bakeoff.required_rates(BY_KEY[k]) for k in ("P1", "P2", "P3"))
        assert p1["min_decode_tps"] < p2["min_decode_tps"] < p3["min_decode_tps"]


class TestMockBackend:
    def test_is_deterministic(self):
        b = bakeoff.MockBackend()
        a1 = b.measure(Path("SmolLM2-135M.gguf"), 140, 28, 3, None)
        a2 = b.measure(Path("SmolLM2-135M.gguf"), 140, 28, 3, None)
        assert a1 == a2

    def test_is_never_labelled_as_measured(self):
        tp = bakeoff.MockBackend().measure(Path("Qwen3-0.6B.gguf"), 140, 28, 3, None)
        assert tp.status == bakeoff.STATUS_MOCK
        assert tp.status != bakeoff.STATUS_MEASURED

    def test_recognises_known_models(self):
        b = bakeoff.MockBackend()
        small = b.measure(Path("SmolLM2-135M-Q4.gguf"), 140, 28, 3, None)
        large = b.measure(Path("Qwen3-0.6B-Q4.gguf"), 140, 28, 3, None)
        assert small.decode_tps > large.decode_tps

    def test_unknown_model_falls_back(self):
        tp = bakeoff.MockBackend().measure(Path("mystery.gguf"), 140, 28, 3, None)
        assert tp.prefill_tps > 0 and tp.decode_tps > 0


class TestAccuracyScoring:
    def case(self, **kw):
        base = {"id": "t", "must_contain": [["pump"], ["14:08", "stall"]],
                "must_not_contain": ["brake"]}
        base.update(kw)
        return base

    def test_full_pass(self):
        r = bakeoff.score_answer("The pump stalled at 14:08.", self.case())
        assert r["passed"] and r["hits"] == 2

    def test_missing_fact_fails(self):
        r = bakeoff.score_answer("The pump stopped.", self.case())
        assert not r["passed"] and r["hits"] == 1

    def test_hallucination_fails_even_with_all_facts(self):
        r = bakeoff.score_answer("The pump stalled at 14:08; also a brake fault.",
                                 self.case())
        assert not r["passed"]
        assert r["banned_hit"] == ["brake"]

    def test_case_insensitive(self):
        r = bakeoff.score_answer("THE PUMP STALLED AT 14:08", self.case())
        assert r["passed"]

    def test_alternative_phrasings_accepted(self):
        r = bakeoff.score_answer("Pump duty hit zero.", self.case())
        assert r["hits"] == 1  # "pump" found, no timestamp alternative
        assert not r["passed"]


class TestEvalCases:
    def test_all_cases_have_required_fields(self):
        for case in bakeoff.load_eval_cases():
            assert {"id", "query", "evidence", "must_contain"} <= set(case)
            assert case["must_contain"], f"{case['id']}: no facts to check"

    def test_ids_are_unique(self):
        ids = [c["id"] for c in bakeoff.load_eval_cases()]
        assert len(ids) == len(set(ids))

    def test_prompt_includes_evidence_and_question(self):
        case = bakeoff.load_eval_cases()[0]
        prompt = bakeoff.build_prompt(case)
        assert case["query"] in prompt
        assert "EVIDENCE:" in prompt

    def test_cases_cover_the_brief_examples(self):
        queries = " ".join(c["query"].lower() for c in bakeoff.load_eval_cases())
        assert "adas disengagement" in queries
        assert "range" in queries
        assert "efficiency" in queries
        assert "overheating" in queries
        assert "city and highway" in queries


class TestEndToEndMock:
    def test_full_run_produces_rows_for_every_path(self):
        result = bakeoff.run_bakeoff(
            [Path("SmolLM2-135M.gguf")], bakeoff.MockBackend(), reps=1,
            threads=None, do_eval=True)
        model = result["models"][0]
        assert len(model["rows"]) == len(PATHS)
        assert {r["path"] for r in model["rows"]} == {p.key for p in PATHS}
        assert model["accuracy"]["cases"] == len(bakeoff.load_eval_cases())

    def test_result_records_the_machine(self):
        result = bakeoff.run_bakeoff(
            [Path("m.gguf")], bakeoff.MockBackend(), 1, None, False)
        assert result["machine"]["machine"]
        assert isinstance(result["is_arm"], bool)

    def test_json_serialisable(self):
        import json

        result = bakeoff.run_bakeoff(
            [Path("m.gguf")], bakeoff.MockBackend(), 1, None, False)
        json.dumps(result)  # raises if any field is not serialisable
