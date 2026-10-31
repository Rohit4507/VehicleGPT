"""Guards the invariants that broke this repository once already.

Runs with the stdlib only, so CI needs no dependencies:

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from paths import BY_KEY, CACHED_PREFIX_TOKENS, PATHS, TARGET_LATENCY_MS  # noqa: E402


class TestPathSpecs(unittest.TestCase):
    def test_paths_are_unique_and_ordered(self):
        self.assertEqual([p.key for p in PATHS], ["P0", "P1", "P2", "P3"])

    def test_only_p0_is_model_free(self):
        self.assertFalse(BY_KEY["P0"].needs_llm)
        for key in ("P1", "P2", "P3"):
            self.assertTrue(BY_KEY[key].needs_llm, key)

    def test_prefix_cache_never_exceeds_prompt(self):
        for p in PATHS:
            self.assertGreaterEqual(p.prompt_tokens, 0, p.key)
            if p.needs_llm:
                self.assertLessEqual(CACHED_PREFIX_TOKENS, p.prompt_tokens, p.key)

    def test_reprefill_is_monotonic_in_prompt_size(self):
        llm = [p for p in PATHS if p.needs_llm]
        values = [p.uncached_prefill_tokens for p in llm]
        self.assertEqual(values, sorted(values))

    def test_spec_strings_match_tokens(self):
        for p in PATHS:
            self.assertEqual(
                p.spec, f"{p.prompt_tokens} in / {p.output_tokens} out", p.key)


class TestModelOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "feasibility_model.py"), "--json"],
            capture_output=True, text=True, check=True,
        )
        cls.data = json.loads(out.stdout)

    def test_paths_in_report_match_module(self):
        for p in PATHS:
            self.assertEqual(self.data["paths"][p.key]["spec"], p.spec, p.key)

    def test_target_and_prefix_echoed(self):
        self.assertEqual(self.data["target_latency_ms"], TARGET_LATENCY_MS)
        self.assertEqual(self.data["cached_prefix_tokens"], CACHED_PREFIX_TOKENS)

    def test_no_measured_status_on_any_path_latency(self):
        """The system is not built. Nothing may claim [M] end-to-end."""
        for tier, paths in self.data["latency"].items():
            for tag, cell in paths.items():
                self.assertNotEqual(
                    cell["status"], "[M]",
                    f"{tier}/{tag} claims measured status; the system does not exist",
                )

    def test_tier_a_is_unmeasured(self):
        for tag, cell in self.data["latency"]["A"].items():
            if tag.startswith(("P1", "P2", "P3")):
                self.assertEqual(cell["status"], "[T]", tag)

    def test_tier_c_meets_target_only_on_p0(self):
        within = [x["path"] for x in self.data["verdicts"]["C"]["within_target"]]
        self.assertEqual(within, ["P0"])

    def test_p0_is_tier_independent(self):
        values = {self.data["latency"][t]["P0+cache"]["ms"]
                  for t in self.data["latency"]}
        self.assertEqual(len(values), 1, f"P0 varies by tier: {values}")

    def test_every_tier_c_figure_is_derived_not_benchmarked(self):
        for tag, cell in self.data["latency"]["C"].items():
            self.assertIn(cell["status"], ("[D]", "[T]"), tag)


class TestDocLinter(unittest.TestCase):
    def test_docs_pass_the_linter(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "check_docs.py")],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_linter_detects_spec_drift(self):
        """The linter must catch the exact bug that shipped once already.

        An earlier draft had P1 at 100/10 in one table and 140/28 in another.
        If this test passes, the guard is genuinely working - not vacuously
        passing because it understands no table layout.
        """
        sys.path.insert(0, str(ROOT / "tools"))
        import check_docs

        check_docs.violations.clear()
        drifted = ("| **P1** | SLM intent-only | 100 | 10 | 90 | 50 "
                   "| 130 ms | yes |")
        check_docs.check_a_path_specs("synthetic.md", 1, drifted)
        self.assertEqual(len(check_docs.violations), 1,
                         "linter failed to flag a P1 spec drift")

        check_docs.violations.clear()
        correct = ("| **P1** | SLM intent-only | 140 | 28 | 90 | 50 "
                   "| 130 ms | yes |")
        check_docs.check_a_path_specs("synthetic.md", 1, correct)
        self.assertEqual(check_docs.violations, [],
                         "linter false-positived on a correct row")

    def test_linter_detects_unlabelled_tier_c_figure(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import check_docs

        check_docs.violations.clear()
        check_docs.check_b_tier_c_labelling(
            "synthetic.md", 1, "| P1 | Tier C | 8 073 ms |")
        self.assertEqual(len(check_docs.violations), 1,
                         "linter failed to flag an unlabelled Tier C latency")

        check_docs.violations.clear()
        check_docs.check_b_tier_c_labelling(
            "synthetic.md", 1, "| P1 | Tier C | 8 073 ms `[D]` |")
        self.assertEqual(check_docs.violations, [])


if __name__ == "__main__":
    unittest.main()
