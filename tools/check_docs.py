#!/usr/bin/env python3
"""
Documentation consistency linter.

Exists because an earlier draft of these docs:
  * defined P1 as "140 in / 28 out" in one file and "100 in / 10 out" in
    another, which moved a headline latency number by 15%;
  * stated "under 2 seconds" as a flat claim when only one path meets it on
    the constrained tier;
  * printed a PassMark-derived extrapolation in a table that read like a
    benchmark.

Prose drifts. This script makes it fail CI instead.

Checks
------
  A  Path specs. Any markdown table row for P0..P3 that states a token count
     must match tools/paths.py exactly.

  B  Tier C labelling. Any line quoting a latency in ms for Tier C must carry
     an explicit projection marker ([D], [T], modelled, extrapolated, projected,
     scaled, derived). Tier C has never been measured and must never read as if
     it has.

  C  No unqualified measurement claims. Lines claiming our pipeline was
     "measured", "demonstrated" or "benchmarked" on any tier must be negated or
     qualified, because the system does not exist yet.

  D  Target status. Any statement that a tier "meets" or "fits" the 2-second
     target must be qualified with the path (P0/P1/...) and must not claim
     end-to-end demonstration.

Usage
-----
    python3 tools/check_docs.py        # exit 0 = clean, 1 = violations

Stdlib only. Runs in CI on every push.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths import CACHED_PREFIX_TOKENS, PATHS, TARGET_LATENCY_MS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOC_GLOB = ["README.md", "docs/*.md"]

PROJECTION_MARKERS = (
    "[d]", "[t]", "modelled", "modeled", "extrapolat", "projected", "projection",
    "scaled", "derived", "estimate", "unvalidated", "not measured", "predicted",
)
NEGATIONS = ("not ", "never", "n't ", "no ", "unmeasured", "not yet", "cannot",
             "can't")

violations: list[str] = []


def files() -> list[Path]:
    out: list[Path] = []
    for pattern in DOC_GLOB:
        out.extend(sorted(ROOT.glob(pattern)))
    return out


BARE_INT = re.compile(r"^\s*\d+\s*$")


def _spec_rows(line: str) -> list[tuple[str, int, int]]:
    """Extract (path_key, prompt_tokens, output_tokens) from a markdown line.

    Handles every layout that has appeared in these docs, because the previous
    version only handled the first two and therefore silently passed a
    genuine drift:

      1. columnar table    | **P1** | desc | 140 | 28 | 90 | 50 | 130 ms | yes |
      2. slash form        | **P1** ... | 140 / 28 |
      3. prose form        P1: 100 in + 10 out
    """
    found: list[tuple[str, int, int]] = []

    # 1. columnar table: path in the first cell, then the first two bare-int cells
    if "|" in line:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        m = re.search(r"\b(P[0-3])\b", cells[0]) if cells else None
        if m:
            ints = [int(c) for c in cells if BARE_INT.match(c)]
            if len(ints) >= 2:
                found.append((m.group(1), ints[0], ints[1]))

    # 2 + 3. inline forms
    for pattern in (
        r"\*\*(P[0-3])\*\*[^|]*\|\s*(\d+)\s*/\s*(\d+)\s*\|",
        r"\b(P[0-3])\b[^|\n]*?(\d+)\s*in\s*[/+]\s*(\d+)\s*out",
    ):
        for m in re.finditer(pattern, line):
            entry = (m.group(1), int(m.group(2)), int(m.group(3)))
            if entry not in found:
                found.append(entry)
    return found


def check_a_path_specs(name: str, lineno: int, line: str) -> None:
    """Token specs for a query path must match tools/paths.py."""
    if not re.search(r"\bP[0-3]\b", line):
        return
    for key, pin, pout in _spec_rows(line):
        spec = next((p for p in PATHS if p.key == key), None)
        if spec is None:
            continue
        if (pin, pout) != (spec.prompt_tokens, spec.output_tokens):
            violations.append(
                f"{name}:{lineno} [A] {key} spec is {pin}/{pout} but "
                f"tools/paths.py says {spec.prompt_tokens}/"
                f"{spec.output_tokens} ({spec.spec})\n         {line.strip()[:120]}"
            )


def check_b_tier_c_labelling(name: str, lineno: int, line: str) -> None:
    """Tier C latency figures must be visibly projections."""
    if "tier c" not in line.lower():
        return
    if not re.search(r"\b\d[\d ]*\s*ms\b", line):
        return
    low = line.lower()
    if any(marker in low for marker in PROJECTION_MARKERS):
        return
    violations.append(
        f"{name}:{lineno} [B] Tier C latency quoted without a projection marker "
        f"([D]/[T]/modelled/extrapolated/projected). Tier C has never been "
        f"measured.\n         {line.strip()[:120]}"
    )


def check_c_unqualified_measurement(name: str, lineno: int, line: str) -> None:
    """We must never claim our own pipeline was measured."""
    low = line.lower()
    claims = ("we measured", "we have measured", "demonstrated on",
              "benchmarked on", "measured on tier", "our measured",
              "end-to-end measurement")
    if not any(c in low for c in claims):
        return
    if any(n in low for n in NEGATIONS) or any(m in low for m in PROJECTION_MARKERS):
        return
    violations.append(
        f"{name}:{lineno} [C] unqualified measurement claim; the system has not "
        f"been built.\n         {line.strip()[:120]}"
    )


def check_d_target_qualification(name: str, lineno: int, line: str) -> None:
    """'meets the 2s target' must name a path and not overclaim."""
    low = line.lower()
    if not re.search(r"\b(meets?|fits?|satisf\w+|achiev\w+)\b", low):
        return
    if not re.search(rf"({TARGET_LATENCY_MS}|2[\s-]?s(econd)?s?\b|2 s\b)", low):
        return
    if "tier" not in low:
        return
    if not re.search(r"\bP[0-3]\b", line):
        violations.append(
            f"{name}:{lineno} [D] latency claim does not name a query path. "
            f"Latency is path-specific (see tools/paths.py).\n"
            f"         {line.strip()[:120]}"
        )


def main() -> int:
    for path in files():
        name = path.relative_to(ROOT)
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            # skip fenced code blocks' contents heuristically: they rarely
            # contain tables, and the checks are table-oriented
            if line.strip().startswith("```"):
                continue
            check_a_path_specs(str(name), lineno, line)
            check_b_tier_c_labelling(str(name), lineno, line)
            check_c_unqualified_measurement(str(name), lineno, line)
            check_d_target_qualification(str(name), lineno, line)

    print(f"checked {len(files())} markdown files "
          f"(cached prefix = {CACHED_PREFIX_TOKENS} tok, "
          f"target = {TARGET_LATENCY_MS} ms)")
    if not violations:
        print("PASS  docs are consistent with tools/paths.py")
        return 0

    print(f"FAIL  {len(violations)} violation(s):\n")
    for v in violations:
        print(f"  {v}\n")
    print("Fix the docs, or change tools/paths.py if the spec genuinely moved.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
