"""
Canonical definition of the VehicleGPT query paths.

This module is the SINGLE SOURCE OF TRUTH for P0/P1/P2/P3. Everything else —
the feasibility model, the architecture docs, the latency tables, the roadmap
targets — must agree with the numbers here.

Why this file exists: an earlier draft defined P1 as "140 in / 28 out" in one
document and "100 in / 10 out" in another, which silently changed a headline
latency figure by 15%. Prose drifts; code does not. `tools/check_docs.py`
enforces that the markdown matches this module, and runs in CI.

If you change a number here, you must also change:
  - docs/04-architecture.md  §7 table
  - docs/06-hardware-feasibility.md  §5, §6.3, §11
  - README.md  "three findings" and constraints table
`check_docs.py` will tell you exactly which lines are stale.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tokens of system prompt + few-shot header that are prefill-cached once at
# startup and reused for every query. llama.cpp supports this via slot reuse /
# --prompt-cache. Only the per-query tail is re-prefilled.
CACHED_PREFIX_TOKENS: int = 90

# The brief's headline constraint.
TARGET_LATENCY_MS: int = 2000


@dataclass(frozen=True)
class PathSpec:
    """One way a user query can be answered, with its full cost model."""

    key: str
    name: str
    prompt_tokens: int
    output_tokens: int
    pipeline_ms: float          # everything that is not LLM: parse, SQL, assemble
    needs_llm: bool = True

    @property
    def uncached_prefill_tokens(self) -> int:
        """Tokens actually re-prefilled per query, after prefix caching."""
        if not self.needs_llm:
            return 0
        return max(0, self.prompt_tokens - CACHED_PREFIX_TOKENS)

    @property
    def spec(self) -> str:
        """Short machine-checkable form, e.g. '140 in / 28 out'."""
        if not self.needs_llm:
            return "0 in / 0 out"
        return f"{self.prompt_tokens} in / {self.output_tokens} out"

    @property
    def table_cell(self) -> str:
        """Form used inside the latency table, e.g. '140 / 28'."""
        if not self.needs_llm:
            return "0 / 0"
        return f"{self.prompt_tokens} / {self.output_tokens}"


PATHS: tuple[PathSpec, ...] = (
    PathSpec(
        key="P0",
        name="Deterministic: classifier -> DSL -> SQL -> template",
        prompt_tokens=0,
        output_tokens=0,
        pipeline_ms=110,
        needs_llm=False,
    ),
    PathSpec(
        key="P1",
        name="SLM intent-only (GBNF-constrained JSON) + template answer",
        prompt_tokens=140,
        output_tokens=28,
        pipeline_ms=130,
    ),
    PathSpec(
        key="P2",
        name="SLM short narration over a pre-computed evidence pack",
        prompt_tokens=260,
        output_tokens=70,
        pipeline_ms=130,
    ),
    PathSpec(
        key="P3",
        name="SLM long explanation / multi-hop (deep-explain mode)",
        prompt_tokens=420,
        output_tokens=160,
        pipeline_ms=150,
    ),
)

BY_KEY: dict[str, PathSpec] = {p.key: p for p in PATHS}


def spec_table() -> str:
    """Markdown for the normative spec table. Used to generate docs."""
    rows = [
        "| Path | What it does | Prompt in | Output out | Cached prefix | "
        "Re-prefilled | Pipeline | LLM? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in PATHS:
        rows.append(
            f"| **{p.key}** | {p.name} | {p.prompt_tokens} | {p.output_tokens} "
            f"| {CACHED_PREFIX_TOKENS if p.needs_llm else 0} "
            f"| {p.uncached_prefill_tokens} | {p.pipeline_ms:.0f} ms "
            f"| {'yes' if p.needs_llm else '**no**'} |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    print(spec_table())
    print()
    for p in PATHS:
        print(f"{p.key}: {p.spec}   re-prefill {p.uncached_prefill_tokens} tok"
              f"   pipeline {p.pipeline_ms:.0f} ms")
