#!/usr/bin/env python3
"""Ask the deterministic layer a question.

    python3 tools/phase0/ask.py --db /tmp/vehiclegpt.db \\
        "What happened before the battery overheating warning?"

    python3 tools/phase0/ask.py --db /tmp/vehiclegpt.db --all
    python3 tools/phase0/ask.py --db /tmp/vehiclegpt.db --json "..." 

No model is involved. Every answer comes from SQL over the store, so the same
question on the same data gives the same answer every time, and the numbers can
be checked by hand.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vehiclegpt.analytics import queries  # noqa: E402

BRIEF_QUESTIONS = [
    "Why has my driving range reduced over the last two weeks?",
    "When did fuel efficiency show the largest drop?",
    "Show all ADAS disengagement events from yesterday.",
    "What happened before the battery overheating warning?",
    "Compare energy consumption between city and highway driving.",
]


def open_store(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"no store at {path}; run tools/phase0/build_store.py first")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    have = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'")}
    if "v_daily_rolling" not in have:
        raise SystemExit(
            f"{path} has no analytics views; rebuild it with "
            "tools/phase0/build_store.py")
    return conn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*")
    ap.add_argument("--db", type=Path, default=Path("/tmp/vehiclegpt.db"))
    ap.add_argument("--all", action="store_true", help="ask every brief question")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--evidence", action="store_true", help="dump supporting rows")
    args = ap.parse_args()

    conn = open_store(args.db)
    questions = BRIEF_QUESTIONS if (args.all or not args.question) else [" ".join(args.question)]

    out = []
    for q in questions:
        t0 = time.perf_counter()
        answer = queries.answer(conn, q)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        record = answer.to_dict() | {"latency_ms": round(elapsed_ms, 2), "question": q}
        out.append(record)

        if not args.json:
            print(f"\nQ: {q}")
            print(f"   intent {answer.intent} | path {answer.path} | "
                  f"{elapsed_ms:.1f} ms")
            print(f"\n   {answer.text}")
            if answer.caveats:
                for c in answer.caveats:
                    print(f"   note: {c}")
            if args.evidence and answer.evidence:
                print("   evidence:")
                for row in answer.evidence[:8]:
                    print(f"     {dict(row)}")

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        slowest = max(r["latency_ms"] for r in out)
        print(f"\n{len(out)} questions answered deterministically; "
              f"slowest {slowest:.1f} ms (target 2000 ms, no model used)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
