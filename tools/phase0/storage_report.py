#!/usr/bin/env python3
"""Stress-test the store against the 103 MB budget.

    python3 tools/phase0/storage_report.py --db /tmp/vehiclegpt.db
    python3 tools/phase0/storage_report.py --db /tmp/vg.db --json

Measures the real store, then projects forward under two different sampling
assumptions, because they give very different answers and only one of them is
the one doc 06 models.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vehiclegpt.store import footprint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("/tmp/vehiclegpt.db"))
    ap.add_argument("--days", type=int, default=28, help="projection horizon")
    ap.add_argument("--hot-signals", type=int, default=15,
                    help="signals sampled at 1 Hz in the doc 06 model")
    ap.add_argument("--budget-mb", type=float, default=footprint.DEFAULT_BUDGET_MB)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no store at {args.db}; run tools/phase0/build_store.py")

    fp = footprint.measure(args.db)
    assessment = footprint.assess(fp, args.budget_mb, args.hot_signals, args.days)

    if args.json:
        print(json.dumps({"footprint": asdict(fp), "assessment": assessment}, indent=2))
    else:
        print(footprint.report(fp, assessment))
    return 0 if assessment["driving_within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
