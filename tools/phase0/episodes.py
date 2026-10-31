#!/usr/bin/env python3
"""Search the episode cards, or rebuild them.

    python3 tools/phase0/episodes.py --db /tmp/vehiclegpt.db --list
    python3 tools/phase0/episodes.py --db /tmp/vehiclegpt.db --search "adas disengagement"
    python3 tools/phase0/episodes.py --db /tmp/vehiclegpt.db --search "pump stopped" --json

Cards are derived data, so `--build` is idempotent and rebuilding the store
rebuilds them. This tool is read-mostly: its job is to show what retrieval would
return for a question, with the evidence that produced each hit.
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

from vehiclegpt.analytics import retrieval  # noqa: E402


def open_store(path: Path, write: bool = False) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"no store at {path}; run tools/phase0/build_store.py")
    mode = "" if write else "?mode=ro"
    conn = sqlite3.connect(f"file:{path}{mode}", uri=True)
    conn.row_factory = sqlite3.Row
    have = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    if "card_fts" not in have:
        raise SystemExit(f"{path} has no card index; rebuild with tools/phase0/build_store.py")
    return conn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("/tmp/vehiclegpt.db"))
    ap.add_argument("--search", metavar="TEXT")
    ap.add_argument("--kind", choices=["incident", "trend"])
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="list every card")
    ap.add_argument("--build", action="store_true", help="rebuild cards from the store")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    conn = open_store(args.db, write=args.build)

    if args.build:
        count = retrieval.build_cards(conn)
        print(f"rebuilt {count} cards")
        return 0

    if args.list:
        rows = [dict(r) for r in conn.execute(
            "SELECT id, kind, day, severity, title FROM card ORDER BY start_ns")]
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        for row in rows:
            print(f"  [{row['id']:2d}] {row['day']}  {row['severity']:8s} "
                  f"{row['kind']:8s} {row['title']}")
        return 0

    if not args.search:
        ap.error("give --search TEXT, --list or --build")

    t0 = time.perf_counter()
    hits = retrieval.search(conn, args.search, limit=args.limit, kind=args.kind)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if args.json:
        print(json.dumps({"query": args.search,
                          "expanded": retrieval.fts_query(args.search),
                          "hits": hits, "latency_ms": round(elapsed_ms, 2)},
                         indent=2, default=str))
        return 0

    print(f"\nquery    {args.search}")
    print(f"expanded {retrieval.fts_query(args.search)}")
    print(f"{len(hits)} hit(s) in {elapsed_ms:.2f} ms\n")
    for hit in hits:
        print(f"  [{hit['id']}] {hit['day']}  {hit['severity']:8s}  {hit['title']}")
        print(f"       {hit['body'][:220]}")
        print(f"       evidence: {len(hit['evidence'])} pointer(s), "
              f"signals: {hit['signals']}\n")
    if not hits:
        print("  no card matched those words. This is a lexical index: a question "
              "whose words do not appear\n  in any card returns nothing rather than "
              "a plausible near-miss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
