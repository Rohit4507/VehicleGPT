#!/usr/bin/env python3
"""Ask what moved before an event.

    python3 tools/phase0/precursors.py --db /tmp/vehiclegpt.db --event warning
    python3 tools/phase0/precursors.py --db /tmp/vehiclegpt.db --at 2026-03-11T17:50:00Z \\
        --target battery.cell_temp_max --window 1500 --min-delta 1.0
    python3 tools/phase0/precursors.py --db /tmp/vehiclegpt.db --event warning --json

This answers "what happened before the battery overheating warning?" with two
instruments, because they answer different halves of the question:

* precedence (where each series departed from a settled state) says *when*
  something moved, and is reported first;
* lagged correlation says how similar two series are once one is slid against
  the other, and is reported as support, gated on effective sample size.

They disagree on smooth vehicle data, and the disagreement is the point: every
onboard signal is a slow ramp, so a high correlation is easy and says little.
`crowded` means several signals departed together, which is the signature of one
shared driver rather than a cause.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vehiclegpt.analytics import leadlag  # noqa: E402


def open_store(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"no store at {path}; run tools/phase0/build_store.py")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def parse_at(value: str) -> int:
    try:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"cannot read --at {value!r}: {exc}") from exc
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.UTC)
    return int(stamp.timestamp() * 1e9)


def resolve_event(conn: sqlite3.Connection, kind: str) -> sqlite3.Row:
    order = {"critical": 0, "warning": 1, "info": 2}
    rows = [r for r in conn.execute("SELECT ts_ns, kind, severity, message FROM event")]
    if not rows:
        raise SystemExit("the store has no events; rebuild with tools/phase0/build_store.py")
    if kind == "any":
        kind = "warning"
    matches = [r for r in rows if r["severity"] == kind]
    if not matches:
        seen = ", ".join(sorted({r["severity"] for r in rows}))
        raise SystemExit(f"no {kind} event; have {seen}")
    return min(matches, key=lambda r: (r["ts_ns"], order.get(r["severity"], 9)))


def clock(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return dt.datetime.fromtimestamp(seconds, dt.UTC).strftime("%H:%M:%S")


def human(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    if seconds < 90:
        return f"{seconds} s"
    return f"{seconds // 60} min {seconds % 60:02d} s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("/tmp/vehiclegpt.db"))
    ap.add_argument("--event", default="warning",
                    choices=["critical", "warning", "info"],
                    help="pick the event to explain (default: warning)")
    ap.add_argument("--at", metavar="ISO8601", help="use this instant instead of an event")
    ap.add_argument("--target", default="battery.cell_temp_max",
                    help="the signal the event is about")
    ap.add_argument("--window", type=float, default=1500.0, help="seconds before the event")
    ap.add_argument("--max-lag", type=float, default=600.0, help="longest lead to consider")
    ap.add_argument("--min-delta", type=float, default=1.0,
                    help="smallest departure that counts as an onset")
    ap.add_argument("--settled", type=float, default=300.0,
                    help="seconds of quiet required before an onset")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    conn = open_store(args.db)
    if args.at:
        at_ns, label = parse_at(args.at), args.at
    else:
        row = resolve_event(conn, args.event)
        at_ns = int(row["ts_ns"])
        label = f"{row['severity']}: {row['message']}"

    try:
        result = leadlag.precursors(
            conn, args.target, at_ns, window_s=args.window, max_lag_s=args.max_lag,
            min_delta=args.min_delta, settled_s=args.settled)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps({"event": label, "at_ns": at_ns,
                          "target": result["target"],
                          "target_onset_s": result["target_onset_s"],
                          "crowded": result["crowded"],
                          "precursors": result["precursors"],
                          "table": result["table"]}, indent=2))
        return 0

    when = dt.datetime.fromtimestamp(at_ns / 1e9, dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
    print(f"event    {when} UTC  {label}")
    print(f"target   {result['target']}  (turns at {clock(result['target_onset_s'])})")
    print(f"window   {int(args.window)} s, settled >= {int(args.settled)} s, "
          f"min delta {args.min_delta:g}")
    print()
    if not result["precursors"]:
        print("no signal departed from a settled state before this event.")
        return 0

    print(f"{'signal':30s} {'onset':9s} {'before':>9s} {'corr':>6s}  {'n_eff':>5s}  note")
    for row in result["precursors"][: args.limit]:
        note = "" if row["reliable"] else "smooth, corr not usable"
        print(f"{row['signal']:30s} {clock(row['onset_s']):9s} "
              f"{human(row['lead_s']):>9s} {row['correlation']:6.2f}  "
              f"{row['n_eff']:5.1f}  {note}")

    if result["unsettled"]:
        print()
        print(f"never settled, so no onset to date ({len(result['unsettled'])} signals):")
        for row in result["unsettled"][: args.limit]:
            print(f"  {row['signal']:28s} corr {row['correlation']:6.2f}")
        print("A duty-cycled fault looks like this - it changes without ever holding")
        print("a steady state, so there is no baseline to depart from. "
              "tools/phase0/episodes.py dates those instead.")

    if result["crowded"]:
        print()
        print(f"{result['crowd_size']} signals departed within seconds of each other.")
        print("That is one shared driver moving them all, not one cause among them;")
        print("precedence cannot separate them and should not be read as a ranking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
