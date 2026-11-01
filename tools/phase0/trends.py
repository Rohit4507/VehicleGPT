#!/usr/bin/env python3
"""Find where a series changed level, and which readings broke away from it.

    python3 tools/phase0/trends.py --db /tmp/vehiclegpt.db
    python3 tools/phase0/trends.py --db /tmp/vehiclegpt.db --metric wh_per_km
    python3 tools/phase0/trends.py --db /tmp/vehiclegpt.db --signal battery.cell_temp_max
    python3 tools/phase0/trends.py --db /tmp/vehiclegpt.db --json

The daily series is filtered to days with meaningful distance before anything is
measured, and that filter is the whole argument of this tool. On the shipped
store the last two days are 7.9 km cold trips at 321 Wh/km: without the filter a
change-point detector confidently reports a degradation that starts on 14 March,
when what actually changed was the duty cycle. Consumption per kilometre is only
comparable between days that drove a comparable distance.
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

from vehiclegpt.analytics import anomaly, changepoint  # noqa: E402

DAILY_METRICS = ("wh_per_km", "energy_wh", "distance_km", "ambient_c", "hvac_w")


def daily_series(conn: sqlite3.Connection, metric: str, min_km: float
                 ) -> tuple[list[str], list[float], int]:
    """Comparable days only: returns (days, values, number excluded)."""
    if metric not in DAILY_METRICS:
        raise SystemExit(f"unknown metric {metric!r}; choose from {', '.join(DAILY_METRICS)}")
    rows = [r for r in conn.execute(
        f"SELECT day, {metric} AS v, distance_km FROM daily ORDER BY day")
        if r["v"] is not None]
    kept = [r for r in rows if r["distance_km"] >= min_km]
    return ([r["day"] for r in kept], [float(r["v"]) for r in kept],
            len(rows) - len(kept))


def signal_series(conn: sqlite3.Connection, path: str, hours: int
                  ) -> tuple[list[str], list[float]]:
    row = conn.execute("SELECT id FROM signal WHERE path = ?", (path,)).fetchone()
    if row is None:
        raise SystemExit(f"no signal {path!r} in this store")
    lo = conn.execute("SELECT MAX(ts_s) FROM sample").fetchone()[0] - hours * 3600
    rows = conn.execute(
        "SELECT ts_s, value FROM sample WHERE signal_id = ? AND ts_s >= ? "
        "ORDER BY ts_s", (row["id"], lo)).fetchall()

    def clock(ts_s: int) -> str:
        return dt.datetime.fromtimestamp(ts_s, dt.UTC).strftime("%m-%d %H:%M:%S")

    return ([clock(r["ts_s"]) for r in rows], [float(r["value"]) for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("/tmp/vehiclegpt.db"))
    ap.add_argument("--metric", default="wh_per_km", choices=DAILY_METRICS)
    ap.add_argument("--min-km", type=float, default=10.0,
                    help="exclude days shorter than this (default 10 km)")
    ap.add_argument("--signal", help="analyse a raw signal instead of the daily table")
    ap.add_argument("--hours", type=int, default=24, help="signal lookback (default 24 h)")
    ap.add_argument("--window", type=int, default=7, help="local-level window")
    ap.add_argument("--threshold", type=float, default=anomaly.DEFAULT_THRESHOLD)
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="ignore changes smaller than this in absolute units; "
                         "statistical significance is not practical significance")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no store at {args.db}; run tools/phase0/build_store.py")
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    if args.signal:
        labels, values = signal_series(conn, args.signal, args.hours)
        excluded, name = 0, args.signal
    else:
        labels, values, excluded = daily_series(conn, args.metric, args.min_km)
        name = f"daily {args.metric}"

    if len(values) < 4:
        print(f"only {len(values)} comparable points for {name}; nothing to analyse.")
        return 0

    segs = changepoint.segments(values, min_delta=args.min_delta)
    hits = anomaly.detect(values, threshold=args.threshold, window=args.window,
                          min_delta=args.min_delta)
    boundary = changepoint.first_change(values)

    result = {
        "series": name, "points": len(values),
        "excluded_short_days": excluded, "min_km": args.min_km,
        "min_delta": args.min_delta,
        "segments": segs, "anomalies": hits,
        "current_regime_start": None if boundary is None else labels[boundary],
    }

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(f"\n{name}: {len(values)} comparable points")
    if excluded:
        print(f"  {excluded} day(s) under {args.min_km:.0f} km excluded - "
              "consumption per km is not comparable across trip lengths")
    print("\nlevel changes (PELT, BIC penalty):")
    if len(segs) == 1:
        print(f"  none - the level held at {segs[0]['mean']:.1f} throughout")
    for seg in segs:
        delta = "" if seg["delta"] is None else f"  {seg['delta']:+.1f} vs previous"
        print(f"  from {labels[seg['start']]:>19s}  {seg['mean']:8.1f}{delta}")
    if boundary is not None:
        before = sum(values[:boundary]) / boundary
        after = sum(values[boundary:]) / (len(values) - boundary)
        print(f"\ncurrent regime starts {labels[boundary]} "
              f"({before:.1f} -> {after:.1f}, {len(values) - boundary} points after)")
    print(f"\nanomalies (robust z > {args.threshold}, window {args.window}):")
    if not hits:
        print("  none")
    for hit in hits[:10]:
        print(f"  {labels[hit['index']]:>19s}  value {hit['value']:9.2f}  "
              f"delta {hit['delta']:+8.2f}  z {hit['score']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
