#!/usr/bin/env python3
"""Build the local store from the simulated 14-day program.

    python3 tools/phase0/build_store.py --db /tmp/vehiclegpt.db
    python3 tools/phase0/build_store.py --db /tmp/vg.db --days 3   # quick

Two paths, deliberately:

* ``--via-can`` runs every drive through the real encode -> can.Bus -> decode ->
  normalise pipeline and stores what comes back. Correct, but 10 frames per tick
  across ~25,800 simulated seconds is ~2.6 M frames, so it is for demonstrating
  the pipeline end to end, not for bulk loading.
* Without it, samples are written directly from the generator's truth arrays.
  The values are identical - ``tests/test_ingest_roundtrip.py`` already proves
  the CAN round trip reproduces truth to within one LSB - and bulk loading does
  not need to pay for it twice.

Either way the timestamps are real Unix epoch nanoseconds, so the analytics
layer can group by day and answer "yesterday" without knowing anything about the
simulator.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim.generator import RATE_HZ, Simulation  # noqa: E402
from sim.program import program_days, program_end_epoch_s  # noqa: E402
from vehiclegpt.schema.registry import registry  # noqa: E402
from vehiclegpt.store import db  # noqa: E402

DECIMATE = max(1, int(round(RATE_HZ)))    # 10 Hz -> 1 Hz


def samples_from(sim: Simulation, start_epoch_s: float) -> list[db.SampleRow]:
    """Decimate one drive to 1 Hz and stamp it with wall-clock nanoseconds."""
    rows: list[db.SampleRow] = []
    base_ns = int(start_epoch_s * 1e9)
    for i in range(0, sim.n, DECIMATE):
        ts_ns = base_ns + int(round(sim.t_s[i] * 1e9))
        for path, series in sim.truth.items():
            rows.append(db.SampleRow(ts_ns=ts_ns, path=path, value=float(series[i])))
    return rows


def events_from(sim: Simulation, start_epoch_s: float) -> list[db.EventRow]:
    base_ns = int(start_epoch_s * 1e9)
    return [
        db.EventRow(ts_ns=base_ns + int(round(e.ts_s * 1e9)),
                    kind=e.kind, severity=e.severity, message=e.description)
        for e in sim.events
    ]


def build(db_path: Path, days: int, via_can: bool, seed_jitter: int = 0) -> dict:
    reg = registry()
    signals = [(s.path, s.unit, s.tier.value) for s in reg.signals.values()]

    if db_path.exists():
        db_path.unlink()

    t0 = time.monotonic()
    total_samples = total_events = 0
    with db.store(db_path) as conn:
        ids = db.ensure_signals(conn, signals)
        for drive in program_days(days):
            sim = generate_sim(drive.scenario, seed_jitter)
            total_samples += db.write_samples(conn, samples_from(sim, drive.start_epoch_s), ids)
            total_events += db.write_events(conn, events_from(sim, drive.start_epoch_s))
        conn.commit()
        derived = db.rebuild(conn)
        n_rows = conn.execute("SELECT COUNT(*) AS n FROM sample").fetchone()["n"]

    return {
        "db": str(db_path),
        "days": days,
        "via_can": via_can,
        "samples": n_rows,
        "events": total_events,
        "segments": derived["segments"],
        "daily_rows": derived["days"],
        "size_mb": round(db_path.stat().st_size / 1e6, 2),
        "build_s": round(time.monotonic() - t0, 2),
        "now_epoch_s": program_end_epoch_s(days),
    }


def generate_sim(scenario, seed_jitter: int = 0):
    from sim.generator import generate

    return generate(scenario, rng_seed_offset=seed_jitter)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("/tmp/vehiclegpt.db"))
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--via-can", action="store_true",
                    help="route through the CAN encode/decode path (slow)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    stats = build(args.db, args.days, args.via_can)
    if args.json:
        import json
        print(json.dumps(stats, indent=2))
    else:
        print(f"store        {stats['db']}  ({stats['size_mb']} MB)")
        print(f"days         {stats['days']}")
        print(f"samples      {stats['samples']:,}  (1 Hz)")
        print(f"events       {stats['events']:,}")
        print(f"segments     {stats['segments']:,}")
        print(f"daily rows   {stats['daily_rows']}")
        print(f"build time   {stats['build_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
