"""Storage footprint: measured, then projected.

Doc 06 §7 projects 103 MB for 28 days from an assumed **1.2 bytes per value**
(int16 + delta + zstd). This module measures what the store actually costs and
reports the difference rather than letting the projection stand.

The distinction that matters, and which an earlier version of this analysis
missed: doc 06's model assumes the hot tier is sampled **continuously, 24/7,**
for 15 signals. The store as built records **driving time only** - roughly half
an hour a day in the simulated program. Those are not the same workload, so
comparing their totals directly would flatter the implementation by a factor of
about 46.

So every projection here states its sampling assumption explicitly, and both are
reported: what the store costs for what it actually records, and what it would
cost if it sampled the way the model assumes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BUDGET_MB = 103.17        # docs/06 §7, [D]
MODEL_BYTES_PER_VALUE = 1.2       # docs/06 §7 assumption
SECONDS_PER_DAY = 86_400


@dataclass
class Footprint:
    db_path: str
    db_bytes: int
    samples: int
    signals: int
    sampled_seconds: int          # distinct seconds containing data
    drive_hours: float
    days_covered: int
    objects: dict[str, int] = field(default_factory=dict)

    @property
    def bytes_per_sample(self) -> float:
        return self.db_bytes / self.samples if self.samples else 0.0

    @property
    def samples_per_drive_hour(self) -> float:
        return self.samples / self.drive_hours if self.drive_hours else 0.0

    @property
    def drive_hours_per_day(self) -> float:
        return self.drive_hours / self.days_covered if self.days_covered else 0.0

    @property
    def bytes_per_sample_row_only(self) -> float:
        """Bytes attributable to the `sample` B-tree alone."""
        return self.objects.get("sample", 0) / self.samples if self.samples else 0.0


def measure(db_path: str | Path, conn: sqlite3.Connection | None = None) -> Footprint:
    """Measure an existing store. Read-only and cheap."""
    db_path = Path(db_path)
    if conn is None and not db_path.exists():
        raise FileNotFoundError(f"no store at {db_path}")
    own = conn is None
    conn = conn or sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        samples = conn.execute("SELECT COUNT(*) AS n FROM sample").fetchone()["n"]
        signals = conn.execute(
            "SELECT COUNT(DISTINCT signal_id) AS n FROM sample").fetchone()["n"]
        span = conn.execute(
            "SELECT MIN(ts_s) AS lo, MAX(ts_s) AS hi, "
            "       COUNT(DISTINCT ts_s) AS secs FROM sample").fetchone()
        day_row = conn.execute(
            "SELECT COUNT(DISTINCT date(ts_s, 'unixepoch')) AS d FROM sample"
        ).fetchone()
        try:
            drive_ns = conn.execute(
                "SELECT COALESCE(SUM(end_ns - start_ns), 0) AS ns FROM segment"
            ).fetchone()["ns"]
        except sqlite3.OperationalError:
            drive_ns = 0        # not yet segmented; project_driving reports unavailable

        objects: dict[str, int] = {}
        try:
            for row in conn.execute(
                    "SELECT name, SUM(pgsize) AS b FROM dbstat GROUP BY name"):
                objects[row["name"]] = int(row["b"])
        except sqlite3.OperationalError:
            objects = {"(dbstat unavailable)": 0}
    finally:
        if own:
            conn.close()

    return Footprint(
        db_path=str(db_path),
        db_bytes=db_path.stat().st_size,
        samples=int(samples),
        signals=int(signals),
        sampled_seconds=int(span["secs"] or 0),
        drive_hours=float(drive_ns) / 3.6e12,
        days_covered=int(day_row["d"] or 0),
        objects=objects,
    )


def project_driving(fp: Footprint, days: int = 28) -> dict:
    """Extrapolate from observed driving density, keeping the same duty cycle.

    This is the honest projection for a vehicle that is driven the way the
    program drives: it assumes the same hours-per-day and the same bytes per
    sample. It says nothing about a car parked with the ignition on.
    """
    if fp.days_covered == 0:
        return {"available": False, "reason": "no data"}
    samples = fp.samples / fp.days_covered * days
    return {
        "available": True,
        "assumption": "same driving hours per day as observed",
        "days": days,
        "duty_cycle_hours_per_day": round(fp.drive_hours_per_day, 2),
        "samples": int(samples),
        "bytes": int(samples * fp.bytes_per_sample),
        "mb": round(samples * fp.bytes_per_sample / 1e6, 2),
    }


def project_continuous(fp: Footprint, hot_signals: int = 15,
                       days: int = 28) -> dict:
    """What the hot tier costs at 1 Hz, 24/7 - the workload doc 06 models."""
    samples = hot_signals * days * SECONDS_PER_DAY
    return {
        "available": True,
        "assumption": "1 Hz continuously, 24/7, for the whole horizon",
        "days": days,
        "hot_signals": hot_signals,
        "samples": samples,
        "bytes": int(samples * fp.bytes_per_sample),
        "mb": round(samples * fp.bytes_per_sample / 1e6, 2),
    }


def required_bytes_per_sample(budget_mb: float = DEFAULT_BUDGET_MB,
                              hot_signals: int = 15, days: int = 28) -> float:
    """Density the store would need to hit the budget at 1 Hz, 24/7."""
    samples = hot_signals * days * SECONDS_PER_DAY
    return budget_mb * 1e6 / samples


def assess(fp: Footprint, budget_mb: float = DEFAULT_BUDGET_MB,
           hot_signals: int = 15, days: int = 28) -> dict:
    driving = project_driving(fp, days)
    continuous = project_continuous(fp, hot_signals, days)
    required = required_bytes_per_sample(budget_mb, hot_signals, days)

    return {
        "budget_mb": budget_mb,
        "measured_bytes_per_sample": round(fp.bytes_per_sample, 2),
        "model_bytes_per_value": MODEL_BYTES_PER_VALUE,
        "overhead_factor_vs_model": round(fp.bytes_per_sample / MODEL_BYTES_PER_VALUE, 1),
        "required_bytes_per_sample_for_budget": round(required, 2),
        "driving_projection": driving,
        "continuous_projection": continuous,
        "driving_within_budget": driving.get("available") and driving["mb"] <= budget_mb,
        "continuous_within_budget": continuous["mb"] <= budget_mb,
        "duty_cycle_to_fit_continuous_budget": round(
            budget_mb / continuous["mb"] * 24.0, 2) if continuous["mb"] else None,
    }


def report(fp: Footprint, assessment: dict) -> str:
    W = 76
    lines = [
        "=" * W,
        "Storage footprint".center(W),
        "=" * W,
        "",
        f"store            {fp.db_path}",
        f"size             {fp.db_bytes / 1e6:.2f} MB",
        f"samples          {fp.samples:,} across {fp.signals} signals",
        f"days covered     {fp.days_covered}   driving {fp.drive_hours:.2f} h "
        f"({fp.drive_hours_per_day:.2f} h/day)",
        f"bytes per sample {fp.bytes_per_sample:.1f}  "
        f"(sample B-tree alone: {fp.bytes_per_sample_row_only:.1f})",
        "",
        "-- where the bytes are " + "-" * (W - 23),
    ]
    for name, size in sorted(fp.objects.items(), key=lambda kv: -kv[1])[:6]:
        share = size / fp.db_bytes * 100 if fp.db_bytes else 0
        lines.append(f"   {name:26s} {size / 1e6:8.2f} MB  {share:5.1f}%")

    lines += [
        "",
        "-- against the budget " + "-" * (W - 22),
        f"   budget                      {assessment['budget_mb']:8.2f} MB",
        f"   model assumed               {assessment['model_bytes_per_value']:8.2f} B/value",
        f"   measured                    {assessment['measured_bytes_per_sample']:8.2f} B/sample"
        f"   ({assessment['overhead_factor_vs_model']:.0f}x the model)",
        f"   needed for the budget       "
        f"{assessment['required_bytes_per_sample_for_budget']:8.2f} B/sample at 1 Hz 24/7",
        "",
        "   28-day projection, driving duty cycle",
        f"     {assessment['driving_projection'].get('duty_cycle_hours_per_day')} h/day  ->  "
        f"{assessment['driving_projection'].get('mb')} MB   "
        f"{'WITHIN' if assessment['driving_within_budget'] else 'OVER'} budget",
        "",
        "   28-day projection, 1 Hz 24/7 for 15 signals (what the model assumes)",
        f"     ->  {assessment['continuous_projection']['mb']} MB   "
        f"{'WITHIN' if assessment['continuous_within_budget'] else 'OVER'} budget",
    ]
    if not assessment["continuous_within_budget"]:
        lines += [
            "",
            f"   To fit {assessment['budget_mb']} MB at 1 Hz 24/7 the store would need",
            f"   {assessment['required_bytes_per_sample_for_budget']} B/sample. It uses "
            f"{assessment['measured_bytes_per_sample']}. The gap is the",
            "   compression doc 06 assumed (int16 + delta + zstd) and which is not",
            "   implemented. Quantising the value alone recovers about 15%, because",
            "   SQLite row overhead dominates, not the value width - and on-change",
            "   keyframes do not help here, because the expensive signals are the",
            "   hot continuous ones that change every second by design.",
            "",
            "   The 103 MB figure therefore holds for driving-time sampling, not for",
            "   a continuously-powered vehicle. Either the hot tier moves to",
            "   a columnar or compressed layout, or the claim gets restated.",
        ]
    return "\n".join(lines)
