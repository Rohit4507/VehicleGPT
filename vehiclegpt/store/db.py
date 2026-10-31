"""Local SQLite store.

Design constraints, from docs/06 §7:

* **System of record, not an analytics engine.** SQLite handles the unbatched
  small writes and tiny lookups a vehicle produces; a columnar engine would be
  ~56x too slow at single-row insert, which is the access pattern that actually
  dominates. See docs/03 for the comparison.
* **Decimate on write.** The 10 Hz signal stream is stored at 1 Hz. The raw ring
  stays in RAM (L0) and is deliberately not persisted: it is large, short-lived,
  and worthless without context that the segments already capture.
* **Derived tables are rebuilt, never patched.** ``segment`` and ``daily`` are
  recomputed from ``sample`` on demand, so a schema or logic change cannot leave
  half-stale aggregates behind.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
VIEWS_PATH = (Path(__file__).resolve().parents[1]
              / "analytics" / "views.sql")

DAY_NS = 86_400 * 10**9


@dataclass(frozen=True)
class SampleRow:
    ts_ns: int
    path: str
    value: float
    quality: int = 0


@dataclass(frozen=True)
class EventRow:
    ts_ns: int
    kind: str
    severity: str
    message: str


def connect(path: str | Path, *, create: bool = True) -> sqlite3.Connection:
    """Open (and optionally initialise) the store."""
    path = Path(path)
    if path.exists() is False and not create:
        raise FileNotFoundError(path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if create:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


@contextmanager
def store(path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def ensure_signals(conn: sqlite3.Connection,
                   signals: Sequence[tuple[str, str, str]]) -> dict[str, int]:
    """Insert signal dimension rows, returning ``path -> id``."""
    conn.executemany(
        "INSERT OR IGNORE INTO signal(path, unit, tier) VALUES (?, ?, ?)",
        [(p, u, t) for p, u, t in signals],
    )
    return {row["path"]: row["id"] for row in conn.execute("SELECT id, path FROM signal")}


def write_samples(conn: sqlite3.Connection, rows: Iterable[SampleRow],
                  signal_ids: dict[str, int]) -> int:
    """Insert decimated samples. Unknown paths are skipped, not invented."""
    payload = [
        (r.ts_ns, signal_ids[r.path], r.value, r.quality)
        for r in rows if r.path in signal_ids
    ]
    conn.executemany(
        "INSERT INTO sample(ts_ns, signal_id, value, quality) VALUES (?, ?, ?, ?)",
        payload,
    )
    return len(payload)


def write_events(conn: sqlite3.Connection, rows: Iterable[EventRow]) -> int:
    payload = [(r.ts_ns, r.kind, r.severity, r.message) for r in rows]
    conn.executemany(
        "INSERT INTO event(ts_ns, kind, severity, message) VALUES (?, ?, ?, ?)",
        payload,
    )
    return len(payload)


# ---------------------------------------------------------------------------
# Building derived tables
# ---------------------------------------------------------------------------

SPEED_PATH = "vehicle.speed"
CURRENT_PATH = "battery.pack_current"
VOLTAGE_PATH = "battery.pack_voltage"
AMBIENT_PATH = "thermal.ambient"
HVAC_PATH = "hvac.power"
SOC_PATH = "battery.soc"

REGIME_IDLE_BELOW_KPH = 5.0
REGIME_HIGHWAY_AT_LEAST_KPH = 80.0
MIN_SEGMENT_S = 30.0
DRIVE_BREAK_NS = 10 * 60 * 10**9   # 10 minutes of silence starts a new drive


def signal_id(conn: sqlite3.Connection, path: str) -> int | None:
    row = conn.execute("SELECT id FROM signal WHERE path = ?", (path,)).fetchone()
    return None if row is None else int(row["id"])


def _series(conn: sqlite3.Connection, path: str) -> list[tuple[int, float]]:
    sid = signal_id(conn, path)
    if sid is None:
        return []
    return [(int(r["ts_ns"]), float(r["value"])) for r in conn.execute(
        "SELECT ts_ns, value FROM sample WHERE signal_id = ? ORDER BY ts_ns", (sid,))]


def classify_regime(speed_kph: float) -> str:
    """Speed regime from the signal itself, not from scenario metadata.

    The thresholds are crude on purpose. What matters is that the classification
    is reproducible from stored data and would work identically on a real
    vehicle - where nobody hands you a label saying "this was motorway driving".
    """
    if speed_kph < REGIME_IDLE_BELOW_KPH:
        return "idle"
    if speed_kph >= REGIME_HIGHWAY_AT_LEAST_KPH:
        return "highway"
    return "city"


def _contiguous_blocks(speed: list[tuple[int, float]]) -> list[list[int]]:
    """Split the timeline into drives at silences longer than DRIVE_BREAK_NS.

    This must happen BEFORE regime segmentation. Doing it afterwards lets a
    segment span the overnight gap, and integrating distance across nine hours
    of parked time at the last known speed produces a 415 km "commute" - which
    is exactly what the first version of this function did.
    """
    if not speed:
        return []
    blocks: list[list[int]] = [[0]]
    for i in range(1, len(speed)):
        if speed[i][0] - speed[i - 1][0] > DRIVE_BREAK_NS:
            blocks.append([i])
        else:
            blocks[-1].append(i)
    return blocks


def _runs(speed: list[tuple[int, float]], idx: list[int]) -> list[list]:
    """Runs of one speed regime within a drive, merging blips."""
    runs: list[list] = []
    first = classify_regime(speed[idx[0]][1])
    lo = 0
    for k in range(1, len(idx)):
        regime = classify_regime(speed[idx[k]][1])
        if regime != first:
            runs.append([first, lo, k])
            lo = k
            first = regime
    runs.append([first, lo, len(idx) - 1])

    merged: list[list] = []
    for regime, a, b in runs:
        duration = (speed[idx[b]][0] - speed[idx[a]][0]) / 1e9
        if merged and duration < MIN_SEGMENT_S:
            # A single sample across a threshold is a sensor blip, not a stint.
            merged[-1][2] = b
        else:
            merged.append([regime, a, b])
    return merged


def build_segments(conn: sqlite3.Connection) -> int:
    """Segment the timeline by speed regime and integrate energy and distance.

    Energy is ``V x I`` integrated over time - the physical definition - not the
    ``energy.per_km`` convenience signal, so the figure does not depend on the
    vehicle's own estimator being correct.
    """
    speed = _series(conn, SPEED_PATH)
    conn.execute("DELETE FROM segment")
    conn.execute("DELETE FROM drive")
    if len(speed) < 2:
        return 0

    current = dict(_series(conn, CURRENT_PATH))
    voltage = dict(_series(conn, VOLTAGE_PATH))

    cur = conn.cursor()
    n_segments = 0
    for block in _contiguous_blocks(speed):
        if len(block) < 2:
            continue
        rows = []
        for regime, a, b in _runs(speed, block):
            dist_km = energy_wh = 0.0
            for k in range(a, b):
                ts0, ts1 = speed[block[k]][0], speed[block[k + 1]][0]
                dt_s = (ts1 - ts0) / 1e9
                if dt_s <= 0 or dt_s > 2.0:
                    # 1 Hz data; anything else is a gap, and integrating across
                    # it would invent distance and energy that never happened.
                    continue
                v_kph = 0.5 * (speed[block[k]][1] + speed[block[k + 1]][1])
                dist_km += v_kph * dt_s / 3600.0
                i_a, v_v = current.get(ts0), voltage.get(ts0)
                if i_a is not None and v_v is not None:
                    energy_wh += v_v * i_a * dt_s / 3600.0

            seg_start, seg_end = speed[block[a]][0], speed[block[b]][0]
            span_s = (seg_end - seg_start) / 1e9
            mean_kph = dist_km / (span_s / 3600.0) if span_s > 0 else 0.0
            rows.append((seg_start, seg_end, regime, round(dist_km, 4),
                         round(energy_wh, 3), round(mean_kph, 2)))

        if not rows:
            continue
        cur.execute(
            "INSERT INTO drive(start_ns, end_ns, distance_km, energy_wh) VALUES (?, ?, ?, ?)",
            (rows[0][0], rows[-1][1], round(sum(r[3] for r in rows), 4),
             round(sum(r[4] for r in rows), 3)))
        drive_id = cur.lastrowid
        cur.executemany(
            "INSERT INTO segment(drive_id, start_ns, end_ns, regime, distance_km, "
            "energy_wh, mean_speed_kph) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(drive_id, *r) for r in rows])
        n_segments += len(rows)
    return n_segments


def build_daily(conn: sqlite3.Connection) -> int:
    """One row per UTC day: distance, energy, consumption, ambient, HVAC, SOC.

    Consumption is deliberately computed from integrated Wh and km, not averaged
    from per-sample Wh/km values - averaging a ratio weights every second
    equally, including the seconds when the car was stationary.
    """
    conn.execute("DELETE FROM daily")
    days = [r["day"] for r in conn.execute(
        "SELECT DISTINCT date(start_ns / 1000000000, 'unixepoch') AS day "
        "FROM segment ORDER BY day")]

    for day in days:
        agg = conn.execute(
            "SELECT COALESCE(SUM(distance_km), 0) AS km, "
            "       COALESCE(SUM(energy_wh), 0) AS wh "
            "FROM segment WHERE date(start_ns / 1000000000, 'unixepoch') = ?",
            (day,)).fetchone()
        km, wh = float(agg["km"]), float(agg["wh"])
        ambient = _day_mean(conn, AMBIENT_PATH, day)
        hvac = _day_mean(conn, HVAC_PATH, day)
        soc = _day_range(conn, SOC_PATH, day)
        conn.execute(
            "INSERT INTO daily(day, distance_km, energy_wh, wh_per_km, ambient_c, "
            "hvac_w, soc_start, soc_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (day, round(km, 3), round(wh, 2),
             round(wh / km, 2) if km > 0 else None,
             ambient, hvac, soc[0], soc[1]))
    return len(days)


def _day_mean(conn: sqlite3.Connection, path: str, day: str) -> float | None:
    row = conn.execute(
        "SELECT AVG(s.value) AS v FROM sample s JOIN signal g ON g.id = s.signal_id "
        "WHERE g.path = ? AND date(s.ts_ns / 1000000000, 'unixepoch') = ?",
        (path, day)).fetchone()
    return None if row["v"] is None else round(float(row["v"]), 2)


def _day_range(conn: sqlite3.Connection, path: str, day: str) -> tuple[float | None, float | None]:
    sid = signal_id(conn, path)
    if sid is None:
        return (None, None)
    rows = [float(r["value"]) for r in conn.execute(
        "SELECT value FROM sample WHERE signal_id = ? "
        "AND date(ts_ns / 1000000000, 'unixepoch') = ? ORDER BY ts_ns", (sid, day))]
    return (round(rows[0], 2), round(rows[-1], 2)) if rows else (None, None)


def create_views(conn: sqlite3.Connection) -> None:
    """(Re)create the analytics views. Needs a writable connection."""
    conn.executescript(VIEWS_PATH.read_text(encoding="utf-8"))


def rebuild(conn: sqlite3.Connection) -> dict[str, int]:
    """Rebuild every derived table from ``sample`` and ``event``."""
    n_seg = build_segments(conn)
    n_daily = build_daily(conn)
    create_views(conn)
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('rebuilt_at', datetime('now'))")
    return {"segments": n_seg, "days": n_daily}
