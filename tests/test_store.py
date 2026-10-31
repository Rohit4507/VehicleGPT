"""Store and segmentation tests.

The segmentation bug these guard against was not subtle: the first version
integrated distance across the 23-hour gap between drives, producing a 415 km
"commute" and negative consumption. Both are physically impossible and neither
raised an error, so only an assertion catches it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("yaml", reason="PyYAML not installed")

import sqlite3  # noqa: E402
from pathlib import Path  # noqa: E402

from vehiclegpt.store import db  # noqa: E402

DAY_NS = 86_400 * 10**9
HOUR_NS = 3_600 * 10**9
BASE = 1_772_000_000 * 10**9   # arbitrary fixed epoch


def make_store(tmp_path: Path) -> sqlite3.Connection:
    return db.connect(tmp_path / "t.db")


def add(conn, path, values, start_ns, step_ns=10**9, unit="x", tier="hot"):
    ids = db.ensure_signals(conn, [(path, unit, tier)])
    rows = [db.SampleRow(ts_ns=start_ns + i * step_ns, path=path, value=float(v))
            for i, v in enumerate(values)]
    db.write_samples(conn, rows, ids)


def synth_drive(conn, start_ns: int, kph: float, seconds: int,
                current_a: float = 40.0, voltage_v: float = 390.0):
    add(conn, "vehicle.speed", [kph] * seconds, start_ns)
    add(conn, "battery.pack_current", [current_a] * seconds, start_ns)
    add(conn, "battery.pack_voltage", [voltage_v] * seconds, start_ns)


class TestSchema:
    def test_tables_created(self, tmp_path):
        conn = make_store(tmp_path)
        names = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"signal", "sample", "event", "drive", "segment", "daily", "meta"} <= names

    def test_reconnect_keeps_data(self, tmp_path):
        path = tmp_path / "t.db"
        conn = db.connect(path)
        add(conn, "vehicle.speed", [10, 20], BASE)
        conn.commit()
        conn.close()
        again = db.connect(path)
        assert again.execute("SELECT COUNT(*) AS n FROM sample").fetchone()["n"] == 2


class TestWriting:
    def test_unknown_signal_is_skipped_not_invented(self, tmp_path):
        conn = make_store(tmp_path)
        db.ensure_signals(conn, [("vehicle.speed", "kph", "hot")])
        ids = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM signal")}
        n = db.write_samples(conn, [
            db.SampleRow(ts_ns=BASE, path="vehicle.speed", value=5.0),
            db.SampleRow(ts_ns=BASE, path="vehicle.sparkle", value=1.0),
        ], ids)
        assert n == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM sample").fetchone()["n"] == 1

    def test_events_round_trip(self, tmp_path):
        conn = make_store(tmp_path)
        db.write_events(conn, [db.EventRow(BASE, "adas.disengagement", "info", "x")])
        assert conn.execute("SELECT COUNT(*) AS n FROM event").fetchone()["n"] == 1


class TestRegimeClassification:
    @pytest.mark.parametrize("kph,expected", [
        (0.0, "idle"), (4.9, "idle"), (5.0, "city"), (30.0, "city"),
        (79.9, "city"), (80.0, "highway"), (130.0, "highway"),
    ])
    def test_thresholds(self, kph, expected):
        assert db.classify_regime(kph) == expected


class TestSegmentation:
    def test_does_not_integrate_across_a_gap(self, tmp_path):
        """The regression: two drives a day apart must stay two drives."""
        conn = make_store(tmp_path)
        synth_drive(conn, BASE, kph=100.0, seconds=600)          # 10 min
        synth_drive(conn, BASE + DAY_NS, kph=100.0, seconds=600)  # next day
        db.rebuild(conn)

        drives = conn.execute("SELECT COUNT(*) AS n FROM drive").fetchone()["n"]
        assert drives == 2, "a day-long silence should split the drives"

        for row in conn.execute("SELECT distance_km FROM segment"):
            # 100 kph for 10 minutes is 16.7 km. Anything vastly larger means
            # distance was integrated across the parked gap.
            assert 15.0 < row["distance_km"] < 18.0, row["distance_km"]

    def test_energy_is_positive_and_physical(self, tmp_path):
        conn = make_store(tmp_path)
        synth_drive(conn, BASE, kph=100.0, seconds=600)
        db.rebuild(conn)
        row = conn.execute("SELECT distance_km, energy_wh FROM segment").fetchone()
        # 390 V * 40 A = 15.6 kW for 10 min = 2.6 kWh.
        assert row["energy_wh"] == pytest.approx(2600, rel=0.02)
        assert row["energy_wh"] / row["distance_km"] == pytest.approx(156, rel=0.05)

    def test_gap_within_a_drive_is_not_integrated(self, tmp_path):
        """A dropped sample inside a drive must not add phantom distance."""
        conn = make_store(tmp_path)
        add(conn, "vehicle.speed", [100.0] * 300, BASE)
        add(conn, "battery.pack_current", [40.0] * 300, BASE)
        add(conn, "battery.pack_voltage", [390.0] * 300, BASE)
        # A 5-minute hole, then more driving.
        later = BASE + 300 * 10**9 + 300 * 10**9
        add(conn, "vehicle.speed", [100.0] * 300, later)
        add(conn, "battery.pack_current", [40.0] * 300, later)
        add(conn, "battery.pack_voltage", [390.0] * 300, later)
        db.rebuild(conn)
        total = conn.execute("SELECT SUM(distance_km) AS d FROM segment").fetchone()["d"]
        # 600 s at 100 kph = 16.7 km; the 5-minute hole must not count.
        assert total == pytest.approx(16.67, rel=0.05)

    def test_minority_regime_blip_is_merged(self, tmp_path):
        conn = make_store(tmp_path)
        speeds = [30.0] * 120 + [90.0] * 2 + [30.0] * 120   # 2 s of 'highway'
        add(conn, "vehicle.speed", speeds, BASE)
        db.rebuild(conn)
        regimes = [r["regime"] for r in conn.execute("SELECT regime FROM segment")]
        assert "highway" not in regimes, f"blip became a segment: {regimes}"

    def test_idle_at_start_is_its_own_regime(self, tmp_path):
        conn = make_store(tmp_path)
        speeds = [0.0] * 60 + [50.0] * 120
        add(conn, "vehicle.speed", speeds, BASE)
        db.rebuild(conn)
        regimes = [r["regime"] for r in conn.execute(
            "SELECT regime FROM segment ORDER BY start_ns")]
        assert regimes[0] == "idle" and "city" in regimes


class TestDaily:
    def test_groups_by_utc_day(self, tmp_path):
        conn = make_store(tmp_path)
        synth_drive(conn, BASE, kph=100.0, seconds=600)
        synth_drive(conn, BASE + DAY_NS, kph=100.0, seconds=600)
        db.rebuild(conn)
        days = [r["day"] for r in conn.execute("SELECT day FROM daily ORDER BY day")]
        assert len(days) == 2

    def test_consumption_is_ratio_of_sums(self, tmp_path):
        conn = make_store(tmp_path)
        synth_drive(conn, BASE, kph=100.0, seconds=600)
        db.rebuild(conn)
        row = conn.execute("SELECT wh_per_km, distance_km, energy_wh FROM daily").fetchone()
        # wh_per_km is stored rounded to 2 dp, so compare at that resolution.
        assert row["wh_per_km"] == pytest.approx(
            row["energy_wh"] / row["distance_km"], abs=0.01)

    def test_soc_reported_as_start_and_end(self, tmp_path):
        conn = make_store(tmp_path)
        add(conn, "battery.soc", [80.0, 79.0, 78.0], BASE)
        add(conn, "vehicle.speed", [50.0] * 3, BASE)
        db.rebuild(conn)
        row = conn.execute("SELECT soc_start, soc_end FROM daily").fetchone()
        assert (row["soc_start"], row["soc_end"]) == (80.0, 78.0)


class TestRebuild:
    def test_rebuild_is_idempotent(self, tmp_path):
        conn = make_store(tmp_path)
        synth_drive(conn, BASE, kph=100.0, seconds=600)
        first = db.rebuild(conn)
        second = db.rebuild(conn)
        assert first == second
        assert conn.execute("SELECT COUNT(*) AS n FROM segment").fetchone()["n"] == \
            first["segments"]

    def test_empty_store_rebuilds_without_error(self, tmp_path):
        conn = make_store(tmp_path)
        assert db.rebuild(conn) == {"segments": 0, "days": 0}
