"""Footprint projection tests.

footprint.py is the module that decides whether the 103 MB claim survives. It is
tested against a synthetic store whose size is known exactly, so a projection
that is quietly wrong cannot hide behind a real database.
"""

from __future__ import annotations

import sqlite3

import pytest

from vehiclegpt.store import footprint

SCHEMA = """
CREATE TABLE sample (
    signal_id INTEGER NOT NULL,
    ts_s      INTEGER NOT NULL,
    value     REAL    NOT NULL,
    quality   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (signal_id, ts_s)
) WITHOUT ROWID;
CREATE TABLE segment (id INTEGER PRIMARY KEY, drive_id INTEGER NOT NULL,
                      start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL,
                      regime TEXT NOT NULL, distance_km REAL NOT NULL,
                      energy_wh REAL NOT NULL, mean_speed_kph REAL NOT NULL);
"""


@pytest.fixture
def store(tmp_path):
    """10 hot signals x 3 600 s = one hour of driving."""
    db = tmp_path / "fp.db"
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    rows = [(sig, ts, float(sig), 0)
            for sig in range(10) for ts in range(1_000_000, 1_003_600)]
    conn.executemany("INSERT INTO sample VALUES (?,?,?,?)", rows)
    # one 1-hour segment; measure() derives drive hours from the segment table
    conn.execute("INSERT INTO segment VALUES (?,?,?,?,?,?,?,?)",
                 (1, 1, 1_000_000 * 10**9, 1_003_600 * 10**9,
                  "city", 40.0, 6000.0, 40.0))
    conn.commit()
    conn.close()
    return db


class TestMeasure:
    def test_counts_every_sample(self, store):
        fp = footprint.measure(store)
        assert fp.samples == 36_000

    def test_bytes_per_sample_is_positive_and_plausible(self, store):
        fp = footprint.measure(store)
        assert 8.0 < fp.bytes_per_sample < 200.0

    def test_derives_duty_cycle_from_segments(self, store):
        fp = footprint.measure(store)
        assert pytest.approx(1.0) == fp.drive_hours_per_day

    def test_survives_a_missing_segment_table(self, tmp_path):
        db = tmp_path / "noseg.db"
        conn = sqlite3.connect(db)
        conn.executescript(SCHEMA.split("CREATE TABLE segment")[0])
        conn.execute("INSERT INTO sample VALUES (1,1,1.0,0)")
        conn.commit()
        conn.close()
        fp = footprint.measure(db)          # must not raise
        assert fp.samples == 1
        assert fp.drive_hours == 0.0

    def test_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            footprint.measure(tmp_path / "nope.db")


class TestProjections:
    def test_driving_scales_linearly_with_days(self, store):
        fp = footprint.measure(store)
        one, four = footprint.project_driving(fp, 7), footprint.project_driving(fp, 28)
        assert pytest.approx(4.0, rel=0.02) == four["mb"] / one["mb"]

    def test_continuous_is_much_larger_than_driving(self, store):
        fp = footprint.measure(store)
        assert (footprint.project_continuous(fp, 15, 28)["mb"]
                > footprint.project_driving(fp, 28)["mb"])

    def test_continuous_matches_hand_calculation(self, store):
        fp = footprint.measure(store)
        proj = footprint.project_continuous(fp, hot_signals=15, days=28)
        expected = (fp.bytes_per_sample * 15 * footprint.SECONDS_PER_DAY * 28) / 1e6
        assert pytest.approx(expected, rel=0.02) == proj["mb"]

    def test_zero_duty_cycle_cannot_project(self):
        fp = footprint.Footprint(db_path="x", db_bytes=0, samples=0, signals=0,
                                 sampled_seconds=0, drive_hours=0.0,
                                 days_covered=0)
        proj = footprint.project_driving(fp, 28)
        assert proj.get("available") is False


class TestBudget:
    def test_required_bytes_per_sample_is_under_three(self):
        r = footprint.required_bytes_per_sample(103.17, 15, 28)
        assert pytest.approx(2.84, abs=0.05) == r

    def test_requirement_falls_as_signals_rise(self):
        assert (footprint.required_bytes_per_sample(103.17, 60, 28)
                < footprint.required_bytes_per_sample(103.17, 15, 28))

    def test_verdict_is_not_a_bare_boolean(self, store):
        """Both assumptions must be reported, or the number flatters the code."""
        a = footprint.assess(footprint.measure(store))
        assert a["driving_within_budget"] is True
        assert a["continuous_within_budget"] is False
        assert a["overhead_factor_vs_model"] > 1.0

    def test_report_names_both_assumptions(self, store):
        fp = footprint.measure(store)
        text = footprint.report(fp, footprint.assess(fp))
        assert "24/7" in text and "duty cycle" in text
