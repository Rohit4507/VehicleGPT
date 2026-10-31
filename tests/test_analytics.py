"""Analytics layer tests.

These run against a small hand-built store rather than the 14-day program, so
the expected answers are known exactly and a failure names the arithmetic that
broke rather than "the number moved".
"""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("yaml", reason="PyYAML not installed")

from pathlib import Path  # noqa: E402

from vehiclegpt.analytics import queries as Q  # noqa: E402
from vehiclegpt.store.db import (  # noqa: E402  # noqa: E402
    EventRow,
    SampleRow,
    connect,
    ensure_signals,
    rebuild,
    write_events,
    write_samples,
)

VIEWS = Path(__file__).resolve().parents[1] / "vehiclegpt" / "analytics" / "views.sql"
DAY = 86_400
BASE = dt.datetime(2026, 3, 2, tzinfo=dt.UTC).timestamp()   # Monday


def day_ns(i: int, hour: int = 9) -> int:
    return int((BASE + i * DAY + hour * 3600) * 1e9)


def build(tmp_path, days=14, kph=60.0, amps=40.0, hvac=1000.0, ambient=15.0):
    """A synthetic store: one 10-minute drive per day with constant load."""
    conn = connect(tmp_path / "a.db")
    ids = ensure_signals(conn, [
        ("vehicle.speed", "kph", "hot"), ("battery.pack_current", "A", "hot"),
        ("battery.pack_voltage", "V", "hot"), ("hvac.power", "W", "hot"),
        ("thermal.ambient", "C", "slow"), ("battery.soc", "%", "hot"),
        ("thermal.coolant_pump", "%", "hot"), ("battery.cell_temp_max", "C", "slow"),
    ])
    n = 1200
    for d in range(days):
        start = day_ns(d)
        rows = []
        for i in range(n):
            ts = start + i * 10**9
            rows += [
                SampleRow(ts, "vehicle.speed", kph),
                SampleRow(ts, "battery.pack_current", amps),
                SampleRow(ts, "battery.pack_voltage", 390.0),
                SampleRow(ts, "hvac.power", hvac),
                SampleRow(ts, "thermal.ambient", ambient),
                SampleRow(ts, "battery.soc", 80.0 - d),
                SampleRow(ts, "thermal.coolant_pump", 20.0),
                SampleRow(ts, "battery.cell_temp_max", 25.0),
            ]
        write_samples(conn, rows, ids)
    conn.commit()
    rebuild(conn)
    conn.executescript(VIEWS.read_text())
    return conn


class TestRouting:
    @pytest.mark.parametrize("question,intent", [
        ("Why has my driving range reduced over the last two weeks?", "range_reduction"),
        ("When did fuel efficiency show the largest drop?", "efficiency_drop"),
        ("Show all ADAS disengagement events from yesterday.", "das_events"),
        ("What happened before the battery overheating warning?", "before_warning"),
        ("Compare energy consumption between city and highway driving.", "city_vs_highway"),
    ])
    def test_all_brief_questions_route(self, question, intent):
        assert Q.classify(question) == intent

    def test_unknown_question_escalates_rather_than_guessing(self):
        assert Q.classify("what is the meaning of life") is None

    def test_unknown_returns_escalation_answer(self, tmp_path):
        conn = build(tmp_path, days=3)
        a = Q.answer(conn, "sing me a song")
        assert a.intent == "unknown" and a.path == "escalate"

    def test_routing_is_case_insensitive(self):
        assert Q.classify("WHY HAS MY RANGE REDUCED?") == "range_reduction"


def two_regime(tmp_path, city_kph=30.0, city_a=20.0, hwy_kph=100.0, hwy_a=100.0):
    """One city stint then one motorway stint, with known currents."""
    conn = connect(tmp_path / "b.db")
    ids = ensure_signals(conn, [
        ("vehicle.speed", "kph", "hot"), ("battery.pack_current", "A", "hot"),
        ("battery.pack_voltage", "V", "hot")])
    rows = []
    start = day_ns(0)
    for i in range(600):
        ts = start + i * 10**9
        rows += [SampleRow(ts, "vehicle.speed", city_kph),
                 SampleRow(ts, "battery.pack_current", city_a),
                 SampleRow(ts, "battery.pack_voltage", 390.0)]
    start2 = start + 3600 * 10**9      # an hour later: a separate drive
    for i in range(600):
        ts = start2 + i * 10**9
        rows += [SampleRow(ts, "vehicle.speed", hwy_kph),
                 SampleRow(ts, "battery.pack_current", hwy_a),
                 SampleRow(ts, "battery.pack_voltage", 390.0)]
    write_samples(conn, rows, ids)
    conn.commit()
    rebuild(conn)
    conn.executescript(VIEWS.read_text())
    return conn


class TestCityVsHighway:
    def test_matches_hand_arithmetic(self, tmp_path):
        """600 s at 390 V x 20 A over 600 s of 30 kph is 260 Wh/km, exactly."""
        conn = two_regime(tmp_path, city_a=20.0, hwy_a=100.0)
        a = Q.city_vs_highway(conn)
        city_km = 599 * 30.0 / 3600
        hwy_km = 599 * 100.0 / 3600
        # 600 samples span 599 intervals, so the integrated energy uses 599 s.
        assert a.data["city"]["wh_per_km"] == pytest.approx(
            390 * 20 * (599 / 3600) / city_km, rel=1e-3)
        assert a.data["highway"]["wh_per_km"] == pytest.approx(
            390 * 100 * (599 / 3600) / hwy_km, rel=1e-3)

    def test_detects_highway_being_worse(self, tmp_path):
        conn = two_regime(tmp_path, city_a=20.0, hwy_a=100.0)
        assert Q.city_vs_highway(conn).data["highway_over_city"] > 1.0

    def test_reports_mean_speed_because_ratio_alone_misleads(self, tmp_path):
        conn = two_regime(tmp_path)
        a = Q.city_vs_highway(conn)
        assert "mean_speed_kph" in a.data["city"]
        assert "kph" in a.text

    def test_missing_regime_is_reported_not_guessed(self, tmp_path):
        conn = build(tmp_path, days=3)   # constant 60 kph -> city only
        a = Q.city_vs_highway(conn)
        assert a.data == {}
        assert a.caveats


class TestEfficiencyDrop:
    def test_finds_a_planted_step(self, tmp_path):
        """A step change on day 8 must be reported, not smoothed away."""
        conn = build(tmp_path, days=14, amps=40.0)
        ids = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM signal")}
        rows = []
        for i in range(600):        # day 8 at double the current
            ts = day_ns(8) + i * 10**9
            rows.append(SampleRow(ts, "battery.pack_current", 80.0))
        # Replace is not supported, so append higher-current samples for day 8's
        # tail: the step is large enough to dominate the daily average.
        for i in range(600, 1200):
            ts = day_ns(8) + i * 10**9
            rows.append(SampleRow(ts, "battery.pack_current", 80.0))
            rows.append(SampleRow(ts, "vehicle.speed", 60.0))
            rows.append(SampleRow(ts, "battery.pack_voltage", 390.0))
        write_samples(conn, rows, ids)
        conn.commit()
        rebuild(conn)
        conn.executescript(VIEWS.read_text())

        a = Q.efficiency_largest_drop(conn)
        assert a.intent == "efficiency_largest_drop"
        assert a.data["delta_wh_per_km"] > 0

    def test_excludes_short_days(self, tmp_path):
        conn = build(tmp_path, days=14)
        a = Q.efficiency_largest_drop(conn, min_km=10.0)
        assert a.data, "20 km days should clear a 10 km filter"
        assert "excluded" in " ".join(a.caveats)

    def test_filter_can_exclude_everything(self, tmp_path):
        conn = build(tmp_path, days=14)
        a = Q.efficiency_largest_drop(conn, min_km=1000.0)
        assert a.data == {}

    def test_flat_history_reports_small_delta(self, tmp_path):
        conn = build(tmp_path, days=14)
        a = Q.efficiency_largest_drop(conn)
        assert abs(a.data["delta_wh_per_km"]) < 5.0


class TestRangeReduction:
    def test_needs_two_weeks(self, tmp_path):
        conn = build(tmp_path, days=5)
        a = Q.answer(conn, "Why has my range reduced?")
        assert a.data == {}
        assert a.caveats

    def test_reports_hvac_share_as_a_percentage_of_total(self, tmp_path):
        conn = build(tmp_path, days=14)
        a = Q.answer(conn, "Why has my range reduced over the last two weeks?")
        assert 0 <= a.data["hvac_share_now_pct"] <= 100
        assert "Wh/km" in a.text

    def test_hvac_share_rises_with_hvac_draw(self, tmp_path):
        (tmp_path / "low").mkdir()
        (tmp_path / "high").mkdir()
        low = build(tmp_path / "low", days=14, hvac=500.0)
        high = build(tmp_path / "high", days=14, hvac=6000.0)
        a_low = Q.answer(low, "Why has my range reduced?")
        a_high = Q.answer(high, "Why has my range reduced?")
        assert a_high.data["hvac_share_now_pct"] > a_low.data["hvac_share_now_pct"]


class TestAdasEvents:
    def test_lists_events_for_a_day(self, tmp_path):
        conn = build(tmp_path, days=5)
        write_events(conn, [
            EventRow(day_ns(3) + 10**9, "adas.disengagement", "info", "ADAS disengaged"),
            EventRow(day_ns(3) + 100 * 10**9, "adas.disengagement", "info", "ADAS disengaged"),
            EventRow(day_ns(4) + 10**9, "adas.disengagement", "info", "ADAS disengaged"),
        ])
        conn.commit()
        conn.executescript(VIEWS.read_text())
        day = dt.datetime.fromtimestamp(day_ns(3) / 1e9, dt.UTC).date().isoformat()
        a = Q.adas_events(conn, day)
        assert a.data["count"] == 2
        assert len(a.evidence) == 2

    def test_injected_faults_are_not_counted_as_vehicle_events(self, tmp_path):
        """Harness metadata must not appear in an answer about the vehicle."""
        conn = build(tmp_path, days=3)
        write_events(conn, [
            EventRow(day_ns(2) + 10**9, "fault.adas_disengagement", "info", "injected"),
        ])
        conn.commit()
        conn.executescript(VIEWS.read_text())
        day = dt.datetime.fromtimestamp(day_ns(2) / 1e9, dt.UTC).date().isoformat()
        assert Q.adas_events(conn, day).data["count"] == 0

    def test_empty_day_says_so(self, tmp_path):
        conn = build(tmp_path, days=3)
        a = Q.adas_events(conn, "2020-01-01")
        assert a.data["count"] == 0 and "No ADAS" in a.text


class TestBeforeWarning:
    def test_requires_a_sustained_stall_not_a_traffic_stop(self):
        """A pump dipping to zero at a red light is not a failure."""
        pump = [(i * 10**9, v) for i, v in
                enumerate([20.0, 18.0, 0.0, 18.0, 0.0, 18.0])]
        assert Q._sustained_stall(pump, min_s=60) is None

    def test_short_zero_run_is_ignored(self):
        """A 30-second stop is a junction, not a failed pump."""
        pump = [(i * 10**9, 18.0) for i in range(10)]
        pump += [(i * 10**9, 0.0) for i in range(10, 40)]
        pump += [(i * 10**9, 18.0) for i in range(40, 200)]
        assert Q._sustained_stall(pump, min_s=60) is None

    def test_detects_a_stall_that_persists(self):
        pump = [(i * 10**9, 18.0) for i in range(10)]
        pump += [(i * 10**9, 0.0) for i in range(10, 200)]
        assert Q._sustained_stall(pump, min_s=60) == 10 * 10**9

    def test_returns_the_first_sample_of_the_zero_run(self):
        pump = [(0, 20.0), (5 * 10**9, 0.0), (6 * 10**9, 0.0), (100 * 10**9, 0.0)]
        assert Q._sustained_stall(pump, min_s=60) == 5 * 10**9

    def test_no_warning_returns_explanatory_answer(self, tmp_path):
        conn = build(tmp_path, days=3)
        a = Q.before_warning(conn)
        assert a.intent == "before_warning"
        assert a.data == {}
        assert a.caveats

    def test_builds_the_sequence_from_signals(self, tmp_path):
        conn = build(tmp_path, days=3)
        warn_ns = day_ns(2) + 3600 * 10**9
        write_events(conn, [
            EventRow(warn_ns, "battery.overheat_warning", "warning", "pack hot"),
            # A harness row that must not be used as evidence.
            EventRow(warn_ns - 1200 * 10**9, "fault.coolant_pump_failure", "info", "x"),
        ])
        ids = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM signal")}
        rows = []
        for i in range(0, 1800):
            ts = warn_ns - 1800 * 10**9 + i * 10**9
            stalled = i >= 600
            rows.append(SampleRow(ts, "thermal.coolant_pump", 0.0 if stalled else 18.0))
            rows.append(SampleRow(ts, "battery.cell_temp_max", 60.0 if i >= 1500 else 30.0))
        write_samples(conn, rows, ids)
        conn.commit()
        conn.executescript(VIEWS.read_text())

        a = Q.before_warning(conn, lookback_min=30)
        assert a.data["stall_time"] is not None
        assert a.data["lead_minutes"] > 0
        assert "pump" in a.text
