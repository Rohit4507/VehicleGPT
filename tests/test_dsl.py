"""Adversarial tests for the query DSL.

The DSL exists so a language model can choose *what to ask* without being able
to write SQL. That is a security claim, so it is tested adversarially rather
than illustrated with one happy path.

The central property: **the compiled SQL depends on the kind alone.** Every
value is a bound parameter, so an injection payload is not escaped - it is a
string sitting where SQLite expects a value, and SQLite binds it as such. If
someone later "optimises" the compiler into an f-string, the byte-for-byte SQL
comparison below fails immediately.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import sqlite3

import pytest

from vehiclegpt.analytics import dsl
from vehiclegpt.analytics.dsl import DslError, Kind

# Payloads a hostile or confused model might emit where a date belongs.
HOSTILE = [
    "2026-03-01'; DROP TABLE sample; --",
    "2026-03-01' OR '1'='1",
    "2026-03-01 UNION SELECT * FROM signal",
    "../../etc/passwd",
    "2026-03-01\x00",
    "'; ATTACH DATABASE '/tmp/x' AS x; --",
    "2026-03-01'--",
    "not-a-date",
    "",
]


def good(kind=Kind.ADAS_EVENTS, **over):
    payload = {"kind": str(kind), "start_day": "2026-03-01", "end_day": "2026-03-05"}
    payload.update(over)
    return payload


class TestClosedSchema:
    def test_unknown_top_level_key_is_rejected(self):
        """`{"sql": ...}` must fail loudly, not be silently dropped."""
        with pytest.raises(DslError, match="unknown field"):
            dsl.parse(good(sql="DROP TABLE sample"))

    def test_unknown_param_is_rejected(self):
        with pytest.raises(DslError, match="does not take"):
            dsl.parse(good(params={"limit": 10}))

    def test_missing_required_field(self):
        with pytest.raises(DslError, match="missing field"):
            dsl.parse({"kind": "adas_events", "start_day": "2026-03-01"})

    def test_unknown_kind_lists_the_valid_ones(self):
        with pytest.raises(DslError, match="unknown kind"):
            dsl.parse(good(kind="drop_everything"))

    def test_kind_must_be_a_string(self):
        with pytest.raises(DslError):
            dsl.parse({"kind": ["a"], "start_day": "2026-03-01",
                       "end_day": "2026-03-02"})

    def test_query_object_is_frozen(self):
        q = dsl.build(Kind.ADAS_EVENTS, "2026-03-01", "2026-03-02")
        with pytest.raises(dataclasses.FrozenInstanceError):
            q.kind = Kind.PRECURSOR


class TestDateValidation:
    @pytest.mark.parametrize("hostile", HOSTILE)
    def test_hostile_dates_are_rejected(self, hostile):
        with pytest.raises(DslError):
            dsl.parse({"kind": "adas_events", "start_day": hostile,
                       "end_day": "2026-03-05"})

    def test_reversed_window(self):
        with pytest.raises(DslError, match="before start_day"):
            dsl.parse(good(start_day="2026-03-05", end_day="2026-03-01"))

    def test_window_length_is_capped(self):
        with pytest.raises(DslError, match="limit is"):
            dsl.parse(good(start_day="2020-01-01", end_day="2026-01-01"))

    def test_defaults_never_produce_an_invalid_query(self):
        """A 2-day window with no baseline_days must not be rejected."""
        q = dsl.parse(good(kind=Kind.CONSUMPTION_TREND,
                           start_day="2026-03-01", end_day="2026-03-02"))
        assert q.params["baseline_days"] == 2

    def test_single_day_window_is_allowed(self):
        q = dsl.parse(good(start_day="2026-03-05", end_day="2026-03-05"))
        assert q.start_day == q.end_day


class TestParamValidation:
    def test_bool_is_not_an_integer(self):
        """`True` would sail through an isinstance(x, int) check."""
        with pytest.raises(DslError, match="must be an integer"):
            dsl.parse(good(kind=Kind.PRECURSOR, params={"lookback_min": True}))

    def test_negative_lookback(self):
        with pytest.raises(DslError, match="between"):
            dsl.parse(good(kind=Kind.PRECURSOR, params={"lookback_min": -5}))

    def test_string_where_a_number_belongs(self):
        with pytest.raises(DslError, match="must be an integer"):
            dsl.parse(good(kind=Kind.PRECURSOR,
                           params={"lookback_min": "60 OR 1=1"}))

    def test_baseline_longer_than_the_window(self):
        with pytest.raises(DslError, match="exceed the window"):
            dsl.parse(good(kind=Kind.CONSUMPTION_TREND,
                           params={"baseline_days": 30}))

    def test_malformed_json_is_a_dsl_error(self):
        with pytest.raises(DslError, match="not valid JSON"):
            dsl.from_json("{not json")

    def test_json_array_is_rejected(self):
        with pytest.raises(DslError, match="must be an object"):
            dsl.from_json("[1, 2, 3]")


class TestSqlIsConstantPerKind:
    """The security property: values never reach the query text."""

    @pytest.mark.parametrize("kind", list(Kind))
    def test_sql_text_is_identical_across_wildly_different_inputs(self, kind):
        params_a: dict = {}
        params_b: dict = {}
        if kind is Kind.CONSUMPTION_TREND:
            params_a, params_b = {"baseline_days": 0}, {"baseline_days": 7}
        elif kind is Kind.EFFICIENCY_DROP:
            params_a, params_b = {"min_km": 0.0}, {"min_km": 9999.0}
        elif kind is Kind.PRECURSOR:
            params_a, params_b = {"lookback_min": 1}, {"lookback_min": 1440}

        a = dsl.build(kind, "2026-03-01", "2026-03-02", **params_a)
        b = dsl.build(kind, "2025-03-09", "2026-03-09", **params_b)
        assert dsl.compile_query(a).sql == dsl.compile_query(b).sql

    @pytest.mark.parametrize("kind", list(Kind))
    @pytest.mark.parametrize("hostile", HOSTILE)
    def test_hostile_dates_never_approach_the_sql(self, kind, hostile):
        # Rejected at validation; if it were not, the SQL is still unchanged.
        try:
            q = dsl.parse({"kind": str(kind), "start_day": hostile,
                           "end_day": "2026-03-05"})
        except DslError:
            return
        compiled = dsl.compile_query(q)
        assert "DROP" not in compiled.sql.upper()
        assert hostile not in compiled.sql

    def test_every_value_is_bound_not_interpolated(self):
        q = dsl.build(Kind.CONSUMPTION_TREND, "2026-03-01", "2026-03-08",
                      baseline_days=7)
        compiled = dsl.compile_query(q)
        assert "?" not in compiled.sql, "positional placeholders hide the binding"
        assert "2026-03-01" not in compiled.sql
        for name in ("start_day", "end_day", "min_day_km"):
            assert name in compiled.params
            assert f":{name}" in compiled.sql

    def test_no_statement_separator_survives(self):
        """A trailing `;` would let a second statement be appended."""
        for kind in Kind:
            sql = dsl.compile_query(
                dsl.build(kind, "2026-03-01", "2026-03-02")).sql
            assert ";" not in sql.replace(";\n", "")


@pytest.fixture
def store(tmp_path):
    """A tiny store with the schema the DSL reads from."""
    db = tmp_path / "dsl.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE daily (day TEXT PRIMARY KEY, distance_km REAL, energy_wh REAL,
                            wh_per_km REAL, ambient_c REAL, hvac_w REAL,
                            soc_start REAL, soc_end REAL);
        CREATE TABLE segment (id INTEGER PRIMARY KEY, drive_id INTEGER,
                              start_ns INTEGER, end_ns INTEGER, regime TEXT,
                              distance_km REAL, energy_wh REAL, mean_speed_kph REAL);
        CREATE TABLE event (id INTEGER PRIMARY KEY, ts_ns INTEGER, kind TEXT,
                            severity TEXT, message TEXT);
    """)
    rows = [("2026-03-01", 40.0, 8000.0, 200.0, 10.0, 1000.0),
            ("2026-03-02", 40.0, 9200.0, 230.0, 2.0, 3000.0),
            ("2026-03-03", 40.0, 8000.0, 200.0, 3.0, 2500.0)]
    conn.executemany(
        "INSERT INTO daily (day, distance_km, energy_wh, wh_per_km, ambient_c, hvac_w) "
        "VALUES (?,?,?,?,?,?)", rows)
    conn.execute("INSERT INTO segment VALUES (1,1,0,3600000000000,'city',40,8000,25)")
    conn.execute("INSERT INTO segment VALUES (2,2,0,3600000000000,'highway',40,7000,105)")
    day_ns = int(dt.datetime(2026, 3, 2, 8, 0, tzinfo=dt.UTC).timestamp() * 1e9)
    conn.executemany("INSERT INTO event (ts_ns, kind, severity, message) VALUES (?,?,?,?)", [
        (day_ns, "adas.disengagement", "info", "ADAS disengaged"),
        (day_ns + 1_000_000_000, "fault.adas_disengagement", "info", "injected"),
    ])
    from vehiclegpt.store.db import VIEWS_PATH

    conn.executescript(VIEWS_PATH.read_text())
    conn.commit()
    conn.close()
    return db


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


class TestExecution:
    def test_adas_events_exclude_harness_faults(self, store):
        conn = connect(store)
        rows = dsl.run(conn, dsl.build(Kind.ADAS_EVENTS, "2026-03-01", "2026-03-05"))
        assert len(rows) == 1
        assert not rows[0]["message"].startswith("injected")

    def test_consumption_trend_returns_both_windows(self, store):
        conn = connect(store)
        rows = dsl.run(conn, dsl.build(Kind.CONSUMPTION_TREND, "2026-03-02",
                                       "2026-03-03", baseline_days=1))
        row = rows[0]
        assert row["a_wh"] == pytest.approx(9200 + 8000)
        assert row["b_wh"] == pytest.approx(8000)      # 1 March only

    def test_efficiency_drop_finds_the_step(self, store):
        conn = connect(store)
        rows = dsl.run(conn, dsl.build(Kind.EFFICIENCY_DROP, "2026-03-01",
                                       "2026-03-03", min_km=1.0))
        assert rows[0]["day"] == "2026-03-02"
        assert rows[0]["delta_wh_per_km"] == pytest.approx(30.0)

    def test_regime_compare_returns_both_regimes(self, store):
        conn = connect(store)
        rows = dsl.run(conn, dsl.build(Kind.REGIME_COMPARE, "2026-03-01",
                                       "2026-03-05"))
        assert {r["regime"] for r in rows} == {"city", "highway"}

    def test_empty_window_returns_no_rows_rather_than_raising(self, store):
        conn = connect(store)
        rows = dsl.run(conn, dsl.build(Kind.ADAS_EVENTS, "2025-01-01", "2025-01-02"))
        assert rows == []

    def test_a_query_built_without_parse_cannot_reach_the_database(self, store):
        """A payload smuggled in by constructing Query directly is still refused."""
        conn = connect(store)
        q = dsl.Query(kind=Kind.ADAS_EVENTS, start_day="2026-03-01",
                      end_day="2026-03-05", params={})
        object.__setattr__(q, "start_day", "2026-03-01'; DROP TABLE event; --")
        with pytest.raises(DslError, match="not ISO dates"):
            dsl.run(conn, q)
        assert conn.execute("SELECT COUNT(*) FROM event").fetchone()[0] == 2
