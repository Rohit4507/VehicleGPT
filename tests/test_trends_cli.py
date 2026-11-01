"""Tests for the trends CLI.

The behaviour worth pinning is not the printing - it is which days the tool
refuses to compare. On the shipped store the two most recent days are 7.9 km
cold trips at 321 Wh/km; a detector that includes them reports a degradation
starting the day the driving pattern changed.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "phase0" / "trends.py"


@pytest.fixture
def store(tmp_path):
    """10 comparable days at ~190 Wh/km, then 2 at ~250, plus 2 short outliers."""
    db = tmp_path / "trends.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE daily (day TEXT PRIMARY KEY, distance_km REAL,
                    energy_wh REAL, wh_per_km REAL, ambient_c REAL, hvac_w REAL,
                    soc_start REAL, soc_end REAL)""")
    conn.execute("""CREATE TABLE signal (id INTEGER PRIMARY KEY, path TEXT UNIQUE, unit TEXT)""")
    conn.execute("""CREATE TABLE sample (signal_id INTEGER, ts_s INTEGER,
                    value REAL, quality INTEGER, PRIMARY KEY (signal_id, ts_s))""")
    rows = []
    for i in range(10):
        rows.append((f"2026-03-{i + 1:02d}", 40.0, 7600.0, 190.0, 12.0, 1500.0))
    for i in range(2):
        rows.append((f"2026-03-{i + 11:02d}", 40.0, 10000.0, 250.0, -4.0, 5000.0))
    # two short days that would dominate an unfiltered analysis
    rows.append(("2026-03-13", 7.9, 2500.0, 316.0, -6.0, 5600.0))
    rows.append(("2026-03-14", 7.9, 2500.0, 316.0, -6.0, 5600.0))
    conn.executemany("INSERT INTO daily VALUES (?,?,?,?,?,?,NULL,NULL)", rows)
    conn.execute("INSERT INTO signal VALUES (1, 'battery.cell_temp_max', 'C')")
    conn.executemany("INSERT INTO sample VALUES (1, ?, ?, 0)",
                     [(1_772_000_000 + i, 20.0 + (0.001 * i)) for i in range(120)])
    conn.commit()
    conn.close()
    return db


def run(store, *args):
    out = subprocess.run([sys.executable, str(SCRIPT), "--db", str(store), *args],
                         capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_short_days_are_excluded_by_default(store):
    out = run(store, "--json")
    data = json.loads(out)
    assert data["excluded_short_days"] == 2
    assert data["points"] == 12


def test_change_point_is_the_manufactured_step(store):
    data = json.loads(run(store, "--json"))
    assert data["current_regime_start"] == "2026-03-11"
    means = [seg["mean"] for seg in data["segments"]]
    assert means[0] == pytest.approx(190.0, abs=1.0)
    assert means[-1] == pytest.approx(250.0, abs=1.0)


def test_short_days_change_the_answer_when_not_excluded(store):
    """The point of the filter, demonstrated rather than asserted."""
    filtered = json.loads(run(store, "--json"))
    unfiltered = json.loads(run(store, "--json", "--min-km", "0"))
    assert unfiltered["points"] == 14
    assert unfiltered["current_regime_start"] != filtered["current_regime_start"]


def test_min_delta_larger_than_the_step_removes_it(store):
    data = json.loads(run(store, "--json", "--min-delta", "1000"))
    assert len(data["segments"]) == 1


def test_signal_mode_reports_clock_times(store):
    out = run(store, "--signal", "battery.cell_temp_max", "--hours", "24")
    assert "level changes" in out
    assert ":" in out.split("level changes")[1], "timestamps should be readable"


def test_unknown_signal_is_a_clean_error(store):
    out = subprocess.run([sys.executable, str(SCRIPT), "--db", str(store),
                          "--signal", "nope.nothing"],
                         capture_output=True, text=True, check=False)
    assert out.returncode != 0
    assert "no signal" in out.stderr


def test_missing_store_is_a_clean_error(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), "--db", str(tmp_path / "x.db")],
                         capture_output=True, text=True, check=False)
    assert out.returncode != 0
    assert "no store" in out.stderr


def test_too_few_points_does_not_crash(store):
    out = run(store, "--metric", "ambient_c", "--min-km", "999")
    assert "nothing to analyse" in out
