"""Episode-distillation tests.

An episode card is a *claim* about what happened, written in prose, and prose is
hard to check. These tests pin the rules that keep the prose honest: which rows
may become evidence, when two events are one episode, and which signals are
relevant to which incident.
"""

from __future__ import annotations

import pytest

from vehiclegpt.analytics import episodes
from vehiclegpt.store import db

BASE = 1_773_200_000        # arbitrary epoch second used as t0


@pytest.fixture
def store(tmp_path):
    conn = db.connect(tmp_path / "ep.db")
    conn.executemany("INSERT INTO signal (path, unit, tier) VALUES (?, ?, 'hot')",
                     [("thermal.coolant_pump", "%"), ("battery.cell_temp_max", "C"),
                      ("battery.cell_imbalance", "C")])
    ids = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM signal")}

    pump, temp, imb = ids["thermal.coolant_pump"], ids["battery.cell_temp_max"], \
        ids["battery.cell_imbalance"]
    rows = []
    # pump runs at 18 % then stops at t0+600 for 15 minutes
    for t in range(0, 600, 60):
        rows.append((pump, BASE + t, 18.0, 0))
    for t in range(600, 1380, 60):
        rows.append((pump, BASE + t, 0.0, 0))
    rows.append((pump, BASE + 1440, 18.0, 0))      # the pump restarts
    # temperature climbs through 55 and 60 C
    for i, t in enumerate(range(0, 1500, 60)):
        rows.append((temp, BASE + t, 20.0 + i * 3.0, 0))
    for i, t in enumerate(range(0, 1500, 60)):
        rows.append((imb, BASE + t, 1.0 + (i % 3) * 0.1, 0))
    # Named columns on purpose: `sample` is (ts_s, signal_id, value, quality)
    # and a positional insert silently puts the signal id in the timestamp.
    conn.executemany(
        "INSERT INTO sample (signal_id, ts_s, value, quality) VALUES (?,?,?,?)", rows)
    conn.commit()
    yield conn
    conn.close()


def add_event(conn, offset_s, kind, severity="info"):
    conn.execute("INSERT INTO event (ts_ns, kind, severity, message) VALUES (?,?,?,?)",
                 ((BASE + offset_s) * 10**9, kind, severity, "x"))
    conn.commit()


class TestGrouping:
    def test_events_within_the_gap_become_one_episode(self, store):
        add_event(store, 1000, "battery.overheat_warning", "warning")
        add_event(store, 1200, "battery.overheat_critical", "critical")
        eps = episodes.incident_episodes(store)
        assert len(eps) == 1, [e.title for e in eps]
        assert eps[0].severity == "critical"

    def test_events_beyond_the_gap_are_separate(self, store):
        add_event(store, 0, "adas.disengagement")
        add_event(store, 5000, "adas.disengagement")
        assert len(episodes.incident_episodes(store)) == 2

    def test_episode_spans_its_events(self, store):
        add_event(store, 1000, "battery.overheat_warning", "warning")
        add_event(store, 1200, "battery.overheat_critical", "critical")
        ep = episodes.incident_episodes(store)[0]
        assert ep.start_ns == (BASE + 1000) * 10**9
        assert ep.end_ns == (BASE + 1200) * 10**9

    def test_no_events_means_no_episodes(self, store):
        assert episodes.incident_episodes(store) == []


class TestEvidenceRules:
    def test_harness_faults_are_never_evidence(self, store):
        add_event(store, 1000, "fault.adas_disengagement")
        add_event(store, 1100, "adas.disengagement")
        eps = episodes.incident_episodes(store)
        assert len(eps) == 1
        kinds = {e["kind"] for e in eps[0].evidence if e["type"] == "event"}
        assert not any(k.startswith("fault.") for k in kinds)

    def test_every_episode_carries_provenance(self, store):
        add_event(store, 1000, "battery.overheat_warning", "warning")
        ep = episodes.incident_episodes(store)[0]
        assert ep.evidence
        assert all({"type"} <= set(e) for e in ep.evidence)

    def test_an_episode_of_only_faults_does_not_exist(self, store):
        add_event(store, 1000, "fault.coolant_pump_failure")
        assert episodes.incident_episodes(store) == []


class TestRelevance:
    def test_adas_episodes_do_not_cite_battery_signals(self, store):
        """Co-located context is not context."""
        add_event(store, 1000, "adas.disengagement")
        ep = episodes.incident_episodes(store)[0]
        assert not any("coolant" in s or "cell" in s for s in ep.signals)
        assert "imbalance" not in ep.body

    def test_battery_episodes_narrate_the_pump_stall(self, store):
        add_event(store, 1400, "battery.overheat_warning", "warning")
        ep = episodes.incident_episodes(store)[0]
        assert "coolant pump stopped" in ep.body

    def test_the_stall_is_reported_with_its_real_length(self, store):
        """Regression: the time needed to prove the stall was reported as its length."""
        add_event(store, 1400, "battery.overheat_warning", "warning")
        body = episodes.incident_episodes(store)[0].body
        # off from t+600 to t+1440, when the pump restarts: 14 minutes
        assert "14 minutes" in body, body

    def test_temperature_milestones_are_reported(self, store):
        add_event(store, 1400, "battery.overheat_warning", "warning")
        body = episodes.incident_episodes(store)[0].body
        assert "55 C" in body or "60 C" in body


class TestTrendEpisodes:
    def test_a_sustained_step_becomes_a_card(self, store):
        rows = [(f"2026-03-{d:02d}", 190.0, 40.0) for d in range(1, 8)]
        rows += [(f"2026-03-{d:02d}", 250.0, 40.0) for d in range(8, 15)]
        store.executemany(
            "INSERT INTO daily (day, wh_per_km, distance_km) VALUES (?,?,?)", rows)
        store.commit()
        eps = episodes.trend_episodes(store)
        assert eps, "a 60 Wh/km step should produce a card"
        assert eps[-1].severity == "warning"

    def test_short_days_are_excluded(self, store):
        rows = [(f"2026-03-{d:02d}", 190.0, 40.0) for d in range(1, 8)]
        # short days at a wildly different level must not create the card
        rows += [(f"2026-03-{d:02d}", 400.0, 5.0) for d in range(8, 15)]
        store.executemany(
            "INSERT INTO daily (day, wh_per_km, distance_km) VALUES (?,?,?)", rows)
        store.commit()
        assert episodes.trend_episodes(store) == []

    def test_too_few_days_is_not_an_episode(self, store):
        store.executemany("INSERT INTO daily (day, wh_per_km, distance_km) VALUES (?,?,?)",
                          [("2026-03-01", 190.0, 40.0), ("2026-03-02", 250.0, 40.0)])
        store.commit()
        assert episodes.trend_episodes(store) == []


class TestDeterminism:
    def test_distill_twice_gives_identical_cards(self, store):
        add_event(store, 1000, "battery.overheat_warning", "warning")
        add_event(store, 1200, "battery.overheat_critical", "critical")
        first = [e.to_row() for e in episodes.distill(store)]
        second = [e.to_row() for e in episodes.distill(store)]
        assert first == second

    def test_cards_are_ordered_by_time(self, store):
        add_event(store, 0, "adas.disengagement")
        add_event(store, 5000, "adas.disengagement")
        stamps = [e.start_ns for e in episodes.distill(store)]
        assert stamps == sorted(stamps)
