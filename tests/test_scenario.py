"""Scenario semantics.

The simulator is only useful if it produces the phenomena the demo later claims
to explain. These tests assert the *causal stories*, not just that numbers
exist - most importantly that the coolant pump fails **before** the overheat
warning fires, because "what happened before the warning?" is a question we must
be able to answer correctly before we can ask a model to answer it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("yaml", reason="PyYAML not installed")

from pathlib import Path

import numpy as np  # noqa: E402

from sim.generator import OVERHEAT_WARN_C, RATE_HZ, generate
from sim.scenario import FaultKind, Mode, load_scenario, scenario_from_dict

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "sim" / "scenarios"
SCENARIOS = ["city_highway_week", "battery_overheat", "range_degradation",
             "adas_disengagement"]


def scenario(name: str):
    """Load by name, independent of the working directory."""
    return load_scenario(SCENARIO_DIR / f"{name}.yaml")


@pytest.fixture(scope="module")
def sims():
    return {name: generate(scenario(name)) for name in SCENARIOS}


def test_all_scenarios_generate(sims):
    for name, sim in sims.items():
        assert sim.n == int(round(sim.scenario.duration_s * RATE_HZ))
        assert sim.t_s.shape == (sim.n,)
        assert sim.truth, f"{name}: no signals produced"
        for path, arr in sim.truth.items():
            assert arr.shape == (sim.n,), f"{name}/{path}: wrong length {arr.shape}"
            assert np.isfinite(arr).all(), f"{name}/{path}: non-finite values"


def test_generation_is_deterministic():
    """Same scenario, same seed, same bytes - or the demo is not repeatable."""
    sc = scenario("battery_overheat")
    a, b = generate(sc), generate(sc)
    for path in a.truth:
        np.testing.assert_array_equal(a.truth[path], b.truth[path],
                                      err_msg=f"{path} is not reproducible")
    assert [e.ts_s for e in a.events] == [e.ts_s for e in b.events]


def test_seed_changes_output():
    sc = scenario("battery_overheat")
    other = scenario_from_dict({**{
        "name": sc.name, "ambient_c": sc.ambient_c,
        "initial_soc": sc.initial_soc, "segments": [
            {"mode": s.mode.value, "duration_s": s.duration_s} for s in sc.segments],
        "faults": [],
    }, "seed": sc.seed + 1})
    a, b = generate(sc), generate(other)
    assert not np.array_equal(a.truth["vehicle.speed"], b.truth["vehicle.speed"])


def test_modes_follow_the_script(sims):
    sim = sims["city_highway_week"]
    modes = sim.modes
    for start, end, mode in sim.scenario.segment_bounds():
        lo, hi = int(start * RATE_HZ), min(int(end * RATE_HZ), sim.n)
        assert set(modes[lo:hi]) == {mode.value}, (
            f"segment {start}-{end}s should be {mode.value}")


def test_highway_is_faster_than_city(sims):
    sim = sims["city_highway_week"]
    speed = sim.truth["vehicle.speed"]
    city = speed[sim.modes == Mode.CITY.value]
    highway = speed[sim.modes == Mode.HIGHWAY.value]
    assert highway.mean() > city.mean(), (
        f"highway mean {highway.mean():.1f} should exceed city {city.mean():.1f}")
    assert highway.mean() > 60.0


def test_pump_fails_before_the_overheat_warning(sims):
    """The causal claim behind Q4: a pump failure precedes the warning.

    If this ever stops holding, the demo is answering a question with a story
    that did not happen.
    """
    sim = sims["battery_overheat"]
    pump = sim.truth["thermal.coolant_pump"]
    temp = sim.truth["battery.cell_temp_max"]
    t = sim.t_s

    assert temp.max() >= OVERHEAT_WARN_C, (
        f"scenario never crosses the warning threshold (max {temp.max():.1f} C)")

    pump_dead = np.flatnonzero(pump == 0.0)
    # ignore the first seconds while the vehicle is stationary at t=0
    pump_dead = pump_dead[t[pump_dead] > 30.0]
    assert pump_dead.size, "pump never actually stops"

    warn = np.flatnonzero(temp >= OVERHEAT_WARN_C)[0]
    assert pump_dead[0] < warn, (
        f"pump stopped at tick {pump_dead[0]} but the warning fired at {warn}; "
        "the precursor must come first")
    lead = (warn - pump_dead[0]) / RATE_HZ
    assert lead > 60.0, f"only {lead:.1f} s between pump failure and warning"


def test_overheat_warning_is_a_registered_event(sims):
    sim = sims["battery_overheat"]
    kinds = {e.kind for e in sim.events}
    assert "battery.overheat_warning" in kinds
    assert "battery.overheat_critical" in kinds
    warn = next(e for e in sim.events if e.kind == "battery.overheat_warning")
    flag = sim.truth["battery.overheat_warning"]
    first_on = sim.t_s[np.flatnonzero(flag > 0)[0]]
    assert warn.ts_s == pytest.approx(first_on, abs=0.2)


def test_baseline_never_overheats(sims):
    sim = sims["city_highway_week"]
    assert sim.truth["battery.cell_temp_max"].max() < OVERHEAT_WARN_C
    assert not sim.truth["battery.overheat_warning"].any()


def test_adas_disengagements_are_detectable(sims):
    """Q3 needs falling edges in the signal, not just an event log."""
    sim = sims["adas_disengagement"]
    adas = sim.truth["adas.state"]
    edges = np.flatnonzero((adas[:-1] > 0) & (adas[1:] == 0))
    injected = sim.scenario.faults_of(FaultKind.ADAS_DISENGAGEMENT)
    assert len(edges) == len(injected) == 4
    logged = [e for e in sim.events if e.kind == "adas.disengagement"]
    assert len(logged) == 4
    for ev, tick in zip(logged, edges, strict=True):
        assert ev.ts_s == pytest.approx(sim.t_s[tick], abs=0.2)


def test_cold_weather_increases_consumption(sims):
    """Q1/Q2 need a real, measurable efficiency difference to explain."""
    cold = sims["range_degradation"].truth["energy.per_km"]
    mild = sims["city_highway_week"].truth["energy.per_km"]
    moving_cold = cold[cold > 0]
    moving_mild = mild[mild > 0]
    assert moving_cold.mean() > moving_mild.mean() * 1.05, (
        f"cold-weather consumption {moving_cold.mean():.1f} Wh/km is not "
        f"meaningfully above mild {moving_mild.mean():.1f} Wh/km")


def test_soc_decreases_and_stays_valid(sims):
    for name, sim in sims.items():
        soc = sim.truth["battery.soc"]
        assert soc.min() >= 0.0 and soc.max() <= 100.0, f"{name}: SOC out of range"
        net_drop = soc[0] - soc[-1]
        assert net_drop > 0, f"{name}: SOC did not decrease over the drive"


def test_scenario_validation_rejects_nonsense():
    from sim.scenario import Scenario, Segment

    with pytest.raises(ValueError):
        Scenario(name="x", seed=1, ambient_c=20, initial_soc=50, segments=())
    with pytest.raises(ValueError):
        Scenario(name="x", seed=1, ambient_c=20, initial_soc=150,
                 segments=(Segment(Mode.CITY, 10),))
    with pytest.raises(ValueError):
        Segment(Mode.CITY, 0)
