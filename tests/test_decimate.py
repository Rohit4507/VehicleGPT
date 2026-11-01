"""Decimation tests.

The failure these guard against is silent. A decimator that takes every Nth
sample returns a plausible-looking series, passes any test that only checks row
counts, and quietly deletes the one-tick events the product exists to report.
"""

from __future__ import annotations

import pytest

pytest.importorskip("yaml", reason="PyYAML not installed")

from vehiclegpt.schema.registry import SignalSpec, Source, Tier  # noqa: E402
from vehiclegpt.store.decimate import (  # noqa: E402
    Policy,
    compression_ratio,
    decimate,
    policy_for,
)

HZ10 = 100_000_000          # 100 ms between samples


def spec(path="test.sig", tier=Tier.HOT, lo=0.0, hi=100.0, deadband=0.0):
    return SignalSpec(path=path, unit="-", minimum=lo, maximum=hi, rate_hz=10,
                      tier=tier, deadband=deadband, description="d",
                      sources=(Source("tesla_can.dbc", "ESP_B", "ESP_vehicleSpeed"),))


def series(values, hz=10, start=1_772_000_000_000_000_000):
    step = int(1e9 / hz)
    return [start + i * step for i in range(len(values))], [float(v) for v in values]


class TestPolicySelection:
    def test_flags_get_any(self):
        assert policy_for(spec(lo=0.0, hi=1.0)) is Policy.ANY

    def test_slow_signals_get_on_change(self):
        assert policy_for(spec(tier=Tier.SLOW, hi=200.0)) is Policy.ON_CHANGE

    def test_hot_continuous_signals_get_last(self):
        assert policy_for(spec(tier=Tier.HOT, hi=250.0)) is Policy.LAST


class TestLastPolicy:
    def test_reduces_10hz_to_1hz(self):
        ts, vals = series([10.0] * 50)          # 5 s at 10 Hz
        out = decimate(ts, vals, spec(), Policy.LAST)
        assert len(out) == 5

    def test_one_row_per_second_of_coverage(self):
        ts, vals = series(list(range(100)))
        out = decimate(ts, vals, spec(), Policy.LAST)
        assert len({s.ts_s for s in out}) == len(out), "duplicate seconds emitted"
        assert len(out) == 10                    # 100 samples at 10 Hz = 10 s

    def test_keeps_the_last_value_in_each_second(self):
        ts, vals = series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        out = decimate(ts, vals, spec(), Policy.LAST)
        assert out[0].value == 10.0

    def test_output_is_strictly_ordered(self):
        ts, vals = series([float(i % 7) for i in range(200)])
        out = decimate(ts, vals, spec(), Policy.LAST)
        assert all(b.ts_s > a.ts_s for a, b in zip(out, out[1:], strict=False))

    def test_matches_storage_granularity(self):
        """Every emitted timestamp must be a whole second, or it collides on write."""
        ts, vals = series([1.0] * 33)
        out = decimate(ts, vals, spec(), Policy.LAST)
        assert all(isinstance(s.ts_s, int) for s in out)


class TestFlagPolicy:
    def test_single_tick_pulse_survives(self):
        """The regression: a 0.1 s disengagement must not vanish."""
        vals = [0.0] * 20 + [1.0] + [0.0] * 20      # one tick set, at 10 Hz
        ts, vals = series(vals)
        out = decimate(ts, vals, spec(lo=0.0, hi=1.0), Policy.ANY)
        assert any(s.value == 1.0 for s in out), "pulse was dropped"

    def test_pulse_within_one_second_is_recorded_as_set(self):
        """0 -> 1 -> 0 inside a single second still means the flag fired."""
        vals = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ts, vals = series(vals)
        out = decimate(ts, vals, spec(lo=0.0, hi=1.0), Policy.ANY)
        assert max(s.value for s in out) == 1.0

    def test_flag_that_never_fires_is_cheap(self):
        vals = [0.0] * 600                          # 60 s
        ts, vals = series(vals)
        out = decimate(ts, vals, spec(lo=0.0, hi=1.0), Policy.ANY)
        assert len(out) <= 2, f"idle flag cost {len(out)} rows"

    def test_each_transition_is_kept(self):
        vals = [0.0] * 30 + [1.0] * 30 + [0.0] * 30
        ts, vals = series(vals)
        out = decimate(ts, vals, spec(lo=0.0, hi=1.0), Policy.ANY)
        transitions = sum(1 for a, b in zip(out, out[1:], strict=False)
                          if a.value != b.value)
        assert transitions == 2

    def test_no_duplicate_seconds(self):
        vals = [0.0, 1.0, 0.0, 1.0, 0.0] * 20
        ts, vals = series(vals)
        out = decimate(ts, vals, spec(lo=0.0, hi=1.0), Policy.ANY)
        assert len({s.ts_s for s in out}) == len(out)


class TestOnChangePolicy:
    def test_constant_series_costs_about_one_row_a_minute(self):
        """Cheap, but not free: a keyframe bounds how long a plateau can hide."""
        ts, vals = series([12.0] * 1200)          # 120 s
        out = decimate(ts, vals, spec(tier=Tier.SLOW), Policy.ON_CHANGE)
        assert len(out) <= 4, f"flat 120 s cost {len(out)} rows"
        assert len(out) >= 2, "a plateau must be bounded by at least two points"

    def test_plateau_duration_is_recoverable(self):
        """Regression: a 15-minute stall must not collapse to two endpoints.

        `_sustained_stall` needs a reading still at zero 60 s after the drop.
        Two endpoints give it nothing to compare, and it reports no stall.
        """
        vals = [18.0] * 10 + [0.0] * 9000 + [18.0] * 10     # 15 min at 10 Hz
        ts, vals = series(vals)
        out = decimate(ts, vals, spec(tier=Tier.SLOW), Policy.ON_CHANGE)
        gaps = [(b.ts_s - a.ts_s) for a, b in zip(out, out[1:], strict=False)]
        assert max(gaps) <= 60, f"gap of {max(gaps)} s makes duration unprovable"

    def test_decimated_stall_is_still_detected_by_the_analytics(self):
        """The end-to-end regression: decimation must not erase the Q4 precursor."""
        from vehiclegpt.analytics.queries import _sustained_stall

        vals = [18.0] * 10 + [0.0] * 9000 + [18.0] * 10
        ts, vals = series(vals)
        out = decimate(ts, vals, spec(tier=Tier.SLOW), Policy.ON_CHANGE)
        stored = [(s.ts_s * 10**9, s.value) for s in out]
        assert _sustained_stall(stored) is not None, "stall lost in decimation"

    def test_keeps_steps_beyond_the_deadband(self):
        vals = [10.0] * 30 + [20.0] * 30
        ts, vals = series(vals)
        out = decimate(ts, vals, spec(tier=Tier.SLOW, deadband=0.5), Policy.ON_CHANGE)
        assert [s.value for s in out][0] == 10.0
        assert 20.0 in [s.value for s in out]

    def test_ignores_changes_inside_the_deadband(self):
        vals = [10.0, 10.1, 10.2, 10.1, 10.0] * 20
        ts, vals = series(vals)
        out = decimate(ts, vals, spec(tier=Tier.SLOW, deadband=0.5), Policy.ON_CHANGE)
        assert len(out) <= 2

    def test_always_keeps_first_and_last(self):
        vals = [5.0] * 100
        ts, vals = series(vals)
        out = decimate(ts, vals, spec(tier=Tier.SLOW), Policy.ON_CHANGE)
        assert out[0].value == 5.0
        assert out[-1].ts_s == ts[-1] // 10**9


class TestInvariants:
    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="timestamps for"):
            decimate([1, 2, 3], [1.0, 2.0], spec())

    def test_empty_series(self):
        assert decimate([], [], spec()) == []

    def test_single_sample_survives_every_policy(self):
        for policy in Policy:
            out = decimate([1_772_000_000_000_000_000], [3.0], spec(), policy)
            assert len(out) == 1 and out[0].value == 3.0

    def test_never_emits_more_than_input(self):
        ts, vals = series([float(i % 3) for i in range(500)])
        for policy in Policy:
            assert len(decimate(ts, vals, spec(lo=0.0, hi=100.0), policy)) <= len(vals)

    def test_compression_ratio(self):
        assert compression_ratio(1000, 100) == 10.0
        assert compression_ratio(0, 0) == float("inf")
