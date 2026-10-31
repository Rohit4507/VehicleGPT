"""Change-point tests, including a cross-check against `ruptures`.

Writing our own PELT is only defensible if it agrees with a reference
implementation, so the agreement test runs against ``ruptures`` on a few hundred
random series. It is skipped if the library is absent - the detector itself has
no dependency on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vehiclegpt.analytics import changepoint as cp


def step(n1=50, n2=50, level1=0.0, level2=10.0, noise=0.5, seed=1):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(level1, noise, n1),
                           rng.normal(level2, noise, n2)])


class TestPelt:
    def test_single_step_is_found(self):
        bounds = cp.pelt(step(), penalty=10.0)
        assert bounds[0] == 0 and bounds[-1] == 100
        assert len(bounds) == 3, f"expected one breakpoint, got {bounds}"
        assert 45 <= bounds[1] <= 55, f"breakpoint at {bounds[1]}, expected ~50"

    def test_flat_series_has_no_breakpoints(self):
        assert cp.pelt([3.0] * 40, penalty=10.0) == [0, 40]

    def test_short_series_returns_one_segment(self):
        assert cp.pelt([1.0, 2.0], penalty=1.0) == [0, 2]
        assert cp.pelt([], penalty=1.0) == [0, 0]

    def test_bounds_are_ordered_and_bounded(self):
        bounds = cp.pelt(step(), penalty=5.0)
        assert bounds == sorted(bounds)
        assert all(0 <= b <= 100 for b in bounds)

    def test_min_segment_is_respected(self):
        # two changes 1 sample apart cannot both be reported
        bounds = cp.pelt(step(50, 50), penalty=1.0, min_segment=10)
        gaps = [b - a for a, b in zip(bounds, bounds[1:], strict=False)]
        assert all(g >= 1 for g in gaps)

    def test_penalty_zero_finds_every_deviation(self):
        """A penalty of zero must not silently behave like 'no changes'."""
        bounds = cp.pelt(step(noise=0.0, level1=0.0, level2=10.0), penalty=0.0)
        assert len(bounds) > 2

    def test_higher_penalty_never_adds_segments(self):
        x = step()
        counts = [len(cp.pelt(x, p)) for p in (1.0, 5.0, 20.0, 100.0)]
        assert counts == sorted(counts, reverse=True), counts

    def test_deterministic(self):
        x = step()
        assert cp.pelt(x, 10.0) == cp.pelt(x, 10.0)

    def test_rejects_two_dimensional_input(self):
        with pytest.raises(ValueError, match="1-D"):
            cp.pelt([[1.0, 2.0], [3.0, 4.0]], penalty=1.0)


class TestSegments:
    def test_reports_means_and_deltas(self):
        segs = cp.segments(step(level1=100.0, level2=200.0, noise=0.5), penalty=10.0)
        assert len(segs) == 2
        assert segs[0]["mean"] == pytest.approx(100.0, abs=1.0)
        assert segs[1]["mean"] == pytest.approx(200.0, abs=1.0)
        assert segs[0]["delta"] is None
        assert segs[1]["delta"] == pytest.approx(100.0, abs=2.0)

    def test_spans_cover_the_series_exactly(self):
        x = step()
        segs = cp.segments(x, penalty=10.0)
        assert segs[0]["start"] == 0
        assert segs[-1]["end"] == len(x)
        for a, b in zip(segs, segs[1:], strict=False):
            assert a["end"] == b["start"], "segments must not overlap or skip"

    def test_first_change_points_at_the_last_boundary(self):
        x = np.concatenate([np.full(20, 1.0), np.full(20, 5.0)])
        assert cp.first_change(x, penalty=1.0) == 20

    def test_first_change_is_none_for_a_flat_series(self):
        assert cp.first_change([2.0] * 30, penalty=10.0) is None

    def test_bic_penalty_scales_with_noise(self):
        quiet = cp.bic_penalty(step(noise=0.1))
        noisy = cp.bic_penalty(step(noise=10.0))
        assert noisy > quiet


class TestCusum:
    def test_detects_an_upward_shift(self):
        alarms = cp.cusum(step(level1=0.0, level2=6.0))
        assert alarms, "no alarm raised for a 6-sigma shift"
        assert alarms[0]["direction"] == "up"
        assert alarms[0]["index"] >= 50, "fired before the change"

    def test_detects_a_downward_shift(self):
        alarms = cp.cusum(step(level1=10.0, level2=4.0))
        assert alarms and alarms[0]["direction"] == "down"

    def test_detects_a_two_sigma_shift(self):
        """The threshold is not so conservative that a real shift slips past."""
        rng = np.random.default_rng(5)
        detected = 0
        for _ in range(20):
            x = np.concatenate([rng.normal(0, 1, 200), rng.normal(2, 1, 300)])
            detected += any(a["index"] >= 200 for a in cp.cusum(x))
        assert detected >= 18, f"only {detected}/20 two-sigma shifts detected"

    def test_false_alarm_rate_is_bounded_and_measured(self):
        """A sequential test always has some false-alarm rate; it is not zero."""
        rate = cp.false_alarm_rate(n=500, h=6.0, trials=120, seed=0)
        assert rate < 1.5, f"{rate:.2f} false alarms per 500 quiet samples"
        assert cp.false_alarm_rate(h=4.0, trials=60) > rate, "h must trade off"

    def test_constant_series_produces_no_alarms(self):
        assert cp.cusum([5.0] * 100) == []

    def test_short_series_produces_no_alarms(self):
        assert cp.cusum([1.0, 2.0, 3.0]) == []

    def test_a_sustained_shift_alarms_once(self):
        """Hysteresis: one change must not become a cluster of alarms."""
        alarms = cp.cusum(step(level1=0.0, level2=6.0))
        early = [a for a in alarms if 50 <= a["index"] <= 99]
        assert len(early) == 1, f"{len(early)} alarms for one change: {early}"

    def test_reports_the_lag_rather_than_pretending(self):
        first = cp.cusum(step(level1=0.0, level2=6.0))[0]
        assert first["lag_samples"] == first["index"] - first["start_index"]
        assert first["lag_samples"] >= 0

    def test_start_index_brackets_the_change(self):
        alarms = cp.cusum(step(level1=0.0, level2=6.0))
        assert 40 <= alarms[0]["start_index"] <= 60

    def test_baseline_window_is_not_contaminated_by_the_shift(self):
        """Standardising with the whole series makes the low half look anomalous."""
        x = np.concatenate([np.full(60, 0.0), np.full(60, 6.0)])
        alarms = cp.cusum(x)
        assert alarms, "a clean step should be detected"
        assert alarms[0]["index"] >= 60


class TestAgainstRuptures:
    """Our PELT must agree with the reference implementation, or be shown not to."""

    @pytest.mark.parametrize("seed", [7, 11, 23])
    def test_breakpoints_match_ruptures(self, seed):
        rpt = pytest.importorskip("ruptures")
        rng = np.random.default_rng(seed)
        agreed = total = 0
        for _ in range(40):
            n = int(rng.integers(20, 160))
            k = int(rng.integers(1, min(4, n // 10) + 1))
            cuts = sorted(rng.choice(np.arange(5, n - 5), size=k, replace=False))
            means = rng.normal(0, 5, k + 1)
            x = np.concatenate([rng.normal(m, 1.0, b - a) for a, b, m in
                                zip([0, *cuts], [*cuts, n], means, strict=False)])
            penalty = 5.0
            mine = [b for b in cp.pelt(x, penalty) if 0 < b < n]
            theirs = [b for b in sorted(
                rpt.Pelt(model="l2", min_size=2, jump=1).fit(x).predict(pen=penalty))
                if 0 < b < n]
            total += 1
            agreed += mine == theirs
        # Random series occasionally admit two equally good segmentations; a
        # handful of ties is fine, a systematic difference is not.
        assert agreed / total >= 0.97, f"agreement {agreed}/{total} with ruptures"
