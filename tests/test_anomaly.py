"""Anomaly-scoring tests.

The claim this module makes is specific: a robust score keeps working when a
sizeable fraction of the data is anomalous, where mean/std does not. That is
tested directly rather than asserted, and the limits (symmetry, one signal at a
time) are tested too, so the documentation cannot drift away from the behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vehiclegpt.analytics import anomaly as an


class TestCentreScale:
    def test_matches_hand_calculation(self):
        centre, scale = an.center_scale([1.0, 2.0, 3.0, 4.0, 5.0])
        assert centre == 3.0
        assert scale == pytest.approx(1.4826 * 1.0)   # MAD here is 1

    def test_constant_series_has_zero_scale(self):
        centre, scale = an.center_scale([7.0] * 10)
        assert (centre, scale) == (7.0, 0.0)

    def test_mad_zero_falls_back_instead_of_dividing_by_zero(self):
        """More than half identical is normal for a vehicle signal, not exotic."""
        x = np.array([3.0] * 8 + [3.5, 9.0])
        centre, scale = an.center_scale(x)
        assert centre == 3.0
        assert scale > 0, "MAD collapsed but the series still varies"
        assert math.isfinite(scale)

    def test_empty_series(self):
        assert an.center_scale([]) == (0.0, 0.0)

    def test_scale_is_not_dragged_by_a_single_outlier(self):
        clean = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        dirty = np.append(clean, 10_000.0)
        assert an.center_scale(dirty)[1] == pytest.approx(
            an.center_scale(clean)[1], rel=1.0)


class TestZScore:
    def test_constant_series_scores_zero_everywhere(self):
        z = an.zscore([4.0] * 20)
        assert np.all(z == 0.0)

    def test_never_returns_nan_or_inf(self):
        for x in ([1.0] * 20, [1.0, 2.0], [0.0] * 50 + [1e9], []):
            z = an.zscore(x)
            assert np.all(np.isfinite(z)), x[:5]

    def test_outlier_scores_highest(self):
        x = [10.0, 10.2, 9.8, 10.1, 9.9, 50.0, 10.0]
        z = an.zscore(x)
        assert int(np.argmax(z)) == 5

    def test_outlier_does_not_hide_itself(self):
        """The failure mode of mean/std: the anomaly inflates the scale.

        With 20 clean points and one 500-unit excursion, the classic z-score is
        capped near sqrt(n) ~ 4.6 no matter how large the excursion gets, and
        with a few such points it drops below the threshold entirely. The robust
        score keeps rising with the size of the excursion.
        """
        clean = np.random.default_rng(0).normal(0.0, 1.0, 20)
        for spike in (6.0, 60.0, 6000.0):
            x = np.append(clean, spike)
            classic = abs(x[-1] - x.mean()) / x.std()
            robust = an.zscore(x)[-1]
            assert robust > classic, f"spike={spike}: robust {robust} <= classic {classic}"

    def test_two_large_spikes_break_the_classic_score_but_not_this_one(self):
        rng = np.random.default_rng(2)
        x = np.concatenate([rng.normal(0, 1, 18), [40.0, 45.0]])
        classic = (x[-1] - x.mean()) / x.std()
        assert classic < 3.5, "mean/std should be fooled here"
        assert an.zscore(x)[-1] > 3.5


class TestLocalLevel:
    def test_a_ramp_is_fitted_exactly(self):
        """Regression: a shrinking median window flagged every edge of a ramp."""
        ramp = np.linspace(20.0, 60.0, 60)
        assert np.abs(an.residuals(ramp, window=9)).max() < 1e-9

    def test_a_step_is_not_fitted_away(self):
        x = np.concatenate([np.zeros(30), np.full(30, 5.0)])
        assert np.abs(an.residuals(x, window=5)[:30]).max() > 1.0

    def test_no_nan_at_the_edges(self):
        assert np.all(np.isfinite(an.local_level(np.arange(20.0), window=7)))


class TestRollingMedian:
    def test_edge_windows_shrink_rather_than_pad(self):
        """Padding would manufacture anomalies at the ends of every series."""
        x = np.arange(10.0)
        med = an.rolling_median(x, window=5)
        assert med[0] == pytest.approx(np.median(x[:3]))
        assert med[-1] == pytest.approx(np.median(x[-3:]))

    def test_odd_and_larger_than_series_windows(self):
        assert an.rolling_median([1.0, 2.0, 3.0], window=99).tolist() == [2.0, 2.0, 2.0]

    def test_rejects_non_positive_window(self):
        with pytest.raises(ValueError, match="must be positive"):
            an.rolling_median([1.0, 2.0], window=0)

    def test_empty(self):
        assert an.rolling_median([], 3).size == 0


class TestResidualsAndDetect:
    def test_a_linear_trend_produces_no_anomalies(self):
        """A slow warm-up is not a fault; the residual removes the local level."""
        ramp = np.linspace(20.0, 60.0, 60)
        assert an.detect(ramp, threshold=3.5, window=9) == []

    def test_a_single_excursion_on_a_flat_signal_is_flagged(self):
        x = np.full(40, 18.0)
        x += np.random.default_rng(1).normal(0, 0.05, 40)
        x[25] = 0.0                     # pump stops for one sample
        hits = an.detect(x, threshold=3.5, window=7)
        assert hits[0]["index"] == 25, hits

    def test_a_spike_outranks_its_own_smear(self):
        """A local fit smears a spike onto its neighbour; ranking must survive.

        The window containing the spike fits through the spike, so the adjacent
        point gets a residual in the opposite direction. That neighbour is an
        artefact, not a second fault, and it must not outrank the real one.
        """
        x = np.full(40, 18.0) + np.random.default_rng(1).normal(0, 0.05, 40)
        x[25] = 0.0
        hits = an.detect(x, threshold=3.5, window=7)
        assert hits[0]["score"] > 10 * hits[1]["score"]

    def test_results_are_sorted_worst_first(self):
        x = np.zeros(50)
        x[10] = 3.0
        x[30] = 30.0
        hits = an.detect(x, threshold=2.0)
        assert hits[0]["index"] == 30
        assert hits[0]["score"] >= hits[-1]["score"]

    def test_top_limits_the_output(self):
        x = np.zeros(60)
        x[[5, 15, 25, 35]] = [5.0, 9.0, 20.0, 40.0]
        assert len(an.detect(x, threshold=1.0, top=2)) == 2

    def test_threshold_is_respected(self):
        x = np.zeros(30)
        x[10] = 2.0
        assert an.detect(x, threshold=100.0) == []

    def test_empty_series(self):
        assert an.detect([]) == []

    def test_flags_include_value_and_score(self):
        x = np.zeros(20)
        x[7] = 5.0
        hit = an.detect(x, threshold=2.0)[0]
        assert set(hit) == {"index", "value", "score", "delta"}
        assert hit["value"] == 5.0
        assert hit["delta"] == pytest.approx(5.0, abs=0.5)

    def test_min_delta_filters_physically_trivial_extremes(self):
        """A large z on a tiny deviation is statistically real and useless."""
        x = np.full(60, 20.0) + np.random.default_rng(0).normal(0, 0.001, 60)
        x[30] = 20.05                      # 50x the noise, but 0.05 C
        assert an.detect(x, threshold=3.5), "the score should fire"
        assert an.detect(x, threshold=3.5, min_delta=1.0) == []


class TestDocumentedLimits:
    """The caveats in the docstring are behaviour, so they are tested as such."""

    def test_scoring_is_two_sided_by_design(self):
        """A cold excursion scores the same as a hot one of equal size."""
        x = np.zeros(30)
        x[10], x[20] = 5.0, -5.0
        scores = {h["index"]: h["score"] for h in an.detect(x, threshold=2.0)}
        assert scores[10] == pytest.approx(scores[20])

    def test_joint_outlier_is_missed_when_signals_are_scored_alone(self):
        """Per-signal scoring cannot see a point that is normal in each margin."""
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 300)
        b = rng.normal(0, 1, 300)
        a[150], b[150] = 2.2, 2.2          # unremarkable alone, unusual together
        # Not "no alarms at all" - a 300-point Gaussian series produces a few
        # tail hits, which is the false-alarm rate working as measured. The
        # claim is narrower: this particular point is invisible to either
        # signal scored alone.
        assert 150 not in [h["index"] for h in an.detect(a, threshold=3.5)]
        assert 150 not in [h["index"] for h in an.detect(b, threshold=3.5)]
        # ...whereas a multivariate detector separates it - measured, not claimed.
        sklearn = pytest.importorskip("sklearn.ensemble")
        xy = np.column_stack([a, b])
        iso = sklearn.IsolationForest(random_state=0, contamination="auto").fit(xy)
        assert iso.score_samples(xy)[150] < np.median(iso.score_samples(xy))
