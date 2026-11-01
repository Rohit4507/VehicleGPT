"""Lead-lag and precedence tests.

Most of these run on synthetic series where the true lead is known exactly,
because that is the only way to know whether the sign convention and the null
test are right. The store-level tests check the plumbing, not the science.
"""

from __future__ import annotations

import numpy as np
import pytest

from vehiclegpt.analytics import leadlag as ll
from vehiclegpt.store import db


def shift_lead(x, k):
    """y lags x by k samples."""
    y = np.zeros_like(x)
    y[k:] = x[:-k]
    return y


def ar1(n, phi, rng):
    e = rng.normal(0, 1, n)
    v = np.zeros(n)
    v[0] = e[0]
    for i in range(1, n):
        v[i] = phi * v[i - 1] + e[i]
    return v


class TestCrossCorrelation:
    def test_pure_shift_correlates_exactly(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 60)
        for k in (1, 3, 7):
            corr = ll.cross_correlation(x, shift_lead(x, k), 10)
            assert np.nanmax(corr) == pytest.approx(1.0, abs=1e-9)

    def test_matches_a_direct_computation(self):
        """Regression: an index error here silently flipped every lead."""
        def naive(a, b, ml):
            out = np.full(2 * ml + 1, np.nan)
            for lag in range(-ml, ml + 1):
                if lag >= 0:
                    u, v = a[: len(a) - lag], b[lag:]
                else:
                    u, v = a[-lag:], b[: len(b) + lag]
                if u.size >= ll.MIN_OVERLAP:
                    out[lag + ml] = ll._pearson(u, v)
            return out

        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 120)
        b = np.roll(a, 3) + rng.normal(0, 0.5, 120)
        fast = ll.cross_correlation(a, b, 10)
        slow = naive(a, b, 10)
        assert np.nanmax(np.abs(fast - slow)) < 1e-9

    def test_short_overlap_is_nan_not_zero(self):
        """'No correlation' and 'not enough data' are different answers."""
        corr = ll.cross_correlation(np.arange(20.0), np.arange(20.0), 15)
        assert np.isnan(corr[0])
        assert not np.isnan(corr[15])

    def test_constant_series_yields_nan(self):
        assert np.all(np.isnan(ll.cross_correlation(np.ones(40), np.arange(40.0), 5)))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="lengths differ"):
            ll.cross_correlation([1.0, 2.0], [1.0, 2.0, 3.0], 1)


class TestSignConvention:
    def test_positive_lag_means_the_first_series_leads(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 200)
        y = shift_lead(x, 5)
        lag, corr = ll.best_lag(x, y, 20)
        assert lag == 5, "x moved first, so the lag must be positive"
        assert corr > 0.9

    def test_swapping_the_arguments_flips_the_sign(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 200)
        y = shift_lead(x, 5)
        assert ll.best_lag(y, x, 20)[0] == -5

    def test_no_lead_between_independent_series(self):
        rng = np.random.default_rng(4)
        lag, corr = ll.best_lag(rng.normal(0, 1, 400), rng.normal(0, 1, 400), 20)
        assert abs(corr) < 0.35


class TestNullTest:
    def test_a_real_lead_is_significant(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, 300)
        y = shift_lead(x, 4) + rng.normal(0, 0.2, 300)
        lag, corr = ll.best_lag(x, y, 15)
        assert ll.surrogate_p_value(x, y, 15, abs(corr)) < 0.05

    def test_the_null_is_calibrated_on_autocorrelated_noise(self):
        """The trap: independent AR(1) series show a large spurious correlation.

        With phi=0.98 and 600 samples two unrelated series reach |corr| ~ 0.33
        on average. If the p-value simply trusted that coefficient, every smooth
        pair would look causal. The circular-shift null keeps the false-positive
        rate near the nominal level instead.
        """
        rng = np.random.default_rng(0)
        false_positives = 0
        trials = 24
        for _ in range(trials):
            a, b = ar1(600, 0.98, rng), ar1(600, 0.98, rng)
            lag, corr = ll.best_lag(a, b, 30)
            false_positives += ll.surrogate_p_value(a, b, 30, abs(corr)) < 0.05
        assert false_positives / trials <= 0.25, (
            f"{false_positives}/{trials} independent pairs called significant")

    def test_p_value_is_never_zero(self):
        rng = np.random.default_rng(6)
        x = rng.normal(0, 1, 200)
        assert ll.surrogate_p_value(x, shift_lead(x, 3), 10) > 0.0

    def test_short_series_is_not_called_significant(self):
        assert ll.surrogate_p_value([1.0, 2.0], [2.0, 1.0], 2) == 1.0


class TestEffectiveSampleSize:
    def test_white_noise_keeps_most_of_its_samples(self):
        rng = np.random.default_rng(7)
        assert ll.effective_n(rng.normal(0, 1, 500)) > 300

    def test_smooth_series_has_far_fewer(self):
        rng = np.random.default_rng(8)
        assert ll.effective_n(ar1(500, 0.98, rng)) < 30

    def test_smooth_pairs_are_marked_unreliable(self):
        rng = np.random.default_rng(9)
        a, b = ar1(400, 0.98, rng), ar1(400, 0.98, rng)
        row = ll.lead_lag_table({"a": a, "b": b}, "a", 10)[0]
        assert row["reliable"] is False
        assert row["n_eff"] < ll.MIN_EFFECTIVE_N

    def test_noisy_pairs_are_marked_reliable(self):
        rng = np.random.default_rng(10)
        a = rng.normal(0, 1, 400)
        b = shift_lead(a, 3) + rng.normal(0, 0.3, 400)
        row = ll.lead_lag_table({"a": a, "b": b}, "a", 10)[0]
        assert row["reliable"] is True


class TestOnset:
    def test_finds_a_settled_step(self):
        x = np.concatenate([np.zeros(200), np.full(200, 5.0)])
        assert ll.onset(x, sample_s=1.0, settled_s=60) == 200

    def test_flat_series_has_no_onset(self):
        assert ll.onset(np.zeros(300), settled_s=60) is None

    def test_ignores_a_change_in_the_opening_seconds(self):
        """Every signal starts somewhere; that is not a precursor."""
        x = np.concatenate([np.zeros(20), np.full(400, 5.0)])
        assert ll.onset(x, sample_s=1.0, skip_s=60, settled_s=60) is None

    def test_requires_a_settled_baseline(self):
        """A signal that has been changing all along has no baseline to depart."""
        x = np.zeros(400)
        x[100] = 1.0
        x[200] = 3.0
        assert ll.onset(x, settled_s=300) is None

    def test_too_short_returns_none(self):
        assert ll.onset([1.0, 2.0, 3.0]) is None


class TestAlign:
    def test_forward_fill_is_the_sensor_semantics(self):
        series = {"s": [(0, 1.0), (3, 9.0)]}
        aligned, dt_s = ll.align(series, 0, 5)
        assert aligned["s"] == [1.0, 1.0, 1.0, 9.0, 9.0, 9.0]
        assert dt_s == 1.0

    def test_before_the_first_sample_uses_the_first_value(self):
        aligned, _ = ll.align({"s": [(5, 7.0)]}, 0, 5)
        assert aligned["s"] == [7.0] * 6

    def test_unsorted_input_is_sorted(self):
        aligned, _ = ll.align({"s": [(3, 9.0), (0, 1.0)]}, 0, 3)
        assert aligned["s"][0] == 1.0 and aligned["s"][-1] == 9.0

    def test_empty_series_is_dropped(self):
        aligned, _ = ll.align({"a": [], "b": [(0, 1.0)]}, 0, 2)
        assert "a" not in aligned


@pytest.fixture
def store(tmp_path):
    """A store where a.step moves 300 s before b.ramp begins."""
    conn = db.connect(tmp_path / "ll.db")
    conn.executemany("INSERT INTO signal (path, unit, tier) VALUES (?, ?, 'hot')",
                     [("a.step", "-"), ("b.ramp", "-"), ("c.flat", "-")])
    ids = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM signal")}
    rows = []
    for t in range(0, 1200):
        rows.append((ids["a.step"], t, 0.0 if t < 300 else 5.0, 0))
        rows.append((ids["b.ramp"], t, 0.0 if t < 600 else (t - 600) * 0.02, 0))
        rows.append((ids["c.flat"], t, 1.0 + (t % 7) * 0.01, 0))
    conn.executemany(
        "INSERT INTO sample (signal_id, ts_s, value, quality) VALUES (?,?,?,?)", rows)
    conn.commit()
    yield conn
    conn.close()


class TestStorePrecursors:
    def test_finds_the_earlier_signal(self, store):
        res = ll.precursors(store, "b.ramp", 1200 * 10**9, window_s=1200,
                            max_lag_s=200, candidates=["a.step", "b.ramp", "c.flat"],
                            settled_s=60)
        names = [r["signal"] for r in res["precursors"]]
        assert "a.step" in names
        a = next(r for r in res["precursors"] if r["signal"] == "a.step")
        # The step is sharp, so its onset is exact.
        assert a["onset_s"] == pytest.approx(300, abs=2)
        # lead_s is measured to the event; onset_lead_s to the target's own turn.
        # The target ramps rather than steps, so its onset lands somewhere inside
        # the ramp (657 s here) and only the ordering is meaningful, not the gap.
        assert a["lead_s"] == pytest.approx(900, abs=2)
        assert a["onset_s"] < res["target_onset_s"]
        assert a["onset_lead_s"] > 200

    def test_reports_the_full_table_not_just_the_winner(self, store):
        res = ll.precursors(store, "b.ramp", 1200 * 10**9, window_s=1200,
                            max_lag_s=200, candidates=["a.step", "b.ramp", "c.flat"],
                            settled_s=60)
        assert len(res["table"]) == 2, "every candidate that could be compared is listed"

    def test_unknown_target_raises(self, store):
        with pytest.raises(ValueError, match="no samples"):
            ll.precursors(store, "nope.nothing", 1200 * 10**9, window_s=600)

    def test_a_crowd_is_flagged(self, tmp_path):
        """Several signals departing together means a shared driver, not a cause."""
        conn = db.connect(tmp_path / "crowd.db")
        paths = ["a.one", "b.two", "c.three", "d.four"]
        conn.executemany("INSERT INTO signal (path, unit, tier) VALUES (?, '-', 'hot')",
                         [(p,) for p in paths])
        ids = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM signal")}
        rows = []
        for t in range(0, 1200):
            for offset, p in enumerate(paths):
                # all four depart at t=600 together
                rows.append((ids[p], t, 0.0 if t < 600 else 3.0 + offset, 0))
        conn.executemany(
            "INSERT INTO sample (signal_id, ts_s, value, quality) VALUES (?,?,?,?)", rows)
        conn.commit()
        res = ll.precursors(conn, "d.four", 1200 * 10**9, window_s=1200,
                            max_lag_s=100, settled_s=60)
        assert res["crowded"] is True
        conn.close()

    def test_never_settled_is_reported_separately(self, tmp_path):
        """A duty-cycled fault has no baseline to depart from, but it is not absent.

        Dropping it would silently hide the actual cause in the 2026-03-11
        overheat: the coolant pump duty-cycles before it fails, so it never
        holds a steady state and precedence can never date it.
        """
        conn = db.connect(tmp_path / "duty.db")
        conn.executemany("INSERT INTO signal (path, unit, tier) VALUES (?, '-', 'hot')",
                         [("pump",), ("temp",)])
        ids = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM signal")}
        rows = []
        for t in range(0, 1200):
            rows.append((ids["pump"], t, 10.0 if t % 40 < 20 else 0.0, 0))
            rows.append((ids["temp"], t, 0.0 if t < 700 else (t - 700) * 0.02, 0))
        conn.executemany(
            "INSERT INTO sample (signal_id, ts_s, value, quality) VALUES (?,?,?,?)", rows)
        conn.commit()
        res = ll.precursors(conn, "temp", 1200 * 10**9, window_s=1200,
                            max_lag_s=100, settled_s=60)
        assert [r["signal"] for r in res["unsettled"]] == ["pump"]
        assert all(r["signal"] != "pump" for r in res["precursors"])
        conn.close()
