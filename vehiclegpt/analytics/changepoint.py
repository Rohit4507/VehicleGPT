"""Change-point detection: when did this series actually shift?

The analytics layer can already say *that* consumption rose (224 vs 185 Wh/km).
The question a driver actually asks is *when* it started, and a day-over-day
maximum answers that badly: it points at the single noisiest step, not at the
regime change.

Two detectors, for two different questions:

* :func:`cusum` - online. Tuned to notice a sustained shift within a few samples
  of it happening, and it reports the sample where the evidence crossed rather
  than the sample where the shift began. That lag is inherent and is reported.
* :func:`pelt` - retrospective. Finds the best piecewise-constant segmentation
  of the whole series, so it can say "the level changed on 12 March" precisely.

**Why PELT is implemented here rather than imported.** ``ruptures`` is a fine
library, but it pulls in SciPy and is another thing to cross-compile onto an ARM
target with 2 GB of RAM. The mean-shift case needs about sixty lines, and
:mod:`tests.test_changepoint` cross-checks our implementation against
``ruptures`` on random series so "we wrote our own" is a claim with evidence
behind it rather than an assertion.

Both detectors assume the thing they are looking at is meaningful. A series with
fewer than a handful of points has no change points, and both functions say so
instead of returning the first index.
"""

from __future__ import annotations

import numpy as np

#: Minimum points a segment must contain. Guards against reporting a "shift"
#: that is one point wide, which is an outlier, not a regime.
MIN_SEGMENT = 2


def _as_1d(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {x.shape}")
    return x


def _robust_scale(x: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based sigma, falling back when MAD collapses to zero.

    MAD is zero whenever more than half the points are identical - common in a
    real signal (a parked vehicle) and a division by zero if unhandled.
    """
    centre = float(np.median(x))
    mad = float(np.median(np.abs(x - centre)))
    if mad > 0:
        return centre, mad * 1.4826
    spread = float(np.mean(np.abs(x - centre)))
    if spread > 0:
        return centre, spread * 1.2533      # mean absolute deviation → sigma
    return centre, 0.0


def cusum(values, k: float = 0.5, h: float = 6.0, baseline: int | None = None,
          min_segment: int = MIN_SEGMENT) -> list[dict]:
    """Two-sided tabular CUSUM, standardised against an initial baseline window.

    ``k`` is the slack in standard deviations: shifts smaller than ``k`` are
    absorbed rather than accumulated, which is what stops slow drift from
    triggering. ``h`` is the decision threshold, and it sets the false-alarm
    rate. Measured on this implementation (see :func:`false_alarm_rate`):

    | h | false alarms per 500 Gaussian samples |
    |---|---|
    | 4 | 4.24 |
    | 5 | 1.85 |
    | 6 | 0.92 |

    The default is 6. For a user-facing answer a false alarm is worse than a
    two-sample delay: on 200 simulated series with a 6-sigma step, h=6 raised an
    alarm before the change in 12.5 % of them, detected 198/200 with a median
    lag of zero samples, and still caught a 2-sigma shift in 100/100 series.

    **The baseline matters more than the thresholds.** Standardising with the
    median of the whole series is the obvious implementation and it is wrong: in
    a series that is half at one level and half at another, the grand median
    sits between them, so the *first* half registers as a sustained excursion in
    its own right and the detector fires before the change it was meant to find.
    Taking centre and scale from the first ``baseline`` samples (a quarter of the
    series by default, never more than half) keeps the reference clean - at the
    cost that a change occurring inside the baseline window cannot be detected.

    Returns one dict per alarm with the index it fired at *and* the index the
    excursion began, because those are different questions and conflating them
    makes the detector look more precise than it is.
    """
    x = _as_1d(values)
    if x.size < 4 * min_segment:
        return []

    n_base = baseline if baseline is not None else max(6 * min_segment, x.size // 4)
    n_base = max(min_segment, min(n_base, x.size // 2))
    centre, scale = _robust_scale(x[:n_base])
    if scale == 0.0:
        # A perfectly flat baseline has no spread to standardise against. Keep
        # the baseline centre - it is still the right reference - and borrow the
        # scale from the whole series. Falling back to the full-series *centre*
        # instead would reintroduce the contamination the baseline exists to
        # avoid. Quantised signals (integer SOC, gear, flags) hit this case.
        _, scale = _robust_scale(x)
    if scale == 0.0:
        return []
    z = (x - centre) / scale

    alarms: list[dict] = []
    hi = lo = 0.0
    hi_start = lo_start = 0
    last_alarm = -min_segment
    # Once a direction has alarmed, it stays disarmed until the series returns
    # to within the slack. Without this a sustained shift re-fires every few
    # samples, turning one change into a cluster of alarms.
    up_armed = down_armed = True

    for i, zi in enumerate(z):
        if up_armed is False and zi < k:
            up_armed = True
        if down_armed is False and zi > -k:
            down_armed = True

        hi = max(0.0, hi + zi - k)
        lo = max(0.0, lo - zi - k)
        if hi == 0.0:
            hi_start = i + 1
        if lo == 0.0:
            lo_start = i + 1

        direction = None
        if hi > h and up_armed:
            direction, start = "up", hi_start
            up_armed, hi = False, 0.0
        elif lo > h and down_armed:
            direction, start = "down", lo_start
            down_armed, lo = False, 0.0

        if direction is None or i - last_alarm < min_segment:
            continue
        start = min(start, i)
        alarms.append({
            "index": i,
            "direction": direction,
            "start_index": start,
            # The alarm fires after roughly h/(shift-k) samples of evidence;
            # reporting the gap stops callers reading `index` as the change.
            "lag_samples": i - start,
            "magnitude_sigma": round(float(abs(z[start:i + 1]).mean()), 2),
        })
        last_alarm = i
    return alarms


def false_alarm_rate(n: int = 500, k: float = 0.5, h: float = 6.0,
                     trials: int = 200, seed: int = 0) -> float:
    """Mean alarms per series on pure Gaussian noise, with no change at all.

    CUSUM is a sequential test, so it has a non-zero false-alarm rate by
    construction; the honest thing is to measure it for the chosen thresholds
    rather than describe it as "rare". Measured on this implementation:
    ``h=4`` gives 4.24 alarms per 500 samples, ``h=5`` 1.85, ``h=6`` 0.92.

    Deterministic for a given seed, so the number in the docs can be reproduced.
    """
    rng = np.random.default_rng(seed)
    total = 0
    for _ in range(trials):
        total += len(cusum(rng.normal(0.0, 1.0, n), k=k, h=h))
    return total / trials


def pelt(values, penalty: float, min_segment: int = MIN_SEGMENT) -> list[int]:
    """Exact PELT segmentation for a piecewise-constant mean (L2 cost).

    Returns segment boundaries as start indices, always beginning with ``0`` and
    ending with ``len(values)`` - so ``len(result) - 1`` segments. An
    empty/short series returns ``[0, len(values)]``: one segment, no change.

    ``penalty`` is the cost of adding a segment, in the same squared units as the
    data. There is no universally correct value; :func:`bic_penalty` gives the
    usual BIC-scaled default.
    """
    x = _as_1d(values)
    n = x.size
    if n < 2 * min_segment:
        return [0, n]

    cs = np.concatenate(([0.0], np.cumsum(x)))
    css = np.concatenate(([0.0], np.cumsum(x * x)))

    def cost(a: int, b: int) -> float:
        """Least-squares cost of the segment ``x[a:b]`` about its own mean."""
        m = b - a
        s = cs[b] - cs[a]
        return float(css[b] - css[a] - s * s / m)

    # F[t] is the optimal cost of segmenting x[:t]. Starting at -penalty means
    # the first segment is not charged for existing, so `penalty` is purely the
    # price of an *additional* change point.
    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    previous = np.zeros(n + 1, dtype=int)
    candidates = [0]

    for t in range(1, n + 1):
        best = np.inf
        best_s = 0
        for s in candidates:
            if t - s < min_segment:
                continue
            candidate = F[s] + cost(s, t) + penalty
            if candidate < best:
                best, best_s = candidate, s
        F[t] = best
        previous[t] = best_s
        # Pruning: a start point that cannot beat the current optimum for any
        # larger t can be dropped. This is what makes PELT linear rather than
        # quadratic; without it the inner loop is O(n^2) overall.
        candidates = [s for s in candidates if F[s] + cost(s, t) <= F[t]]
        candidates.append(t)

    cuts = []
    t = n
    while t > 0:
        cuts.append(t)
        t = int(previous[t])
    cuts.append(0)
    return sorted(cuts)


def bic_penalty(values, min_segment: int = MIN_SEGMENT) -> float:
    """BIC-style penalty: ``sigma^2 * log(n)``.

    With a Gaussian noise assumption, adding a segment costs roughly
    ``sigma^2 log n`` in twice-log-likelihood. ``sigma`` comes from the robust
    scale, so a single spike does not inflate the penalty for the whole series.
    """
    x = _as_1d(values)
    if x.size < 2:
        return 0.0
    _, sigma = _robust_scale(x)
    return float(max(sigma, 1e-12) ** 2 * np.log(max(x.size, 2)))


def segments(values, penalty: float | None = None, min_segment: int = MIN_SEGMENT,
             min_delta: float = 0.0) -> list[dict]:
    """Describe each segment: span, mean, and how far it moved from the last.

    This is the shape the answer renderers want - "the level was 185 Wh/km until
    11 March, then 240" - rather than a bare list of indices.
    """
    x = _as_1d(values)
    if penalty is None:
        penalty = bic_penalty(x, min_segment)
    bounds = pelt(x, penalty, min_segment)

    out: list[dict] = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        if b <= a:
            continue
        mean = float(x[a:b].mean())
        # A change smaller than `min_delta` is a wiggle, not a regime. Merging
        # it into the previous segment keeps the count meaningful on dense
        # signals, where BIC happily splits on differences nobody cares about.
        if out and abs(mean - out[-1]["mean"]) < min_delta:
            prev = out[-1]
            prev["end"] = b
            prev["length"] = b - prev["start"]
            prev["mean"] = float(x[prev["start"]:b].mean())
            continue
        previous = out[-1]["mean"] if out else None
        out.append({
            "start": a, "end": b, "length": b - a, "mean": mean,
            "delta": None if previous is None else mean - previous,
        })
    return out


def first_change(values, penalty: float | None = None,
                 min_segment: int = MIN_SEGMENT) -> int | None:
    """Index where the final segment begins, or ``None`` if the series is flat.

    "Where does the current regime start?" is the question the range-reduction
    answer needs, and it is the last boundary rather than the first.
    """
    x = _as_1d(values)
    bounds = pelt(x, bic_penalty(x) if penalty is None else penalty, min_segment)
    return None if len(bounds) <= 2 else bounds[-2]
