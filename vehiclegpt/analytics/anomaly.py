"""Robust anomaly scoring for vehicle signals.

The naive anomaly score is ``(x - mean) / std``. It fails in a specific and
predictable way on vehicle data: the thing you are trying to detect moves the
very statistics you are measuring with. One 55 C battery excursion inflates the
standard deviation enough to hide itself, and a series that is 80 % "pump off"
has a mean that describes neither state.

So the score here is built from the median and the median absolute deviation:

    z = |x - median| / (1.4826 * MAD)

The 1.4826 factor makes MAD a consistent estimator of sigma for Gaussian noise,
which is the only reason the threshold "3.5 sigma" means anything. The
breakdown point is 50 %: half the series can be anomalous before the estimate
degrades at all, against 0 % for mean and standard deviation.

Three limits are worth stating plainly rather than discovering later:

* **Symmetric by assumption.** MAD treats +5 and -5 alike. A signal that only
  ever fails in one direction - over-temperature - should be scored one-sided,
  or genuinely cold readings will be flagged as faults.
* **Uncorrelated by assumption.** Each signal is scored alone, so a point that
  is unremarkable in every margin but rare in combination is missed. That is
  exactly what a multivariate detector buys, and
  :mod:`tests.test_anomaly` measures the gap against IsolationForest instead of
  claiming one is better.
* **Not a classifier.** A score says "this is far from typical". Whether it
  matters is domain knowledge, so :func:`detect` returns the score and lets the
  caller decide.
"""

from __future__ import annotations

import numpy as np

#: Consistency constant turning MAD into a sigma estimate for Gaussian data.
MAD_TO_SIGMA = 1.4826

#: Conventional robust threshold (Iglewicz & Hoaglin). Deliberately not tuned
#: to this dataset - a threshold fitted to the demo would not transfer.
DEFAULT_THRESHOLD = 3.5


def center_scale(values) -> tuple[float, float]:
    """Median and MAD-scaled sigma, with explicit handling of degenerate cases.

    MAD is exactly zero whenever more than half the points share the median -
    which is normal for a vehicle signal, not a corner case: a coolant pump idles
    at one value for most of a drive. Returning sigma=0 and letting callers
    divide would produce inf or nan; the fallback here uses the mean absolute
    deviation instead, and returns 0 only when the series is truly constant.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0, 0.0
    centre = float(np.median(x))
    mad = float(np.median(np.abs(x - centre)))
    if mad > 0:
        return centre, mad * MAD_TO_SIGMA
    spread = float(np.mean(np.abs(x - centre)))
    if spread > 0:
        # 1.2533 = sqrt(pi/2): mean absolute deviation → sigma for Gaussians.
        return centre, spread * 1.2533
    return centre, 0.0


def zscore(values) -> np.ndarray:
    """Robust z-score. A constant series scores zero everywhere, not NaN."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.array([], dtype=float)
    centre, scale = center_scale(x)
    if scale == 0.0:
        return np.zeros_like(x)
    return np.abs(x - centre) / scale


def rolling_median(values, window: int = 7) -> np.ndarray:
    """Centred rolling median, shrinking at the edges rather than padding.

    Padding with zeros or repeating the edge value would manufacture anomalies at
    the start and end of every series - and the end is exactly where a live
    vehicle spends its time.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return np.array([], dtype=float)
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    half = max(window // 2, 0)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out


def _theil_sen_level(block: np.ndarray, at: int, offset: int) -> float:
    """Robust linear fit of ``block`` evaluated at index ``at`` within it.

    Theil-Sen: the median of all pairwise slopes, then the median intercept. A
    single spike can corrupt at most the pairs it participates in, so with a
    window of 7 it cannot move the median slope at all.
    """
    m = block.size
    if m < 3:
        return float(block[offset])          # two points fit any line exactly
    t = np.arange(m, dtype=float)
    dt = t[:, None] - t[None, :]
    dv = block[:, None] - block[None, :]
    upper = np.triu_indices(m, k=1)
    slopes = dv[upper] / dt[upper]
    slope = float(np.median(slopes))
    intercept = float(np.median(block - slope * t))
    return intercept + slope * offset


def local_level(values, window: int = 7) -> np.ndarray:
    """Robust local linear level, so a smooth trend has zero residual.

    A rolling *median* is the obvious local level and it is biased at the ends of
    a trending series: on a clean ramp the shrinking edge window sits behind the
    current point, so every edge reading looks anomalous. That is a detector that
    reports a fault whenever a battery warms up evenly, which is to say always.

    Fitting a line instead removes the bias: a ramp is fitted exactly, so its
    residual is zero everywhere, while a spike cannot move a median-of-slopes
    fit. The cost is that the outermost points are left unscored - a two-point
    window fits any two points exactly - and that is reported by returning the
    value itself there, giving a residual of zero rather than a false alarm.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return np.array([], dtype=float)
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    half = max(window // 2, 1)
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = _theil_sen_level(x[lo:hi], i, i - lo)
    return out


def residuals(values, window: int = 7) -> np.ndarray:
    """Deviation from the local level, so a slow trend is not called an anomaly.

    A battery warming from 20 C to 60 C over a drive has no single anomalous
    point if you look at the raw series with a global median - the whole series
    is the excursion. Subtracting a local level asks the more useful question:
    which readings broke away from what was happening around them.
    """
    x = np.asarray(values, dtype=float)
    return x - local_level(x, window)


def score(values, window: int = 7) -> np.ndarray:
    """Robust z-score of the residual from the local level.

    A series that the local fit reproduces exactly - any clean ramp - leaves
    only floating-point dust behind. Dividing that dust by its own MAD turns
    rounding error into confident anomalies, so a residual that is negligible
    against the magnitude of the data is reported as zero.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.array([], dtype=float)
    r = residuals(x, window)
    scale = max(float(np.abs(x).max()), 1.0)
    if float(np.abs(r).max()) <= 1e-9 * scale:
        return np.zeros_like(r)
    return zscore(r)


def detect(values, threshold: float = DEFAULT_THRESHOLD, window: int = 7,
           top: int | None = None, min_delta: float = 0.0) -> list[dict]:
    """Indices whose residual score exceeds ``threshold``, worst first.

    ``min_delta`` is a floor on the *absolute* deviation from the local level,
    and it exists because statistical significance is not practical
    significance. On a smooth signal - battery temperature rising half a degree
    over a drive - the residual noise is so small that a 0.1 C wobble scores
    above eleven sigma. That is arithmetically correct and operationally
    useless. A caller who knows that 2 C matters sets ``min_delta=2.0`` and gets
    the readings that matter; the default of 0 leaves the decision to them
    rather than inventing a domain threshold here.

    Returns a list of dicts rather than indices so a caller can see the value,
    the deviation and the score that produced the flag, and argue with it.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return []
    z = score(x, window)
    delta = x - local_level(x, window)
    flagged = np.flatnonzero((z > threshold) & (np.abs(delta) >= min_delta))
    out = [{"index": int(i), "value": float(x[i]), "score": round(float(z[i]), 3),
            "delta": round(float(delta[i]), 4)} for i in flagged]
    out.sort(key=lambda r: r["score"], reverse=True)
    return out if top is None else out[:top]
