"""Lead-lag search: which signal moved first?

"What happened before the battery overheated?" is answered today by a hardcoded
check on the coolant pump. That works for one question and generalises to none.
This module asks the general version: given a target signal and a time, which
other signals moved *before* it, and by how much?

The statistic is lagged Pearson correlation, computed over the overlapping
region at each lag so that a lag costs sample size rather than bias. The sign
convention is fixed and tested, because it is the single easiest thing to get
backwards:

    corr(L) = corr(x[t], y[t + L])

A peak at ``L = +k`` means x's pattern shows up in y k samples later, so **x
leads y**. :func:`best_lag` returns that ``L``.

Two traps this module takes seriously:

* **Correlation is not causation, and lead-lag makes that worse, not better.**
  Two signals driven by the same third variable - HVAC draw and battery
  temperature are both driven by ambient temperature - will show a stable lead
  that means nothing. There is no statistical fix for that inside a pair; the
  fix is to look at the mechanism, which is why every result carries the pairs it
  compared and not just the winner.
* **Autocorrelated series make ordinary significance testing meaningless.** With
  1 Hz samples of a slowly varying temperature, neighbouring points are almost
  duplicates, so a naive p-value will call any lag significant. The null here is
  built by *circularly shifting* one series: that preserves the autocorrelation
  and the marginal distribution while destroying the alignment, which is the
  right question - "is this alignment better than an arbitrary one?"
"""

from __future__ import annotations

import sqlite3

import numpy as np

from vehiclegpt.analytics import changepoint

#: Fewest overlapping samples for a lagged correlation to be reported at all.
MIN_OVERLAP = 10

#: Below this many *effective* samples a correlation is reported but marked
#: unreliable. See :func:`effective_n`.
MIN_EFFECTIVE_N = 10


def _as_1d(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {x.shape}")
    return x


def lag1_autocorrelation(values) -> float:
    """Correlation between consecutive samples. High for smooth signals."""
    x = _as_1d(values)
    if x.size < 3:
        return 0.0
    a, b = x[:-1], x[1:]
    value = _pearson(a, b)
    return 0.0 if np.isnan(value) else max(min(value, 0.999), -0.999)


def effective_n(values) -> float:
    """How many *independent* samples a smooth series really contains.

    For an AR(1) process this is ``n (1 - phi) / (1 + phi)``. It matters because
    the trap this module exists to avoid is quantitative, not qualitative: two
    independent AR(1) series with phi=0.98 and 600 samples show a mean |corr| of
    0.33 and can reach 0.57, because they contain roughly six independent
    samples between them. Reporting 0.57 next to "600 samples" would be true and
    completely misleading, so the count travels with the correlation.
    """
    x = _as_1d(values)
    phi = lag1_autocorrelation(x)
    return float(x.size) * (1.0 - phi) / (1.0 + phi)


def standardize(values) -> np.ndarray:
    """Centre and scale robustly. A constant series becomes all zeros."""
    x = _as_1d(values)
    if x.size == 0:
        return x
    centre = float(np.median(x))
    scale = float(np.median(np.abs(x - centre))) * 1.4826
    if scale <= 0:
        spread = float(np.mean(np.abs(x - centre)))
        scale = spread * 1.2533 if spread > 0 else 0.0
    if scale <= 0:
        return np.zeros_like(x)
    return (x - centre) / scale


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < MIN_OVERLAP:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def cross_correlation(x, y, max_lag: int) -> np.ndarray:
    """``corr(x[t], y[t+L])`` for L from ``-max_lag`` to ``+max_lag``.

    Index ``i`` of the result corresponds to ``L = i - max_lag``. Entries whose
    overlap is smaller than :data:`MIN_OVERLAP`, or where either piece is
    constant, are ``nan`` rather than 0 - "no correlation" and "not enough data
    to say" are different answers.
    """
    x = _as_1d(x)
    y = _as_1d(y)
    if x.size != y.size:
        raise ValueError(f"series lengths differ: {x.size} vs {y.size}")
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative, got {max_lag}")
    n = x.size
    out = np.full(2 * max_lag + 1, np.nan)

    # The cross-product sum for every lag comes from np.correlate, which runs in
    # C. The per-lag means and variances come from prefix sums, so each lag costs
    # O(1) on top. Doing this per lag in Python made a surrogate-tested search
    # take 44 s; it is the difference between a table and a hang.
    #
    # Mapping, verified in test_cross_correlation_matches_a_direct_computation:
    #   np.correlate(y, x, "full")[(n - 1) + L] == sum_i x[i] * y[i + L]
    raw = np.correlate(y, x, mode="full")
    px = np.concatenate(([0.0], np.cumsum(x)))
    py = np.concatenate(([0.0], np.cumsum(y)))
    pxx = np.concatenate(([0.0], np.cumsum(x * x)))
    pyy = np.concatenate(([0.0], np.cumsum(y * y)))

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            i0, i1 = 0, n - lag          # x[i0:i1] against y[i0+lag:i1+lag]
        else:
            i0, i1 = -lag, n
        count = i1 - i0
        if count < MIN_OVERLAP:
            continue
        j0, j1 = i0 + lag, i1 + lag
        sx = px[i1] - px[i0]
        sy = py[j1] - py[j0]
        vx = (pxx[i1] - pxx[i0]) - sx * sx / count
        vy = (pyy[j1] - pyy[j0]) - sy * sy / count
        if vx <= 0 or vy <= 0:
            continue
        sxy = raw[(n - 1) + lag]
        out[lag + max_lag] = (sxy - sx * sy / count) / float(np.sqrt(vx * vy))
    return out


def best_lag(x, y, max_lag: int) -> tuple[int, float]:
    """The lag with the strongest correlation, and its coefficient.

    Positive lag means **x leads y**. Returns ``(0, nan)`` when nothing can be
    computed, so callers can test with ``np.isnan`` rather than for ``None``.
    """
    corr = cross_correlation(x, y, max_lag)
    if not np.any(~np.isnan(corr)):
        return 0, float("nan")
    index = int(np.nanargmax(np.abs(corr)))
    return index - int(max_lag), float(corr[index])


def surrogate_p_value(x, y, max_lag: int, observed: float | None = None,
                      shifts: int = 64, seed: int = 0) -> float:
    """How often a circularly shifted alignment beats the real one.

    Shifting ``y`` by a large offset keeps its autocorrelation and its
    distribution intact and breaks only the alignment, which is exactly the null
    a naive p-value fails to construct. The returned value is the fraction of
    shifts whose best ``|corr|`` is at least the observed one, with the usual
    ``(1 + k) / (1 + n)`` smoothing so it is never exactly zero.

    Deterministic for a given seed.
    """
    x = standardize(x)
    y = standardize(y)
    n = x.size
    if n < MIN_OVERLAP:
        return 1.0
    if observed is None:
        observed = abs(best_lag(x, y, max_lag)[1])
    if np.isnan(observed):
        return 1.0

    rng = np.random.default_rng(seed)
    lo = max(max_lag + 1, MIN_OVERLAP)
    if n - lo <= lo:
        return 1.0
    hits = 0
    for _ in range(shifts):
        offset = int(rng.integers(lo, n - lo))
        shifted = np.roll(y, offset)
        value = abs(best_lag(x, shifted, max_lag)[1])
        hits += bool(value >= observed)
    return (1 + hits) / (1 + shifts)


def lead_lag_table(series: dict[str, list[float]], target: str, max_lag: int,
                   sample_s: float = 1.0) -> list[dict]:
    """Every candidate against one target, with lead time and a null test.

    ``series`` maps a path to an equally spaced, equally lengthed series. The
    caller is responsible for alignment - :func:`align` does it from the store.
    """
    if target not in series:
        raise KeyError(f"target {target!r} is not among the series")
    y = standardize(series[target])
    out: list[dict] = []
    for path, values in series.items():
        if path == target:
            continue
        x = standardize(values)
        lag, corr = best_lag(x, y, max_lag)
        if np.isnan(corr):
            continue
        n_eff = min(effective_n(values), effective_n(series[target]))
        out.append({
            "signal": path,
            "lag_samples": lag,
            "lead_s": lag * sample_s,
            "correlation": round(corr, 4),
            "p_value": round(surrogate_p_value(x, y, max_lag, abs(corr)), 4),
            "n_eff": round(n_eff, 1),
            "reliable": n_eff >= MIN_EFFECTIVE_N,
        })
    # Strongest first, then longest lead: a weak short lead is not a precursor.
    out.sort(key=lambda r: (-abs(r["correlation"]), -r["lead_s"]))
    return out


def align(series: dict[str, list[tuple[int, float]]], start_s: int, end_s: int
          ) -> tuple[dict[str, list[float]], float]:
    """Put irregular, on-change signals on one 1 Hz grid by forward-filling.

    Forward fill is the correct semantics for a sensor: between two reported
    values the last one is still what the sensor said. Interpolating would invent
    readings, and dropping to the common timestamps would silently compare
    different windows for different signals.

    Returns the aligned series and the sample period in seconds.
    """
    grid = np.arange(start_s, end_s + 1, 1, dtype=np.int64)
    aligned: dict[str, list[float]] = {}
    for path, points in series.items():
        if not points:
            continue
        times = np.array([p[0] for p in points], dtype=np.int64)
        values = np.array([p[1] for p in points], dtype=float)
        order = np.argsort(times)
        times, values = times[order], values[order]
        idx = np.searchsorted(times, grid, side="right") - 1
        idx = np.clip(idx, 0, len(times) - 1)
        aligned[path] = values[idx].tolist()
    return aligned, 1.0


def onset(values, sample_s: float = 1.0, min_delta: float = 0.0,
          skip_s: float = 60.0, settled_s: float = 300.0) -> int | None:
    """Index of the first meaningful level change, or ``None`` if the series is flat.

    Correlation and onset answer different questions and disagree on smooth
    data. A temperature that ramps for twenty minutes has an onset — the moment
    it started climbing — but its *level* correlates with everything else that
    drifts, and its effective sample size is a handful. Onset survives both
    problems because it only asks where the series changed, not how similar two
    drifting lines look.
    """
    x = _as_1d(values)
    if x.size < 8:
        return None
    segments = changepoint.segments(x, min_delta=min_delta)
    # A change in the opening seconds is the window starting, not the vehicle
    # doing something: every signal begins somewhere. Without this the drive
    # start wins for all eighteen signals and the "precursor" list is simply a
    # list of everything the car has.
    first_index = int(max(skip_s / sample_s, 0))
    settled = int(max(settled_s / sample_s, 1))
    for previous, segment in zip(segments, segments[1:], strict=False):
        start = int(segment["start"])
        if start < first_index:
            continue
        # A precursor is a departure from a *settled* state. Requiring the
        # preceding segment to have lasted a few minutes is what separates
        # "this signal changed" from "this signal has been changing all along",
        # and it is why a duty-cycled fault (the coolant pump toggling before it
        # fails) does not register here at all: nothing was ever settled.
        if int(previous["length"]) >= settled:
            return start
    return None


def _fetch(conn: sqlite3.Connection, path: str, lo_s: int,
           hi_s: int) -> list[tuple[int, float]]:
    row = conn.execute("SELECT id FROM signal WHERE path = ?", (path,)).fetchone()
    if row is None:
        return []
    return [(int(r["ts_s"]), float(r["value"])) for r in conn.execute(
        "SELECT ts_s, value FROM sample WHERE signal_id = ? AND ts_s BETWEEN ? AND ? "
        "ORDER BY ts_s", (row["id"], lo_s, hi_s))]


def precursors(conn: sqlite3.Connection, target: str, at_ns: int,
               window_s: int = 1800, max_lag_s: int = 600,
               candidates: list[str] | None = None,
               min_abs_corr: float = 0.3,
               min_delta: float = 0.0,
               settled_s: float = 300.0,
               crowd_s: float = 10.0) -> dict:
    """Which signals led ``target`` in the window before ``at_ns``.

    Returns ``{"table": [...], "precursors": [...], "window_s": ..., "n_eff": ...}``
    where ``precursors`` is the subset that actually leads (positive lag) with a
    correlation above ``min_abs_corr``. The full table is returned as well, so a
    reader can see what was compared and rejected instead of only the winner.

    Candidates default to every signal in the store, which is the honest default:
    choosing the candidate list after seeing the answer is how a lead-lag search
    becomes a story.
    """
    end_s = int(at_ns / 1e9)
    start_s = end_s - int(window_s)

    # Anchor to the current drive. Between drives every signal is absent, so a
    # window straddling the start of the drive makes "the car started" the
    # largest change in all eighteen signals - which is true, useless, and
    # identical for everything compared.
    drive = conn.execute(
        "SELECT start_ns FROM drive WHERE start_ns <= ? AND end_ns >= ? "
        "ORDER BY start_ns DESC LIMIT 1", (int(at_ns), int(at_ns))).fetchone()
    if drive is not None:
        start_s = max(start_s, int(drive["start_ns"] / 1e9) + 60)

    if end_s - start_s < 2 * MIN_OVERLAP:
        raise ValueError("window is too short to compare anything")

    paths = candidates or [r["path"] for r in conn.execute("SELECT path FROM signal")]
    series = {path: _fetch(conn, path, start_s, end_s) for path in paths}
    series = {p: v for p, v in series.items() if len(v) >= 4}
    if target not in series:
        raise ValueError(f"target {target!r} has no samples in the window")

    aligned, sample_s = align(series, start_s, end_s)
    max_lag = max(1, int(max_lag_s / sample_s))
    table = lead_lag_table(aligned, target, max_lag, sample_s=sample_s)

    # Precedence: when each series changed, relative to the target's own change.
    skip_s = max(60.0, 0.05 * window_s)
    target_onset = onset(aligned[target], sample_s, min_delta=min_delta, skip_s=skip_s,
                         settled_s=settled_s)
    by_name = {row["signal"]: row for row in table}
    for path, row in by_name.items():
        index = onset(aligned[path], sample_s, min_delta=min_delta, skip_s=skip_s,
                      settled_s=settled_s)
        row["onset_s"] = None if index is None else start_s + index * sample_s
        # Lead against the event the caller asked about, which is usually the
        # warning rather than the target signal's own turn.
        row["lead_s"] = None if index is None else end_s - row["onset_s"]
        row["onset_lead_s"] = (None if index is None or target_onset is None
                               else (target_onset - index) * sample_s)

    # Onset detection does not inherit the correlation's sample-size problem: it
    # asks where a series changed, not how similar two drifting lines look. So
    # `reliable` gates the correlation coefficient only, and a smooth signal can
    # still have a trustworthy onset.
    leading = [row for row in table
               if row["lead_s"] is not None and row["lead_s"] > 0]
    leading.sort(key=lambda r: (-abs(r["correlation"]), -r["lead_s"]))

    # Say when the answer is a crowd. Several signals departing together usually
    # means one shared driver moved them all, and picking the first of them is
    # arbitrary rather than causal.
    ambiguity = 0
    if len(leading) >= 3:
        onsets = sorted(r["onset_s"] for r in leading)
        ambiguity = sum(1 for a in onsets
                        if sum(1 for b in onsets if abs(b - a) <= crowd_s) >= 3)

    # Signals that never settle have no onset to date: a duty-cycled fault is
    # invisible to precedence by construction. They are returned rather than
    # dropped, because "not dated" is not the same as "not involved" - the
    # coolant pump in the 2026-03-11 overheat is exactly this case.
    unsettled = [row for row in table if row["onset_s"] is None]
    unsettled.sort(key=lambda r: -abs(r["correlation"]))

    return {"table": table, "precursors": leading, "unsettled": unsettled,
            "crowded": ambiguity > 0, "crowd_size": ambiguity,
            "target_onset_s": (None if target_onset is None
                               else start_s + target_onset * sample_s),
            "window_s": window_s, "target": target, "at_ns": int(at_ns)}
