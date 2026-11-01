"""1 Hz decimation, owned by the store.

Decimation used to live in the build script as ``range(0, n, 10)``. That is
wrong in two ways and neither announces itself:

1. **It loses pulses.** A one-tick ADAS disengagement or an overheat flag that
   is set for 0.3 s falls between samples and vanishes. The question "show me
   all disengagements" then silently under-reports, and the answer looks
   perfectly healthy.
2. **It loses peaks and troughs.** Point sampling a temperature series reports
   whatever happened to land on the sample instant, which is not the maximum.

So decimation is policy-driven, per signal, and the policies are chosen so that
no discrete event can be dropped:

``LAST``
    One value per second, the last one seen. For continuous measurements.
``ON_CHANGE``
    Emit only when the value moves by more than the signal's deadband (from
    ``signals.yaml``), plus the first sample and the final sample. For slow
    signals this is a large storage win and loses nothing a reader cares about.
``ANY``
    Emit a sample whenever the value changes at all, and additionally emit the
    final value of every second in which any change occurred. Used for flags,
    where a single-tick pulse is the event.

The rule underneath all three: **the first sample and the last sample are always
kept**, and no transition is ever lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vehiclegpt.schema.registry import SignalSpec, Tier


class Policy(StrEnum):
    LAST = "last"
    ON_CHANGE = "on_change"
    ANY = "any"


def policy_for(spec: SignalSpec) -> Policy:
    """Choose a decimation policy from the registry.

    Flags are detected by their declared range rather than by a name list, so a
    new flag added to ``signals.yaml`` is handled without touching this file.
    """
    if spec.minimum == 0.0 and spec.maximum == 1.0:
        return Policy.ANY
    if spec.tier is Tier.SLOW:
        return Policy.ON_CHANGE
    return Policy.LAST


@dataclass(frozen=True)
class DecimatedSample:
    ts_s: int
    value: float


#: A flat stretch keeps one repeat sample this often, so its duration can be
#: reconstructed. Must stay well below MIN_STALL_S (60 s) or a stall of exactly
#: that length becomes unprovable.
KEYFRAME_S = 60.0


def decimate(
    ts_ns: list[int],
    values: list[float],
    spec: SignalSpec,
    policy: Policy | None = None,
    target_hz: float = 1.0,
) -> list[DecimatedSample]:
    """Reduce a series to ``target_hz`` according to ``policy``.

    ``ts_ns`` and ``values`` must be the same length and in time order.

    Output timestamps are whole seconds, which is also the storage key - so the
    reduction has to happen here, not be left to the primary key. An earlier
    version bucketed by ``1e9 / input_rate`` and let ``INSERT OR REPLACE``
    collapse the duplicates, which looked correct in the row counts while
    silently discarding every intra-second transition.
    """
    if len(ts_ns) != len(values):
        raise ValueError(
            f"{spec.path}: {len(ts_ns)} timestamps for {len(values)} values")
    if not values:
        return []

    policy = policy or policy_for(spec)
    if policy is Policy.LAST:
        return _last_per_bucket(ts_ns, values, target_hz)
    if policy is Policy.ANY:
        return _flag_changes(ts_ns, values, target_hz)
    return _on_change(ts_ns, values, spec.deadband, target_hz)


def _bucket(ts_ns: int, target_hz: float) -> int:
    """Index of the output bucket containing ``ts_ns``."""
    period_ns = max(1, int(1e9 / max(1e-9, target_hz)))
    return ts_ns // period_ns


def _bucket_ts(bucket: int, target_hz: float) -> int:
    period_ns = max(1, int(1e9 / max(1e-9, target_hz)))
    return (bucket * period_ns) // 10**9


def _last_per_bucket(ts_ns: list[int], values: list[float],
                     target_hz: float) -> list[DecimatedSample]:
    """One sample per bucket, carrying the last value seen in it."""
    out: list[DecimatedSample] = []
    last_bucket = None
    for ts, v in zip(ts_ns, values, strict=True):
        b = _bucket(ts, target_hz)
        if b != last_bucket:
            out.append(DecimatedSample(_bucket_ts(b, target_hz), v))
            last_bucket = b
        else:
            out[-1] = DecimatedSample(_bucket_ts(b, target_hz), v)
    return out


def _flag_changes(ts_ns: list[int], values: list[float],
                  target_hz: float) -> list[DecimatedSample]:
    """Transitions only, with intra-second pulses preserved.

    Two steps, and both matter:

    1. Collapse to one value per bucket using the **maximum**, so a flag that
       goes 0 -> 1 -> 0 inside a single second is recorded as 1 rather than 0.
       Taking the last value - the obvious choice - destroys exactly the events
       this layer exists to report.
    2. Drop every bucket whose value equals the previous kept value, so a flag
       that never fires costs two rows (first and last) instead of one per
       second.

    A reader reconstructs the flag at time *t* as the last row at or before *t*.
    """
    per_bucket: list[DecimatedSample] = []
    last_bucket = None
    for ts, v in zip(ts_ns, values, strict=True):
        b = _bucket(ts, target_hz)
        if b != last_bucket:
            per_bucket.append(DecimatedSample(_bucket_ts(b, target_hz), v))
            last_bucket = b
        elif v > per_bucket[-1].value:
            per_bucket[-1] = DecimatedSample(per_bucket[-1].ts_s, v)

    out: list[DecimatedSample] = [per_bucket[0]]
    for sample in per_bucket[1:]:
        if sample.value != out[-1].value:
            out.append(sample)
    final = DecimatedSample(ts_ns[-1] // 10**9, values[-1])
    if out[-1].ts_s != final.ts_s:
        out.append(final)
    return out


def _on_change(ts_ns: list[int], values: list[float], deadband: float,
               target_hz: float,
               keyframe_s: float = KEYFRAME_S) -> list[DecimatedSample]:
    """Keep the first sample, then values that move beyond the deadband.

    A plateau also gets a repeat sample every ``keyframe_s`` seconds. Without
    that, a long flat stretch is represented by its two endpoints and nobody can
    tell *how long* it lasted: a 15-minute coolant-pump stall collapsed to two
    rows, and the stall detector - which needs a third reading still at zero
    after 60 s - silently reported that no stall had happened. The keyframe
    bounds the gap, so duration stays recoverable from the stored series.

    The final sample is always kept so a reader can tell how the series ended
    even if it ended mid-plateau.
    """
    kept = [DecimatedSample(ts_ns[0] // 10**9, values[0])]
    last_raw_ns = ts_ns[0]
    keyframe_ns = int(keyframe_s * 10**9)
    for ts, v in zip(ts_ns[1:], values[1:], strict=True):
        if abs(v - kept[-1].value) > deadband or ts - last_raw_ns >= keyframe_ns:
            kept.append(DecimatedSample(ts // 10**9, v))
            last_raw_ns = ts
    if kept[-1].ts_s != ts_ns[-1] // 10**9:
        kept.append(DecimatedSample(ts_ns[-1] // 10**9, values[-1]))

    # Two changes inside one second would collide on the storage key; keep the
    # later one, which is what a last-value-wins reader expects.
    out: list[DecimatedSample] = []
    for sample in kept:
        if out and sample.ts_s == out[-1].ts_s:
            out[-1] = sample
        else:
            out.append(sample)
    return out


def compression_ratio(raw_count: int, kept_count: int) -> float:
    return raw_count / kept_count if kept_count else float("inf")
