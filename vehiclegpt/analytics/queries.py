"""Deterministic query layer.

Five questions from the case study, answered with SQL and arithmetic. No model
is involved, which is the point: on the constrained tier this is the only way
the 2-second target is met at all (see docs/10), and for filter/aggregate
questions the model would add nothing but latency and the chance of getting a
number wrong.

Two rules the handlers follow:

1. **Never use harness metadata as evidence.** ``event.kind`` values beginning
   ``fault.`` record what the *simulator* injected. An answer built from them
   would be circular - the analytics layer would be reading the answer key. The
   precursor for the overheat question is found by looking at the coolant pump
   signal and noticing it stopped, which is what a real vehicle would give you.

2. **Report the mean speed alongside consumption.** "City is worse than
   motorway" is only meaningful if you know the city mean speed was 24 kph; a
   ratio without its conditions invites a wrong conclusion.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass, field

EVIDENCE_KIND_PREFIX = "fault."


@dataclass
class Answer:
    intent: str
    path: str
    text: str
    data: dict = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent, "path": self.path, "text": self.text,
            "data": self.data, "evidence": self.evidence, "caveats": self.caveats,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _day_ns(day: str) -> int:
    return int(dt.datetime.fromisoformat(day).replace(
        tzinfo=dt.UTC).timestamp() * 1e9)


def _fmt_day(day: str) -> str:
    d = dt.date.fromisoformat(day)
    return d.strftime("%a %d %b")


def _signal_id(conn: sqlite3.Connection, path: str) -> int | None:
    row = conn.execute("SELECT id FROM signal WHERE path = ?", (path,)).fetchone()
    return None if row is None else int(row["id"])


def _series(conn: sqlite3.Connection, path: str, lo_ns: int, hi_ns: int) -> list[tuple[int, float]]:
    sid = _signal_id(conn, path)
    if sid is None:
        return []
    return [(int(r["ts_ns"]), float(r["value"])) for r in conn.execute(
        "SELECT ts_ns, value FROM sample WHERE signal_id = ? AND ts_ns BETWEEN ? AND ? "
        "ORDER BY ts_ns", (sid, lo_ns, hi_ns))]


def _drive_hours(conn: sqlite3.Connection, lo_ns: int, hi_ns: int) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(end_ns - start_ns), 0) AS ns FROM segment "
        "WHERE start_ns BETWEEN ? AND ?", (lo_ns, hi_ns)).fetchone()
    return float(row["ns"]) / 3.6e12


def _pct(new: float, old: float) -> float:
    return (new - old) / old * 100.0 if old else float("nan")


# ---------------------------------------------------------------------------
# Q1: why has my range reduced?
# ---------------------------------------------------------------------------

def _window_stats(conn: sqlite3.Connection, start_day: str, end_day: str) -> dict:
    """Distance, energy, driving hours and mean HVAC/ambient draw over a window."""
    row = conn.execute(
        "SELECT COALESCE(SUM(distance_km), 0) AS km, COALESCE(SUM(energy_wh), 0) AS wh, "
        "       AVG(hvac_w) AS hvac_w, AVG(ambient_c) AS ambient_c, COUNT(*) AS days "
        "FROM daily WHERE day BETWEEN ? AND ? AND distance_km > 2",
        (start_day, end_day)).fetchone()
    hours = _drive_hours(conn, _day_ns(start_day), _day_ns(end_day) + 86400 * 10**9)
    km, wh = float(row["km"]), float(row["wh"])
    hvac_w = row["hvac_w"] or 0.0
    return {
        "days": int(row["days"]), "distance_km": km, "energy_wh": wh,
        "wh_per_km": wh / km if km else float("nan"),
        "hours": hours, "hvac_w": float(hvac_w),
        "ambient_c": float(row["ambient_c"] or 0.0),
        # W * h / km = Wh/km: the HVAC energy attributable to each kilometre.
        "hvac_wh_per_km": (hvac_w * hours / km) if km else float("nan"),
    }


def _shift_day(day: str, delta: int) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=delta)).isoformat()


def range_reduction(conn: sqlite3.Connection) -> Answer:
    """Decompose the consumption increase into HVAC and the remainder.

    The HVAC share is *computed* - mean draw in watts times hours driven divided
    by kilometres - in both windows, and the difference is attributed. An
    earlier version of this function used a hardcoded fraction, which is
    indistinguishable from making the number up.
    """
    row = conn.execute(
        "SELECT * FROM v_daily_rolling WHERE prior7_days >= 7 "
        "ORDER BY day DESC LIMIT 1").fetchone()
    if row is None:
        return Answer("range_reduction", "P0",
                      "Not enough history to compare two weeks yet.",
                      caveats=["need at least 14 days of data"])

    day = row["day"]
    cur = _window_stats(conn, _shift_day(day, -6), day)
    prior = _window_stats(conn, _shift_day(day, -13), _shift_day(day, -7))

    delta = cur["wh_per_km"] - prior["wh_per_km"]
    hvac_delta = cur["hvac_wh_per_km"] - prior["hvac_wh_per_km"]
    share_now = (cur["hvac_wh_per_km"] / cur["wh_per_km"] * 100.0
                 if cur["wh_per_km"] else float("nan"))
    share_before = (prior["hvac_wh_per_km"] / prior["wh_per_km"] * 100.0
                    if prior["wh_per_km"] else float("nan"))

    # The HVAC delta can exceed the total delta: heating is an always-positive
    # load while recuperation varies, so the two are not a clean decomposition.
    # Say so instead of forcing the residual to come out positive.
    if hvac_delta > delta > 0:
        closing = ("Heating alone would explain more than the observed increase; "
                   "part of it is offset by lower average speeds in the cold.")
    elif delta > 0:
        closing = (f"The remaining {delta - hvac_delta:+.0f} Wh/km is consistent "
                   "with higher battery and drivetrain losses at low temperature.")
    else:
        closing = "Consumption did not rise over this window."

    text = (
        f"Consumption over the last 7 days averaged {cur['wh_per_km']:.0f} Wh/km, "
        f"up from {prior['wh_per_km']:.0f} Wh/km the week before "
        f"({_pct(cur['wh_per_km'], prior['wh_per_km']):+.0f}%). "
        f"Mean ambient temperature fell from {prior['ambient_c']:.1f} C to "
        f"{cur['ambient_c']:.1f} C. HVAC draw rose from {prior['hvac_w']:.0f} W to "
        f"{cur['hvac_w']:.0f} W, which is {cur['hvac_wh_per_km']:.0f} Wh/km "
        f"({share_now:.0f}% of consumption) against "
        f"{prior['hvac_wh_per_km']:.0f} Wh/km ({share_before:.0f}%) before. {closing}"
    )
    return Answer(
        intent="range_reduction", path="P0", text=text,
        data={"recent": cur, "baseline": prior,
              "delta_wh_per_km": round(delta, 1),
              "hvac_delta_wh_per_km": round(hvac_delta, 1),
              "hvac_share_now_pct": round(share_now, 1),
              "hvac_share_before_pct": round(share_before, 1)},
        evidence=[dict(row)],
        caveats=["temperature and HVAC draw move together, so the split is an "
                 "attribution rather than a controlled decomposition"])


# ---------------------------------------------------------------------------
# Q2: when did efficiency drop most?
# ---------------------------------------------------------------------------

def efficiency_largest_drop(conn: sqlite3.Connection, min_km: float = 10.0) -> Answer:
    """The single day with the largest consumption increase over the day before.

    Day-over-day, not a trailing-week comparison. A trailing window peaks on the
    last day it can see rather than on the day the change happened, so it
    answers "when did the rolling week end" instead of "when did efficiency
    drop". Days below ``min_km`` are excluded because a short errand is noisy.
    """
    row = conn.execute(
        """
        SELECT day, wh_per_km, ambient_c, prev_wh_per_km, prev_ambient_c,
               wh_per_km - prev_wh_per_km AS delta
        FROM (
            SELECT day, wh_per_km, ambient_c,
                   LAG(wh_per_km)    OVER (ORDER BY day) AS prev_wh_per_km,
                   LAG(ambient_c)    OVER (ORDER BY day) AS prev_ambient_c
            FROM daily WHERE distance_km >= ? AND wh_per_km IS NOT NULL
        )
        WHERE prev_wh_per_km IS NOT NULL
        ORDER BY delta DESC LIMIT 1
        """, (min_km,)).fetchone()
    if row is None:
        return Answer("efficiency_largest_drop", "P0",
                      "Not enough comparable days yet.",
                      caveats=["need at least two days with meaningful distance"])

    ctx = conn.execute(
        "SELECT ma7_wh_per_km, prior7_wh_per_km FROM v_daily_rolling "
        "WHERE day >= ? ORDER BY day LIMIT 1", (row["day"],)).fetchone()

    text = (
        f"Consumption jumped on {_fmt_day(row['day'])}: "
        f"{row['wh_per_km']:.0f} Wh/km against {row['prev_wh_per_km']:.0f} Wh/km "
        f"the day before, an increase of {row['delta']:.0f} Wh/km "
        f"({_pct(row['wh_per_km'], row['prev_wh_per_km']):+.0f}%). "
        f"Ambient temperature went from {row['prev_ambient_c']:.1f} C to "
        f"{row['ambient_c']:.1f} C over the same step."
    )
    data = {"day": row["day"], "wh_per_km": round(row["wh_per_km"], 1),
            "prior_wh_per_km": round(row["prev_wh_per_km"], 1),
            "delta_wh_per_km": round(row["delta"], 1),
            "ambient_c": round(row["ambient_c"], 1),
            "prior_ambient_c": round(row["prev_ambient_c"], 1)}
    if ctx is not None:
        data["trailing_week_wh_per_km"] = round(ctx["ma7_wh_per_km"], 1)
        data["prior_week_wh_per_km"] = (
            None if ctx["prior7_wh_per_km"] is None
            else round(ctx["prior7_wh_per_km"], 1))
    return Answer(
        intent="efficiency_largest_drop", path="P0", text=text, data=data,
        evidence=[dict(row)],
        caveats=[f"days under {min_km:.0f} km excluded as unrepresentative"])


# ---------------------------------------------------------------------------
# Q3: ADAS disengagements from yesterday
# ---------------------------------------------------------------------------

def adas_events(conn: sqlite3.Connection, day: str | None = None) -> Answer:
    if day is None:
        day = conn.execute(
            "SELECT date(MAX(ts_ns) / 1000000000, 'unixepoch') AS d FROM event"
        ).fetchone()["d"]
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM v_adas_disengagements WHERE day = ? ORDER BY ts_ns", (day,))]

    if not rows:
        text = f"No ADAS disengagement events recorded on {_fmt_day(day)}."
    else:
        times = ", ".join(r["time_utc"] for r in rows)
        text = (f"{len(rows)} ADAS disengagement event"
                f"{'s' if len(rows) != 1 else ''} on {_fmt_day(day)} at "
                f"{times} UTC.")
    return Answer(
        intent="adas_events", path="P0", text=text,
        data={"day": day, "count": len(rows)},
        evidence=rows)


# ---------------------------------------------------------------------------
# Q4: what happened before the battery overheating warning?
# ---------------------------------------------------------------------------

MIN_STALL_S = 60.0


def _sustained_stall(series: list[tuple[int, float]],
                     threshold_pct: float = 0.5,
                     min_s: float = MIN_STALL_S) -> int | None:
    """First time a pump reading hits ~zero and *stays* there.

    A coolant pump legitimately drops to zero when the car is stationary, so the
    first >5% -> 0% transition is usually a red light. Requiring the zero to
    persist separates a stall from a traffic stop - and unlike reading the
    simulator's injected-fault event, it works on real vehicle data.
    """
    zero_since: int | None = None
    for ts, value in series:
        if value > threshold_pct:
            zero_since = None
        elif zero_since is None:
            zero_since = ts
        elif (ts - zero_since) / 1e9 >= min_s:
            return zero_since
    return None


def before_warning(conn: sqlite3.Connection, lookback_min: int = 60) -> Answer:
    """Reconstruct the precursor sequence from the signals, not the event log.

    ``fault.*`` events are excluded on purpose: they exist only because the
    simulator wrote them, and building an answer from them would be circular.
    The precursor here is found by noticing the coolant pump stopped - which is
    what a real vehicle would give you.
    """
    warn = conn.execute(
        "SELECT * FROM event WHERE severity = 'warning' AND kind NOT LIKE ? "
        "ORDER BY ts_ns DESC LIMIT 1", (EVIDENCE_KIND_PREFIX + "%",)).fetchone()
    if warn is None:
        warn = conn.execute(
            "SELECT * FROM event WHERE severity = 'critical' AND kind NOT LIKE ? "
            "ORDER BY ts_ns DESC LIMIT 1", (EVIDENCE_KIND_PREFIX + "%",)).fetchone()
    if warn is None:
        return Answer("before_warning", "P0",
                      "No warning or critical events recorded.",
                      caveats=["no warning in the retained history"])

    lo = warn["ts_ns"] - lookback_min * 60 * 10**9
    pump = _series(conn, "thermal.coolant_pump", lo, warn["ts_ns"])
    temp = _series(conn, "battery.cell_temp_max", lo, warn["ts_ns"])

    stall_ns = _sustained_stall(pump)
    rise_ns = next((ts for ts, v in temp if v >= 55.0), None)

    parts = []
    if stall_ns is not None:
        lead = (warn["ts_ns"] - stall_ns) / 6e10
        parts.append(f"the coolant pump stopped for good at {_clock(stall_ns)}, "
                     f"{lead:.0f} minutes earlier")
    if rise_ns is not None:
        parts.append(f"pack temperature passed 55 C at {_clock(rise_ns)}")
    parts.append(f"the overheating warning was raised at {_clock(warn['ts_ns'])}")

    caveats = []
    if stall_ns is None:
        caveats.append("no sustained pump stall in the lookback window")
    return Answer(
        intent="before_warning", path="P0",
        text="Before the battery overheating warning: " + "; ".join(parts) + ".",
        data={"warning_kind": warn["kind"],
              "warning_time": _clock(warn["ts_ns"]),
              "stall_time": _clock(stall_ns) if stall_ns else None,
              "lead_minutes": (round((warn["ts_ns"] - stall_ns) / 6e10, 1)
                               if stall_ns else None),
              "lookback_min": lookback_min},
        evidence=[dict(warn)],
        caveats=caveats)


def _clock(ts_ns: int) -> str:
    return dt.datetime.fromtimestamp(ts_ns / 1e9, dt.UTC).strftime("%H:%M UTC")


# ---------------------------------------------------------------------------
# Q5: city vs highway
# ---------------------------------------------------------------------------

def city_vs_highway(conn: sqlite3.Connection) -> Answer:
    rows = {r["regime"]: dict(r) for r in conn.execute("SELECT * FROM v_regime_energy")}
    city, hwy = rows.get("city"), rows.get("highway")
    if not city or not hwy:
        return Answer("city_vs_highway", "P0",
                      "Not enough segmented driving to compare regimes.",
                      caveats=["need both city and motorway segments"])

    ratio = hwy["wh_per_km"] / city["wh_per_km"]
    faster = "motorway" if ratio > 1 else "city"
    text = (
        f"City driving averaged {city['wh_per_km']:.0f} Wh/km over "
        f"{city['distance_km']:.1f} km at a mean {city['mean_speed_kph']:.0f} kph. "
        f"Motorway driving averaged {hwy['wh_per_km']:.0f} Wh/km over "
        f"{hwy['distance_km']:.1f} km at a mean {hwy['mean_speed_kph']:.0f} kph. "
        f"{faster.capitalize()} driving is the more expensive of the two here, by "
        f"a factor of {max(ratio, 1 / ratio):.2f}."
    )
    return Answer(
        intent="city_vs_highway", path="P0", text=text,
        data={"city": city, "highway": hwy, "highway_over_city": round(ratio, 3)},
        evidence=[city, hwy],
        caveats=["regimes are classified from speed thresholds (5 / 80 kph); "
                 "a slow motorway queue is counted as city"])


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

INTENTS: dict[str, tuple[tuple[str, ...], callable]] = {
    "das_events": (("adas", "disengag"), adas_events),
    "before_warning": (("before", "warning"), before_warning),
    "before_warning_b": (("overheat",), before_warning),
    "city_vs_highway": (("city", "highway"), city_vs_highway),
    "efficiency_drop": (("efficien", "drop"), efficiency_largest_drop),
    "efficiency_drop_b": (("largest", "drop"), efficiency_largest_drop),
    "range_reduction": (("range", "reduc"), range_reduction),
    "range_reduction_b": (("range", "why"), range_reduction),
}


def classify(question: str) -> str | None:
    """Rule-based intent routing. No model, so it is free and deterministic.

    This is the part of the system an LLM would normally do. It is a keyword
    matcher instead because intent classification over five known query shapes
    does not need a language model, and because on the constrained tier every
    millisecond spent here is taken from a budget that is already blown.
    """
    q = " ".join(question.lower().split())
    for intent, (keywords, _) in INTENTS.items():
        if all(k in q for k in keywords):
            return intent.rstrip("_b")
    return None


def answer(conn: sqlite3.Connection, question: str) -> Answer:
    intent = classify(question)
    if intent is None:
        return Answer(
            "unknown", "escalate",
            "I could not map that to a known query. Escalating to the language "
            "model would cost several seconds on constrained hardware.",
            caveats=["no deterministic handler for this question"])
    handler = next(fn for name, (_, fn) in INTENTS.items() if name.rstrip("_b") == intent)
    return handler(conn)
