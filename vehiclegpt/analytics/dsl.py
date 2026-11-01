"""A closed query DSL: what the language model is allowed to say.

The model never writes SQL. It emits a small JSON object naming one of a fixed
set of query shapes plus a time window; this module validates that object and
compiles it to parameterised SQL. Three properties make that safe rather than
merely tidy:

1. **The SQL is a function of the kind alone.** ``compile_().sql`` is a constant
   string per ``Kind``; every value arrives as a bound parameter. There is no
   code path in which a value reaches the query text, so an injection payload is
   not escaped - it is a string in a parameter where SQLite expects a value.
   ``test_dsl.py`` asserts this by compiling hostile inputs and comparing the
   SQL byte for byte.

2. **The schema is closed.** Unknown keys are rejected, not ignored. An object
   carrying ``{"sql": "DROP TABLE sample"}`` fails validation instead of being
   quietly dropped, so a model that has drifted from the grammar produces a
   visible error rather than a silent partial query.

3. **Windows are bounded.** A date range is parsed, ordered, and length-limited.
   The alternative - trusting a model's date arithmetic - is how you get a query
   that scans every row on a device with 2 GB of RAM.

The shapes exist because the five brief questions need them, not because a
general SQL dialect was wanted. A general one would be a much larger attack
surface to test, and its extra power would only ever be used by the model.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: Longest window a single query may cover. Two years of history is far beyond
#: the "several weeks" the brief asks for, and the cap keeps a malformed or
#: adversarial range from becoming a full-table scan on the target device.
MAX_WINDOW_DAYS = 730

#: Days below this distance are excluded from consumption averages. A cold-start
#: crawl can read 900 Wh/km and would otherwise dominate the mean. Matches
#: ``v_daily_rolling``.
MIN_DAY_KM = 2.0


class Kind(StrEnum):
    """The closed set of query shapes."""

    CONSUMPTION_TREND = "consumption_trend"
    EFFICIENCY_DROP = "efficiency_drop"
    ADAS_EVENTS = "adas_events"
    PRECURSOR = "precursor"
    REGIME_COMPARE = "regime_compare"


#: Allowed params per kind, with the validator for each. An empty dict means the
#: kind takes no params beyond the window, and any extra key is an error.
PARAM_SPEC: dict[Kind, dict[str, str]] = {
    Kind.CONSUMPTION_TREND: {"baseline_days": "days"},
    Kind.EFFICIENCY_DROP: {"min_km": "distance"},
    Kind.ADAS_EVENTS: {},
    Kind.PRECURSOR: {"lookback_min": "minutes"},
    Kind.REGIME_COMPARE: {},
}

DEFAULT_PARAMS: dict[Kind, dict[str, Any]] = {
    Kind.CONSUMPTION_TREND: {"baseline_days": 7},
    Kind.EFFICIENCY_DROP: {"min_km": 10.0},
    Kind.ADAS_EVENTS: {},
    Kind.PRECURSOR: {"lookback_min": 60},
    Kind.REGIME_COMPARE: {},
}

REGIMES = ("city", "highway")


class DslError(ValueError):
    """The query object is not well formed. Always a caller error, never a crash."""


@dataclass(frozen=True)
class Query:
    kind: Kind
    start_day: str                 # ISO date, inclusive
    end_day: str                   # ISO date, inclusive
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": str(self.kind), "start_day": self.start_day,
                "end_day": self.end_day, "params": dict(self.params)}


@dataclass(frozen=True)
class Compiled:
    """SQL plus bound parameters. The SQL never varies for a given kind."""

    sql: str
    params: dict[str, Any]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _check_day(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise DslError(f"{field_name} must be an ISO date string, got {type(value).__name__}")
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise DslError(f"{field_name} is not an ISO date: {value!r}") from exc


def _check_int(value: Any, field_name: str, lo: int, hi: int) -> int:
    # bool is a subclass of int and would sail through otherwise
    if isinstance(value, bool) or not isinstance(value, int):
        raise DslError(f"{field_name} must be an integer, got {type(value).__name__}")
    if not lo <= value <= hi:
        raise DslError(f"{field_name} must be between {lo} and {hi}, got {value}")
    return value


def _check_distance(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DslError(f"{field_name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not 0.0 <= value <= 10_000.0:
        raise DslError(f"{field_name} must be between 0 and 10000 km, got {value}")
    return value


_VALIDATORS = {
    "days": lambda v, n: _check_int(v, n, 0, MAX_WINDOW_DAYS),
    "minutes": lambda v, n: _check_int(v, n, 1, 1440),
    "distance": _check_distance,
}


def parse(raw: dict[str, Any]) -> Query:
    """Validate a decoded JSON object into a :class:`Query`.

    Deliberately strict: an unknown top-level or param key raises. Silently
    ignoring what a model produced is how a drifted grammar becomes a wrong
    answer that looks well formed.
    """
    if not isinstance(raw, dict):
        raise DslError(f"query must be an object, got {type(raw).__name__}")

    allowed_top = {"kind", "start_day", "end_day", "params"}
    unknown = set(raw) - allowed_top
    if unknown:
        raise DslError(f"unknown field(s): {', '.join(sorted(unknown))}")

    missing = {"kind", "start_day", "end_day"} - set(raw)
    if missing:
        raise DslError(f"missing field(s): {', '.join(sorted(missing))}")

    try:
        kind = Kind(raw["kind"])
    except ValueError as exc:
        raise DslError(
            f"unknown kind {raw['kind']!r}; expected one of "
            f"{', '.join(str(k) for k in Kind)}") from exc

    start = _check_day(raw["start_day"], "start_day")
    end = _check_day(raw["end_day"], "end_day")
    if end < start:
        raise DslError(f"end_day {end} is before start_day {start}")
    span = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
    if span > MAX_WINDOW_DAYS:
        raise DslError(f"window spans {span} days, limit is {MAX_WINDOW_DAYS}")

    given = raw.get("params") or {}
    if not isinstance(given, dict):
        raise DslError("params must be an object")

    spec = PARAM_SPEC[kind]
    unknown = set(given) - set(spec)
    if unknown:
        raise DslError(
            f"{kind} does not take {', '.join(sorted(unknown))}; "
            f"allowed: {', '.join(sorted(spec)) or 'none'}")

    params = dict(DEFAULT_PARAMS[kind])
    supplied = set(given)
    for name, value in given.items():
        params[name] = _VALIDATORS[spec[name]](value, name)

    # The baseline is the stretch immediately *before* the window, so equal
    # length is normal (compare this week against last week). Longer than the
    # window is a sign the caller meant something else - but only when the
    # caller actually asked for it. A default must never produce a query that
    # the validator would reject, so an unasked-for baseline is clamped to the
    # window instead.
    if kind is Kind.CONSUMPTION_TREND:
        if "baseline_days" in supplied and params["baseline_days"] > span:
            raise DslError(f"baseline_days ({params['baseline_days']}) must not "
                           f"exceed the window ({span} days)")
        params["baseline_days"] = min(params["baseline_days"], span)

    return Query(kind=kind, start_day=start, end_day=end, params=params)


def from_json(text: str) -> Query:
    """Parse a JSON string. Malformed JSON is a :class:`DslError`, not a crash."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DslError(f"not valid JSON: {exc.msg}") from exc
    return parse(raw)


def build(kind: Kind | str, start_day: str, end_day: str,
          **params: Any) -> Query:
    """Convenience constructor that validates exactly like :func:`parse`."""
    return parse({"kind": str(kind), "start_day": start_day,
                  "end_day": end_day, "params": params})


# ---------------------------------------------------------------------------
# compilation
# ---------------------------------------------------------------------------

def _day_bounds(start_day: str, end_day: str) -> tuple[int, int]:
    """Half-open nanosecond bounds covering both days inclusive.

    Validates again rather than trusting the caller: a ``Query`` can be built
    directly, bypassing :func:`parse`, and a raw ``ValueError`` escaping from
    here would be indistinguishable from a bug in the caller.
    """
    try:
        lo = dt.datetime.fromisoformat(start_day).replace(tzinfo=dt.UTC)
        hi = dt.datetime.fromisoformat(end_day).replace(tzinfo=dt.UTC)
    except (TypeError, ValueError) as exc:
        raise DslError(f"start_day/end_day are not ISO dates: "
                       f"{start_day!r}, {end_day!r}") from exc
    return int(lo.timestamp() * 1e9), int((hi.timestamp() + 86400) * 1e9)


_SQL: dict[Kind, str] = {
    # Window and baseline aggregate in one statement. The baseline CASE simply
    # matches nothing when baseline_days is 0, which keeps the SQL static.
    Kind.CONSUMPTION_TREND: """
SELECT
    COALESCE(SUM(CASE WHEN day BETWEEN :start_day AND :end_day
                      AND distance_km > :min_day_km THEN distance_km END), 0) AS a_km,
    COALESCE(SUM(CASE WHEN day BETWEEN :start_day AND :end_day
                      AND distance_km > :min_day_km THEN energy_wh END), 0) AS a_wh,
    AVG(CASE WHEN day BETWEEN :start_day AND :end_day
             AND distance_km > :min_day_km THEN hvac_w END)             AS a_hvac_w,
    AVG(CASE WHEN day BETWEEN :start_day AND :end_day
             AND distance_km > :min_day_km THEN ambient_c END)          AS a_ambient_c,
    COUNT(CASE WHEN day BETWEEN :start_day AND :end_day
               AND distance_km > :min_day_km THEN 1 END)                AS a_days,
    COALESCE(SUM(CASE WHEN day BETWEEN :base_day AND :base_day_end
                      AND distance_km > :min_day_km THEN distance_km END), 0) AS b_km,
    COALESCE(SUM(CASE WHEN day BETWEEN :base_day AND :base_day_end
                      AND distance_km > :min_day_km THEN energy_wh END), 0) AS b_wh,
    AVG(CASE WHEN day BETWEEN :base_day AND :base_day_end
             AND distance_km > :min_day_km THEN hvac_w END)             AS b_hvac_w,
    AVG(CASE WHEN day BETWEEN :base_day AND :base_day_end
             AND distance_km > :min_day_km THEN ambient_c END)          AS b_ambient_c,
    COUNT(CASE WHEN day BETWEEN :base_day AND :base_day_end
               AND distance_km > :min_day_km THEN 1 END)                AS b_days,
    (SELECT COALESCE(SUM(end_ns - start_ns), 0) FROM segment
      WHERE start_ns >= :start_ns AND start_ns < :end_ns)               AS a_drive_ns,
    (SELECT COALESCE(SUM(end_ns - start_ns), 0) FROM segment
      WHERE start_ns >= :base_ns AND start_ns < :base_end_ns)           AS b_drive_ns
FROM daily
""",
    # Day-over-day step, using LAG rather than a trailing window: a trailing
    # window peaks on its own final day and would always report the last day.
    Kind.EFFICIENCY_DROP: """
WITH ordered AS (
    SELECT day, distance_km, energy_wh, ambient_c, wh_per_km,
           LAG(day)       OVER (ORDER BY day) AS prev_day,
           LAG(wh_per_km) OVER (ORDER BY day) AS prev_wh_per_km,
           LAG(ambient_c) OVER (ORDER BY day) AS prev_ambient_c
    FROM daily
    WHERE distance_km >= :min_km AND wh_per_km IS NOT NULL
)
SELECT day, prev_day, wh_per_km, prev_wh_per_km, ambient_c, prev_ambient_c,
       wh_per_km - prev_wh_per_km AS delta_wh_per_km
FROM ordered
WHERE prev_wh_per_km IS NOT NULL
  AND day BETWEEN :start_day AND :end_day
ORDER BY delta_wh_per_km DESC
LIMIT 1
""",
    Kind.ADAS_EVENTS: """
SELECT id, ts_ns, day, time_utc, severity, message
FROM v_adas_disengagements
WHERE ts_ns >= :start_ns AND ts_ns < :end_ns
ORDER BY ts_ns
""",
    # Predecessors of each warning, excluding the harness's own fault records.
    Kind.PRECURSOR: """
SELECT w.id AS warning_id, w.ts_ns AS warning_ns, w.kind AS warning_kind,
       w.severity AS warning_severity, w.message AS warning_message,
       p.ts_ns AS precursor_ns, p.kind AS precursor_kind, p.message AS precursor_message,
       (w.ts_ns - p.ts_ns) / 1000000000.0 AS lead_s
FROM event w
JOIN event p
  ON p.ts_ns < w.ts_ns
 AND p.ts_ns >= w.ts_ns - :lookback_ns
WHERE w.severity IN ('warning', 'critical')
  AND w.kind NOT LIKE :fault_prefix
  AND p.kind NOT LIKE :fault_prefix
  AND w.ts_ns >= :start_ns AND w.ts_ns < :end_ns
ORDER BY w.ts_ns, p.ts_ns
""",
    Kind.REGIME_COMPARE: """
SELECT regime, segments, distance_km, energy_wh, wh_per_km, mean_speed_kph
FROM v_regime_energy
WHERE regime IN (:regime_a, :regime_b)
ORDER BY wh_per_km DESC
""",
}


def compile_query(q: Query) -> Compiled:
    """Compile a validated query. The returned SQL depends only on ``q.kind``."""
    if q.kind not in _SQL:            # pragma: no cover - Kind is closed
        raise DslError(f"no compiler for kind {q.kind}")

    start_ns, end_ns = _day_bounds(q.start_day, q.end_day)
    params: dict[str, Any] = {"start_day": q.start_day, "end_day": q.end_day,
                              "start_ns": start_ns, "end_ns": end_ns}

    if q.kind is Kind.CONSUMPTION_TREND:
        baseline = int(q.params["baseline_days"])
        base_end = dt.date.fromisoformat(q.start_day) - dt.timedelta(days=1)
        base_start = base_end - dt.timedelta(days=max(baseline - 1, 0))
        base_ns, base_end_ns = _day_bounds(base_start.isoformat(), base_end.isoformat())
        params |= {"base_day": base_start.isoformat(),
                   "base_day_end": base_end.isoformat(),
                   "base_ns": base_ns, "base_end_ns": base_end_ns,
                   "min_day_km": MIN_DAY_KM}
    elif q.kind is Kind.EFFICIENCY_DROP:
        params["min_km"] = float(q.params["min_km"])
    elif q.kind is Kind.PRECURSOR:
        params |= {"lookback_ns": int(q.params["lookback_min"]) * 60 * 10**9,
                   "fault_prefix": "fault.%"}
    elif q.kind is Kind.REGIME_COMPARE:
        params |= {"regime_a": REGIMES[0], "regime_b": REGIMES[1]}

    return Compiled(sql=_SQL[q.kind].strip(), params=params)


def run(conn: sqlite3.Connection, q: Query) -> list[dict]:
    """Execute a validated query and return rows as dicts."""
    compiled = compile_query(q)
    cursor = conn.execute(compiled.sql, compiled.params)
    return [dict(row) for row in cursor.fetchall()]


def rows_value(rows: list[dict], key: str, default: Any = 0) -> Any:
    """Read one aggregate out of a single-row result, tolerating NULL."""
    if not rows:
        return default
    value = rows[0].get(key)
    return default if value is None else value
