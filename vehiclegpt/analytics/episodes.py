"""Episode distillation: turn rows into the things people actually ask about.

A vehicle produces two kinds of searchable object and neither is the right one.
Raw *signals* are too granular - "coolant pump = 0 at 17:35:04" is not an
episode. Raw *events* are too granular in the other direction: the store records
four separate rows for one overheating incident, and the driver remembers one
incident.

An episode card is the middle: a bounded stretch of behaviour with a narrative,
the signals it touched, a severity, and provenance pointers back to the rows that
justify it. Retrieval then searches cards, not samples, which is the only way
"show me the ADAS trouble from last week" can return something in a few
milliseconds.

Two rules carried over from the analytics layer, both enforced by tests:

* **`fault.*` events are harness metadata and never become evidence.** They
  record what the simulator injected. A card built from them would be the answer
  key wearing a narrative.
* **A card says what produced it.** Every card carries `evidence`, a list of
  provenance references, so a claim in the text can be traced to an event id or
  a signal window rather than trusted because it reads well.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass, field

from vehiclegpt.analytics import anomaly, changepoint

#: Events closer together than this are one episode. Fifteen minutes is chosen
#: to group a fault cascade (pump stops, temperature climbs, warning raised,
#: critical reached) without swallowing an unrelated drive.
EPISODE_GAP_S = 900

#: Which signals are worth narrating for which kind of event.
#:
#: This mapping exists because the first version scanned every interesting signal
#: for every episode, and the ADAS card came out citing cell imbalance - a
#: plausible-looking sentence with nothing to do with ADAS. Context that is
#: merely co-located is not context.
RELEVANT_SIGNALS: dict[str, tuple[str, ...]] = {
    "battery": ("thermal.coolant_pump", "battery.cell_temp_max", "battery.cell_imbalance"),
    "thermal": ("thermal.coolant_pump", "battery.cell_temp_max"),
    "adas": (),
}

#: Temperature thresholds worth mentioning in a narrative, in C.
TEMP_MILESTONES = (55.0, 60.0, 75.0)

#: A pump below this share of its range counts as stopped.
PUMP_STOPPED_PCT = 1.0


@dataclass
class Episode:
    kind: str                       # "incident" | "trend"
    start_ns: int
    end_ns: int
    severity: str                   # info | warning | critical
    title: str
    body: str
    signals: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)

    @property
    def day(self) -> str:
        return dt.datetime.fromtimestamp(self.start_ns / 1e9, dt.UTC).strftime("%Y-%m-%d")

    def to_row(self) -> tuple:
        return (self.kind, self.start_ns, self.end_ns, self.day, self.severity,
                self.title, self.body, ",".join(self.signals),
                json.dumps(self.evidence, sort_keys=True))


SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def _clock(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9, dt.UTC).strftime("%H:%M")


def _day_label(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9, dt.UTC).strftime("%a %d %b")


def _clock_span(start_ns: int, end_ns: int) -> str:
    if _day_label(start_ns) != _day_label(end_ns):
        return f"from {_clock(start_ns)} on {_day_label(start_ns)} to {_clock(end_ns)}"
    return f"between {_clock(start_ns)} and {_clock(end_ns)} on {_day_label(start_ns)}"


def _series(conn: sqlite3.Connection, path: str, lo_s: int,
            hi_s: int) -> list[tuple[int, float]]:
    row = conn.execute("SELECT id FROM signal WHERE path = ?", (path,)).fetchone()
    if row is None:
        return []
    return [(int(r["ts_s"]) * 10**9, float(r["value"])) for r in conn.execute(
        "SELECT ts_s, value FROM sample WHERE signal_id = ? AND ts_s BETWEEN ? AND ? "
        "ORDER BY ts_s", (row["id"], lo_s, hi_s))]


def _signal_observations(conn: sqlite3.Connection, start_ns: int, end_ns: int,
                         paths: tuple[str, ...], lookback_s: int = 1800,
                         after_s: int = 3600) -> list[dict]:
    """Notable readings in and shortly before the episode, for relevant signals.

    The lookback is generous on purpose: the interesting precursor to a warning
    is usually what happened *before* the window, which is the entire premise of
    the "what happened before X" question.

    Each signal gets the detector its physics calls for. A single generic outlier
    scan was tried first and missed the most important fact in the store: the
    coolant pump toggles on and off for minutes before it fails, so its stall is
    not an outlier - the pre-stall swings are already extreme. Persistence is the
    signature, and :func:`changepoint.sustained_below` is what finds it.
    """
    observations: list[dict] = []
    # The window runs past the episode on purpose: a stall that is still going
    # when the episode ends must be measured to where it actually stopped, not
    # truncated at the event that made us look.
    lo_s = int(start_ns / 1e9) - lookback_s
    hi_s = int(end_ns / 1e9) + after_s

    for path in paths:
        series = _series(conn, path, lo_s, hi_s)
        if len(series) < 4:
            continue
        values = [v for _, v in series]

        if path == "thermal.coolant_pump":
            stall = changepoint.sustained_below(series, PUMP_STOPPED_PCT, 60.0)
            if stall is not None:
                observations.append({
                    "signal": path, "ts_ns": stall[0], "kind": "stall",
                    "value": PUMP_STOPPED_PCT, "duration_s": stall[1],
                    "ended": stall[2],
                })

        elif path == "battery.cell_temp_max":
            for milestone in TEMP_MILESTONES:
                crossings = [ts for ts, v in series if v >= milestone]
                if crossings:
                    observations.append({
                        "signal": path, "ts_ns": crossings[0], "kind": "threshold",
                        "value": milestone, "duration_s": 0.0,
                    })

        hits = anomaly.detect(values, threshold=3.5, window=7,
                              min_delta=2.0 if path != "battery.cell_imbalance" else 0.5,
                              top=1)
        for hit in hits:
            observations.append({
                "signal": path, "ts_ns": series[hit["index"]][0], "kind": "excursion",
                "value": hit["value"], "delta": hit["delta"], "duration_s": 0.0,
            })

    observations.sort(key=lambda o: o["ts_ns"])
    return observations


def _describe(observation: dict) -> str:
    """One clause a reader can act on, in the units of the signal."""
    clock = _clock(observation["ts_ns"])
    name = observation["signal"].split(".", 1)[-1].replace("_", " ")
    kind = observation["kind"]

    if kind == "stall":
        minutes = observation["duration_s"] / 60.0
        unit = "minute" if round(minutes) == 1 else "minutes"
        if not observation.get("ended", True):
            return (f"the coolant pump stopped at {clock} and was still off "
                    f"{minutes:.0f} {unit} later, when the data ends")
        return (f"the coolant pump stopped at {clock} and stayed off for "
                f"{minutes:.0f} {unit}")
    if kind == "threshold":
        return f"pack temperature reached {observation['value']:.0f} C at {clock}"
    delta = observation.get("delta", 0.0)
    if observation["signal"].endswith("imbalance"):
        return (f"cell imbalance reached {observation['value']:.1f} C at {clock} "
                f"({delta:+.1f} C from its local level)")
    return (f"{name} read {observation['value']:.1f} at {clock} "
            f"({delta:+.1f} from its local level)")


def incident_episodes(conn: sqlite3.Connection, gap_s: int = EPISODE_GAP_S) -> list[Episode]:
    """Group real events into episodes and narrate what the signals were doing."""
    rows = conn.execute(
        "SELECT id, ts_ns, kind, severity, message FROM event "
        "WHERE kind NOT LIKE 'fault.%' ORDER BY ts_ns").fetchall()
    if not rows:
        return []

    clusters: list[list] = [[rows[0]]]
    for row in rows[1:]:
        if (row["ts_ns"] - clusters[-1][-1]["ts_ns"]) / 1e9 <= gap_s:
            clusters[-1].append(row)
        else:
            clusters.append([row])

    episodes: list[Episode] = []
    for cluster in clusters:
        start_ns = int(cluster[0]["ts_ns"])
        end_ns = int(cluster[-1]["ts_ns"])
        severity = max((r["severity"] for r in cluster), key=lambda s: SEVERITY_RANK.get(s, 0))
        prefixes = {r["kind"].split(".")[0] for r in cluster}
        paths = tuple(sorted({p for pre in prefixes for p in RELEVANT_SIGNALS.get(pre, ())}))
        observations = _signal_observations(conn, start_ns, end_ns, paths)
        observations = [o for o in observations if o["ts_ns"] <= start_ns] or observations

        labels = []
        for row in cluster:
            short = row["kind"].split(".")[-1].replace("_", " ")
            labels.append(f"{short} at {_clock(row['ts_ns'])}")
        title = f"{severity.capitalize()}: {cluster[0]['kind'].split('.')[0]} " \
                f"{'incident' if len(cluster) == 1 else 'sequence'} on {_day_label(start_ns)}"

        parts = [f"{len(cluster)} recorded event"
                 f"{'' if len(cluster) == 1 else 's'} {_clock_span(start_ns, end_ns)}: "
                 f"{'; '.join(labels)}."]
        for obs in observations[:3]:
            parts.append(f"Shortly before, {_describe(obs)}.")
        if any(r["severity"] in ("warning", "critical") for r in cluster):
            parts.append("The precursor readings above are signal evidence; the events "
                         "themselves only record that the threshold was crossed.")

        signals = sorted({o["signal"] for o in observations})
        episodes.append(Episode(
            kind="incident", start_ns=start_ns, end_ns=end_ns, severity=severity,
            title=title, body=" ".join(parts),
            signals=signals + sorted({r["kind"] for r in cluster}),
            evidence=[{"type": "event", "id": int(r["id"]), "kind": r["kind"]}
                      for r in cluster]
                     + [{"type": "signal", "path": o["signal"], "ts_ns": o["ts_ns"]}
                        for o in observations[:3]]))
    return episodes


def trend_episodes(conn: sqlite3.Connection, min_km: float = 10.0) -> list[Episode]:
    """One card per sustained level change in daily consumption.

    Short days are excluded for the reason doc 11 §11.7 gives at length: on this
    store they invent a degradation on 14 March that is really a change in trip
    length.
    """
    rows = [r for r in conn.execute(
        "SELECT day, wh_per_km, distance_km FROM daily ORDER BY day")
        if r["wh_per_km"] is not None and r["distance_km"] >= min_km]
    if len(rows) < 6:
        return []
    values = [float(r["wh_per_km"]) for r in rows]
    segs = changepoint.segments(values, min_delta=5.0)
    episodes: list[Episode] = []
    for seg in segs[1:]:
        day = rows[seg["start"]]["day"]
        start_ns = int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.UTC).timestamp() * 1e9)
        delta = seg["delta"] or 0.0
        direction = "rose" if delta > 0 else "fell"
        severity = "warning" if abs(delta) >= 20 else "info"
        title = (f"Consumption {direction} {abs(delta):.0f} Wh/km "
                 f"from {_day_label(start_ns)}")
        body = (
            f"Mean consumption {direction} from {seg['mean'] - delta:.0f} to "
            f"{seg['mean']:.0f} Wh/km starting {_day_label(start_ns)}, a change of "
            f"{delta:+.0f} Wh/km. The level then held for {seg['length']} comparable "
            f"driving day{'' if seg['length'] == 1 else 's'}. "
            f"Days under {min_km:.0f} km are excluded, so this compares trips of "
            f"comparable length rather than a mix of commutes and errands.")
        episodes.append(Episode(
            kind="trend", start_ns=start_ns, end_ns=start_ns, severity=severity,
            title=title, body=body, signals=["daily.wh_per_km"],
            evidence=[{"type": "daily", "day": r["day"], "wh_per_km": float(r["wh_per_km"])}
                      for r in rows[seg["start"]:seg["end"]]]))
    return episodes


def distill(conn: sqlite3.Connection) -> list[Episode]:
    """All episodes, newest last. Deterministic: same store, same cards."""
    episodes = incident_episodes(conn) + trend_episodes(conn)
    episodes.sort(key=lambda e: (e.start_ns, e.kind, e.title))
    return episodes
