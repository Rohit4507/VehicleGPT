"""Scenario DSL.

"Why did the battery overheat?" is not answerable from any public dataset, and no
CARLA scenario contains it either - you would still be scripting it. So we script
it properly: a small, typed, seeded, version-controlled description of a drive
and the faults injected into it.

That buys three things a heavyweight simulator does not:

1. **Determinism.** Same seed, same bytes. A demo that works twice.
2. **Ground truth.** We know the answer to "what happened before the warning?"
   because we put it there. Evaluation becomes possible at all.
3. **Diffability.** Scenarios live in YAML and show up in code review.

The base vehicle dynamics are modelled from first principles (see
``sim/generator.py``), not recorded. That limitation is stated in
``vehiclegpt/dbc/README.md`` and in the submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml


class Mode(StrEnum):
    """Driving regime. Drives the speed profile and the energy model."""

    IDLE = "idle"
    CITY = "city"
    HIGHWAY = "highway"


class FaultKind(StrEnum):
    BATTERY_OVERHEAT = "battery_overheat"
    ADAS_DISENGAGEMENT = "adas_disengagement"
    COOLANT_PUMP_FAILURE = "coolant_pump_failure"
    SENSOR_DRIFT = "sensor_drift"


@dataclass(frozen=True)
class Segment:
    mode: Mode
    duration_s: float

    def __post_init__(self) -> None:
        if self.duration_s <= 0:
            raise ValueError(f"segment duration must be > 0, got {self.duration_s}")


@dataclass(frozen=True)
class Fault:
    kind: FaultKind
    start_s: float
    duration_s: float
    # Free-form tuning knobs, validated per-kind in the generator.
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_s < 0:
            raise ValueError("fault start_s must be >= 0")
        if self.duration_s <= 0:
            raise ValueError("fault duration_s must be > 0")


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    ambient_c: float
    initial_soc: float
    segments: tuple[Segment, ...]
    faults: tuple[Fault, ...] = ()
    hvac_setpoint_c: float = 22.0

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("scenario needs at least one segment")
        if not 0 <= self.initial_soc <= 100:
            raise ValueError("initial_soc must be within [0, 100]")

    @property
    def duration_s(self) -> float:
        return sum(s.duration_s for s in self.segments)

    def segment_bounds(self) -> list[tuple[float, float, Mode]]:
        """(start_s, end_s, mode) for each segment, in order."""
        out, cursor = [], 0.0
        for seg in self.segments:
            out.append((cursor, cursor + seg.duration_s, seg.mode))
            cursor += seg.duration_s
        return out

    def faults_of(self, kind: FaultKind) -> list[Fault]:
        return [f for f in self.faults if f.kind is kind]


# ---------------------------------------------------------------------------
# YAML round-trip, so scenarios are data rather than code
# ---------------------------------------------------------------------------

def scenario_from_dict(d: dict) -> Scenario:
    return Scenario(
        name=d["name"],
        seed=int(d["seed"]),
        ambient_c=float(d["ambient_c"]),
        initial_soc=float(d["initial_soc"]),
        hvac_setpoint_c=float(d.get("hvac_setpoint_c", 22.0)),
        segments=tuple(
            Segment(mode=Mode(s["mode"]), duration_s=float(s["duration_s"]))
            for s in d["segments"]
        ),
        faults=tuple(
            Fault(
                kind=FaultKind(f["kind"]),
                start_s=float(f["start_s"]),
                duration_s=float(f["duration_s"]),
                params=dict(f.get("params", {})),
            )
            for f in d.get("faults", [])
        ),
    )


def load_scenario(path: str | Path) -> Scenario:
    return scenario_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def scenario_to_dict(s: Scenario) -> dict:
    return {
        "name": s.name,
        "seed": s.seed,
        "ambient_c": s.ambient_c,
        "initial_soc": s.initial_soc,
        "hvac_setpoint_c": s.hvac_setpoint_c,
        "segments": [{"mode": x.mode.value, "duration_s": x.duration_s}
                     for x in s.segments],
        "faults": [{"kind": f.kind.value, "start_s": f.start_s,
                    "duration_s": f.duration_s, "params": f.params}
                   for f in s.faults],
    }
