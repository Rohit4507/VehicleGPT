"""Multi-day driving program.

A single scenario is ~30 minutes; the questions in the brief span *weeks*
("why has my range reduced over the last two weeks", "show all ADAS
disengagements from yesterday"). So the simulator composes a program of day-long
drives on a fixed calendar, and the store keeps them as real timestamps.

The calendar is explicit rather than "now minus N days" so that every query
result is reproducible and every date in the demo is stable.

What the program is built to contain
------------------------------------
* A **cold snap** between day 9 and day 10 (12 C -> -4 C). This is the honest
  explanation for the range reduction, and it gives Q2 a single unambiguous
  largest drop to find.
* A **coolant pump failure and thermal excursion** on day 9, so Q4 has a real
  precursor sequence in the data.
* **ADAS disengagements on the final day**, so "from yesterday" returns
  something without the query layer needing special cases.

None of these are hard-coded into the answers: the analytics layer has to
rediscover them from `/sample` and `/event` like any other data.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass

from sim.generator import Simulation, generate
from sim.scenario import Fault, FaultKind, Mode, Scenario, Segment

# Day 0 of the program. Chosen to be a Monday so week-over-week grouping is
# meaningful without ISO-week gymnastics.
PROGRAM_START = dt.datetime(2026, 3, 2, tzinfo=dt.UTC)   # Monday
PROGRAM_DAYS = 14

# Ambient by day index. A 16 C step between day 9 and day 10 is the cold snap.
AMBIENT_BY_DAY = [18, 17, 16, 18, 15, 16, 14, 15, 13, 12, -4, -5, -6, -6]

COLD_SNAP_DAY = 10
OVERHEAT_DAY = 9
ADAS_DAY = PROGRAM_DAYS - 1          # "yesterday" relative to PROGRAM_END

# Drives start at 08:00 and 17:30 local, which keeps a clean day boundary.
MORNING_DRIVE_HOUR = 8
EVENING_DRIVE_HOUR = 17


@dataclass(frozen=True)
class DayDrive:
    day_index: int
    label: str
    start_epoch_s: float
    scenario: Scenario


def _drive_times(day: dt.datetime) -> list[tuple[str, dt.datetime]]:
    out = [("morning", day.replace(hour=MORNING_DRIVE_HOUR, minute=0))]
    if day.weekday() < 5:
        out.append(("evening", day.replace(hour=EVENING_DRIVE_HOUR, minute=30)))
    return out


def _segments_for(day: dt.datetime, label: str, day_index: int) -> list[Segment]:
    weekend = day.weekday() >= 5
    if weekend:
        return [Segment(Mode.CITY, 600), Segment(Mode.CITY, 600)]
    if label == "morning":
        return [Segment(Mode.CITY, 300), Segment(Mode.HIGHWAY, 600), Segment(Mode.CITY, 180)]
    if day_index == OVERHEAT_DAY:
        # A longer motorway stint, so the thermal excursion has time to develop
        # inside the drive instead of being cut off by the car being switched off.
        return [Segment(Mode.CITY, 300), Segment(Mode.HIGHWAY, 900), Segment(Mode.CITY, 420)]
    return [Segment(Mode.CITY, 420), Segment(Mode.HIGHWAY, 480), Segment(Mode.CITY, 120)]


def _faults_for(day_index: int, label: str) -> list[Fault]:
    faults: list[Fault] = []
    if day_index == OVERHEAT_DAY and label == "evening":
        # The heat source must stay active for the whole drive, or the pack
        # plateaus below the warning threshold and the scenario has no warning
        # for the analytics layer to explain.
        faults.append(Fault(FaultKind.COOLANT_PUMP_FAILURE, start_s=300, duration_s=1300))
        faults.append(Fault(FaultKind.BATTERY_OVERHEAT, start_s=120, duration_s=1600,
                            params={"watts": 18000}))
    if day_index == ADAS_DAY and label == "morning":
        for start in (200.0, 430.0, 610.0):
            faults.append(Fault(FaultKind.ADAS_DISENGAGEMENT, start_s=start, duration_s=4))
    if day_index == ADAS_DAY and label == "evening":
        faults.append(Fault(FaultKind.ADAS_DISENGAGEMENT, start_s=280, duration_s=4))
    return faults


def program_days(days: int = PROGRAM_DAYS) -> Iterator[DayDrive]:
    """Yield every drive in the program, in chronological order."""
    for day_index in range(days):
        day = PROGRAM_START + dt.timedelta(days=day_index)
        ambient = AMBIENT_BY_DAY[day_index % len(AMBIENT_BY_DAY)]
        for label, when in _drive_times(day):
            start_epoch_s = when.timestamp()
            scenario = Scenario(
                name=f"day{day_index:02d}-{label}",
                seed=1000 + day_index * 7 + (0 if label == "morning" else 3),
                ambient_c=float(ambient),
                initial_soc=float(80 - (day_index % 5) * 4),
                hvac_setpoint_c=22.0,
                segments=tuple(_segments_for(day, label, day_index)),
                faults=tuple(_faults_for(day_index, label)),
            )
            yield DayDrive(day_index=day_index, label=label,
                           start_epoch_s=start_epoch_s, scenario=scenario)


def program_end_epoch_s(days: int = PROGRAM_DAYS) -> float:
    """Start of the day after the last program day - i.e. "now" for the demo."""
    return (PROGRAM_START + dt.timedelta(days=days)).timestamp()


def generate_program(days: int = PROGRAM_DAYS) -> Iterator[tuple[DayDrive, Simulation]]:
    for drive in program_days(days):
        yield drive, generate(drive.scenario)
