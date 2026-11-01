"""Time expressions, resolved deterministically.

A language model is bad at date arithmetic and a car is bad at waiting for one.
"the last two weeks" has exactly one correct interpretation, so it is parsed by
rule, in microseconds, and never handed to the model.

Three decisions are worth stating because they change the answers:

1. **"Now" is a parameter, not the wall clock.** On a parked vehicle the wall
   clock is months past the last recording, so a query anchored to it returns
   nothing. The caller passes the newest sample time, which is what the driver
   means by "recently".

2. **A bare relative phrase ends now.** "the last two weeks" is *two weeks up to
   the latest data*, not two whole calendar weeks. Calendar-aligned windows are
   available explicitly ("last week"), but they are not the default.

3. **Nothing parsed is a valid answer.** ``parse`` returns ``None`` rather than
   guessing a default window; the caller decides whether a missing window is
   acceptable. A silent default of "all time" is how a query becomes a
   full-table scan on a device with 2 GB of RAM.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

#: Longest window any single query may request (matches ``dsl.MAX_WINDOW_DAYS``).
MAX_DAYS = 730

_UNIT_DAYS = {
    "minute": 1.0 / 1440,
    "hour": 1.0 / 24,
    "day": 1.0,
    "week": 7.0,
    "fortnight": 14.0,
    "month": 30.0,
    "year": 365.0,
}

_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
}

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_QUANTITY = re.compile(
    r"\b(?:(?P<n>\d+)|(?P<w>" + "|".join(_WORDS) + r"))\s*"
    r"(?P<unit>minute|hour|day|week|fortnight|month|year)s?\b", re.I)


@dataclass(frozen=True)
class Window:
    """A resolved half-open time range, in nanoseconds since the epoch."""

    start_ns: int
    end_ns: int
    label: str

    @property
    def days(self) -> float:
        return (self.end_ns - self.start_ns) / 86400e9

    def to_dict(self) -> dict:
        return {"start_ns": self.start_ns, "end_ns": self.end_ns,
                "label": self.label, "days": round(self.days, 3)}


def _ns(stamp: dt.datetime) -> int:
    return int(stamp.timestamp() * 1e9)


def _at_offset(now_ns: int, tz_minutes: int) -> dt.datetime:
    """The local wall-clock reading that 'now_ns' corresponds to."""
    return (dt.datetime.fromtimestamp(now_ns / 1e9, dt.UTC)
            + dt.timedelta(minutes=tz_minutes))


def _from_local(local: dt.datetime, tz_minutes: int) -> int:
    return _ns(local - dt.timedelta(minutes=tz_minutes))


def _clamp(window: Window) -> Window:
    span = window.end_ns - window.start_ns
    if span <= 0:
        raise ValueError("window ends before it starts")
    if span > MAX_DAYS * 86400e9:
        raise ValueError(f"window spans more than {MAX_DAYS} days")
    return window


def _relative(match: re.Match, now_ns: int) -> Window | None:
    unit = match.group("unit").lower()
    raw = match.group("n")
    count = int(raw) if raw else _WORDS[match.group("w").lower()]
    if count <= 0:
        return None
    days = count * _UNIT_DAYS[unit]
    return _clamp(Window(int(now_ns - days * 86400e9), now_ns,
                         f"last {count} {unit}{'s' if count != 1 else ''}"))


def _calendar(text: str, now_ns: int, tz_minutes: int) -> Window | None:
    """Named days and calendar periods, which are not relative to the hour."""
    local = _at_offset(now_ns, tz_minutes)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)

    if re.search(r"\btoday\b|\bthis morning\b|\bso far today\b", text):
        return _clamp(Window(_from_local(midnight, tz_minutes), now_ns,
                             "today so far"))
    if re.search(r"\b(?:the\s+)?day before yesterday\b", text):
        start = midnight - dt.timedelta(days=2)
        return _clamp(Window(_from_local(start, tz_minutes),
                             _from_local(start + dt.timedelta(days=1), tz_minutes),
                             "the day before yesterday"))
    if re.search(r"\byesterday\b", text):
        start = midnight - dt.timedelta(days=1)
        return _clamp(Window(_from_local(start, tz_minutes),
                             _from_local(midnight, tz_minutes), "yesterday"))
    if re.search(r"\blast night\b", text):
        start = midnight - dt.timedelta(days=1, hours=6)
        return _clamp(Window(_from_local(start, tz_minutes),
                             _from_local(midnight, tz_minutes), "last night"))
    if re.search(r"\bthis week\b", text):
        start = midnight - dt.timedelta(days=local.weekday())
        return _clamp(Window(_from_local(start, tz_minutes), now_ns, "this week"))
    # "last week" is the previous calendar week; "the last week" is the seven
    # days up to now. The article is the only thing distinguishing them, so it
    # decides which branch runs.
    if re.search(r"(?<!the )\blast week\b", text):
        start = midnight - dt.timedelta(days=local.weekday() + 7)
        return _clamp(Window(_from_local(start, tz_minutes),
                             _from_local(start + dt.timedelta(days=7), tz_minutes),
                             "last week"))
    if re.search(r"(?<!the )\blast month\b", text):
        start = midnight - dt.timedelta(days=local.weekday()) \
            if False else (local.replace(day=1, hour=0, minute=0, second=0,
                                         microsecond=0))
        prior = (start - dt.timedelta(days=1)).replace(day=1)
        return _clamp(Window(_from_local(prior, tz_minutes),
                             _from_local(start, tz_minutes), "last month"))
    return None


def _explicit(text: str, now_ns: int, tz_minutes: int) -> Window | None:
    """Absolute dates: ISO, and '11 March' / 'March 11th' with an optional year."""
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso:
        day = dt.datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        return _clamp(Window(_from_local(day, tz_minutes),
                             _from_local(day + dt.timedelta(days=1), tz_minutes),
                             f"{day:%Y-%m-%d}"))

    months = "|".join(sorted(_MONTHS, key=len, reverse=True))
    day_first = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({months})\b",
                          text)
    month_first = re.search(rf"\b({months})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text)
    if day_first:
        day_s, month_s = day_first.group(1), day_first.group(2)
    elif month_first:
        month_s, day_s = month_first.group(1), month_first.group(2)
    else:
        return None

    year_match = re.search(r"\b(20\d{2})\b", text)
    year = int(year_match.group(1)) if year_match else _at_offset(
        now_ns, tz_minutes).year
    try:
        day = dt.datetime(year, _MONTHS[month_s.lower()], int(day_s))
    except ValueError:
        return None
    return _clamp(Window(_from_local(day, tz_minutes),
                         _from_local(day + dt.timedelta(days=1), tz_minutes),
                         f"{day:%Y-%m-%d}"))


def parse(text: str, now_ns: int, tz_minutes: int = 0) -> Window | None:
    """Resolve the first time expression in ``text``, or return ``None``.

    Order matters: an explicit range beats a named day, which beats a relative
    phrase. "between 11 March and 12 March" contains two named days, so the
    range has to be tried first or it would silently become one day.
    """
    if not text:
        return None
    lowered = " ".join(text.lower().split())

    between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!,]|$)", lowered)
    if between:
        left = _explicit(between.group(1), now_ns, tz_minutes) or _calendar(
            between.group(1), now_ns, tz_minutes)
        right = _explicit(between.group(2), now_ns, tz_minutes) or _calendar(
            between.group(2), now_ns, tz_minutes)
        if left and right:
            start, end = min(left.start_ns, right.start_ns), max(
                left.end_ns, right.end_ns)
            label = f"{left.label} to {right.label}"
            return _clamp(Window(start, end, label))

    for finder in (_explicit, _calendar):
        found = finder(lowered, now_ns, tz_minutes)
        if found is not None:
            return found

    # "the last week" and "past two weeks" both mean 'up to now'.
    relative = _QUANTITY.search(lowered)
    if relative and re.search(r"\b(last|past|previous|recent)\b", lowered):
        return _relative(relative, now_ns)
    if re.search(r"\bthe last (week|month|year)\b", lowered):
        unit = re.search(r"\bthe last (week|month|year)\b", lowered).group(1)
        return _clamp(Window(int(now_ns - _UNIT_DAYS[unit] * 86400e9), now_ns,
                             f"last {unit}"))
    if relative and re.search(r"\b(over|in|during|for)\b", lowered):
        return _relative(relative, now_ns)
    return None
