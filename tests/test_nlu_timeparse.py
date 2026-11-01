"""Time expressions, resolved without a model.

The cases that matter are the ones where a plausible implementation is subtly
wrong: "now" taken from the wall clock instead of the data, "the day before
yesterday" matching inside "yesterday", and a range collapsing into one day.
"""

from __future__ import annotations

import datetime as dt

import pytest

from vehiclegpt.nlu import timeparse as tp

NOW = int(dt.datetime(2026, 3, 17, 12, 0, tzinfo=dt.UTC).timestamp() * 1e9)


def utc(window: tp.Window) -> tuple[dt.datetime, dt.datetime]:
    return (dt.datetime.fromtimestamp(window.start_ns / 1e9, dt.UTC),
            dt.datetime.fromtimestamp(window.end_ns / 1e9, dt.UTC))


class TestRelative:
    def test_word_numbers_are_parsed(self):
        window = tp.parse("range fell over the last two weeks", NOW)
        assert window.days == pytest.approx(14.0)

    def test_digits_are_parsed(self):
        assert tp.parse("past 3 days", NOW).days == pytest.approx(3.0)

    def test_bare_unit_means_one(self):
        assert tp.parse("the last week", NOW).days == pytest.approx(7.0)

    def test_hours_and_minutes(self):
        assert tp.parse("in the last 2 hours", NOW).days == pytest.approx(2 / 24)

    def test_ends_at_now_not_at_the_calendar(self):
        window = tp.parse("last two weeks", NOW)
        assert window.end_ns == NOW

    def test_now_is_the_parameter_not_the_wall_clock(self):
        """A parked car's clock is months ahead of its newest sample."""
        stale = int(dt.datetime(2026, 3, 12, 8, 0, tzinfo=dt.UTC).timestamp() * 1e9)
        window = tp.parse("the last week", stale)
        assert utc(window)[1] == dt.datetime(2026, 3, 12, 8, 0, tzinfo=dt.UTC)

    def test_the_article_decides_calendar_versus_rolling(self):
        """'last week' is the previous calendar week; 'the last week' ends now."""
        calendar = tp.parse("last week", NOW)
        rolling = tp.parse("the last week", NOW)
        assert utc(calendar)[0] == dt.datetime(2026, 3, 9, tzinfo=dt.UTC)
        assert utc(calendar)[1] == dt.datetime(2026, 3, 16, tzinfo=dt.UTC)
        assert rolling.end_ns == NOW


class TestCalendar:
    def test_yesterday_is_a_whole_day(self):
        start, end = utc(tp.parse("from yesterday", NOW))
        assert start == dt.datetime(2026, 3, 16, tzinfo=dt.UTC)
        assert end == dt.datetime(2026, 3, 17, tzinfo=dt.UTC)

    def test_day_before_yesterday_is_not_yesterday(self):
        """Regression: 'yesterday' is a substring, so order of checks matters."""
        window = tp.parse("the day before yesterday", NOW)
        assert window.label == "the day before yesterday"
        assert utc(window)[0] == dt.datetime(2026, 3, 15, tzinfo=dt.UTC)

    def test_without_the_article(self):
        assert tp.parse("day before yesterday at 5pm", NOW).label == \
            "the day before yesterday"

    def test_this_week_starts_on_monday(self):
        window = tp.parse("this week", NOW)
        start, _ = utc(window)
        assert start == dt.datetime(2026, 3, 16, tzinfo=dt.UTC)

    def test_today_ends_now(self):
        window = tp.parse("today", NOW)
        assert window.end_ns == NOW


class TestExplicit:
    def test_iso_date(self):
        start, end = utc(tp.parse("on 2026-03-11", NOW))
        assert start == dt.datetime(2026, 3, 11, tzinfo=dt.UTC)
        assert end == dt.datetime(2026, 3, 12, tzinfo=dt.UTC)

    def test_day_month(self):
        assert tp.parse("on 11 March", NOW).label == "2026-03-11"

    def test_month_day(self):
        assert tp.parse("March 11", NOW).label == "2026-03-11"

    def test_ordinal_suffix(self):
        assert tp.parse("the 11th of March", NOW).label == "2026-03-11"

    def test_explicit_year_is_respected(self):
        assert tp.parse("11 March 2025", NOW).label == "2025-03-11"

    def test_range_does_not_collapse_to_one_day(self):
        """Both ends are dates; taking the first would silently mean one day."""
        window = tp.parse("between 11 March and 12 March", NOW)
        assert window.days == pytest.approx(2.0)
        assert utc(window)[0] == dt.datetime(2026, 3, 11, tzinfo=dt.UTC)


class TestBoundsAndFailure:
    def test_no_expression_returns_none(self):
        assert tp.parse("compare city and highway", NOW) is None

    def test_empty_returns_none(self):
        assert tp.parse("", NOW) is None

    def test_absurd_window_is_rejected(self):
        with pytest.raises(ValueError, match="more than"):
            tp.parse("last 5000 days", NOW)

    def test_timezone_moves_the_boundary(self):
        plus_five = tp.parse("yesterday", NOW, tz_minutes=330)
        utc_day = tp.parse("yesterday", NOW, tz_minutes=0)
        assert plus_five.start_ns != utc_day.start_ns
        start, _ = utc(plus_five)
        assert start == dt.datetime(2026, 3, 15, 18, 30, tzinfo=dt.UTC)
