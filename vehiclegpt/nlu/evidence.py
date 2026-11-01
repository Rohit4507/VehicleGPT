"""Facts with provenance, and text built only from them.

The gate for this project is *numeric grounding rate = 1.000*: every number in
an answer must be traceable to a computed fact. Sampling does not deliver that,
and no amount of prompt engineering does either, because the failure is silent -
a narrator that writes "about 250 Wh/km" when the fact says 249.4 is wrong in a
way a human reviewer will not catch.

So the pack is the single source of numbers, and it works in three steps:

* ``Pack.add`` registers a computed value with a name, a unit, a source, and
  **the format it will be printed in**;
* the renderer substitutes named slots, and the *only* values available are the
  pack's;
* ``guard.verify_numbers`` re-reads the finished text and requires every number
  token to be one the pack could have produced under some declared format.

Keeping the format in the fact rather than at the call site is deliberate: the
verifier has to know the exact rounding that was permitted, or "249.4" and
"249" both pass and the check degrades into a tolerance game.

Provenance is an ID, not a docstring. ``E3`` in the answer refers to row 3 of
the pack, which names the table and the row it came from, so a service engineer
can go and look.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass, field

#: Formats a fact may declare. An allowlist, not a free string: the verifier
#: has to be able to re-render a value the same way the renderer did.
FORMATS: dict[str, str] = {
    "int": "{:.0f}",
    "one": "{:.1f}",
    "two": "{:.2f}",
    "text": "{}",
    "clock": "%H:%M",
    "date": "%Y-%m-%d",
    "day": "%a %d %b",
}


@dataclass(frozen=True)
class Fact:
    """One computed value, with everything needed to print and to trace it."""

    id: str
    name: str
    value: object
    unit: str = ""
    fmt: str = "one"
    source: str = ""
    time_ns: int | None = None

    def __post_init__(self) -> None:
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown format {self.fmt!r}; use one of {sorted(FORMATS)}")
        if isinstance(self.value, float) and math.isnan(self.value):
            raise ValueError(f"fact {self.name!r} is NaN; a missing number must not "
                             "reach the renderer, or it prints as 'nan'")

    def render(self) -> str:
        if self.fmt in ("clock", "date", "day"):
            import datetime as dt

            stamp = dt.datetime.fromtimestamp((self.time_ns or 0) / 1e9, dt.UTC)
            return stamp.strftime(FORMATS[self.fmt])
        spec = FORMATS[self.fmt]
        if self.fmt == "text":
            return str(self.value)
        try:
            return spec.format(float(self.value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"fact {self.name!r} cannot print as {self.fmt}: "
                             f"{self.value!r}") from exc

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "value": self.value,
                "unit": self.unit, "fmt": self.fmt, "rendered": self.render(),
                "source": self.source, "time_ns": self.time_ns}


class Pack:
    """An ordered set of facts. Order is provenance: ``E1`` is the first added."""

    def __init__(self) -> None:
        self._facts: list[Fact] = []
        self._by_name: dict[str, Fact] = {}

    def add(self, name: str, value: object, unit: str = "", fmt: str = "one",
            source: str = "", time_ns: int | None = None) -> Fact:
        if name in self._by_name:
            raise KeyError(f"fact {name!r} already registered; names must be unique "
                           "so a slot cannot silently pick the wrong number")
        fact = Fact(id=f"E{len(self._facts) + 1}", name=name, value=value,
                    unit=unit, fmt=fmt, source=source, time_ns=time_ns)
        self._facts.append(fact)
        self._by_name[name] = fact
        return fact

    def get(self, name: str) -> Fact:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"no fact named {name!r}; the text asked for a number "
                           "nobody computed") from None

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self):
        return iter(self._facts)

    @property
    def names(self) -> list[str]:
        return [fact.name for fact in self._facts]

    def allowed_strings(self) -> set[str]:
        """Every string the renderer is permitted to produce for these facts.

        Variants exist because the same value is legitimately printed differently:
        ``250.0`` as ``250`` or ``250.0``, a signed delta as ``+59`` or ``-59``,
        a value with thousands separators. The set is closed, so it still cannot
        admit a number nobody computed.
        """
        allowed: set[str] = set()
        for fact in self._facts:
            rendered = fact.render()
            allowed.add(rendered)
            if fact.fmt in ("int", "one", "two"):
                number = float(fact.value)
                allowed.add(f"{number:.0f}")
                allowed.add(f"{number:.1f}")
                allowed.add(f"{number:.2f}")
                allowed.add(f"{number:.0f}".replace("-", ""))
                allowed.add(f"{abs(number):.0f}")
                allowed.add(f"{abs(number):.1f}")
                allowed.add(f"{number:+,.0f}")
                allowed.add(f"{abs(number):,.0f}")
                allowed.add(f"{abs(number):,.1f}")
        return allowed

    def to_dict(self) -> dict:
        return {"facts": [fact.to_dict() for fact in self._facts]}


@dataclass
class Rendered:
    """Finished text plus the provenance of every slot it used."""

    text: str = ""
    used: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text": self.text, "used": self.used, "missing": self.missing}


def render(template: str, pack: Pack, *, miss: str = "raise") -> Rendered:
    """Substitute ``{fact_name}`` slots from ``pack``.

    Unknown slots raise by default. A renderer that silently leaves ``{x}`` in
    the output has produced a sentence with a brace in it, which is the sort of
    thing that reaches a driver, and ``miss="blank"`` is available only for
    clearly-labelled draft paths.
    """
    out = Rendered()
    result: list[str] = []
    for literal, name, spec, conversion in _iter_slots(template):
        result.append(literal)
        if name is None:
            continue
        if name not in pack.names:
            out.missing.append(name)
            if miss == "raise":
                raise KeyError(f"template asks for {name!r}, which no fact provides; "
                               f"available: {', '.join(pack.names) or '(none)'}")
            result.append("")
            continue
        fact = pack.get(name)
        text = fact.render()
        if spec:  # an explicit spec wins, and the verifier re-renders it the same way
            text = format(_numeric(fact), spec)
        if conversion:
            text = f"{text!s}" if conversion == "s" else f"{_numeric(fact)!r}"
        out.used.append(name)
        result.append(text)
    out.text = "".join(result)
    return out


def _numeric(fact: Fact) -> float:
    try:
        return float(fact.value)
    except (TypeError, ValueError):
        raise ValueError(f"fact {fact.name!r} is not numeric but was formatted "
                         "as a number") from None


def _iter_slots(template: str):
    """Yield (literal, name, spec, conversion) for a ``str.format``-like template."""
    yield from string.Formatter().parse(template)
