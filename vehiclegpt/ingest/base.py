"""Ingest contracts.

Three types matter here:

``RawRecord``    what an adapter emits, still in its native vocabulary
``Sample``       a canonical value the rest of the system understands
``VehicleDataSource``  the one interface every protocol adapter implements

The protocol is deliberately **read-only and one-way**: ``stream()`` yields
records and nothing else. There is no ``send``, no ``write``, no ``apply``. That
is how design principle P5 ("read-only by construction", docs/04 §1) is enforced
at the type level rather than by discipline — an adapter physically cannot
command the vehicle, because the interface has no method that would let it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Quality(StrEnum):
    """Why a value might not be trustworthy."""

    OK = "ok"
    OUT_OF_BOUNDS = "out_of_bounds"   # outside the registry's physical limits
    UNKNOWN_SIGNAL = "unknown_signal"  # decoded, but no canonical mapping


@dataclass(frozen=True, slots=True)
class RawRecord:
    """One decoded-but-not-yet-canonical record.

    ``payload`` maps the adapter's native signal name to a physical value. For
    CAN this is a DBC message decode; for SOME/IP it would be a service event;
    for replay it is a row from the source file.
    """

    ts_ns: int
    source: str                 # adapter name, e.g. "can:tesla_can.dbc"
    message: str                # native message name, e.g. "ESP_B"
    payload: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Sample:
    """A canonical signal value, ready to store."""

    ts_ns: int
    path: str                   # canonical registry path, e.g. "vehicle.speed"
    value: float
    unit: str
    quality: Quality = Quality.OK

    @property
    def is_ok(self) -> bool:
        return self.quality is Quality.OK


@runtime_checkable
class VehicleDataSource(Protocol):
    """Every protocol adapter implements this and nothing more."""

    name: str

    def open(self) -> None: ...

    def stream(self) -> Iterator[RawRecord]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_SAFE_PATH = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


class Normaliser:
    """Maps ``RawRecord`` payloads onto canonical ``Sample`` objects.

    Uses the registry's declared sources to build a
    ``(message, native_signal) -> canonical_path`` table once, then applies
    physical-bounds checking on every value.

    Out-of-range values are **kept but flagged**, not silently dropped. A sensor
    reporting 400 kph is evidence of a fault, and throwing that evidence away is
    exactly the wrong thing to do in a diagnostic system.
    """

    def __init__(self, reg=None) -> None:
        from vehiclegpt.schema.registry import registry as default_registry

        self.registry = reg or default_registry()
        self._map: dict[tuple[str, str], str] = {}
        for spec in self.registry.signals.values():
            if not _SAFE_PATH.match(spec.path):
                raise ValueError(f"invalid canonical path {spec.path!r}")
            for src in spec.sources:
                key = (src.message, src.signal)
                if key in self._map and self._map[key] != spec.path:
                    raise ValueError(
                        f"{key} maps to both {self._map[key]!r} and {spec.path!r}"
                    )
                self._map[key] = spec.path

    @property
    def mapping_size(self) -> int:
        return len(self._map)

    def messages(self) -> set[str]:
        return {msg for msg, _ in self._map}

    def normalise(self, record: RawRecord) -> list[Sample]:
        out: list[Sample] = []
        for native_signal, value in record.payload.items():
            path = self._map.get((record.message, native_signal))
            if path is None:
                # Decoded but unmapped: normal for DBC fields we do not yet
                # care about. Not an error, and deliberately not logged per
                # frame - that would flood a vehicle's log.
                continue
            spec = self.registry[path]
            quality = Quality.OK if spec.in_bounds(value) else Quality.OUT_OF_BOUNDS
            out.append(
                Sample(
                    ts_ns=record.ts_ns,
                    path=path,
                    value=float(value),
                    unit=spec.unit,
                    quality=quality,
                )
            )
        return out
