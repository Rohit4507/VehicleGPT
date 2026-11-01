"""Canonical signal registry.

The registry is the contract between the data plane and everything downstream.
Adapters translate whatever they receive (CAN frames, SOME/IP notifications,
VHAL property changes, replay files) into *these* paths, and nothing downstream
ever sees a DBC signal name.

Design notes
------------
* Declared physical bounds are load-bearing: the normaliser uses them to reject
  impossible values as sensor faults rather than storing them. See
  ``vehiclegpt/ingest/base.py::Quality``.
* A canonical path may have zero sources (e.g. ``vehicle.accel_long`` is derived
  at normalise time) or many (future multi-bus vehicles).
* Loading is strict. A typo in the YAML should fail at import, not at 3 a.m. in
  a vehicle.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

SCHEMA_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = SCHEMA_DIR / "signals.yaml"
DBC_DIR = SCHEMA_DIR.parent / "dbc"


class Tier(StrEnum):
    """Retention class, drives the storage model in docs/06 §7."""

    HOT = "hot"
    SLOW = "slow"
    BURST = "burst"


@dataclass(frozen=True)
class Source:
    """Where one canonical signal comes from on a specific bus."""

    dbc: str
    message: str
    signal: str


@dataclass(frozen=True)
class SignalSpec:
    path: str
    unit: str
    minimum: float
    maximum: float
    rate_hz: float
    tier: Tier
    deadband: float
    description: str
    sources: tuple[Source, ...] = field(default_factory=tuple)

    def in_bounds(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum


class RegistryError(ValueError):
    """Raised when the registry is malformed or contradicts the DBCs."""


@dataclass
class Registry:
    version: int
    vehicle: dict[str, Any]
    signals: dict[str, SignalSpec]

    def __contains__(self, path: object) -> bool:
        return path in self.signals

    def __getitem__(self, path: str) -> SignalSpec:
        try:
            return self.signals[path]
        except KeyError as exc:
            raise RegistryError(
                f"unknown canonical signal {path!r}; known paths: "
                f"{', '.join(sorted(self.signals))}"
            ) from exc

    def __len__(self) -> int:
        return len(self.signals)

    def paths(self) -> list[str]:
        return sorted(self.signals)

    def by_tier(self, tier: Tier) -> list[SignalSpec]:
        return [s for s in self.signals.values() if s.tier is tier]

    def dbc_files(self) -> set[str]:
        return {src.dbc for spec in self.signals.values() for src in spec.sources}


def _parse_signal(raw: dict[str, Any], seen: set[str]) -> SignalSpec:
    path = raw["path"]
    if path in seen:
        raise RegistryError(f"duplicate signal path {path!r}")
    seen.add(path)

    try:
        tier = Tier(raw["tier"])
    except KeyError as exc:
        raise RegistryError(f"{path}: missing field {exc.args[0]!r}") from exc
    except ValueError as exc:
        raise RegistryError(
            f"{path}: invalid tier {raw['tier']!r}; expected one of "
            f"{[t.value for t in Tier]}"
        ) from exc

    minimum, maximum = float(raw["min"]), float(raw["max"])
    if minimum >= maximum:
        raise RegistryError(f"{path}: min ({minimum}) must be < max ({maximum})")

    sources = tuple(
        Source(dbc=s["dbc"], message=s["message"], signal=s["signal"])
        for s in raw.get("sources", [])
    )
    for src in sources:
        if not (DBC_DIR / src.dbc).exists():
            raise RegistryError(
                f"{path}: references missing DBC {src.dbc!r} in {DBC_DIR}"
            )

    return SignalSpec(
        path=path,
        unit=str(raw.get("unit", "-")),
        minimum=minimum,
        maximum=maximum,
        rate_hz=float(raw.get("rate_hz", 10)),
        tier=tier,
        deadband=float(raw.get("deadband", 0.0)),
        description=str(raw.get("description", "")),
        sources=sources,
    )


def load_registry(path: Path | None = None) -> Registry:
    """Load and validate the registry."""
    path = path or REGISTRY_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "signals" not in data:
        raise RegistryError(f"{path}: expected a mapping with a 'signals' key")

    seen: set[str] = set()
    signals = [_parse_signal(s, seen) for s in data["signals"]]
    return Registry(
        version=int(data.get("version", 1)),
        vehicle=dict(data.get("vehicle", {})),
        signals={s.path: s for s in signals},
    )


@functools.lru_cache(maxsize=1)
def registry() -> Registry:
    """Process-wide cached registry."""
    return load_registry()


def check_against_dbcs(reg: Registry | None = None) -> list[str]:
    """Verify every declared source actually exists in its DBC.

    Returns a list of human-readable problems; empty means clean. Called by the
    test suite and by ``CanSource`` at construction, because a registry/DBC
    mismatch otherwise surfaces as silently missing signals.
    """
    import cantools

    reg = reg or registry()
    problems: list[str] = []
    cache: dict[str, Any] = {}

    for spec in reg.signals.values():
        for src in spec.sources:
            if src.dbc not in cache:
                try:
                    cache[src.dbc] = cantools.database.load_file(DBC_DIR / src.dbc)
                except Exception as exc:  # noqa: BLE001 - report, don't raise
                    problems.append(f"{src.dbc}: failed to load ({exc})")
                    cache[src.dbc] = None
            db = cache[src.dbc]
            if db is None:
                continue
            try:
                msg = db.get_message_by_name(src.message)
            except KeyError:
                problems.append(
                    f"{spec.path}: message {src.message!r} not in {src.dbc}"
                )
                continue
            if src.signal not in {s.name for s in msg.signals}:
                problems.append(
                    f"{spec.path}: signal {src.signal!r} not in "
                    f"{src.dbc}::{src.message}"
                )
    return problems
