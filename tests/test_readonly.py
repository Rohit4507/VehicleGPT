"""Safety property: the data plane cannot command the vehicle.

This is the executable half of the freedom-from-interference argument in
docs/04 §1 (P5). It is deliberately structural rather than behavioural: we do
not test that adapters *don't* transmit, we test that they *can't*, because
``VehicleDataSource`` has no method through which a transmission could be
expressed.

A reviewer should be able to read this file and conclude that VehicleGPT is
architecturally incapable of actuating anything, independent of how carefully
its code is written.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("yaml", reason="PyYAML not installed")

from vehiclegpt.ingest.base import VehicleDataSource
from vehiclegpt.schema.registry import registry

# Verbs that would imply control of the vehicle.
FORBIDDEN = ("send", "write", "transmit", "actuate", "command", "set_",
             "apply", "control", "enable", "engage", "request")


def _public_methods(cls_or_protocol) -> set[str]:
    return {
        name for name, _ in inspect.getmembers(cls_or_protocol, inspect.isfunction)
        if not name.startswith("_")
    }


def test_protocol_is_read_only():
    """The adapter interface exposes no way to send anything."""
    methods = _public_methods(VehicleDataSource)
    assert methods == {"open", "stream", "close"}, (
        f"VehicleDataSource grew methods: {sorted(methods)}. Adding a write path "
        "breaks the read-only guarantee that the safety argument rests on."
    )
    for name in methods:
        assert not any(verb in name for verb in FORBIDDEN), (
            f"{name} looks like a control operation")


def test_cansource_has_no_transmit():
    from vehiclegpt.ingest.can_source import CanSource

    methods = _public_methods(CanSource)
    offenders = [m for m in methods if any(v in m.lower() for v in FORBIDDEN)]
    assert offenders == [], f"CanSource exposes transmit-like methods: {offenders}"


def test_cansource_does_not_accept_own_messages_by_default():
    """Echoing our own frames back would fake a data source."""
    from vehiclegpt.ingest.can_source import CanSource

    sig = inspect.signature(CanSource.__init__)
    assert sig.parameters["receive_own_messages"].default is False


def test_registry_contains_no_actuation_paths():
    """Every canonical signal must be an observation, never a setting."""
    suspicious = []
    for path in registry().paths():
        leaf = path.split(".")[-1]
        if any(leaf.startswith(v) or f"_{v}" in leaf for v in FORBIDDEN):
            suspicious.append(path)
    assert suspicious == [], (
        f"registry contains command-like paths: {suspicious}")


def test_registry_paths_are_namespaced():
    """``subsystem.quantity`` keeps permissions auditable per subsystem."""
    bad = [p for p in registry().paths() if p.count(".") != 1]
    assert bad == [], f"paths must be subsystem.quantity, got {bad}"


def test_normaliser_rejects_malformed_paths():
    """A prototype-pollution style path must not reach the store."""
    from vehiclegpt.ingest.base import Normaliser
    from vehiclegpt.schema.registry import Registry, SignalSpec, Source, Tier

    evil = SignalSpec(
        path="../../etc/passwd", unit="-", minimum=0, maximum=1, rate_hz=1,
        tier=Tier.HOT, deadband=0, description="x",
        sources=(Source("tesla_can.dbc", "ESP_B", "ESP_vehicleSpeed"),))
    reg = Registry(version=1, vehicle={}, signals={evil.path: evil})
    with pytest.raises(ValueError, match="invalid canonical path"):
        Normaliser(reg)
