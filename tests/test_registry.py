"""Registry validation.

The registry is the contract every downstream layer trusts. A typo must fail at
load time with a message that names the offending entry, never at 3 a.m. in a
vehicle - so these tests pin the error behaviour, not just the happy path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("yaml", reason="PyYAML not installed")

import yaml  # noqa: E402

from vehiclegpt.schema.registry import (
    RegistryError,
    Tier,
    load_registry,
    registry,
)

MINIMAL = {
    "version": 1,
    "vehicle": {"model": "test"},
    "signals": [{
        "path": "vehicle.speed", "unit": "kph", "min": 0, "max": 250,
        "rate_hz": 10, "tier": "hot", "deadband": 0.1,
        "description": "d",
        "sources": [{"dbc": "tesla_can.dbc", "message": "ESP_B",
                     "signal": "ESP_vehicleSpeed"}],
    }],
}


def _write(tmp_path, data):
    p = tmp_path / "signals.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_loads_minimal(tmp_path):
    reg = load_registry(_write(tmp_path, MINIMAL))
    assert len(reg) == 1
    assert reg["vehicle.speed"].tier is Tier.HOT
    assert "vehicle.speed" in reg


def test_duplicate_path_is_rejected(tmp_path):
    data = {**MINIMAL, "signals": MINIMAL["signals"] * 2}
    with pytest.raises(RegistryError, match="duplicate signal path"):
        load_registry(_write(tmp_path, data))


def test_bad_tier_is_rejected(tmp_path):
    bad = {**MINIMAL, "signals": [{**MINIMAL["signals"][0], "tier": "lukewarm"}]}
    with pytest.raises(RegistryError, match="invalid tier"):
        load_registry(_write(tmp_path, bad))


def test_inverted_bounds_are_rejected(tmp_path):
    bad = {**MINIMAL, "signals": [{**MINIMAL["signals"][0], "min": 10, "max": 5}]}
    with pytest.raises(RegistryError, match="min .* must be < max"):
        load_registry(_write(tmp_path, bad))


def test_missing_dbc_is_rejected(tmp_path):
    bad = {**MINIMAL, "signals": [{
        **MINIMAL["signals"][0],
        "sources": [{"dbc": "nope.dbc", "message": "M", "signal": "S"}],
    }]}
    with pytest.raises(RegistryError, match="references missing DBC"):
        load_registry(_write(tmp_path, bad))


def test_missing_key_is_named(tmp_path):
    broken = {**MINIMAL, "signals": [{"path": "vehicle.speed"}]}
    with pytest.raises(RegistryError, match="missing field 'tier'"):
        load_registry(_write(tmp_path, broken))


def test_unknown_path_error_is_helpful():
    with pytest.raises(RegistryError, match="known paths:"):
        registry()["vehicle.flux_capacitance"]


def test_real_registry_shape():
    reg = registry()
    assert len(reg) >= 20, "expected a substantial signal set"
    assert reg.dbc_files() == {"tesla_can.dbc", "vehiclegpt_vendor_ext.dbc"}
    # tiers actually used
    assert reg.by_tier(Tier.HOT), "no hot signals - retention model is unused"
    assert reg.by_tier(Tier.SLOW), "no slow signals - retention model is unused"


def test_every_source_message_is_unique_across_dbcs():
    """Two DBCs defining the same message name would make normalisation ambiguous."""
    import cantools

    from vehiclegpt.schema.registry import DBC_DIR

    owners: dict[str, str] = {}
    for name in registry().dbc_files():
        for msg in cantools.database.load_file(DBC_DIR / name).messages:
            assert msg.name not in owners, (
                f"message {msg.name} appears in {owners[msg.name]} and {name}")
            owners[msg.name] = name
