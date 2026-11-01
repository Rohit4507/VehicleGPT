"""DBC and registry integrity.

These tests are the reason the synthetic vendor extension is trustworthy: they
pin the registry to the actual bit layouts, so a plausible-looking signal that
the bus cannot physically carry fails here rather than silently clamping at
runtime (which is how we found the 1-bit brake-pedal bug).
"""

from __future__ import annotations

import pytest

pytest.importorskip("cantools", reason="cantools not installed")
pytest.importorskip("yaml", reason="PyYAML not installed")

import cantools  # noqa: E402

from vehiclegpt.schema.registry import DBC_DIR, Tier, check_against_dbcs, registry


@pytest.fixture(scope="module")
def reg():
    return registry()


@pytest.fixture(scope="module")
def dbs(reg):
    return {name: cantools.database.load_file(DBC_DIR / name) for name in reg.dbc_files()}


def test_dbc_files_exist(reg):
    assert reg.dbc_files(), "registry declares no DBC sources"
    for name in reg.dbc_files():
        assert (DBC_DIR / name).is_file(), f"missing {name}"


def test_dbcs_load(dbs):
    for name, db in dbs.items():
        assert len(db.messages) > 0, f"{name} parsed but contains no messages"


def test_no_frame_id_collisions(dbs):
    """Two DBCs on one bus must not both define the same arbitration ID."""
    seen: dict[int, str] = {}
    for name, db in dbs.items():
        for msg in db.messages:
            if msg.frame_id in seen:
                pytest.fail(
                    f"0x{msg.frame_id:03X} defined in both {seen[msg.frame_id]} "
                    f"and {name} ({msg.name})"
                )
            seen[msg.frame_id] = name


def test_registry_sources_exist_in_dbcs(reg):
    problems = check_against_dbcs(reg)
    assert problems == [], "registry/DBC disagreement:\n" + "\n".join(problems)


def test_declared_bounds_are_representable(reg, dbs):
    """A canonical range wider than the DBC field can carry is a schema lie.

    Signedness matters: ``DI_torqueMotor`` is ``16|13@1-``, a *signed* 13-bit
    field, so +-500 Nm is representable even though the raw field is unsigned
    if you only look at the bit count. Getting this wrong produced three false
    mismatches the first time this check was written.
    """
    bad = []
    for spec in reg.signals.values():
        for src in spec.sources:
            msg = dbs[src.dbc].get_message_by_name(src.message)
            sig = next(s for s in msg.signals if s.name == src.signal)
            lo, hi = _physical_span(sig)
            if spec.minimum < lo - 1e-9 or spec.maximum > hi + 1e-9:
                bad.append(
                    f"{spec.path}: registry [{spec.minimum}, {spec.maximum}] "
                    f"exceeds {src.dbc}::{src.signal} span [{lo:.2f}, {hi:.2f}]"
                )
    assert bad == [], "\n".join(bad)


def _physical_span(sig) -> tuple[float, float]:
    """Physical [min, max] the raw field can carry, honouring signedness."""
    if sig.is_signed:
        raw_lo, raw_hi = -(2 ** (sig.length - 1)), 2 ** (sig.length - 1) - 1
    else:
        raw_lo, raw_hi = 0, 2**sig.length - 1
    a = raw_lo * sig.scale + sig.offset
    b = raw_hi * sig.scale + sig.offset
    return (min(a, b), max(a, b))


def test_every_signal_has_complete_metadata(reg):
    for spec in reg.signals.values():
        assert spec.unit, f"{spec.path}: no unit"
        assert spec.description, f"{spec.path}: no description"
        assert spec.rate_hz > 0, f"{spec.path}: rate_hz must be positive"
        assert spec.deadband >= 0, f"{spec.path}: deadband must be non-negative"
        assert spec.tier in Tier


def test_known_tiers_only(reg):
    assert {s.tier for s in reg.signals.values()} <= set(Tier)


def test_derived_signal_has_no_source(reg):
    """``vehicle.accel_long`` is computed from speed; it must not claim a bus."""
    assert reg["vehicle.accel_long"].sources == ()


def test_vendor_dbc_is_reproducible():
    """The synthetic extension must be regenerable from source, not hand-edited."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    # Compare bytes: cantools emits CRLF and read_text() would normalise it.
    before = (repo / "vehiclegpt" / "dbc" / "vehiclegpt_vendor_ext.dbc").read_bytes()
    result = subprocess.run(
        [sys.executable, str(repo / "tools" / "make_vendor_dbc.py"), "--stdout"],
        capture_output=True, cwd=repo,   # bytes: text mode would rewrite CRLF
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.strip() == before.strip(), (
        "vehiclegpt_vendor_ext.dbc differs from what tools/make_vendor_dbc.py "
        "generates - regenerate it instead of editing by hand"
    )
