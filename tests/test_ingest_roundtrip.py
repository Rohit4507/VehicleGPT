"""End-to-end CAN round-trip.

Generator -> DBC encode -> python-can virtual bus -> python-can receive ->
cantools decode -> normalise -> canonical Sample.

The point of this test is that nothing between the generator and the normaliser
is faked. Values are packed into real 8-byte frames at the bit offsets in the
real DBCs, pushed through a real ``can.Bus``, and unpacked again. The only
substitution is the transport (``interface='virtual'`` instead of
``'socketcan'``), which this sandbox cannot provide - see ``dbc/README.md``.

Assertions are therefore about DBC *quantisation*: the decoded value must be
within one LSB of the truth, not bit-identical.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

can = pytest.importorskip("can", reason="python-can not installed")
cantools = pytest.importorskip("cantools", reason="cantools not installed")

from sim.emitter import replay  # noqa: E402
from sim.generator import generate  # noqa: E402
from sim.scenario import load_scenario  # noqa: E402
from vehiclegpt.ingest.base import Normaliser, Quality  # noqa: E402
from vehiclegpt.ingest.can_source import CanSource  # noqa: E402
from vehiclegpt.schema.registry import DBC_DIR, registry  # noqa: E402

CHANNEL = "vehiclegpt-roundtrip"
TICKS = 400   # 40 s at 10 Hz - long enough to cross mode changes


SCENARIO = (Path(__file__).resolve().parents[1]
            / "sim" / "scenarios" / "battery_overheat.yaml")


@pytest.fixture(scope="module")
def sim():
    return generate(load_scenario(SCENARIO))


@pytest.fixture(scope="module")
def roundtrip(sim):
    """Run the full pipeline once and share the result across assertions."""
    sender = can.Bus(interface="virtual", channel=CHANNEL)
    receiver = can.Bus(interface="virtual", channel=CHANNEL)
    try:
        n_frames = replay(sim, sender, limit_ticks=TICKS)
        source = CanSource(bus=receiver)
        normaliser = Normaliser()
        records = list(source.stream(limit=n_frames))
        samples = [s for r in records for s in normaliser.normalise(r)]
        return {"frames": n_frames, "records": records, "samples": samples,
                "source": source}
    finally:
        sender.shutdown()
        receiver.shutdown()


def test_frames_are_produced(sim, roundtrip):
    enc_messages = 10  # 7 tesla + 3 vendor, see emitter
    assert roundtrip["frames"] == TICKS * enc_messages, (
        f"expected {TICKS * enc_messages} frames, got {roundtrip['frames']}"
    )


def test_every_frame_decodes(roundtrip):
    stats = roundtrip["source"].stats
    assert stats["undecodable"] == 0, f"undecodable frames: {stats}"
    assert stats["unknown_ids"] == [], f"unmapped arbitration IDs: {stats}"
    assert stats["decoded"] == roundtrip["frames"]


def test_no_samples_out_of_bounds(roundtrip):
    """Every emitted value must pass the registry's physical-bounds check."""
    bad = [s for s in roundtrip["samples"] if s.quality is not Quality.OK]
    assert bad == [], (
        "generator emitted values the registry considers impossible: "
        f"{[(s.path, s.value, s.quality) for s in bad[:10]]}"
    )


def test_canonical_coverage(roundtrip):
    """All sourced signals must survive the round-trip."""
    reg = registry()
    sourced = {p for p, spec in reg.signals.items() if spec.sources}
    seen = {s.path for s in roundtrip["samples"]}
    missing = sourced - seen
    assert missing == set(), f"signals lost in transit: {sorted(missing)}"


def _ordinal_pairs(roundtrip, path):
    """(tick, value) pairs for one signal, aligned by occurrence order.

    We cannot align on timestamps: python-can's virtual transport stamps each
    frame with *receive* time and discards the sender's, which is what a real
    bus without hardware timestamping also does. Because
    ``test_every_frame_decodes`` establishes that no frame was dropped, the
    k-th decoded value of a signal is the k-th simulated tick.
    """
    vals: list[float] = []
    for s in roundtrip["samples"]:
        if s.path == path and s.quality is Quality.OK:
            vals.append(s.value)
    return list(enumerate(vals))


def test_values_within_one_lsb(sim, roundtrip):
    """Decoded values must match the generator to within DBC quantisation."""
    dbs = {n: cantools.database.load_file(DBC_DIR / n) for n in registry().dbc_files()}
    scale: dict[str, float] = {}
    for spec in registry().signals.values():
        for src in spec.sources:
            msg = dbs[src.dbc].get_message_by_name(src.message)
            sig = next(s for s in msg.signals if s.name == src.signal)
            scale[spec.path] = max(abs(sig.scale), 1e-9)

    paths = {s.path for s in roundtrip["samples"] if s.quality is Quality.OK}
    failures = []
    for path in sorted(paths):
        pairs = _ordinal_pairs(roundtrip, path)
        idx = np.array([i for i, _ in pairs])
        vals = np.array([v for _, v in pairs])
        assert idx.size <= sim.n, f"{path}: more samples than simulated ticks"
        err = np.abs(vals - sim.truth[path][idx])
        tol = scale[path] * 1.01
        if err.max() > tol:
            worst = int(np.argmax(err))
            failures.append(
                f"{path}: max error {err.max():.4f} > 1 LSB ({scale[path]:g}) "
                f"at tick {idx[worst]} (truth {sim.truth[path][idx[worst]]:.4f}, "
                f"decoded {vals[worst]:.4f})"
            )
    assert failures == [], "\n".join(failures)


def test_low_resolution_signals_are_exact(sim, roundtrip):
    """Boolean and small-integer signals must survive bit-exactly."""
    exact = ["chassis.brake_active", "battery.overheat_warning",
             "adas.state", "powertrain.gear"]
    for path in exact:
        pairs = _ordinal_pairs(roundtrip, path)
        assert pairs, f"{path} never decoded"
        for idx, value in pairs:
            assert value == pytest.approx(sim.truth[path][idx], abs=1e-9), (
                f"{path} tick {idx}: decoded {value} != truth {sim.truth[path][idx]}"
            )


def test_timestamps_are_receive_time(roundtrip):
    """Documented behaviour: the transport stamps arrival, not send time.

    ``CanSource`` therefore reports when *we* saw a frame, which is the honest
    answer on a bus without hardware timestamping. A transport that does supply
    hardware timestamps (SocketCAN on most adapters) passes them through.
    """
    ts = [s.ts_ns for s in roundtrip["samples"]]
    assert ts, "no samples"
    assert all(t > 1_000_000_000_000_000_000 for t in ts), (
        "timestamps look like simulated seconds; the virtual transport should "
        "have stamped wall-clock receive time"
    )
    assert ts == sorted(ts), "sample timestamps must be non-decreasing"


def test_unknown_frame_is_ignored_not_fatal(sim):
    """A frame from an unmodelled ECU must not break the stream."""
    sender = can.Bus(interface="virtual", channel="vehiclegpt-unknown")
    receiver = can.Bus(interface="virtual", channel="vehiclegpt-unknown")
    try:
        # 0x7FF is not defined in either DBC.
        sender.send(can.Message(arbitration_id=0x7FF, data=b"\x01" * 8,
                                is_extended_id=False))
        sender.send(can.Message(arbitration_id=0x155,
                                data=bytes.fromhex("0000000000000b09"),
                                is_extended_id=False))
        source = CanSource(bus=receiver)
        records = list(source.stream(limit=1))
        assert len(records) == 1
        assert records[0].message == "ESP_B"
        assert source.stats["unknown_ids"] == ["0x7FF"]
    finally:
        sender.shutdown()
        receiver.shutdown()


def test_short_frame_does_not_kill_stream():
    """A truncated frame must be counted, not raised."""
    sender = can.Bus(interface="virtual", channel="vehiclegpt-short")
    receiver = can.Bus(interface="virtual", channel="vehiclegpt-short")
    try:
        sender.send(can.Message(arbitration_id=0x155, data=b"\x00",
                                is_extended_id=False))  # ESP_B needs 7 bytes
        sender.send(can.Message(arbitration_id=0x155,
                                data=bytes.fromhex("0000000000000b09"),
                                is_extended_id=False))
        source = CanSource(bus=receiver)
        records = list(source.stream(limit=1))
        assert len(records) == 1, "stream died on a short frame"
        assert source.stats["undecodable"] >= 1
    finally:
        sender.shutdown()
        receiver.shutdown()
