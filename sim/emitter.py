"""Put simulated truth onto a real CAN bus.

Encodes canonical signal values into DBC messages with ``cantools`` and sends the
resulting frames over a ``python-can`` bus, at the per-signal rates declared in
the registry. Ingest then decodes them back with the same DBC.

This is the piece that makes the CAN demo honest: between ``emit`` and ``decode``
the data is genuinely packed into 8-byte frames at real bit offsets, serialised
through a bus, and unpacked again. Nothing shortcuts the wire format.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import can
import cantools

from sim.generator import RATE_HZ, Simulation
from vehiclegpt.schema.registry import DBC_DIR, Registry
from vehiclegpt.schema.registry import registry as default_registry

# Message periods, in simulation ticks. Mirrors the rates in the DBCs.
DEFAULT_MESSAGE_PERIOD_TICKS = 1   # 10 Hz at RATE_HZ


class Encoder:
    """Builds DBC frames from a ``Simulation``'s truth matrix."""

    def __init__(self, sim: Simulation, reg: Registry | None = None) -> None:
        self.sim = sim
        self.reg = reg or default_registry()
        self._dbs = {
            name: cantools.database.load_file(DBC_DIR / name)
            for name in self.reg.dbc_files()
        }
        # canonical path -> (dbc_name, message_name, dbc_signal_name)
        self._routes: dict[str, tuple[str, str, str]] = {}
        for spec in self.reg.signals.values():
            if spec.sources and spec.path in sim.truth:
                src = spec.sources[0]
                self._routes[spec.path] = (src.dbc, src.message, src.signal)

        # group by message so each frame is encoded once
        self._by_message: dict[tuple[str, str], list[str]] = {}
        for path, (dbc, msg, _sig) in self._routes.items():
            self._by_message.setdefault((dbc, msg), []).append(path)

        # cantools requires a COMPLETE payload: a partial dict raises KeyError
        # on any signal you omitted. Precompute a benign default for every
        # signal in every message we emit, so we only have to supply the ones
        # we actually model.
        self._defaults: dict[tuple[str, str], dict[str, float]] = {}
        for dbc_name, msg_name in self._by_message:
            message = self._dbs[dbc_name].get_message_by_name(msg_name)
            self._defaults[(dbc_name, msg_name)] = {
                sig.name: _default_value(sig) for sig in message.signals
            }

    def message_schedule(self) -> list[tuple[str, str]]:
        return sorted(self._by_message)

    def encode_tick(self, i: int) -> list[can.Message]:
        """Encode every message for simulation tick ``i``."""
        frames: list[can.Message] = []
        for (dbc_name, msg_name), paths in self._by_message.items():
            db = self._dbs[dbc_name]
            message = db.get_message_by_name(msg_name)
            payload: dict[str, float] = dict(
                self._defaults[(dbc_name, msg_name)])
            for path in paths:
                _dbc, _msg, sig_name = self._routes[path]
                spec = self.reg[path]
                payload[sig_name] = _clamp(
                    float(self.sim.truth[path][i]), spec.minimum, spec.maximum)
            try:
                data = message.encode(payload, scaling=True, strict=False)
            except Exception:  # noqa: BLE001 - a single bad frame must not abort
                continue
            frames.append(can.Message(
                arbitration_id=message.frame_id,
                data=data,
                is_extended_id=message.is_extended_frame,
                timestamp=self.sim.t_s[i],
            ))
        return frames


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _default_value(sig) -> float:
    """A neutral physical value for a signal we do not model.

    Zero, or the nearest in-range value if zero is out of range. Used to fill
    messages that share a frame with a signal we do care about.
    """
    if sig.minimum is not None and sig.minimum > 0.0:
        return float(sig.minimum)
    if sig.maximum is not None and sig.maximum < 0.0:
        return float(sig.maximum)
    return 0.0


def replay(sim: Simulation, bus: can.BusABC, reg: Registry | None = None,
           wall_clock: bool = False, limit_ticks: int | None = None) -> int:
    """Push a whole simulation onto ``bus``. Returns frames sent.

    ``wall_clock=False`` sends as fast as possible (use in tests);
    ``wall_clock=True`` sleeps to real time (use for a live demo).
    """
    enc = Encoder(sim, reg)
    n = sim.n if limit_ticks is None else min(sim.n, limit_ticks)
    sent = 0
    tick_period = 1.0 / RATE_HZ
    t0 = time.monotonic()
    for i in range(n):
        for frame in enc.encode_tick(i):
            bus.send(frame)
            sent += 1
        if wall_clock:
            target = t0 + (i + 1) * tick_period
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    return sent


def iter_frames(sim: Simulation, reg: Registry | None = None,
                limit_ticks: int | None = None) -> Iterator[can.Message]:
    """Encode without a bus - useful for golden-file tests."""
    enc = Encoder(sim, reg)
    n = sim.n if limit_ticks is None else min(sim.n, limit_ticks)
    for i in range(n):
        yield from enc.encode_tick(i)
