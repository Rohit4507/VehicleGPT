"""CAN / CAN-FD ingest adapter.

Decodes real CAN frames against real DBT files using ``cantools``, on top of
``python-can``. The *same code path* runs against:

* ``interface="socketcan", channel="vcan0"``  - virtual bus on Linux (demo)
* ``interface="socketcan", channel="can0"``   - a real USB-CAN adapter
* ``interface="virtual",   channel="...``     - in-process bus (CI, no kernel
  support needed)

Only the transport differs. Frame layout, bit packing, byte order, scaling and
DBC lookup are identical in all three, which is what makes the demo credible:
the bus is real, the DBC is real, the frames are real.

See ``vehiclegpt/dbc/README.md`` for what is genuine and what is synthetic.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import can
import cantools
from cantools.database.can.database import Database

from vehiclegpt.ingest.base import RawRecord
from vehiclegpt.schema.registry import DBC_DIR, Registry
from vehiclegpt.schema.registry import registry as default_registry


class CanSourceError(RuntimeError):
    pass


class CanSource:
    """Reads CAN frames and yields decoded ``RawRecord`` objects.

    Never transmits. The bus is opened in receive-only mode, which on SocketCAN
    also blocks accidental TX at the kernel level.
    """

    def __init__(
        self,
        interface: str = "virtual",
        channel: str = "vehiclegpt",
        dbc_files: Iterable[str] | None = None,
        bus: can.BusABC | None = None,
        receive_own_messages: bool = False,
        reg: Registry | None = None,
    ) -> None:
        self.name = f"can:{interface}:{channel}"
        self._reg = reg or default_registry()
        self._interface = interface
        self._channel = channel
        self._receive_own = receive_own_messages
        self._bus = bus
        self._own_bus = bus is None

        files = list(dbc_files) if dbc_files else sorted(self._reg.dbc_files())
        self._databases: list[Database] = [
            cantools.database.load_file(DBC_DIR / f) for f in files
        ]

        # arbitration_id -> (message, db). Collisions across DBCs are a real
        # hazard: two databases can legitimately define the same frame ID, and
        # silently picking one would produce confidently wrong data.
        self._by_id: dict[int, tuple[object, str]] = {}
        for db, fname in zip(self._databases, files, strict=True):
            for msg in db.messages:
                if msg.frame_id in self._by_id:
                    other = self._by_id[msg.frame_id][1]
                    raise CanSourceError(
                        f"frame id 0x{msg.frame_id:03X} defined in both "
                        f"{other} and {fname} - refusing to guess"
                    )
                self._by_id[msg.frame_id] = (msg, fname)

        self._dbcs = dict(zip(files, self._databases, strict=True))
        self._unknown_ids: set[int] = set()
        self._decoded = 0
        self._undecodable = 0

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        if self._bus is not None:
            return
        try:
            self._bus = can.Bus(
                interface=self._interface,
                channel=self._channel,
                receive_own_messages=self._receive_own,
            )
        except Exception as exc:  # noqa: BLE001 - re-raise with context
            raise CanSourceError(
                f"could not open CAN bus {self._interface}:{self._channel}: {exc}. "
                "On Linux with no hardware, create a virtual bus first:\n"
                "  sudo modprobe vcan && sudo ip link add dev vcan0 type vcan "
                "&& sudo ip link set up vcan0\n"
                "For CI with no kernel support, use interface='virtual'."
            ) from exc

    def close(self) -> None:
        if self._bus is not None and self._own_bus:
            self._bus.shutdown()
        if self._own_bus:
            self._bus = None

    def __enter__(self) -> CanSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- streaming ----------------------------------------------------------

    def stream(self, timeout: float | None = None, limit: int | None = None
               ) -> Iterator[RawRecord]:
        """Yield decoded records until ``timeout`` elapses or ``limit`` is hit.

        A ``None`` timeout blocks forever, which is what a vehicle does.
        """
        if self._bus is None:
            raise CanSourceError("open() must be called before stream()")

        deadline = None if timeout is None else time.monotonic() + timeout
        emitted = 0

        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                return
            frame = self._bus.recv(timeout=remaining if remaining is not None else 0.1)
            if frame is None:
                if deadline is None:
                    continue
                return

            record = self._decode(frame)
            if record is not None:
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def _decode(self, frame: can.Message) -> RawRecord | None:
        entry = self._by_id.get(frame.arbitration_id)
        if entry is None:
            self._unknown_ids.add(frame.arbitration_id)
            return None
        message, dbc_name = entry

        try:
            decoded = message.decode(frame.data, decode_choices=False)
        except Exception:  # noqa: BLE001 - a bad frame must not kill the stream
            self._undecodable += 1
            return None

        self._decoded += 1
        return RawRecord(
            ts_ns=int(frame.timestamp * 1e9) if frame.timestamp else time.time_ns(),
            source=f"can:{dbc_name}",
            message=message.name,
            payload={k: float(v) for k, v in decoded.items() if v is not None},
        )

    # -- diagnostics --------------------------------------------------------

    @property
    def stats(self) -> dict[str, object]:
        return {
            "decoded": self._decoded,
            "undecodable": self._undecodable,
            "unknown_ids": sorted(f"0x{i:03X}" for i in self._unknown_ids),
            "messages_known": len(self._by_id),
        }

    def message_names(self) -> set[str]:
        return {m.name for m, _ in self._by_id.values()}


def dbc_path(name: str) -> Path:
    return DBC_DIR / name
