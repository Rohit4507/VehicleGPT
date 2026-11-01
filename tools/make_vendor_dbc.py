#!/usr/bin/env python3
"""Generate the VehicleGPT vendor-extension DBC.

The DBC is built from cantools objects rather than hand-written, so bit packing
is valid by construction. Run this instead of editing the .dbc by hand:

    python3 tools/make_vendor_dbc.py

Requires cantools (it is a runtime dependency of the ingest path anyway).
"""

from __future__ import annotations

from pathlib import Path

import cantools
import cantools.database.conversion as conv
from cantools.database.can import Database, Message, Signal

OUT = Path(__file__).resolve().parent.parent / "vehiclegpt" / "dbc" / "vehiclegpt_vendor_ext.dbc"


def sig(name, start, length, scale=1, offset=0, minimum=None, maximum=None,
        unit=None, is_signed=False) -> Signal:
    return Signal(
        name=name, start=start, length=length, byte_order="little_endian",
        is_signed=is_signed,
        conversion=conv.BaseConversion.factory(scale=scale, offset=offset),
        minimum=minimum, maximum=maximum, unit=unit, receivers=[],
    )


def build() -> Database:
    db = Database()

    # 0x6A0 - HV pack status @ 10 Hz
    db.messages.append(Message(
        frame_id=0x6A0, name="BMS_packStatus", length=8,
        is_extended_frame=False, senders=["BMS"], signals=[
            sig("BMS_packVoltage_V", 0, 16, scale=0.1, minimum=0, maximum=800, unit="V"),
            sig("BMS_packCurrent_A", 16, 16, scale=0.1, offset=-500,
                minimum=-500, maximum=500, unit="A", is_signed=True),
            sig("BMS_soc_pct", 32, 10, scale=0.1, minimum=0, maximum=100, unit="%"),
            sig("BMS_packTempMax_C", 42, 8, offset=-40, minimum=-40, maximum=125, unit="C"),
            sig("BMS_packTempMin_C", 50, 8, offset=-40, minimum=-40, maximum=125, unit="C"),
            sig("BMS_contactorState", 58, 3),
            sig("BMS_faultLevel", 61, 3),
        ]))

    # 0x6A1 - pack thermal detail @ 10 Hz
    db.messages.append(Message(
        frame_id=0x6A1, name="BMS_packThermal", length=8,
        is_extended_frame=False, senders=["BMS"], signals=[
            sig("BMS_cellTempSpread_C", 0, 8, scale=0.5, minimum=0, maximum=60, unit="C"),
            sig("BMS_coolantInlet_C", 8, 8, offset=-40, minimum=-40, maximum=125, unit="C"),
            sig("BMS_coolantOutlet_C", 16, 8, offset=-40, minimum=-40, maximum=125, unit="C"),
            sig("BMS_coolantPump_pct", 24, 8, scale=0.5, minimum=0, maximum=100, unit="%"),
            sig("BMS_thermalRunawayFlag", 32, 1),
            sig("BMS_overheatWarning", 33, 1),
            sig("BMS_cellImbalance_mV", 34, 14, minimum=0, maximum=16383, unit="mV"),
        ]))

    # 0x6A2 - HVAC + energy @ 10 Hz
    db.messages.append(Message(
        frame_id=0x6A2, name="VCU_energy", length=8,
        is_extended_frame=False, senders=["VCU"], signals=[
            sig("VCU_energyPerKm_Wh", 0, 16, scale=0.5, minimum=0, maximum=2000, unit="Wh/km"),
            sig("VCU_rangeEstimate_km", 16, 16, scale=0.1, minimum=0, maximum=1000, unit="km"),
            sig("HVAC_power_W", 32, 14, scale=5, minimum=0, maximum=8000, unit="W"),
            sig("HVAC_setpoint_C", 46, 8, scale=0.5, minimum=0, maximum=60, unit="C"),
            sig("VCU_regenPower_W", 54, 10, scale=100, minimum=0, maximum=102300, unit="W"),
        ]))

    return db


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stdout", action="store_true",
        help="print the DBC instead of writing it (used by the reproducibility test)")
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed file is out of date")
    args = parser.parse_args()

    db = build()
    text = db.as_dbc_string()

    if args.stdout:
        print(text)
        return
    if args.check:
        # cantools emits CRLF; Path.read_text() would normalise it to LF and
        # make every check fail spuriously. Compare bytes.
        if OUT.read_bytes() != text.encode("utf-8"):
            raise SystemExit(f"{OUT} is out of date; re-run without --check")
        print(f"{OUT.name}: up to date")
        return

    OUT.write_text(text, encoding="utf-8")

    # Round-trip every message through the on-disk file. If this fails, the
    # ingest path would fail later in a much more confusing way.
    check = cantools.database.load_file(OUT)
    for msg in check.messages:
        sample = {}
        for s in msg.signals:
            if s.minimum is None or s.maximum is None:
                # No declared range (e.g. a 1-bit flag): use the lowest
                # representable physical value rather than inventing one.
                sample[s.name] = s.offset
            else:
                sample[s.name] = (s.minimum + s.maximum) / 2
        msg.decode(msg.encode(sample))
    print(f"wrote {OUT.relative_to(OUT.parents[3])}")
    print(f"  messages: {[(m.name, hex(m.frame_id)) for m in check.messages]}")
    print("  round-trip: OK")


if __name__ == "__main__":
    main()
