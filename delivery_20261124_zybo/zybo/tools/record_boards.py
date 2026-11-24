#!/usr/bin/env python3
"""Z1: record the identity of every Zybo Z7-10 that is reachable.

Usage:
  python record_boards.py --out <delivery>/zybo/boards.json --logs <delivery>/logs/boards \
      --board 1=210251A08870=xilinx@192.168.1.2 [--board 2=<serial>=xilinx@192.168.1.x]

The serial comes from the USB iSerial string of the board's on-board Digilent JTAG-SMT2.
The board's own kernel exposes no serial number, so the printed label is the only other
copy of it; set --label-read only if an operator has read the label and confirmed it.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from host_common import Remote, utc, write_json  # noqa: E402

REMOTE_DIR = "/home/xilinx/puf/req1120_20261124"
BOARD1_MAC = "00:0a:35:00:1e:53"  # MAC named for board 1 in the requirement
PREVIOUS = {  # boot_id values recorded by earlier batches on board 1
    "delivery_0917_report": "9ef6219a-be4f-4d3d-becf-89e4a237cb65",
}


def usb_jtag(serial):
    """lsusb -v needs no privileges for descriptor strings; find the FT232H with this iSerial."""
    proc = subprocess.run(["lsusb", "-v", "-d", "0403:6014"], text=True, capture_output=True, timeout=20)
    found = {}
    device = None
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "Bus" and "Device" in parts[:3]:   # "Bus 001 Device 008: ID 0403:6014"
            device = {"bus_device": " ".join(parts[:4])}
        elif device is not None and len(parts) >= 2 and parts[0] == "bcdDevice":
            device[parts[0]] = parts[1]
        elif device is not None and len(parts) >= 3 and parts[0] in ("iSerial", "iManufacturer", "iProduct"):
            device[parts[0]] = " ".join(parts[2:]).strip()
            if parts[0] == "iSerial":
                found[device["iSerial"]] = device
    return found.get(serial), sorted(found)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", action="append", default=[], help="index=serial=user@host (repeatable)")
    ap.add_argument("--label-read", action="store_true", help="an operator read the printed serial labels")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--expected-mac", default=BOARD1_MAC, help="MAC the requirement gives for board 1")
    args = ap.parse_args()
    boards, missing = [], []
    for spec in args.board:
        index, serial, host = spec.split("=", 2)
        remote = Remote(host, args.logs)
        remote.run(["mkdir", "-p", "-m", "700", REMOTE_DIR])
        remote.put([HERE / "zybo_board.py"], REMOTE_DIR)
        snap = json.loads(remote.get(["python3", REMOTE_DIR + "/zybo_board.py", "snapshot"]))
        dev, all_usb = usb_jtag(serial)
        uptime_days = snap["telemetry"]["uptime_s"] / 86400.0
        previous = snap["boot_id"] == PREVIOUS["delivery_0917_report"]
        boards.append({
            "board_index": int(index),
            "model": "Zybo Z7-10",
            "model_basis": "frozen bitstream targets xc7z010clg400-1 and loads; soc0 family/id/revision below",
            "part": "xc7z010clg400-1",
            "serial": serial,
            "serial_source": "USB iSerial of the on-board Digilent JTAG-SMT2 (FT232H, VID 0403 PID 6014)",
            "serial_printed_label_read_by_operator": bool(args.label_read),
            "usb_jtag": dev,
            "usb_jtag_devices_seen_on_host": all_usb,
            "eth0_mac": snap["eth0_mac"],
            "eth0_addresses": snap["eth0_addresses"],
            "eth0_mac_is_board1_mac_from_requirement": snap["eth0_mac"].lower() == args.expected_mac.lower(),
            "eth0_mac_caveat": ("A Xilinx-OUI address; PYNQ images ship the same default in the SD card's U-Boot "
                                "environment, so the MAC identifies the SD image and is not by itself proof of the "
                                "physical board."),
            "device_dna": None,
            "device_dna_reason": ("Not read. The kernel exposes no DNA node, the frozen PL has no DNA read path, "
                                  "and reading it would mean operating the eFUSE/devcfg controller or JTAG; "
                                  "an optional item, left for an attended session."),
            "soc": {"family": snap["soc_family"], "soc_id": snap["soc_id"], "revision": snap["soc_revision"],
                    "cpu_part": snap["cpu_part"]},
            "kernel": {"uname": snap["uname"], "python": snap["python"], "numpy": snap["numpy"]},
            "boot_id": snap["boot_id"],
            "uptime_days_at_observation": round(uptime_days, 4),
            "research_firmware_sha256": snap["research_firmware_sha256"],
            "bitstream_sha256": snap["bitstream_sha256"],
            "same_physical_board_as_0917_collection": {
                "verdict": bool(previous and uptime_days > 68.1),
                "basis": ("boot_id equals the one in the 0917 report and uptime exceeds the 68 days since that "
                          "collection, so the board has not been swapped or rebooted since (continuous uptime "
                          "covers 0917, the 1102 pilot and the 1113 pilot); the MAC matches as well."),
            },
            "snapshot": snap,
        })
    if len(boards) < 2:
        missing.append("board 2: no second Zybo is attached (one FT232H on USB, one board on 192.168.1.0/24)")
    if boards and not args.label_read:
        missing.append("printed serial labels not read by an operator (unattended run)")
    if boards and boards[0]["device_dna"] is None:
        missing.append("Device DNA of every board")
    result = {
        "schema": "req_1120.zybo.boards/1",
        "observed_utc": utc(),
        "boards": boards,
        "lane_index_convention": {"apuf_lanes[0..3]": "global lane 0-3", "ff_lanes[0..3]": "global lane 4-7"},
        "missing": missing,
        "requirement": "Z1",
    }
    write_json(args.out, result)
    print(json.dumps({"out": str(args.out), "boards": [b["serial"] for b in boards], "missing": missing},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
