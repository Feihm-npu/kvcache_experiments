#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z5: calibrate c, the KDF iteration count that makes unlock's kdf_s at least 1.0 s on this board.

  python3 module_calibrate.py --device-id 210251A08870 --out calibration_board.json \
      [--target-s 1.0] [--margin 1.08] [--trials 10]

Runs the service's own unlock_key() (module_core) with throw-away random roots; nothing here
touches the PUF or an enrollment. c is raised until every one of --trials runs takes at least
--target-s seconds. Both boards must then be served with the same c: take the larger of the two.
"""
import argparse
import json
import platform
import secrets
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import module_core as core  # noqa: E402


def run_kdf(iterations):
    root = bytearray(secrets.token_bytes(32))
    try:
        t0 = time.monotonic()
        z = core.unlock_key(root, b"calibration passphrase", iterations)
        dt = time.monotonic() - t0
        core.zero([z])
        return dt
    finally:
        core.zero([root])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-s", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=1.08)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to overwrite " + str(args.out))
    # rate: seconds per iteration from a 2000-iteration probe, best (minimum) of five
    probes = [run_kdf(2000) / 2000.0 for _ in range(5)]
    per_iteration = min(probes)
    c = max(1, int(args.target_s * args.margin / per_iteration) + 1)
    history = []
    for attempt in range(6):
        seconds = [run_kdf(c) for _ in range(args.trials)]
        history.append({"c": c, "kdf_s": seconds, "min_s": min(seconds), "median_s": statistics.median(seconds)})
        if min(seconds) >= args.target_s:
            break
        c = int(c * 1.1) + 1
    else:
        raise SystemExit("could not reach the target on this board")
    result = {
        "schema": "req_1120.zybo.calibration/1", "requirement": "Z5", "device_id": args.device_id,
        "c": c, "target_kdf_s": args.target_s, "margin": args.margin, "trials": args.trials,
        "kdf_s": seconds, "min_kdf_s": min(seconds), "median_kdf_s": statistics.median(seconds),
        "max_kdf_s": max(seconds), "seconds_per_iteration_us": per_iteration * 1e6,
        "probe_seconds_per_iteration_us": [p * 1e6 for p in probes],
        "history": [{k: v for k, v in h.items() if k != "kdf_s"} for h in history],
        "method": ("module_core.unlock_key with 32-byte random throw-away roots and a fixed "
                   "calibration passphrase, c chosen so that the minimum of the trials is at or above "
                   "the target; the same code path the deployment service runs for every unlock"),
        "environment": {"python": sys.version.split()[0], "machine": platform.machine(),
                        "kernel": platform.release(), "hostname": platform.node()},
        "module_core_sha256": core.sha256_file(str(HERE / "module_core.py")),
        "note": "c must be the same on both boards; use the larger of the two calibrations",
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.write_text(text)
    print(json.dumps({k: result[k] for k in ("device_id", "c", "min_kdf_s", "median_kdf_s", "max_kdf_s",
                                            "seconds_per_iteration_us")}, sort_keys=True))


if __name__ == "__main__":
    main()
