#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enrollment-mode firmware image of the Zybo root module (req_1120 Z4). Run as root, ONCE per board.

  python3 module_enroll.py --lanes 0,1,2,3 --t 32 --device-id 210251A08870 \
      --k challenges_K.npy --research puf_server.py --bit system_wrapper_apuf_ff_20260917.bit \
      --out NEW_enrollment.json [--no-program]

`lanes` and `t` come from the analysis side after Z2. Lane numbers: 0-3 = apuf_lanes 0-3,
4-7 = ff_lanes 0-3. Steps are exactly those of the requirement:

  1. nine reads of K; per read the selected lanes are XORed bit by bit (4096-bit response);
     w is the per-bit majority of the nine;
  2. data = packbits(w, big) (512 bytes); ecc = bchlib.BCH(t, m=13).encode(data);
  3. seed = 544 bytes from the operating system RNG;
  4. r = packbits((rows @ x) % 2, little), rows = 256 windows of 4096 bits of s;
  5. tag = HMAC-SHA256(r, "BoundKV-tag|" + helper) over the seven public fields;
  6. only enrollment.json is written.

The only thing printed is a short JSON summary without w, r or any raw response.
The output file must not exist yet; the tool never overwrites an enrollment.

Test-only: --replay-* (recorded real reads instead of the PL) and --dry-run (nothing is written).
After a board is deployed it must not run this image again.
"""
import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import module_core as core  # noqa: E402


def log(event, **fields):
    rec = {"board_monotonic_s": time.monotonic(), "event": event}
    rec.update(fields)
    sys.stderr.write(json.dumps(rec, sort_keys=True) + "\n")
    sys.stderr.flush()


def enroll(source, lanes, t, device_id):
    """Returns (fields with tag_hex, per-read seconds). Every intermediate buffer is zeroed here."""
    bch = core.Bch(t)
    bch.selftest()
    families = core.families_of(lanes)
    reads, seconds = [], []
    for index in range(9):                                   # step 1
        t0 = time.monotonic()
        reads.append(core.measure_word(source, lanes))
        seconds.append(time.monotonic() - t0)
        log("read", index=index + 1, of=9, seconds=seconds[-1], evaluations=source.evaluations)
    stack = None
    root = ecc = None
    try:
        stack = np.stack(reads)
        core.zero(reads)
        del reads[:]
        w = (stack.sum(axis=0, dtype=np.int32) >= 5).astype(np.uint8)
        core.zero([stack])
        stack = None
        data = bytearray(core.pack_bits(w))                  # step 2
        core.zero([w])
        ecc = bch.encode(data)
        seed = bytearray(secrets.token_bytes(544))           # step 3
        root = core.toeplitz_root(data, seed)                # step 4
        core.zero([data])
        fields = core.enrollment_fields(device_id, lanes, t, ecc, seed)     # step 5
        tag = bytearray(core.hmac_sha256(root, core.TAG_DOMAIN, core.helper_bytes(fields)))
        fields["tag_hex"] = bytes(tag).hex()
        core.zero([tag, seed])
        return fields, seconds
    finally:
        core.zero([x for x in (stack, root, ecc) if x is not None] + reads)
        core.scrub()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", required=True, help="comma separated, e.g. 0,1,2,3 or 0,4 (sorted, distinct)")
    ap.add_argument("--t", type=int, required=True)
    ap.add_argument("--device-id", required=True)
    ap.add_argument("--k", required=True)
    ap.add_argument("--out", required=True, help="enrollment.json to create (must not exist)")
    ap.add_argument("--research", default=None)
    ap.add_argument("--bit", default=None)
    ap.add_argument("--base", default="0x40000000")
    ap.add_argument("--no-program", action="store_true",
                    help="reuse the PL that is already configured (default: program it from the frozen bitstream)")
    ap.add_argument("--dry-run", action="store_true", help="do everything except writing the file")
    ap.add_argument("--replay-apuf", default=None)
    ap.add_argument("--replay-ff", default=None)
    ap.add_argument("--replay-start", type=int, default=0)
    ap.add_argument("--replay-count", type=int, default=9)
    args = ap.parse_args()

    try:
        lanes = sorted(set(int(x) for x in args.lanes.split(",")))
        core._check_lanes(lanes)
    except (ValueError, core.ModuleError):
        raise core.ModuleError("--lanes must be sorted distinct integers 0-7")
    out = Path(args.out)
    if not args.dry_run and (out.exists() or out.is_symlink()):
        raise core.ModuleError("refusing to overwrite " + str(out))
    k = core.load_k(args.k)
    if args.replay_apuf:
        source = core.ReplaySource(args.replay_apuf, args.replay_ff, args.replay_start, args.replay_count)
    else:
        if os.geteuid() != 0:
            raise core.ModuleError("live enrollment needs root (/dev/mem and FPGA Manager)")
        source = core.MmioSource(k, args.research, args.bit, not args.no_program, int(args.base, 0))
    log("start", backend=source.backend, lanes=lanes, t=args.t, device_id=args.device_id,
        pl_programmed_by_module=getattr(source, "pl_programmed", None),
        k_raw_sha256=core.K_SHA, research_firmware_sha256=core.sha256_file(args.research) if args.research else None,
        bitstream_sha256=core.sha256_file(args.bit) if args.bit else None,
        module_enroll_sha256=core.sha256_file(str(HERE / "module_enroll.py")),
        module_core_sha256=core.sha256_file(str(HERE / "module_core.py")))
    fields, seconds = enroll(source, lanes, args.t, args.device_id)
    public = dict(fields)
    if args.dry_run:
        log("dry_run", written=False, evaluations=source.evaluations)
        print(json.dumps({"written": None, "dry_run": True, "lanes": lanes, "t": args.t,
                          "seconds_per_read": seconds, "evaluations": source.evaluations,
                          "backend": source.backend}, sort_keys=True))
        return
    text = json.dumps(public, indent=2, sort_keys=True) + "\n"
    core.validate_enrollment(json.loads(text))                # the file we are about to write parses
    fd = os.open(str(out), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    log("written", path=str(out), evaluations=source.evaluations)
    print(json.dumps({"written": str(out), "sha256": core.sha256_file(str(out)), "device_id": args.device_id,
                      "lanes": lanes, "t": args.t, "m": 13, "ecc_bytes": len(bytes.fromhex(public["ecc_hex"])),
                      "seconds_per_read": seconds, "evaluations": source.evaluations,
                      "backend": source.backend}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except core.ModuleError as exc:
        log("error", detail=str(exc))
        sys.exit(2)
