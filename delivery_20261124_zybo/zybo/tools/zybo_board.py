#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Board-side helper for req_1120 part Z (Zybo Z7-10, PYNQ 2.5.6).

Runs on the board with Python 3.6 and numpy 1.13. Every lane response is read
through the FROZEN research service on 127.0.0.1:3040 (puf_server.py, SHA-256
a057df2c...); this file never touches the PL registers and never programs PL.

Subcommands
  snapshot                       identity / provenance / telemetry as JSON on stdout
  read ...                       raw four-lane reads of a challenge file, no processing
  sweep ...                      same-boot temperature sweep: warms the die with a lowest-priority
                                 CPU load, reads K during and after it (NOT a power cycle)
  load ...                       the CPU load used by `sweep` (self-terminating)

Lane bytes are stored exactly as the research service returns them (first four
bytes of its five-byte record); the service's XOR byte is dropped, not computed.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BIT = "/home/xilinx/puf/system_wrapper_apuf_ff_20260917.bit"
RESEARCH = "/home/xilinx/puf/puf_server_apuf_ff_20260917.py"
BIT_SHA = "224cc1fdd3234b070ef7f6568c21fecf1683cbe45508593597e3e2b1bb7b26dd"
RESEARCH_SHA = "a057df2c6136e72ed9ae10be55dbcbc2ab12f8a16f15d656e7bc1e17c488c1c5"
K_SHA = "6743daa679eaf86c3cba25c028728774bcf4a80f214b17d9b9e2e31cec55acd5"
FAMILIES = ("apuf_lanes", "ff_lanes")
HEADER = struct.Struct("<4sHI")
CMD_HELLO, CMD_CONFIGURE, CMD_GET_PARAMS = 0x01, 0x10, 0x31
CMD_SEND_CHALLENGES, CMD_GET_RESPONSES = 0x20, 0x21


def read(path):
    return Path(path).read_text().strip()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def telemetry():
    base = "/sys/bus/iio/devices/iio:device0/"
    v = {n: float(read(base + n)) for n in ("in_temp0_raw", "in_temp0_offset", "in_temp0_scale",
                                            "in_voltage0_vccint_raw", "in_voltage0_vccint_scale")}
    return {"board_monotonic_s": time.monotonic(),
            "uptime_s": float(read("/proc/uptime").split()[0]),
            "board_wall_clock_utc_unaligned": datetime.datetime.utcnow().isoformat() + "Z",
            "xadc_die_temperature_C": (v["in_temp0_raw"] + v["in_temp0_offset"]) * v["in_temp0_scale"] / 1000.0,
            "xadc_vccint_mV": v["in_voltage0_vccint_raw"] * v["in_voltage0_vccint_scale"],
            "xadc_raw": v}


def temperature():
    return telemetry()["xadc_die_temperature_C"]


def run(argv):
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def research_processes():
    out = []
    for path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            argv = Path(path).read_bytes().decode().rstrip("\0").split("\0")
            if RESEARCH in argv:
                out.append({"pid": int(path.split("/")[2]), "argv": argv})
        except (OSError, UnicodeError, IndexError):
            pass
    return sorted(out, key=lambda p: p["pid"])


def snapshot():
    soc = "/sys/devices/soc0/"
    optional = {}
    for name, path in (("devicetree_serial_number", "/sys/firmware/devicetree/base/serial-number"),
                       ("devicetree_model", "/proc/device-tree/model"),
                       ("cpuinfo_serial", "/proc/cpuinfo")):
        try:
            text = read(path)
            optional[name] = text if name != "cpuinfo_serial" else [
                l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("Serial")]
        except OSError:
            optional[name] = None
    rc_ss, ss = run(["ss", "-ltn"])
    rc_est, est = run(["ss", "-tn", "state", "established", "( sport = :3040 or dport = :3040 )"])
    return {
        "boot_id": read("/proc/sys/kernel/random/boot_id"),
        "eth0_mac": read("/sys/class/net/eth0/address"),
        "eth0_addresses": run(["ip", "-4", "-br", "addr", "show", "eth0"])[1].strip().splitlines(),
        "board_serial_number": optional["devicetree_serial_number"],
        "device_tree_model": optional["devicetree_model"],
        "cpuinfo_serial": optional["cpuinfo_serial"],
        "soc_family": read(soc + "family"), "soc_id": read(soc + "soc_id"), "soc_revision": read(soc + "revision"),
        "cpu_part": sorted(set(l.split(":", 1)[1].strip() for l in read("/proc/cpuinfo").splitlines()
                               if l.startswith("CPU part"))),
        "uname": list(os.uname()),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "fpga_manager_state": read("/sys/class/fpga_manager/fpga0/state"),
        "bitstream_path": BIT, "bitstream_sha256": sha256_file(BIT),
        "research_firmware_path": RESEARCH, "research_firmware_sha256": sha256_file(RESEARCH),
        "research_processes": research_processes(),
        "listeners": [l for l in ss.splitlines() if l.strip()],
        "established_tcp_3040": [l.strip() for l in est.splitlines() if l.strip() and not l.startswith("Recv-Q")],
        "telemetry": telemetry(),
        "configuration_provenance": ("running argv of the research service, on-disk SHA-256 of the frozen "
                                     "bitstream/firmware, FPGA Manager state; no configuration readback"),
    }


class Client(object):
    def __init__(self, host="127.0.0.1", port=3040, timeout=30):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.sock.settimeout(timeout)

    def close(self):
        self.sock.close()

    def send(self, cmd, data=b""):
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        self.sock.sendall(HEADER.pack(b"PUF1", cmd, len(data)) + data)

    def exact(self, n):
        data = bytearray()
        while len(data) < n:
            part = self.sock.recv(min(1 << 20, n - len(data)))
            if not part:
                raise IOError("connection closed mid-frame")
            data.extend(part)
        return bytes(data)

    def receive(self):
        magic, cmd, n = HEADER.unpack(self.exact(HEADER.size))
        if magic != b"PUF1" or n > (1 << 22):
            raise ValueError("bad response frame")
        return cmd, self.exact(n)

    def request(self, cmd, data=b""):
        self.send(cmd, data)
        actual, payload = self.receive()
        if actual == 0xFF:
            raise IOError("research service error: " + payload.decode("utf-8", "replace"))
        if actual != cmd:
            raise ValueError("unexpected response command {}".format(actual))
        return payload


def load_challenges(path):
    c = np.load(path, allow_pickle=False)
    if c.ndim != 2 or c.shape[1] != 64 or c.dtype != np.int8:
        raise ValueError("challenge file must be int8 with shape (N, 64)")
    c = np.ascontiguousarray(c)
    if not bool(((c == 0) | (c == 1)).all()):
        raise ValueError("challenge file must be binary")
    raw_sha = hashlib.sha256(c.tobytes()).hexdigest()
    # row bit i -> word bit i; rows are little-endian uint64 on the wire (protocol.md section 5)
    pw = np.uint64(1) << np.arange(64, dtype=np.uint64)
    words = (c.astype(np.uint64) * pw).sum(axis=1).astype("<u8")
    for i in range(min(3, len(c))):
        ref = sum(int(bit) << j for j, bit in enumerate(c[i]))
        if int(words[i]) != ref:
            raise ValueError("wire packing self-check failed")
    return c, raw_sha, words.tobytes()


def check_service(snapshot_dict):
    if snapshot_dict["bitstream_sha256"] != BIT_SHA or snapshot_dict["research_firmware_sha256"] != RESEARCH_SHA:
        raise ValueError("frozen board file mismatch")
    if snapshot_dict["fpga_manager_state"] != "operating":
        raise ValueError("PL is not in state operating")
    procs = snapshot_dict["research_processes"]
    if len(procs) != 1:
        raise ValueError("expected exactly one research service process")
    if BIT not in procs[0]["argv"] or "--no-program" in procs[0]["argv"]:
        raise ValueError("research service was not started from the frozen bitstream")


def configure(client, family):
    cfg = json.loads(client.request(CMD_CONFIGURE, {"family": family, "instance_id": 0, "seed": 20260917}))
    if cfg.get("ok") is not True:
        raise ValueError("configure failed")


def family_params(client, family):
    """What the frozen service reports for a family (GET_PARAMS needs the family configured)."""
    configure(client, family)
    return json.loads(client.request(CMD_GET_PARAMS))


def collect_family(client, family, wire, n, chunk):
    configure(client, family)
    out = []
    for off in range(0, n, chunk):
        m = min(chunk, n - off)
        client.send(CMD_SEND_CHALLENGES, wire[off * 8:(off + m) * 8])
        payload = client.request(CMD_GET_RESPONSES, struct.pack("<I", m))
        if len(payload) != m * 5:
            raise ValueError("unexpected raw response length")
        lanes = np.frombuffer(payload, dtype=np.int8).reshape(m, 5)[:, :4]
        if not bool(((lanes == 0) | (lanes == 1)).all()):
            raise ValueError("nonbinary raw lanes")
        out.append(lanes.copy())
    return np.concatenate(out) if len(out) > 1 else out[0]


def do_read(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    c, raw_sha, wire = load_challenges(args.challenges)
    n = len(c)
    families = tuple(f for f in args.families.split(",") if f)
    for f in families:
        if f not in FAMILIES:
            raise ValueError("unsupported family " + f)
    before = snapshot()
    check_service(before)
    if before["established_tcp_3040"]:
        raise ValueError("another client is connected to the research service")
    counts = dict.fromkeys(families, 0)
    arrays = {f: [] for f in families}
    meta = {"kind": "research_mode_raw_lanes", "label": args.label, "challenge_file": args.challenges,
            "challenge_shape": list(c.shape), "challenge_dtype": str(c.dtype), "challenges_raw_sha256": raw_sha,
            "is_frozen_K": bool(n == 4096 and raw_sha == K_SHA), "repeats_requested": args.repeats,
            "read_order": "for each repeat: " + " then ".join(families) + " (whole challenge file, fixed order)",
            "raw_response_processing": "save wire bytes 0..3 in lane order; drop the service's XOR byte",
            "true_power_cycles": 0, "counted_as_Z2": False,
            "tool": {"path": __file__, "sha256": sha256_file(__file__)},
            "boot_id_start": before["boot_id"], "eth0_mac": before["eth0_mac"],
            "bitstream_sha256": before["bitstream_sha256"], "research_firmware_sha256": before["research_firmware_sha256"],
            "k_raw_sha256": K_SHA if (n == 4096 and raw_sha == K_SHA) else None,
            "complete": False, "reads": []}
    write_json(out / "board_before.json", before)
    write_json(out / "meta.json", meta)
    client = Client(port=args.port)
    try:
        meta["hello"] = json.loads(client.request(CMD_HELLO))
        meta["params"] = {f: family_params(client, f) for f in families}
        for rep in range(args.repeats):
            for f in families:
                t0 = telemetry()
                start = time.monotonic()
                lanes = collect_family(client, f, wire, n, args.chunk)
                elapsed = time.monotonic() - start
                t1 = telemetry()
                arrays[f].append(lanes)
                meta["reads"].append({"index": len(meta["reads"]), "repeat_index": rep, "family": f,
                                      "array_row": counts[f], "challenges": n,
                                      "read_start_board_monotonic_s": start, "read_duration_s": elapsed,
                                      "before": t0, "after": t1})
                counts[f] += 1
                write_json(out / "meta.json", meta)
                print("{} repeat {} {}: {} challenges {:.3f} s {:.3f} C".format(
                    args.label, rep + 1, f, n, elapsed, t0["xadc_die_temperature_C"]), flush=True)
                if args.interval and not (rep == args.repeats - 1 and f == families[-1]):
                    time.sleep(args.interval)
        after = snapshot()
        write_json(out / "board_after.json", after)
        if before["boot_id"] != after["boot_id"] or before["research_processes"] != after["research_processes"]:
            raise ValueError("research service or boot changed during the read")
        shapes = {}
        for f in families:
            arr = np.stack(arrays[f]).astype(np.int8)
            shapes[f] = list(arr.shape)
            np.save(str(out / (f + ".npy")), arr, allow_pickle=False)
        meta.update(complete=True, boot_id_end=after["boot_id"], array_shape=shapes)
        write_json(out / "meta.json", meta)
    except Exception as exc:
        meta["error"] = "%s: %s" % (type(exc).__name__, exc)
        write_json(out / "meta.json", meta)
        raise
    finally:
        client.close()


def load_state(pid):
    try:
        return open("/proc/%d/stat" % pid).read().rsplit(") ", 1)[1].split()[0]
    except (OSError, IndexError):
        return "?"


class Load(object):
    """Lowest-priority CPU load in child processes; the reader always outruns it (nice 0 vs 19).

    mode sha: sha256 chain (integer work).  mode mem: a numpy multiply over two 32 MiB arrays
    (memory traffic + NEON); it warms the die about twice as fast as sha on this board.
    Every child exits by itself when the stop file appears, when its lifetime ends, or when
    the XADC reading exceeds max_temp.
    """

    def __init__(self, mode, cores, max_temp, max_seconds, stop_file):
        self.stop_file = Path(stop_file)
        if self.stop_file.exists():
            self.stop_file.unlink()
        self.mode = mode
        cmd = [sys.executable, __file__, "load", "--mode", mode, "--stop-file", str(self.stop_file),
               "--max-temp", str(max_temp), "--max-seconds", str(max_seconds)]
        self.procs = [subprocess.Popen(cmd, preexec_fn=lambda: os.nice(19),
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                      for _ in range(cores)]

    def state(self):
        return ["%s:%s" % ("exited" if p.poll() is not None else load_state(p.pid), p.returncode)
                for p in self.procs]

    def alive(self):
        return [p for p in self.procs if p.poll() is None]

    def pause(self):
        for p in self.alive():
            p.send_signal(signal.SIGSTOP)

    def resume(self):
        for p in self.alive():
            p.send_signal(signal.SIGCONT)

    def stop(self):
        self.stop_file.touch()
        for p in self.procs:
            try:
                if p.poll() is None:
                    p.send_signal(signal.SIGCONT)
                    p.wait(timeout=8)
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=5)
            except Exception:
                pass


def do_load(args):
    """Body of one load child: stops on the stop file, on its lifetime limit or above max_temp."""
    stop, t_end, h = Path(args.stop_file), time.monotonic() + args.max_seconds, b"x"
    if args.mode == "mem":
        a, b = np.ones(1 << 22), np.zeros(1 << 22)
    n = 0
    while True:
        if args.mode == "mem":
            np.multiply(a, 1.000001, out=b)
            a, b = b, a
        else:
            slice_end = time.monotonic() + 0.5
            while time.monotonic() < slice_end:
                for _ in range(200):
                    h = hashlib.sha256(h).digest()
        n += 1
        if args.mode == "mem" and n % 4:
            continue                     # about one check per second
        if stop.exists() or time.monotonic() > t_end:
            return 0
        if temperature() > args.max_temp:
            return 3


def do_sweep(args):
    """Same-boot temperature sweep. NOT a power cycle and not Z2.

    heat:  full load, no reads, until the XADC reading passes --hot-target, stops rising,
           or --heat-max-seconds elapse (events every 10 s are written to meta.json).
    hot:   --under-load-pairs read while the load is still running (the reader outruns it).
    cool:  load killed; --hot-pairs read back to back, then one pair per --cool-interval
           until the die is back near idle or --cool-max-seconds elapse.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    c, raw_sha, wire = load_challenges(args.challenges)
    if len(c) != 4096 or raw_sha != K_SHA:
        raise ValueError("sweep reads the frozen K only")
    before = snapshot()
    check_service(before)
    if before["established_tcp_3040"]:
        raise ValueError("another client is connected to the research service")
    t_idle = before["telemetry"]["xadc_die_temperature_C"]
    meta = {"kind": "same_boot_temperature_sweep", "label": args.label, "challenges_raw_sha256": K_SHA,
            "true_power_cycles": 0, "counted_as_Z2": False,
            "tool": {"path": __file__, "sha256": sha256_file(__file__)},
            "method": ("K is read through the frozen research service before, during and after the die is warmed "
                       "by a nice-19 CPU load on both cores. 'heat': load running, no reads. 'under_load': load "
                       "running, the reader (nice 0) outruns it. 'cool': load killed, reads continue while the die "
                       "cools by itself. No external heat source, no power cut; the XADC reading rises by roughly "
                       "0.5 C while the load runs, so the 'cool' readings are the settled die temperature."),
            "heating": {"load_mode": args.mode, "load_cores": args.cores, "load_nice": 19,
                        "load_max_temperature_C": args.max_temp, "load_max_seconds": args.load_max_seconds,
                        "hot_target_C": args.hot_target, "heat_max_seconds": args.heat_max_seconds},
            "idle_temperature_C_at_start": t_idle, "boot_id_start": before["boot_id"], "eth0_mac": before["eth0_mac"],
            "bitstream_sha256": before["bitstream_sha256"],
            "research_firmware_sha256": before["research_firmware_sha256"],
            "raw_response_processing": "save wire bytes 0..3 in lane order; drop the service's XOR byte",
            "complete": False, "reads": [], "events": []}
    write_json(out / "board_before.json", before)
    write_json(out / "meta.json", meta)
    arrays = {"apuf_lanes": [], "ff_lanes": []}
    load = Load(args.mode, args.cores, args.max_temp, args.load_max_seconds, out / "load.stop")
    client = Client(port=args.port)
    start_mono = time.monotonic()

    def one_read(phase):
        recs = []
        for f in ("apuf_lanes", "ff_lanes"):
            t0 = telemetry()
            s = time.monotonic()
            lanes = collect_family(client, f, wire, 4096, args.chunk)
            e = time.monotonic() - s
            t1 = telemetry()
            arrays[f].append(lanes)
            rec = {"index": len(meta["reads"]), "family": f, "array_row": len(arrays[f]) - 1, "phase": phase,
                   "challenges": 4096, "read_start_board_monotonic_s": s,
                   "read_start_since_sweep_start_s": s - start_mono, "read_duration_s": e,
                   "load_state": load.state(), "before": t0, "after": t1}
            meta["reads"].append(rec)
            recs.append(rec)
        write_json(out / "meta.json", meta)
        return recs

    def event(name, write=False, **kw):
        ev = {"event": name, "board_monotonic_s": time.monotonic(),
              "since_sweep_start_s": time.monotonic() - start_mono, "temperature_C": temperature(),
              "load_state": load.state()}
        ev.update(kw)
        meta["events"].append(ev)
        if write:
            write_json(out / "meta.json", meta)
        print(json.dumps(ev, sort_keys=True), flush=True)

    try:
        meta["hello"] = json.loads(client.request(CMD_HELLO))
        event("sweep_start", write=True)
        # ---- heat ----
        heat_deadline = time.monotonic() + args.heat_max_seconds
        window_start, window_temp, n = time.monotonic(), temperature(), 0
        while True:
            now, temp = time.monotonic(), temperature()
            if temp >= args.hot_target:
                event("hot_target_reached", write=True, temperature_C=temp)
                break
            if now >= heat_deadline:
                event("heat_time_limit", write=True, temperature_C=temp)
                break
            if now - window_start >= args.plateau_window:
                if temp - window_temp < args.plateau_delta:
                    event("heat_plateau", write=True, temperature_C=temp, window_start_temperature_C=window_temp)
                    break
                window_start, window_temp = now, temp
            if not load.alive():
                event("load_exited", write=True, temperature_C=temp)
                break
            if n % 2 == 0:      # every ~10 s
                event("heating", write=(n % 6 == 0), temperature_C=temp)
            n += 1
            time.sleep(5)
        # ---- under load ----
        for _ in range(args.under_load_pairs):
            one_read("under_load")
        event("under_load_done", write=True)
        # ---- cool ----
        load.stop()
        time.sleep(args.settle)
        event("load_stopped", write=True)
        for _ in range(args.hot_pairs):
            one_read("cool_hot")
        event("hot_pairs_done", write=True)
        cool_deadline = time.monotonic() + args.cool_max_seconds
        floor = min(t_idle, args.cool_floor)
        while time.monotonic() < cool_deadline:
            one_read("cool")
            temp = temperature()
            if temp <= floor:
                event("cool_floor_reached", write=True, temperature_C=temp)
                break
            time.sleep(args.cool_interval)
        else:
            event("cool_time_limit", write=True)
        (out / "load.stop").unlink()
        after = snapshot()
        write_json(out / "board_after.json", after)
        if before["boot_id"] != after["boot_id"] or before["research_processes"] != after["research_processes"]:
            raise ValueError("research service or boot changed during the sweep")
        shapes = {}
        for f in arrays:
            arr = np.stack(arrays[f]).astype(np.int8)
            shapes[f] = list(arr.shape)
            np.save(str(out / (f + ".npy")), arr, allow_pickle=False)
        meta.update(complete=True, boot_id_end=after["boot_id"], array_shape=shapes)
        write_json(out / "meta.json", meta)
    except Exception as exc:
        meta["error"] = "%s: %s" % (type(exc).__name__, exc)
        write_json(out / "meta.json", meta)
        raise
    finally:
        load.stop()
        client.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("snapshot")
    r = sub.add_parser("read")
    r.add_argument("--challenges", required=True)
    r.add_argument("--families", default="apuf_lanes,ff_lanes")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--chunk", type=int, default=4096)
    r.add_argument("--interval", type=float, default=0.0, help="seconds between reads")
    r.add_argument("--port", type=int, default=3040)
    r.add_argument("--label", default="")
    r.add_argument("--out", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--challenges", required=True)
    s.add_argument("--chunk", type=int, default=4096)
    s.add_argument("--port", type=int, default=3040)
    s.add_argument("--label", default="")
    s.add_argument("--mode", choices=("mem", "sha"), default="mem")
    s.add_argument("--cores", type=int, default=2)
    s.add_argument("--max-temp", type=float, default=71.0, help="the load kills itself above this XADC reading")
    s.add_argument("--load-max-seconds", type=int, default=3600)
    s.add_argument("--hot-target", type=float, default=67.0, help="stop heating at this (load running) reading")
    s.add_argument("--heat-max-seconds", type=float, default=720.0)
    s.add_argument("--plateau-window", type=float, default=120.0)
    s.add_argument("--plateau-delta", type=float, default=0.3, help="rise below this within the window = plateau")
    s.add_argument("--under-load-pairs", type=int, default=10)
    s.add_argument("--hot-pairs", type=int, default=60, help="back-to-back pairs right after the load is killed")
    s.add_argument("--settle", type=float, default=3.0)
    s.add_argument("--cool-max-seconds", type=float, default=720.0)
    s.add_argument("--cool-interval", type=float, default=3.0)
    s.add_argument("--cool-floor", type=float, default=57.0, help="stop cooling reads at or below this reading")
    s.add_argument("--out", required=True)
    l = sub.add_parser("load")
    l.add_argument("--mode", choices=("mem", "sha"), default="mem")
    l.add_argument("--stop-file", required=True)
    l.add_argument("--max-temp", type=float, default=71.0)
    l.add_argument("--max-seconds", type=int, default=3600)
    args = ap.parse_args()
    if args.cmd == "snapshot":
        print(json.dumps(snapshot(), indent=2, sort_keys=True))
    elif args.cmd == "read":
        do_read(args)
    elif args.cmd == "sweep":
        do_sweep(args)
    elif args.cmd == "load":
        sys.exit(do_load(args))
    else:
        ap.error("choose a subcommand")


if __name__ == "__main__":
    main()
