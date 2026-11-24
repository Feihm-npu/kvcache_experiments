#!/usr/bin/env python3
"""Z4-Z8 engineering run on the real Zybo with a TEST enrollment built from recorded reads.

  python run_module_board_tests.py --out <delivery> --serial 210251A08870 \
      [--steps prepare,calibrate,live_acquire,matrix,timing,closure,ocm_check,cleanup,final]

No formal enrollment is made here. Each test enrollment is computed on the board by the real
enrollment image (module_enroll.py) from nine recorded K reads of the same boot (the pilot's
idle reads), used by the real deployment image (module_serve.py) with --backend replay for
later recorded reads (idle and warmer), and deleted again. Nothing of them is delivered.

  prepare        upload the module and the replay files, check hashes, check the OCM page is empty
  calibrate      Z5  c for kdf_s >= 1.0 s on this board (module_calibrate.py)
  live_acquire   Z4  the enrollment image's live path (frozen research functions, real PL) in
                     --dry-run: nine reads of K per family, nothing written; real acquire time
  matrix         Z4  reconstruction success of several (lanes, t) test enrollments against the
                     recorded reads, with the board's own bchlib
  timing         Z7  20 unlocks, then 20 derives, through module_client.py (replay backend)
  closure        Z8  research service stopped, 100 research-protocol commands + 10 JSON impostors
                     sent to the deployment service, then one unlock; research service restarted
  j6_layout      two deployment services at once on the board (ports 3041 and 3043, two OCM pages): the
                     normal one unlocks, the one started with an enrollment whose ecc_hex has one changed
                     byte answers "reconstruction" (J6's "modified" setting)
  ocm_check      the OCM pages that held r and k_u are all zeros after the services stopped
  cleanup        removes test enrollments and replay files from the board
  final          board snapshot
"""
import argparse
import hashlib
import json
import secrets
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / "module"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODULE))
from host_common import Remote, utc, write_json  # noqa: E402
import research_frames as rf  # noqa: E402

REMOTE_DIR = "/home/xilinx/puf/req1120_20261124"
REMOTE_MODULE = REMOTE_DIR + "/module"
BCH_PATH = "/home/xilinx/.local/lib/python3.6/site-packages"      # bchlib 2.1.3 lives in xilinx's user site
FROZEN_RESEARCH = "/home/xilinx/puf/puf_server_apuf_ff_20260917.py"
FROZEN_BIT = "/home/xilinx/puf/system_wrapper_apuf_ff_20260917.bit"
PORT = 3041
OCM_ADDRESS = "0xFFFE8000"
ALL_STEPS = ("prepare", "calibrate", "live_acquire", "matrix", "timing", "closure", "j6_layout", "ocm_check",
             "cleanup", "final")
MATRIX = [  # (lanes, t) of the test enrollments
    ([0], 8), ([0], 16), ([0], 64), ([1], 16), ([2], 16), ([3], 16),
    ([4], 16), ([5], 16), ([6], 16), ([7], 16),
    ([0, 1, 2, 3], 16), ([0, 1, 2, 3], 32), ([0, 1, 2, 3], 64),
    ([4, 5, 6, 7], 64), ([0, 4], 32), ([0, 1, 2, 3, 4, 5, 6, 7], 64),
]
TIMING_LANES, TIMING_T = [0, 1, 2, 3], 64
UPLOAD = ["module_core.py", "module_serve.py", "module_enroll.py", "module_calibrate.py", "module_client.py",
          "research_frames.py", "challenges_K.npy", "frozen/puf_server.py", "frozen/system_wrapper_apuf_ff_20260917.bit"]
REPLAY = {"apuf_lanes": REMOTE_DIR + "/replay/replay_sel_apuf_lanes.npy",
          "ff_lanes": REMOTE_DIR + "/replay/replay_sel_ff_lanes.npy"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cfg_name(lanes, t):
    return "lanes" + "".join(str(l) for l in lanes) + "_t%d" % t


def combined(sel, row, lanes):
    """XOR of the selected lanes of recorded read `row`; sel maps family -> (R, 4096, 4) array."""
    w = np.zeros(4096, dtype=np.uint8)
    for lane in lanes:
        fam = "apuf_lanes" if lane < 4 else "ff_lanes"
        w ^= sel[fam][row][:, lane % 4].astype(np.uint8)
    return w


# ---------------------------------------------------------------- recorded reads -> replay selection
def build_selection(pilot, work):
    """Selection rows 0-8 are the idle reads every test enrollment is made from; rows 9.. are the
    other idle reads and a spread of the sweep's warmer reads, in replay order."""
    parts, index = {}, []
    idle_meta = json.loads((pilot / "idle_pre" / "meta.json").read_text())
    temps = {r["array_row"]: r for r in idle_meta["reads"] if r["family"] == "apuf_lanes"}
    for family in ("apuf_lanes", "ff_lanes"):
        parts[family] = [np.load(pilot / "idle_pre" / (family + ".npy"), allow_pickle=False)]
    for row in sorted(temps):
        r = temps[row]
        index.append({"row": row, "source": "idle_pre", "array_row": row, "phase": "idle",
                      "temperature_before_C": r["before"]["xadc_die_temperature_C"],
                      "temperature_after_C": r["after"]["xadc_die_temperature_C"]})
    sweep_meta = json.loads((pilot / "sweep" / "meta.json").read_text())
    by_phase = {}
    for r in (x for x in sweep_meta["reads"] if x["family"] == "apuf_lanes"):
        by_phase.setdefault(r["phase"], []).append(r)
    chosen = []
    for phase, every in (("under_load", 1), ("cool_hot", 2), ("cool", 4)):
        chosen.extend(by_phase.get(phase, [])[::every])
    for family in ("apuf_lanes", "ff_lanes"):
        arr = np.load(pilot / "sweep" / (family + ".npy"), allow_pickle=False)
        parts[family].append(arr[[r["array_row"] for r in chosen]])
    base = len(index)
    for i, r in enumerate(chosen):
        index.append({"row": base + i, "source": "sweep", "array_row": r["array_row"], "phase": r["phase"],
                      "temperature_before_C": r["before"]["xadc_die_temperature_C"],
                      "temperature_after_C": r["after"]["xadc_die_temperature_C"]})
    work.mkdir(parents=True, exist_ok=True)
    sel = {}
    for family in ("apuf_lanes", "ff_lanes"):
        sel[family] = np.ascontiguousarray(np.concatenate(parts[family])).astype(np.int8)
        np.save(work / ("replay_sel_" + family + ".npy"), sel[family], allow_pickle=False)
    write_json(work / "replay_sel_index.json", {"rows": index, "enrollment_rows": [0, 8]})
    return sel, index


def load_selection(work):
    sel = {f: np.load(work / ("replay_sel_" + f + ".npy"), allow_pickle=False) for f in ("apuf_lanes", "ff_lanes")}
    return sel, json.loads((work / "replay_sel_index.json").read_text())["rows"]


# ---------------------------------------------------------------- board helpers
def sudo_python(args, timeout=600):
    return ["sudo", "-n", "env", "PYTHONPATH=" + BCH_PATH, "PYTHONDONTWRITEBYTECODE=1", "python3"] + [str(a) for a in args]


def wait_port(ip, port, timeout=40.0):
    end = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((ip, port), timeout=2):
                return True
        except OSError:
            if time.monotonic() >= end:
                return False
            time.sleep(0.3)


def ask(ip, port, payload, timeout=60.0):
    """One request over a fresh connection; returns (reply line or None, seconds, transport error)."""
    t0 = time.perf_counter()
    try:
        conn = socket.create_connection((ip, port), timeout=10)
        conn.settimeout(timeout)
        try:
            conn.sendall(payload if payload.endswith(b"\n") else payload + b"\n")
            data = bytearray()
            while b"\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data.extend(chunk)
        finally:
            conn.close()
        return bytes(data).split(b"\n", 1)[0], time.perf_counter() - t0, None
    except OSError as exc:
        return None, time.perf_counter() - t0, type(exc).__name__


def ocm_page(remote, log_name=None, address=OCM_ADDRESS):
    code = ("import mmap,os,json\n"
            "fd=os.open('/dev/mem',os.O_RDONLY|os.O_SYNC)\n"
            "m=mmap.mmap(fd,4096,mmap.MAP_SHARED,mmap.PROT_READ,offset=%d)\n"
            "os.close(fd)\n"
            "b=bytes(m[0:4096])\n"
            "print(json.dumps({'address':'%s','bytes':4096,'all_zero':b==bytes(4096),"
            "'nonzero_bytes':sum(1 for x in b if x)}))\n") % (int(address, 16), address)
    return json.loads(remote.get(sudo_python(["-c", code]), log_name=log_name))


def board_snapshot(remote):
    return json.loads(remote.get(["python3", REMOTE_DIR + "/zybo_board.py", "snapshot"]))


def start_service(remote, name, extra, ip):
    """Starts module_serve.py as root, detached; returns (pid from its own startup line, log path)."""
    log = REMOTE_MODULE + "/logs/%s.log" % name
    remote.shell("rm -f %s" % shlex.quote(log))
    argv = sudo_python([REMOTE_MODULE + "/module_serve.py"] + extra)
    remote.shell("nohup setsid sh -c %s >%s 2>&1 </dev/null &" % (shlex.quote(shlex.join(argv)), shlex.quote(log)))
    text = remote.shell("cat " + shlex.quote(log), check=False).stdout
    if not wait_port(ip, PORT, timeout=25):
        raise RuntimeError("module service %s did not listen; log:\n%s" % (name, text))
    first = json.loads(remote.shell("head -n 1 " + shlex.quote(log)).stdout)
    if first.get("event") != "startup":
        raise RuntimeError("unexpected first log line: %r" % first)
    return int(first["pid"]), log


def serve(remote, name, extra, ip, state):
    """start_service with the OCM secret store, falling back to process memory if the board refuses."""
    try:
        pid, log = start_service(remote, name, extra + ["--secret-store", "ocm"], ip)
        return pid, log, "ocm"
    except RuntimeError as exc:
        text = str(exc)
        if "secret" not in text and "OCM" not in text and "ocm" not in text:
            raise
        state.setdefault("store_fallback", []).append({"step": name, "reason": text[-600:]})
        pid, log = start_service(remote, name, extra + ["--secret-store", "memory"], ip)
        return pid, log, "memory"


def stop_service(remote, pid, log, ip):
    remote.run(["sudo", "-n", "kill", "-TERM", str(pid)], check=False)
    end = time.monotonic() + 20
    while time.monotonic() < end and wait_port(ip, PORT, timeout=0.5):
        time.sleep(0.3)
    time.sleep(0.5)
    return remote.shell("cat " + shlex.quote(log)).stdout


def last_event(text):
    lines = [l for l in text.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def test_enrollment(remote, lanes, t, serial, log_name):
    """Makes and returns the path of a test enrollment from selection rows 0-8 (deleted by the caller)."""
    name = cfg_name(lanes, t)
    enr = REMOTE_MODULE + "/out/test_%s.json" % name
    remote.run(["sudo", "-n", "rm", "-f", enr], check=False)
    remote.run(sudo_python([REMOTE_MODULE + "/module_enroll.py", "--lanes", ",".join(map(str, lanes)), "--t", str(t),
                            "--device-id", serial, "--k", REMOTE_MODULE + "/challenges_K.npy",
                            "--replay-apuf", REPLAY["apuf_lanes"], "--replay-ff", REPLAY["ff_lanes"],
                            "--replay-start", "0", "--replay-count", "9", "--out", enr]),
               timeout=300, log_name=log_name)
    return enr


def replay_args(enr, rows, c):
    return ["--enrollment", enr, "--k", REMOTE_MODULE + "/challenges_K.npy", "--iterations", str(c),
            "--backend", "replay", "--replay-apuf", REPLAY["apuf_lanes"], "--replay-ff", REPLAY["ff_lanes"],
            "--replay-start", "9", "--replay-count", str(rows - 9), "--host", "0.0.0.0", "--port", str(PORT)]


def stop_research(remote, snap):
    remote.run(["sudo", "-n", "kill", "-TERM", str(snap["research_processes"][0]["pid"])], check=False)
    end = time.monotonic() + 20
    while time.monotonic() < end:
        if ":3040 " not in remote.get(["ss", "-ltn"]):
            return True
        time.sleep(0.3)
    return False


def start_research(remote):
    inner = ("nohup setsid python3 %s --bit %s --port 3040 --base 0x40000000 --no-program "
             ">>/home/xilinx/puf/puf_server_apuf_ff_20260917.log 2>&1 </dev/null & echo $!" % (FROZEN_RESEARCH, FROZEN_BIT))
    remote.shell(shlex.join(["sudo", "-n", "sh", "-c", inner]))
    end = time.monotonic() + 30
    while time.monotonic() < end:
        if ":3040 " in remote.get(["ss", "-ltn"]):
            return True
        time.sleep(0.3)
    return False


# ---------------------------------------------------------------- steps
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--serial", required=True)
    ap.add_argument("--host", default="xilinx@192.168.1.2")
    ap.add_argument("--steps", default=",".join(ALL_STEPS))
    args = ap.parse_args()
    out = args.out.resolve()
    zybo, logs = out / "zybo", out / "logs" / "module"
    remote = Remote(args.host, logs)
    ip = args.host.split("@")[-1]
    pilot = zybo / "sameboot_pilot" / args.serial
    old_status = json.loads((logs / "run_status.json").read_text()) if (logs / "run_status.json").exists() else {}
    status = {"first_started_utc": old_status.get("first_started_utc", utc()), "host": args.host, "serial": args.serial,
              "steps": old_status.get("steps", {}), "invocations": old_status.get("invocations", []) + [utc()],
              "note": "steps of earlier invocations are kept; timeline.json lists every remote call in order"}
    passphrase = secrets.token_bytes(16)                       # lives in this process only
    state = {}
    if (logs / "inputs" / "replay_sel_index.json").exists():
        state["sel"], state["index"] = load_selection(logs / "inputs")
    if (zybo / "module" / "calibration.json").exists():
        state["c"] = json.loads((zybo / "module" / "calibration.json").read_text())["c"]
    write_json(logs / "run_status.json", status)

    def step(name, info):
        snap0 = board_snapshot(remote)
        info["board"] = {"boot_id": snap0["boot_id"], "research_pid": snap0["research_processes"][0]["pid"] if snap0["research_processes"] else None}
        if name == "prepare":
            state["sel"], state["index"] = build_selection(pilot, logs / "inputs")
            remote.shell("mkdir -p -m 700 %s/logs %s/out %s/frozen %s/replay %s/incoming" % (
                REMOTE_MODULE, REMOTE_MODULE, REMOTE_MODULE, REMOTE_DIR, REMOTE_DIR))
            remote.put([MODULE / f for f in UPLOAD] + [
                logs / "inputs" / "replay_sel_apuf_lanes.npy", logs / "inputs" / "replay_sel_ff_lanes.npy",
                logs / "inputs" / "replay_sel_index.json", zybo / "crp_board1" / "challenges.npy",
                HERE / "zybo_board.py"], REMOTE_DIR + "/incoming")
            moves = ["mv -f %s/incoming/%s %s/%s" % (REMOTE_DIR, Path(f).name, REMOTE_MODULE, f) for f in UPLOAD]
            moves += ["mv -f %s/incoming/replay_sel_apuf_lanes.npy %s/incoming/replay_sel_ff_lanes.npy %s/incoming/replay_sel_index.json %s/replay/"
                      % ((REMOTE_DIR,) * 4),
                      "mv -f %s/incoming/challenges.npy %s/replay/z3_challenges.npy" % (REMOTE_DIR, REMOTE_DIR),
                      "mv -f %s/incoming/zybo_board.py %s/zybo_board.py" % (REMOTE_DIR, REMOTE_DIR),
                      "rm -rf %s/incoming" % REMOTE_DIR]
            remote.shell(" && ".join(moves))
            lines = remote.get(["sha256sum"] + [REMOTE_MODULE + "/" + f for f in UPLOAD]
                               + list(REPLAY.values()), log_name="prepare_sha256sum.txt")
            on_board = dict((l.split(None, 1)[1].strip(), l.split()[0]) for l in lines.splitlines() if l.strip())
            expected = dict((REMOTE_MODULE + "/" + f, sha256(MODULE / f)) for f in UPLOAD)
            expected.update((REPLAY[f], sha256(logs / "inputs" / ("replay_sel_" + f + ".npy"))) for f in REPLAY)
            bad = [p for p, h in expected.items() if on_board.get(p) != h]
            if bad:
                raise RuntimeError("uploaded files differ on the board: %s" % bad)
            info.update(files={Path(p).name: h for p, h in expected.items()}, replay_rows=len(state["index"]))
            info["ocm_precheck"] = ocm_page(remote, "ocm_precheck.log")
            if not info["ocm_precheck"]["all_zero"]:
                raise RuntimeError("the OCM page is not empty before the first use: %r" % info["ocm_precheck"])

        elif name == "calibrate":
            remote.run(sudo_python([REMOTE_MODULE + "/module_calibrate.py", "--device-id", args.serial,
                                    "--out", REMOTE_MODULE + "/out/calibration.json"]),
                       timeout=600, log_name="calibrate.log")
            cal = json.loads(remote.shell("cat %s/out/calibration.json" % REMOTE_MODULE).stdout)
            cal["board"] = {"index": 1, "serial": args.serial, "boot_id": snap0["boot_id"]}
            cal["provisional"] = True
            cal["provisional_note"] = ("One board only: board 2 is absent, so c is calibrated on board 1. Both boards "
                                       "must use one c: calibrate board 2 when it arrives and keep the larger c.")
            write_json(zybo / "module" / "calibration.json", cal)
            state["c"] = cal["c"]
            info.update(c=cal["c"], min_kdf_s=cal["min_kdf_s"], median_kdf_s=cal["median_kdf_s"])

        elif name == "live_acquire":
            proc = remote.run(sudo_python([
                REMOTE_MODULE + "/module_enroll.py", "--lanes", "0,4", "--t", "16", "--device-id", args.serial,
                "--k", REMOTE_MODULE + "/challenges_K.npy", "--research", FROZEN_RESEARCH, "--bit", FROZEN_BIT,
                "--no-program", "--dry-run", "--out", REMOTE_MODULE + "/out/never_written.json"]),
                timeout=600, log_name="live_acquire.log")
            summary = json.loads(proc.stdout.strip().splitlines()[-1])
            if summary.get("dry_run") is not True or summary["evaluations"] != 18 or summary["backend"] != "mmio":
                raise RuntimeError("unexpected live dry-run summary: %r" % summary)
            if remote.shell("test -e %s/out/never_written.json && echo EXISTS || echo absent" % REMOTE_MODULE).stdout.strip() != "absent":
                raise RuntimeError("the dry-run wrote a file")
            write_json(zybo / "module" / "validation" / "live_acquire.json", {
                "utc": utc(), "board": info["board"],
                "what": ("module_enroll.py --dry-run on the live PL: module_core.MmioSource calls the frozen research "
                         "firmware's measure_apuf_lanes / measure_ff_lanes for nine reads of K per family (lanes 0 and 4, "
                         "one of each). Every result was discarded and no file was written. This is the real acquire "
                         "time of one read of K per family on this board."),
                "seconds_per_read": summary["seconds_per_read"], "evaluations": summary["evaluations"],
                "backend": summary["backend"], "pl_programmed_by_module": False,
                "research_firmware_sha256": snap0["research_firmware_sha256"],
                "bitstream_sha256": snap0["bitstream_sha256"], "service_log": proc.stderr.splitlines(),
                "xadc_temperature_C_after": board_snapshot(remote)["telemetry"]["xadc_die_temperature_C"]})
            info.update(seconds_per_read=summary["seconds_per_read"])

        elif name == "matrix":
            rows = len(state["index"])
            results = []
            for lanes, t in MATRIX:
                cname = cfg_name(lanes, t)
                enr = test_enrollment(remote, lanes, t, args.serial, "matrix_enroll_%s.log" % cname)
                pid, log, store = serve(remote, "matrix_" + cname, replay_args(enr, rows, 1), ip, state)
                req = json.dumps({"op": "unlock", "passphrase_hex": passphrase.hex()}).encode()
                attempts = []
                for row in range(9, rows):
                    reply, secs, err = ask(ip, PORT, req)
                    obj = json.loads(reply) if reply else None
                    attempts.append({"row": row, "success": bool(obj and obj.get("success")),
                                     "error": (obj or {}).get("error", err), "round_trip_s": secs})
                text = stop_service(remote, pid, log, ip)
                remote.run(["sudo", "-n", "rm", "-f", enr], check=False)
                stack = np.stack([combined(state["sel"], r, lanes) for r in range(9)])
                w_enroll = (stack.sum(axis=0) >= 5).astype(np.uint8)
                for a in attempts:
                    a["errors_vs_enrollment_word"] = int(np.count_nonzero(combined(state["sel"], a["row"], lanes) != w_enroll))
                    a["source"] = state["index"][a["row"]]["source"]
                    a["phase"] = state["index"][a["row"]]["phase"]
                    a["temperature_C"] = state["index"][a["row"]]["temperature_before_C"]
                last = last_event(text)
                results.append({
                    "lanes": lanes, "t": t, "secret_store": store, "attempts": len(attempts),
                    "successes": sum(a["success"] for a in attempts),
                    "success_iff_errors_within_t": all((a["errors_vs_enrollment_word"] <= t) == a["success"] for a in attempts),
                    "max_errors_seen": max(a["errors_vs_enrollment_word"] for a in attempts),
                    "service_counters": {k: v for k, v in last.items() if isinstance(v, int) and k != "signum"},
                    "per_attempt": attempts})
                print(cname, results[-1]["successes"], "/", len(attempts), "max errors", results[-1]["max_errors_seen"],
                      "rule", results[-1]["success_iff_errors_within_t"], flush=True)
            warm = [i for i in state["index"][9:] if i["source"] == "sweep"]
            write_json(zybo / "module" / "validation" / "reconstruction_matrix.json", {
                "utc": utc(), "board": info["board"],
                "what": ("Firmware validation on the board. Every test enrollment was made by module_enroll.py from nine "
                         "recorded idle reads of this boot (selection rows 0-8). module_serve.py (--backend replay, KDF "
                         "c = 1) then tried one unlock per remaining recorded read: rows 9-%d are the other idle reads, "
                         "rows %d-%d are %d warmer reads of the same-boot sweep (%.1f to %.1f C). The test enrollments "
                         "were deleted; they are not enrollments of the board." % (
                             19, 20, rows - 1, len(warm), min(i["temperature_before_C"] for i in warm),
                             max(i["temperature_before_C"] for i in warm))),
                "counted_as_formal_enrollment": False, "replay_index": "logs/module/inputs/replay_sel_index.json",
                "configurations": results})
            info.update(configurations=len(results), rows=rows)

        elif name in ("timing", "closure"):
            rows = len(state["index"])
            enr = test_enrollment(remote, TIMING_LANES, TIMING_T, args.serial, name + "_enroll.log")
            pid, log, store = serve(remote, name, replay_args(enr, rows, state["c"]), ip, state)
            req = json.dumps({"op": "unlock", "passphrase_hex": passphrase.hex()}).encode()
            try:
                if name == "timing":
                    def client(request):
                        t0 = time.perf_counter()
                        proc = subprocess.run([sys.executable, str(MODULE / "module_client.py"), "--host", ip,
                                               "--port", str(PORT)], input=json.dumps(request).encode() + b"\n",
                                              capture_output=True, timeout=180)
                        secs = time.perf_counter() - t0
                        return (json.loads(proc.stdout) if proc.stdout else None), secs, proc.returncode

                    record = {"utc": utc(), "board": info["board"], "c": state["c"], "lanes": TIMING_LANES, "t": TIMING_T,
                              "secret_store": store, "provisional": True, "counted_as_formal_Z7": False,
                              "what": ("Same boot, no power-up before the run. The deployment image ran with a test "
                                       "enrollment made from recorded reads (replay backend), so fe_s and kdf_s are real "
                                       "timings of this board and acquire_s is only the replay time; the real read time is "
                                       "in module/validation/live_acquire.json. 20 unlocks in a row (each followed by a "
                                       "lock), then one unlock, 20 derives and a lock, every request through "
                                       "module_client.py from the PC over Ethernet. Keys and the passphrase are not recorded.")}
                    reply, secs, rc = client({"op": "derive", "label_hex": "00"})
                    record["derive_before_unlock"] = {"reply": reply, "host_round_trip_s": secs, "exit": rc}
                    record["unlocks"] = []
                    for i in range(20):
                        reply, secs, rc = client({"op": "unlock", "passphrase_hex": passphrase.hex()})
                        record["unlocks"].append({"index": i, "reply": reply, "host_round_trip_s": secs, "exit": rc})
                        if not (reply and reply.get("success")):
                            raise RuntimeError("unlock %d failed: %r" % (i, reply))
                        client({"op": "lock"})
                    reply, secs, rc = client({"op": "unlock", "passphrase_hex": passphrase.hex()})
                    record["unlock_before_derives"] = {"reply": reply, "host_round_trip_s": secs, "exit": rc}
                    record["derives"], first_key = [], None
                    for i in range(20):
                        reply, secs, rc = client({"op": "derive", "label_hex": b"cold|aead|model=qwen3|store=binding-test".hex()})
                        key = reply.pop("key_hex") if reply else None
                        first_key = key if first_key is None else first_key
                        record["derives"].append({"index": i, "success": bool(reply.get("success")),
                                                  "same_key_as_first": key == first_key, "host_round_trip_s": secs,
                                                  "exit": rc, "reply_without_key": reply})
                    record["lock"] = client({"op": "lock"})
                    record["derive_after_lock"] = client({"op": "derive", "label_hex": "00"})
                else:
                    before = remote.get(["ss", "-ltnp"], sudo=True, log_name="listeners_before_closure.txt")
                    stopped = stop_research(remote, snap0)
                    during = remote.get(["ss", "-ltnp"], sudo=True, log_name="listeners_during_closure.txt")
                    challenges = np.load(zybo / "crp_board1" / "challenges.npy", allow_pickle=False)
                    k_rows = {r.tobytes() for r in np.load(MODULE / "challenges_K.npy", allow_pickle=False)}
                    commands = rf.research_commands(challenges[~np.array([r.tobytes() in k_rows for r in challenges])], 100)
                    impostors = rf.json_impostors(challenges, 10)
                    record = {"utc": utc(), "board": info["board"], "c": state["c"], "lanes": TIMING_LANES, "t": TIMING_T,
                              "secret_store": store, "deployment_service_port": PORT,
                              "what": ("Z8 on the real board. The research service (port 3040) was stopped with kill -TERM "
                                       "and its listener was confirmed gone; the deployment image ran with a test "
                                       "enrollment (replay backend, KDF c = %d). From the PC: 100 research-protocol commands "
                                       "(PUF-COLP v1 frames that read responses, challenges from the Z3 set, none in K), then "
                                       "10 line-JSON impostors, then one unlock with the right passphrase, one derive and a "
                                       "lock. 'evaluations' in the service log counts reads of K and must rise only for the "
                                       "unlock. The research service was then restarted (same frozen firmware, --no-program)." % state["c"]),
                              "research_service_before": snap0["research_processes"],
                              "research_service_stopped": stopped, "commands": [], "impostors": []}
                    for c in commands:
                        reply, secs, err = ask(ip, PORT, c["payload"])
                        obj = json.loads(reply) if reply else None
                        record["commands"].append({
                            "index": c["index"], "description": c["description"], "bytes_sent": len(c["payload"]),
                            "payload_sha256": hashlib.sha256(c["payload"]).hexdigest(), "reply": obj,
                            "rejected": bool(obj and obj.get("success") is False and obj.get("error") == "unsupported_request"),
                            "round_trip_s": secs, "transport_error": err})
                    for c in impostors:
                        reply, secs, err = ask(ip, PORT, c["payload"])
                        obj = json.loads(reply) if reply else None
                        record["impostors"].append({"index": c["index"], "description": c["description"], "reply": obj,
                                                    "rejected": bool(obj and obj.get("error") == "unsupported_request"),
                                                    "round_trip_s": secs})
                    reply, secs, err = ask(ip, PORT, req)
                    record["unlock_after"] = {"reply": json.loads(reply) if reply else None, "round_trip_s": secs}
                    reply, secs, err = ask(ip, PORT, json.dumps({"op": "derive", "label_hex": "00"}).encode())
                    record["derive_after"] = {"success": bool(reply and json.loads(reply).get("success")), "round_trip_s": secs}
                    ask(ip, PORT, b'{"op":"lock"}')
                    record["all_rejected"] = all(c["rejected"] for c in record["commands"] + record["impostors"])
                    record["unlock_ok_after"] = bool(record["unlock_after"]["reply"] and record["unlock_after"]["reply"].get("success"))
                    record["listeners"] = {"before": before, "during": during}
            finally:
                text = stop_service(remote, pid, log, ip)
                remote.run(["sudo", "-n", "rm", "-f", enr], check=False)
            record["service_counters"] = {k: v for k, v in last_event(text).items() if isinstance(v, int) and k != "signum"}
            (zybo / "timing").mkdir(parents=True, exist_ok=True)
            if name == "timing":
                (logs / "timing_service.log").write_text(text)
                write_json(zybo / "timing" / ("unlock_timing_%s_replay_test.json" % args.serial), record)
                info.update(unlocks=len(record["unlocks"]), derives=len(record["derives"]), store=store)
            else:
                events = [json.loads(l) for l in text.splitlines() if l.startswith("{")]
                record["evaluations_by_event"] = [{"event": e.get("event"), "evaluations": e.get("evaluations")}
                                                  for e in events if "evaluations" in e]
                (zybo / "interface").mkdir(parents=True, exist_ok=True)
                (zybo / "interface" / ("module_service_%s.log" % args.serial)).write_text(text)
                record["research_service_restarted"] = start_research(remote)
                record["listeners"]["after"] = remote.get(["ss", "-ltnp"], sudo=True, log_name="listeners_after_closure.txt")
                record["research_service_after"] = board_snapshot(remote)["research_processes"]
                write_json(zybo / "interface" / ("closure_%s.json" % args.serial), record)
                info.update(all_rejected=record["all_rejected"], unlock_ok=record["unlock_ok_after"],
                            restarted=record["research_service_restarted"], store=store)

        elif name == "j6_layout":
            rows = len(state["index"])
            enr = test_enrollment(remote, TIMING_LANES, TIMING_T, args.serial, "j6_enroll.log")
            bad = REMOTE_MODULE + "/out/test_modified.json"
            # the "modified" setting of J6: one byte of ecc_hex changed in a copy of the enrollment
            code = ("import json,sys\n"
                    "src,dst=sys.argv[1],sys.argv[2]\n"
                    "o=json.load(open(src))\n"
                    "e=bytearray(bytes.fromhex(o['ecc_hex']))\n"
                    "e[5]^=0x01\n"
                    "o['ecc_hex']=bytes(e).hex()\n"
                    "open(dst,'w').write(json.dumps(o,indent=2,sort_keys=True)+'\\n')\n")
            remote.run(["sudo", "-n", "rm", "-f", bad], check=False)
            remote.run(sudo_python(["-c", code, enr, bad]), log_name="j6_make_modified.log")
            pid_a, log_a, store_a = serve(remote, "j6_owner", replay_args(enr, rows, 1) + ["--ocm-address", OCM_ADDRESS], ip, state)
            try:
                pid_b, log_b, store_b = serve(remote, "j6_modified", replay_args(bad, rows, 1) + [
                    "--ocm-address", "0xFFFE9000", "--port", "3043"], ip, state)
            except Exception:
                stop_service(remote, pid_a, log_a, ip)
                raise
            try:
                req = json.dumps({"op": "unlock", "passphrase_hex": passphrase.hex()}).encode()
                results = {}
                for label, port in (("owner", PORT), ("modified", 3043)):
                    reply, secs, err = ask(ip, port, req)
                    results[label] = {"reply": json.loads(reply) if reply else None, "round_trip_s": secs, "port": port}
                ok = (results["owner"]["reply"] and results["owner"]["reply"].get("success") is True
                      and results["modified"]["reply"] == {"success": False, "device_id": args.serial, "error": "reconstruction"})
            finally:
                text_a = stop_service(remote, pid_a, log_a, ip)
                text_b = stop_service(remote, pid_b, log_b, ip)
                remote.run(["sudo", "-n", "rm", "-f", enr, bad], check=False)
            pages = {"owner_page": ocm_page(remote, "ocm_j6_owner_page.log"),
                     "modified_page": ocm_page(remote, "ocm_j6_modified_page.log", "0xFFFE9000")}
            write_json(zybo / "module" / "validation" / "j6_two_services.json", {
                "utc": utc(), "board": info["board"], "what": ("Two module_serve.py processes at the same time on the board, "
                    "ports 3041 and 3043, OCM pages 0xFFFE8000 and 0xFFFE9000, both with a test enrollment from recorded "
                    "reads (replay backend, c = 1); the second one started from a copy whose ecc_hex differs in one byte."),
                "results": results, "as_expected": ok, "secret_stores": {"owner": store_a, "modified": store_b},
                "pages_after_stop": pages,
                "service_counters": {"owner": {k: v for k, v in last_event(text_a).items() if isinstance(v, int) and k != "signum"},
                                     "modified": {k: v for k, v in last_event(text_b).items() if isinstance(v, int) and k != "signum"}}})
            if not ok:
                raise RuntimeError("two-service layout did not behave as expected: %r" % results)

        elif name == "ocm_check":
            pages = [ocm_page(remote, "ocm_check.log"), ocm_page(remote, "ocm_check_page2.log", "0xFFFE9000")]
            page = {"pages": pages, "all_zero": all(p["all_zero"] for p in pages), "bytes": sum(p["bytes"] for p in pages)}
            write_json(zybo / "module" / "validation" / "ocm_check.json", page)
            info.update(ocm=page)

        elif name == "cleanup":
            remote.run(["sudo", "-n", "rm", "-rf", REMOTE_MODULE + "/out", REMOTE_DIR + "/replay",
                        REMOTE_DIR + "/probe_ro.py", REMOTE_DIR + "/heat_probe.py", REMOTE_DIR + "/poll_sweep.py",
                        REMOTE_DIR + "/__pycache__", "/home/xilinx/probe_ro.py"], check=False)
            listing = remote.shell("ls -la %s %s/module %s/module/logs 2>&1; ls %s/out %s/replay %s/module/out 2>&1" % (
                REMOTE_DIR, REMOTE_DIR, REMOTE_DIR, REMOTE_DIR, REMOTE_DIR, REMOTE_DIR), check=False).stdout
            listing += "\n" + remote.shell("find %s -name '*test_*' -o -name '*enrollment*' 2>/dev/null; echo find-done" % REMOTE_DIR,
                                           check=False).stdout
            (logs / "cleanup_ls.txt").write_text(listing)
            info.update(board_dir_listing=listing)

        elif name == "final":
            snap = board_snapshot(remote)
            write_json(zybo / "env" / "snapshot_after_module_tests.json", snap)
            info.update(research_processes=snap["research_processes"], fpga_manager_state=snap["fpga_manager_state"])

    for name in [s for s in args.steps.split(",") if s]:
        print("[{}] step {}".format(utc(), name), flush=True)
        info = {"started_utc": utc()}
        try:
            step(name, info)
        except Exception as exc:
            status["steps"][name] = dict(status="failed", error="%s: %s" % (type(exc).__name__, exc), **info)
            write_json(logs / "run_status.json", status)
            raise
        info["finished_utc"] = utc()
        status["steps"][name] = dict(status="done", **info)
        write_json(logs / "run_status.json", status)
    status["store_fallback"] = state.get("store_fallback")
    status["finished_utc"] = utc()
    write_json(logs / "run_status.json", status)
    print(json.dumps({"status": "complete", "steps": {k: v.get("status", "n/a") for k, v in status["steps"].items()}}, indent=2))


if __name__ == "__main__":
    main()
