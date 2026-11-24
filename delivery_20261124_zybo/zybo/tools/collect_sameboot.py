#!/usr/bin/env python3
"""Same-boot pilot on the original Zybo (board 1), through the frozen research service.

This is NOT Z2: nothing here cuts power. It records (a) repeated K reads at idle,
(b) the Z3 challenge set on board 1, (c) K reads while the die is warmed by a
lowest-priority CPU load and while it cools, (d) repeated K reads at idle again.

Usage:
  python collect_sameboot.py --out <delivery> --serial 210251A08870 --host xilinx@192.168.1.2
      [--steps upload,snapshot_start,idle_pre,crp_board1,sweep,idle_post,snapshot_end,verify]
Re-running with the same --out keeps every step whose output is already complete.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from host_common import Remote, utc, write_json  # noqa: E402

# Frozen inputs of the requirement; the first two live in the repository's history on this host.
FROZEN = {
    "bitstream": ("new_req_09_08/delivery_0916_req/evidence/apuf_ff_image/system_wrapper_apuf_ff_20260917.bit",
                  "224cc1fdd3234b070ef7f6568c21fecf1683cbe45508593597e3e2b1bb7b26dd"),
    "research_firmware": ("new_req_09_08/delivery_0916_req/evidence/apuf_ff_image/puf_server.py",
                          "a057df2c6136e72ed9ae10be55dbcbc2ab12f8a16f15d656e7bc1e17c488c1c5"),
    "challenges": ("new_req_09_08/delivery_0916_req/data/apuf_ff_20260917/challenges.npy",
                   "69324bbc2daf94b7b9c401f3e912f2a675b59bdac9adcd87e0abff3a5ddd1905"),
}
K_SHA = "6743daa679eaf86c3cba25c028728774bcf4a80f214b17d9b9e2e31cec55acd5"
Z3_SEED = 20261120
REMOTE_DIR = "/home/xilinx/puf/req1120_20261124"
ALL_STEPS = ("upload", "snapshot_start", "idle_pre", "crp_board1", "sweep", "idle_post", "snapshot_end", "verify")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_paths():
    """The frozen files are not in the git repository; take them from the workspace, verified by hash."""
    base = Path("/home/fei-zl-share/fei_space/puf_exp")
    out = {}
    for name, (rel, expected) in FROZEN.items():
        path = base / rel
        actual = sha256(path)
        if actual != expected:
            raise SystemExit("frozen input {} has SHA-256 {} (expected {})".format(path, actual, expected))
        out[name] = (path, expected)
    return out


def import_step(remote, remote_name, dest):
    """Copy one finished board-side read directory into the delivery."""
    dest = Path(dest)
    tmp = dest.parent / (dest.name + ".incoming")
    if tmp.exists():
        shutil.rmtree(tmp)
    got = remote.get_dir(REMOTE_DIR + "/out/" + remote_name, tmp)
    got = tmp / remote_name
    meta = json.loads((got / "meta.json").read_text())
    if not meta.get("complete"):
        raise SystemExit("board-side step {} is incomplete".format(remote_name))
    dest.mkdir(parents=True, exist_ok=True)
    for name in sorted(p.name for p in got.iterdir()):
        shutil.move(str(got / name), str(dest / name))
    shutil.rmtree(tmp)
    return meta


def verify_read_dir(path, n, meta_name="meta.json", families=("apuf_lanes", "ff_lanes"), drop_read_axis=False):
    path = Path(path)
    meta = json.loads((path / meta_name).read_text())
    checks = {"complete": meta["complete"], "files": {}}
    for f in families:
        arr = np.load(path / (f + ".npy"), allow_pickle=False)
        rows = sum(1 for r in meta["reads"] if r["family"] == f)
        want = (n, 4) if drop_read_axis else (rows, n, 4)
        checks["files"][f] = {
            "shape": list(arr.shape), "dtype": str(arr.dtype), "binary": bool(np.isin(arr, [0, 1]).all()),
            "shape_ok": (rows == 1 if drop_read_axis else True) and arr.shape == want,
            "sha256": sha256(path / (f + ".npy"))}
    checks["ok"] = bool(checks["complete"] and all(v["shape_ok"] and v["binary"] for v in checks["files"].values()))
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="delivery directory")
    ap.add_argument("--serial", required=True)
    ap.add_argument("--host", default="xilinx@192.168.1.2")
    ap.add_argument("--steps", default=",".join(ALL_STEPS))
    ap.add_argument("--idle-repeats", type=int, default=20)
    ap.add_argument("--idle-interval", type=float, default=5.0)
    ap.add_argument("--sweep-arg", action="append", default=[], help="extra argument passed to the board-side sweep")
    args = ap.parse_args()
    out = args.out.resolve()
    zybo = out / "zybo"
    logs = out / "logs" / "sameboot"
    pilot = zybo / "sameboot_pilot" / args.serial
    remote = Remote(args.host, logs)
    wanted = [s for s in args.steps.split(",") if s]
    frozen = frozen_paths()
    status_path = logs / "run_status.json"
    old = json.loads(status_path.read_text()) if status_path.exists() else {}
    env = {"first_started_utc": old.get("first_started_utc", utc()), "last_started_utc": utc(), "host": args.host,
           "serial": args.serial, "true_power_cycles": 0, "counted_as_Z2": False, "frozen_inputs": old.get("frozen_inputs", {}),
           "steps": old.get("steps", {}), "note": "steps of earlier invocations are kept; timeline.json lists every remote call"}
    for name, (path, expected) in frozen.items():
        env["frozen_inputs"][name] = {"host_path": str(path), "sha256": expected}

    def run_step(name):
        if name == "upload":
            remote.run(["mkdir", "-p", "-m", "700", REMOTE_DIR + "/out"])
            remote.run(["rm", "-f"] + [REMOTE_DIR + "/" + n for n in
                                       ("zybo_board.py", "challenges_K.npy", "crp_challenges.npy")])
            remote.put([HERE / "zybo_board.py"], REMOTE_DIR)
            src = np.load(frozen["challenges"][0], allow_pickle=False)
            k = np.ascontiguousarray(src[:4096])
            if k.shape != (4096, 64) or k.dtype != np.int8 or hashlib.sha256(k.tobytes()).hexdigest() != K_SHA:
                raise SystemExit("K mismatch")
            work = logs / "inputs"
            work.mkdir(parents=True, exist_ok=True)
            np.save(work / "challenges_K.npy", k, allow_pickle=False)
            # Z3 challenge set: the generator named in the requirement, minus rows equal to a K row
            raw = np.random.default_rng(Z3_SEED).integers(0, 2, (100000, 64), dtype=np.int8)
            k_rows = {r.tobytes() for r in k}
            keep = np.array([r.tobytes() not in k_rows for r in raw], dtype=bool)
            crp = np.ascontiguousarray(raw[keep])
            np.save(work / "crp_challenges.npy", crp, allow_pickle=False)
            info["crp"] = {"seed": Z3_SEED, "generated": int(len(raw)), "after_removing_K_rows": int(len(crp)),
                           "removed": int((~keep).sum()), "sha256": sha256(work / "crp_challenges.npy"),
                           "raw_sha256": hashlib.sha256(crp.tobytes()).hexdigest()}
            remote.put([work / "challenges_K.npy", work / "crp_challenges.npy"], REMOTE_DIR)
            remote.run(["chmod", "444", REMOTE_DIR + "/challenges_K.npy", REMOTE_DIR + "/crp_challenges.npy"])
            remote.run(["sha256sum", REMOTE_DIR + "/challenges_K.npy", REMOTE_DIR + "/crp_challenges.npy",
                        REMOTE_DIR + "/zybo_board.py"], log_name="upload_sha256sum.txt")
            info["k"] = {"shape": list(k.shape), "raw_sha256": K_SHA, "npy_sha256": sha256(work / "challenges_K.npy")}
        elif name.startswith("snapshot_"):
            text = remote.get(["python3", REMOTE_DIR + "/zybo_board.py", "snapshot"], log_name=name + ".json")
            write_json(zybo / "env" / (name + ".json"), json.loads(text))
        elif name in ("idle_pre", "idle_post"):
            rname = name
            remote.run(["python3", REMOTE_DIR + "/zybo_board.py", "read", "--challenges", REMOTE_DIR + "/challenges_K.npy",
                        "--repeats", str(args.idle_repeats), "--interval", str(args.idle_interval),
                        "--label", name, "--out", REMOTE_DIR + "/out/" + rname], log_name=name + ".log",
                       timeout=60 + args.idle_repeats * (2 * 20 + args.idle_interval))
            import_step(remote, rname, pilot / name)
        elif name == "crp_board1":
            remote.run(["python3", REMOTE_DIR + "/zybo_board.py", "read",
                        "--challenges", REMOTE_DIR + "/crp_challenges.npy", "--repeats", "1",
                        "--label", "crp_board1", "--out", REMOTE_DIR + "/out/crp_board1"],
                       log_name=name + ".log", timeout=900)
            dest = zybo / "crp_board1"
            board_meta = import_step(remote, "crp_board1", dest)
            (dest / "meta.json").rename(dest / "board_read_meta.json")
            for extra in ("board_before.json", "board_after.json"):
                shutil.move(str(dest / extra), str(zybo / "env" / ("crp_board1_" + extra)))
            shutil.copyfile(logs / "inputs" / "crp_challenges.npy", dest / "challenges.npy")
            for family in ("apuf_lanes", "ff_lanes"):          # one read: drop the read axis -> (N, 4)
                arr = np.load(dest / (family + ".npy"), allow_pickle=False)
                if arr.shape[0] != 1:
                    raise SystemExit("crp_board1 expects exactly one read")
                np.save(dest / (family + ".npy"), np.ascontiguousarray(arr[0]), allow_pickle=False)
            c = np.load(dest / "challenges.npy", allow_pickle=False)
            k = np.load(logs / "inputs" / "challenges_K.npy", allow_pickle=False)
            k_rows = {r.tobytes() for r in k}
            removed = int(sum(r.tobytes() in k_rows for r in c))
            snap = json.loads((zybo / "env" / "crp_board1_board_before.json").read_text())
            reads = board_meta["reads"]
            write_json(dest / "meta.json", {
                "schema": "req_1120.zybo.crp/1",
                "requirement": "Z3 challenge set read on board 1; Z3 itself asks for board 2, which is absent",
                "board": {"index": 1, "serial": args.serial, "eth0_mac": board_meta["eth0_mac"],
                          "boot_id": board_meta["boot_id_start"],
                          "uptime_days": round(snap["telemetry"]["uptime_s"] / 86400.0, 4)},
                "challenge_generation": {
                    "generator": "np.random.default_rng(%d).integers(0, 2, (100000, 64), dtype=np.int8)" % Z3_SEED,
                    "seed": Z3_SEED, "generated": 100000, "removed_rows_equal_to_K": removed,
                    "N_after_dedup": int(len(c)), "challenge_shape": list(c.shape), "challenge_dtype": str(c.dtype),
                    "challenges_npy_sha256": sha256(dest / "challenges.npy"),
                    "challenges_raw_sha256": hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest()},
                "reads_per_challenge": 1, "read_order": board_meta["read_order"],
                "xadc_die_temperature_C": {r["family"]: [r["before"]["xadc_die_temperature_C"],
                                                         r["after"]["xadc_die_temperature_C"]] for r in reads},
                "xadc_vccint_mV": {r["family"]: [r["before"]["xadc_vccint_mV"], r["after"]["xadc_vccint_mV"]]
                                   for r in reads},
                "read_duration_s": {r["family"]: r["read_duration_s"] for r in reads},
                "bitstream_sha256": board_meta["bitstream_sha256"],
                "research_firmware_sha256": board_meta["research_firmware_sha256"],
                "raw_response_processing": board_meta["raw_response_processing"],
                "array_shape": {f: [int(len(c)), 4] for f in ("apuf_lanes", "ff_lanes")},
                "array_note": ("one read per challenge, so the read axis is dropped: int8 (N, 4), the layout the "
                               "requirement gives for crp_board2; board_read_meta.json describes the read"),
                "host_imported_utc": utc(), "board_read_meta": "board_read_meta.json"})
        elif name == "sweep":
            remote.run(["python3", REMOTE_DIR + "/zybo_board.py", "sweep", "--challenges", REMOTE_DIR + "/challenges_K.npy",
                        "--label", "sweep", "--out", REMOTE_DIR + "/out/sweep"] + args.sweep_arg,
                       log_name="sweep.log", timeout=2400)
            import_step(remote, "sweep", pilot / "sweep")
        elif name == "verify":
            checks, ok = {}, True
            for name2 in ("idle_pre", "sweep", "idle_post"):
                if (pilot / name2 / "meta.json").exists():
                    checks[name2] = verify_read_dir(pilot / name2, 4096)
                    ok = ok and checks[name2]["ok"]
            if (zybo / "crp_board1" / "board_read_meta.json").exists():
                n = len(np.load(zybo / "crp_board1" / "challenges.npy", allow_pickle=False))
                checks["crp_board1"] = verify_read_dir(zybo / "crp_board1", n, "board_read_meta.json", drop_read_axis=True)
                ok = ok and checks["crp_board1"]["ok"]
            k_file = logs / "inputs" / "challenges_K.npy"
            if not k_file.exists():                            # the delivered copy, once the inputs are cleaned up
                k_file = zybo / "module" / "challenges_K.npy"
            k_raw = hashlib.sha256(np.ascontiguousarray(np.load(k_file, allow_pickle=False)).tobytes()).hexdigest()
            checks["K"] = {"raw_sha256": k_raw, "expected": K_SHA, "ok": k_raw == K_SHA}
            ok = ok and checks["K"]["ok"]
            write_json(pilot / "verify.json", {"utc": utc(), "checks": checks, "ok": ok})
            info["verify_ok"] = ok

    for name in wanted:
        if name in ("idle_pre", "idle_post") and (pilot / name / "meta.json").exists():
            if json.loads((pilot / name / "meta.json").read_text()).get("complete"):
                print("keep completed step", name, flush=True)
                env["steps"][name] = {"status": "done", "kept_from_earlier_run": True}
                continue
        if name == "crp_board1" and (zybo / "crp_board1" / "meta.json").exists():
            print("keep completed step", name, flush=True)
            env["steps"][name] = {"status": "done", "kept_from_earlier_run": True}
            continue
        if name == "sweep" and (pilot / "sweep" / "meta.json").exists():
            if json.loads((pilot / "sweep" / "meta.json").read_text()).get("complete"):
                print("keep completed step", name, flush=True)
                env["steps"][name] = {"status": "done", "kept_from_earlier_run": True}
                continue
        print("[{}] step {}".format(utc(), name), flush=True)
        info = {"started_utc": utc()}
        try:
            run_step(name)
        except Exception as exc:  # keep the record honest, then stop
            env["steps"][name] = dict(status="failed", error="%s: %s" % (type(exc).__name__, exc),
                                      finished_utc=utc(), **info)
            write_json(logs / "run_status.json", env)
            raise
        info["finished_utc"] = utc()
        env["steps"][name] = dict(status="done", **info)
        write_json(logs / "run_status.json", env)
    env["finished_utc"] = utc()
    write_json(logs / "run_status.json", env)
    print(json.dumps({"status": "complete", "steps": {k: v.get("status", "n/a") for k, v in env["steps"].items()}}, indent=2))


if __name__ == "__main__":
    main()
