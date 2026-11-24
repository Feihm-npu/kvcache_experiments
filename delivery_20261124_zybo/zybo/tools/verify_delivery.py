#!/usr/bin/env python3
"""Checks the Zybo batch of a delivery directory against the requirement's formats.

  python verify_delivery.py <delivery_dir> [--sha]

Exits 0 when every check passes. --sha also runs `sha256sum -c SHA256SUMS.txt` inside the delivery.
Nothing here analyses the PUF data; it only checks formats, hashes, completeness and the absence of
key material in the delivered outputs.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

K_SHA = "6743daa679eaf86c3cba25c028728774bcf4a80f214b17d9b9e2e31cec55acd5"
FROZEN = {"zybo/module/frozen/puf_server.py": "a057df2c6136e72ed9ae10be55dbcbc2ab12f8a16f15d656e7bc1e17c488c1c5",
          "zybo/module/frozen/system_wrapper_apuf_ff_20260917.bit":
              "224cc1fdd3234b070ef7f6568c21fecf1683cbe45508593597e3e2b1bb7b26dd"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("delivery", type=Path)
    ap.add_argument("--sha", action="store_true")
    args = ap.parse_args()
    root = args.delivery.resolve()
    checks = []

    def check(name, ok, detail=None):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    def load(path):
        return np.load(str(path), allow_pickle=False)

    def is_lanes(arr, rows=None):
        return (arr.dtype == np.int8 and arr.ndim == 3 and arr.shape[1:] == (4096, 4)
                and (rows is None or arr.shape[0] == rows) and bool(np.isin(arr, [0, 1]).all()))

    jsons = sorted(root.rglob("*.json"))
    bad = []
    for p in jsons:
        try:
            json.loads(p.read_text())
        except ValueError as exc:
            bad.append("%s: %s" % (p, exc))
    check("all JSON files parse", not bad, {"files": len(jsons), "bad": bad})

    boards = json.loads((root / "zybo/boards.json").read_text())
    serial = boards["boards"][0]["serial"]
    pilot = root / "zybo/sameboot_pilot" / serial
    for step in ("idle_pre", "idle_post", "sweep"):
        meta = json.loads((pilot / step / "meta.json").read_text())
        rows = {f: sum(1 for r in meta["reads"] if r["family"] == f) for f in ("apuf_lanes", "ff_lanes")}
        ok = bool(meta["complete"]) and meta["challenges_raw_sha256"] == K_SHA and meta["true_power_cycles"] == 0
        for f in rows:
            ok = ok and is_lanes(load(pilot / step / (f + ".npy")), rows[f])
        check("same-boot pilot %s: K read once per family per row, int8 (R, 4096, 4), complete" % step, ok,
              {"rows": rows, "temperature_C": [
                  round(min(r["before"]["xadc_die_temperature_C"] for r in meta["reads"]), 2),
                  round(max(r["before"]["xadc_die_temperature_C"] for r in meta["reads"]), 2)]})

    crp = root / "zybo/crp_board1"
    c = load(crp / "challenges.npy")
    regen = np.random.default_rng(20261120).integers(0, 2, (100000, 64), dtype=np.int8)
    k = load(root / "zybo/module/challenges_K.npy")
    k_rows = {r.tobytes() for r in k}
    meta = json.loads((crp / "meta.json").read_text())
    if meta["challenge_generation"]["removed_rows_equal_to_K"] == 0:
        same = np.array_equal(c, regen)                       # nothing removed: the set is the generator's output
    else:
        regen_rows = {r.tobytes() for r in regen}
        same = all(r.tobytes() in regen_rows for r in c)      # removed: every delivered row must come from it
    ok = (c.dtype == np.int8 and c.shape == (meta["challenge_generation"]["N_after_dedup"], 64)
          and hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest() == meta["challenge_generation"]["challenges_raw_sha256"]
          and same and not any(r.tobytes() in k_rows for r in c))
    for f in ("apuf_lanes", "ff_lanes"):
        arr = load(crp / (f + ".npy"))
        ok = ok and arr.dtype == np.int8 and arr.shape == (len(c), 4) and bool(np.isin(arr, [0, 1]).all()) \
            and hashlib.sha256(Path(crp / (f + ".npy")).read_bytes()).hexdigest() == meta["array_sha256"][f]
    check("crp_board1: Z3 challenge set (seed 20261120, no K row), one read per family, int8 (N, 4)", ok,
          {"N": int(len(c)), "challenges_raw_sha256": meta["challenge_generation"]["challenges_raw_sha256"]})

    check("frozen K raw SHA-256", hashlib.sha256(np.ascontiguousarray(k).tobytes()).hexdigest() == K_SHA)
    check("frozen bitstream and research firmware SHA-256", all(sha256(root / p) == h for p, h in FROZEN.items()),
          {p: sha256(root / p) == h for p, h in FROZEN.items()})

    cal = json.loads((root / "zybo/module/calibration.json").read_text())
    check("calibration: c set and every kdf_s >= 1.0 s on the board", cal["c"] >= 1 and min(cal["kdf_s"]) >= 1.0,
          {"c": cal["c"], "min_kdf_s": cal["min_kdf_s"]})

    matrix = json.loads((root / "zybo/module/validation/reconstruction_matrix.json").read_text())
    check("reconstruction matrix: success iff errors <= t in every configuration",
          all(c["success_iff_errors_within_t"] for c in matrix["configurations"]),
          {"configurations": len(matrix["configurations"]),
           "successes": {"lanes%s_t%d" % ("".join(map(str, c["lanes"])), c["t"]): [c["successes"], c["attempts"]]
                         for c in matrix["configurations"]}})

    timing = json.loads((root / ("zybo/timing/unlock_timing_%s_replay_test.json" % serial)).read_text())
    kd = [u["reply"]["timing"]["kdf_s"] for u in timing["unlocks"]]
    check("timing record: 20 unlocks and 20 derives succeeded, no key stored",
          len(timing["unlocks"]) == 20 and all(u["reply"]["success"] for u in timing["unlocks"])
          and len(timing["derives"]) == 20 and all(d["success"] and d["same_key_as_first"] for d in timing["derives"])
          and not any("key_hex" in json.dumps(u) for u in timing["unlocks"] + timing["derives"]),
          {"kdf_s": [min(kd), max(kd)], "fe_s": [min(u["reply"]["timing"]["fe_s"] for u in timing["unlocks"]),
                                                 max(u["reply"]["timing"]["fe_s"] for u in timing["unlocks"])]})

    closure = json.loads((root / ("zybo/interface/closure_%s.json" % serial)).read_text())
    evals = sum(e["evaluations"] for e in closure["evaluations_by_event"])
    check("closure: 100 research commands and 10 impostors rejected, one unlock afterwards, no evaluation for them",
          len(closure["commands"]) == 100 and len(closure["impostors"]) == 10 and closure["all_rejected"]
          and closure["unlock_ok_after"] and evals == 1 and closure["research_service_stopped"]
          and closure["research_service_restarted"] and all(c["transport_error"] is None for c in closure["commands"]),
          {"evaluations_total": evals, "service_counters": closure["service_counters"]})

    ocm = json.loads((root / "zybo/module/validation/ocm_check.json").read_text())
    check("OCM pages that held r and k_u are all zeros after the services stopped", ocm["all_zero"], ocm)

    # no key, root or passphrase in anything that is an output (source code and tests may name the fields)
    leaks = []
    for p in sorted(root.rglob("*.json")):
        if "zybo/module/test_" in str(p) or "replay_sel_index" in p.name or "challenges" in p.name:
            continue
        text = p.read_text()
        for m in re.finditer(r'"(key_hex|passphrase_hex|root_hex|k_u_hex)"\s*:\s*"([0-9a-fA-F]*)"', text):
            if m.group(2):
                leaks.append("%s: %s" % (p.relative_to(root), m.group(1)))
    check("no key, passphrase or root value in any JSON output", not leaks, leaks)
    log_leaks = []
    for p in sorted(root.rglob("*.log")) + sorted(root.rglob("*.txt")):
        for line in p.read_text(errors="replace").splitlines():
            if re.search(r'"(key_hex|passphrase_hex|root_hex)"\s*:\s*"[0-9a-fA-F]+"', line):
                log_leaks.append(p.relative_to(root).as_posix())
    check("logs hold no key, passphrase or root value", not log_leaks, log_leaks)

    if args.sha:
        proc = subprocess.run(["sha256sum", "-c", "SHA256SUMS.txt"], cwd=str(root), capture_output=True, text=True)
        check("sha256sum -c SHA256SUMS.txt", proc.returncode == 0, {"lines": len(proc.stdout.splitlines())})

    for c in checks:
        print("%s  %s" % ("PASS" if c["ok"] else "FAIL", c["check"]))
    ok = all(c["ok"] for c in checks)
    print(json.dumps({"ok": ok, "checks": checks}, indent=2, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
