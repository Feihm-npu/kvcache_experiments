#!/usr/bin/env python3
"""Groups logs/<run>/commands.jsonl (every ssh/scp call of a run) into steps: logs/<run>/timeline.json.

  python make_timeline.py --logs <delivery>/logs/sameboot --run sameboot --merge-status
  python make_timeline.py --logs <delivery>/logs/module   --run module   --merge-status
  python make_timeline.py --logs <delivery>/logs/boards   --run boards

commands.jsonl is the complete record; this only makes it readable. A call gets its step from the
name of the output log the tool attached to it or from its command line; calls that fit several
steps (a snapshot, a service stop) belong to the step that was running at that moment.
--merge-status also rebuilds run_status.json from the timeline.
"""
import argparse
import json
import re
from pathlib import Path


def label(entry, run, current):
    argv, log = entry["argv"], entry.get("output_log") or ""
    text = " ".join(argv) if isinstance(argv, list) else str(argv)
    if run == "sameboot":
        if entry["kind"] == "scp_get":
            return "import " + text.split("out/")[-1].split()[0].strip()
        if "zybo_board.py snapshot" in text:
            return "snapshot"
        m = re.search(r"zybo_board\.py (read|sweep)", text)
        if m:
            out = re.search(r"--out \S+/out/(\S+)", text)
            return out.group(1) if out else m.group(1)
        return "upload"                 # uploads of the tools and challenge files, mkdir/chmod
    if run == "module":
        if log.startswith("matrix_"):
            return "matrix"
        if log.startswith("j6_") or "/logs/j6_" in text:
            return "j6_layout"
        if log in ("timing_enroll.log",):
            return "timing"
        if log in ("closure_enroll.log",) or log.startswith("listeners_"):
            return "closure"
        if log in ("calibrate.log",):
            return "calibrate"
        if log in ("live_acquire.log",):
            return "live_acquire"
        if log == "ocm_precheck.log":
            return "prepare"
        if log.startswith("ocm_"):
            return "ocm_check"
        if log == "cleanup_ls.txt":
            return "cleanup"
        if log == "prepare_sha256sum.txt" or entry["kind"] == "scp_put" or "mkdir -p -m 700" in text or current == "start":
            return "prepare"
        if ("rm -rf" in text or "ls -la" in text or "find " in text) and current == "ocm_check":
            return "cleanup"
        if "zybo_board.py snapshot" in text:
            return "final" if current == "cleanup" else current
        return current
    return run                      # any other run (boards, power_cycle rehearsal): one step named after the run


def merge_status(logs):
    """Rewrites run_status.json from timeline.json: one entry per step with every occurrence of it."""
    path = logs / "run_status.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    tl = json.loads((logs / "timeline.json").read_text())
    merged = {k: v for k, v in old.items() if k != "steps"}
    merged["note"] = ("assembled from timeline.json, which lists every remote call in order; a step that ran in "
                      "several invocations has several occurrences")
    merged["steps"] = {}
    for st in tl["steps"]:
        merged["steps"].setdefault(st["step"], {"occurrences": []})["occurrences"].append(
            {"first_utc": st["first_utc"], "last_utc": st["last_utc"], "n_calls": st["n_calls"],
             "failed_calls": st["failed_calls"]})
    for name, step in merged["steps"].items():
        if not name.startswith("import "):
            step["status"] = "done"
    path.write_text(json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--merge-status", action="store_true")
    args = ap.parse_args()
    entries = [json.loads(l) for l in (args.logs / "commands.jsonl").read_text().splitlines() if l.strip()]
    steps, current = [], "start"
    for e in entries:
        current = label(e, args.run, current)
        if steps and steps[-1]["step"] == current:
            steps[-1]["calls"].append(e)
            steps[-1]["last_utc"] = e["utc"]
        else:
            steps.append({"step": current, "first_utc": e["utc"], "last_utc": e["utc"], "calls": [e]})
    for st in steps:
        st["n_calls"] = len(st["calls"])
        st["failed_calls"] = sum(1 for c in st["calls"] if c.get("returncode"))
        st["commands"] = [" ".join(c["argv"]) if isinstance(c["argv"], list) else str(c["argv"]) for c in st["calls"]]
        del st["calls"]
    (args.logs / "timeline.json").write_text(json.dumps(
        {"run": args.run, "source": "commands.jsonl", "n_calls": len(entries), "steps": steps}, indent=2, ensure_ascii=False) + "\n")
    for st in steps:
        print("%-28s %s -> %s (UTC)  calls %3d  failed %d" % (st["step"], st["first_utc"][11:19], st["last_utc"][11:19],
                                                           st["n_calls"], st["failed_calls"]))
    if args.merge_status:
        merge_status(args.logs)


if __name__ == "__main__":
    main()
