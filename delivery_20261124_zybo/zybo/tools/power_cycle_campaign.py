#!/usr/bin/env python3
"""Z2: power-cycle campaign for one Zybo (20 true power-offs at room temperature or at about 60 C).

  python power_cycle_campaign.py --out <delivery> --serial 210251A08870 --condition room \
      --board-host xilinx@192.168.1.2 [--fallback-host xilinx@192.168.2.99] \
      --power operator | --power switch --power-off-cmd CMD --power-on-cmd CMD \
      [--rounds 20] [--min-off-s 60]

The PC cannot cut the board's power by itself. Either a person at the bench does it (--power
operator: the tool says when to cut and when to restore power) or a switched outlet is driven by
two shell commands. Per round the tool then:

  1. takes a snapshot of the running board (boot_id A, uptime);
  2. asks for power-off and waits until SSH stops answering (and notes whether the USB JTAG
     serial disappeared too, which shows that the USB cable carries the board's power);
  3. holds the outage for at least --min-off-s; a board that answers again earlier is invalid;
  4. asks for power-on, waits for SSH (DHCP address first, fallback address second) and checks
     that boot_id changed and that the uptime is that of a fresh boot;
  5. verifies the SHA-256 of the frozen bitstream and firmware on the board, starts the frozen
     research service WITH PL programming, and reads K once per family through zybo_board.py,
     recording the XADC temperature, the board uptime at the read and the host-side timing;
  6. stores the raw arrays under logs/power_cycle/<condition>/rounds/boot_NN/.

Invalid rounds are logged with their reason and never counted. When --rounds valid rounds exist,
--steps assemble stacks them into zybo/temperature/<serial>_<condition>_apuf_lanes.npy,
_ff_lanes.npy (int8, shape (rounds, 4096, 4)) and <serial>_<condition>_meta.json.

Before starting: the PC must answer DHCP on the board link (dnsmasq; needs root on the PC) or
own an address on 192.168.2.0/24 for the board's fallback address, otherwise the board is
unreachable after its first power-up. The board keeps /home/xilinx/puf/req1120_20261124 on its
SD card, so nothing is uploaded between rounds.
"""
import argparse
import glob
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from host_common import Remote, ssh_ok, utc, write_json  # noqa: E402

REMOTE_DIR = "/home/xilinx/puf/req1120_20261124"
FROZEN_RESEARCH = "/home/xilinx/puf/puf_server_apuf_ff_20260917.py"
FROZEN_BIT = "/home/xilinx/puf/system_wrapper_apuf_ff_20260917.bit"
ALL_STEPS = ("prepare", "campaign", "assemble")
EXTRA_STEPS = ("rehearsal",)          # --steps rehearsal: exercise the tool on a board that stays powered
LIMITS = {"wait_off": 900.0, "wait_on": 3600.0, "wait_ssh": 300.0}


# ---------------------------------------------------------------- power control
class OperatorPower(object):
    """A person cuts and restores power when told to. The Enter key is the person's own record of
    having cut the power; restoring power is not confirmed by a key, the tool watches for the board."""
    basis = "operator confirmation (process convention, not a measurement)"

    def __init__(self, prompt=print, answer=input):
        self.prompt, self.answer = prompt, answer

    def off(self, round_no, min_off):
        self.prompt(">>> ROUND %d: cut the power of the board NOW (unplug it or switch its outlet off), "
                    "then press Enter. Keep it off until the tool says otherwise." % round_no)
        self.answer()

    def on(self, round_no, min_off):
        self.prompt(">>> ROUND %d: the board has been off for at least %d s. Restore power NOW." % (round_no, min_off))


class SwitchPower(object):
    basis = "timestamps of the two switch commands (keep the switch's own log as well)"

    def __init__(self, off_cmd, on_cmd):
        self.off_cmd, self.on_cmd = off_cmd, on_cmd

    @staticmethod
    def _run(cmd):
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode:
            raise RuntimeError("power command failed: %s\n%s" % (cmd, proc.stdout))

    def off(self, round_no, min_off):
        self._run(self.off_cmd)

    def on(self, round_no, min_off):
        self._run(self.on_cmd)


# ---------------------------------------------------------------- probes
class RealWorld(object):
    """Clock, USB JTAG presence and SSH reachability of a real board."""

    def __init__(self, serial, hosts):
        self.serial, self.hosts = serial, list(hosts)

    clock = staticmethod(time.monotonic)
    wall = staticmethod(time.time)
    sleep = staticmethod(time.sleep)

    def usb_present(self):
        for path in glob.glob("/sys/bus/usb/devices/*/serial"):
            try:
                if Path(path).read_text().strip() == self.serial:
                    return True
            except OSError:
                pass
        return False

    def host_up(self):
        for host in self.hosts:
            if ssh_ok(host, timeout=2):
                return host
        return None


class Invalid(Exception):
    def __init__(self, reason, record):
        super(Invalid, self).__init__(reason)
        self.record = record


# ---------------------------------------------------------------- one round
def run_round(world, power, round_no, min_off, limits=LIMITS):
    """Cuts and restores the power of the board once. Returns the record of a valid outage or raises
    Invalid (record in exc.record) when the round cannot be counted."""
    rec = {"round": round_no, "valid": False, "started_utc": utc()}
    usb_before = world.usb_present()
    rec["usb_jtag_present_before"] = usb_before
    # ---- off ----
    t_request = world.clock()
    power.off(round_no, min_off)
    t_ssh_down = t_usb_gone = None
    while world.clock() - t_request < limits["wait_off"]:
        now, host = world.clock(), world.host_up()
        if t_usb_gone is None and usb_before and not world.usb_present():
            t_usb_gone = now
        if t_ssh_down is None and not host:
            t_ssh_down = now
            break                                   # SSH stopped answering: power is gone
        world.sleep(0.5)
    rec.update(t_power_off_request=t_request, t_usb_gone=t_usb_gone, t_ssh_gone=t_ssh_down)
    if t_ssh_down is None:
        raise Invalid("the board kept answering on SSH: it was not powered off", rec)
    off_start = t_ssh_down
    # ---- hold ----
    while world.clock() - off_start < min_off:
        if world.host_up():
            raise Invalid("the board answered again %.1f s after the cut, before the minimum of %.0f s" % (
                world.clock() - off_start, min_off), rec)
        world.sleep(0.25)
    t_on_request = world.clock()
    power.on(round_no, min_off)
    rec["t_power_on_request"] = t_on_request
    # ---- back ----
    def back():
        if t_usb_gone is not None and world.usb_present():
            return True                              # the USB JTAG chip is powered from the board's supply
        return bool(world.host_up())
    t_back = None
    while world.clock() - t_on_request < limits["wait_on"]:
        if back():
            t_back = world.clock()
            break
        world.sleep(0.5)
    if t_back is None:
        raise Invalid("the board did not come back within %d s" % limits["wait_on"], rec)
    rec["t_power_back"] = t_back
    rec["power_off_duration_s"] = t_back - off_start       # held interval, measured on the PC clock
    rec["power_on_request_to_back_s"] = t_back - t_on_request
    host, t_ssh = None, None
    while world.clock() - t_back < limits["wait_ssh"]:
        host = world.host_up()
        if host:
            t_ssh = world.clock()
            break
        world.sleep(1.0)
    if not host:
        raise Invalid("SSH did not come back within %d s (no DHCP lease and no fallback route?)" % limits["wait_ssh"], rec)
    rec.update(host_after_power_up=host, back_to_ssh_s=t_ssh - t_back,
               power_back_host_monotonic=t_back, power_back_host_wall=world.wall() - (world.clock() - t_back))
    return rec


# ---------------------------------------------------------------- board side after a power-up
def snapshot(remote):
    return json.loads(remote.get(["python3", REMOTE_DIR + "/zybo_board.py", "snapshot"]))


def start_research_service(remote):
    """After a power-up nothing runs: start the frozen research service WITH PL programming."""
    if ":3040 " in remote.get(["ss", "-ltn"]):
        return False
    inner = ("nohup setsid python3 %s --bit %s --port 3040 --base 0x40000000 "
             ">>/home/xilinx/puf/puf_server_apuf_ff_20260917.log 2>&1 </dev/null &" % (FROZEN_RESEARCH, FROZEN_BIT))
    remote.shell(shlex.join(["sudo", "-n", "sh", "-c", inner]))
    end = time.monotonic() + 60
    while time.monotonic() < end:
        if ":3040 " in remote.get(["ss", "-ltn"]):
            return True
        time.sleep(0.5)
    raise RuntimeError("the research service did not start after power-up")


def frozen_service_running(snap):
    """One research service started from the frozen bitstream with PL programming, PL operating."""
    procs = snap["research_processes"]
    return bool(snap["fpga_manager_state"] == "operating" and len(procs) == 1 and FROZEN_BIT in procs[0]["argv"]
                and "--no-program" not in procs[0]["argv"]
                and snap["bitstream_sha256"] == "224cc1fdd3234b070ef7f6568c21fecf1683cbe45508593597e3e2b1bb7b26dd"
                and snap["research_firmware_sha256"] == "a057df2c6136e72ed9ae10be55dbcbc2ab12f8a16f15d656e7bc1e17c488c1c5")


def fetch_round(remote, name, work):
    """Copies out/<name> from the board to work/<name>; returns the directory."""
    incoming = work / ".incoming"
    if incoming.exists():
        shutil.rmtree(incoming)
    got = remote.get_dir(REMOTE_DIR + "/out/" + name, incoming)
    dest = work / name
    if dest.exists():
        shutil.rmtree(dest)
    got.replace(dest)
    shutil.rmtree(incoming, ignore_errors=True)
    return dest


def assemble(logs, zybo, serial, condition, meta):
    """Stacks the valid rounds into the arrays the requirement asks for."""
    rows = sorted(meta["rounds"], key=lambda r: r["round"])
    dest, tmp = zybo / "temperature", zybo / "temperature" / ".assembling"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for family in ("apuf_lanes", "ff_lanes"):
        arr = np.stack([np.load(logs / r["directory"] / (family + ".npy"), allow_pickle=False)[0] for r in rows])
        np.save(str(tmp / ("%s_%s_%s.npy" % (serial, condition, family))), arr, allow_pickle=False)
    write_json(tmp / ("%s_%s_meta.json" % (serial, condition)), meta)
    for path in sorted(tmp.iterdir()):
        path.replace(dest / path.name)
    tmp.rmdir()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="delivery directory")
    ap.add_argument("--serial", required=True)
    ap.add_argument("--condition", choices=("room", "60c"), required=True)
    ap.add_argument("--board-host", default="xilinx@192.168.1.2")
    ap.add_argument("--fallback-host", default=None, help="e.g. xilinx@192.168.2.99 (PYNQ's static address)")
    ap.add_argument("--power", choices=("operator", "switch"), default="operator")
    ap.add_argument("--power-off-cmd", default=None)
    ap.add_argument("--power-on-cmd", default=None)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--min-off-s", type=float, default=60.0)
    ap.add_argument("--steps", default=",".join(ALL_STEPS))
    args = ap.parse_args()
    if args.power == "switch" and not (args.power_off_cmd and args.power_on_cmd):
        ap.error("--power switch needs --power-off-cmd and --power-on-cmd")
    out = args.out.resolve()
    zybo, logs = out / "zybo", out / "logs" / "power_cycle" / args.condition
    work, meta_path = logs / "rounds", logs / "meta.json"
    hosts = [args.board_host] + ([args.fallback_host] if args.fallback_host else [])
    power = OperatorPower() if args.power == "operator" else SwitchPower(args.power_off_cmd, args.power_on_cmd)
    world = RealWorld(args.serial, hosts)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"schema": "req_1120.zybo.temperature/1", "requirement": "Z2", "serial": args.serial,
                "condition": args.condition, "power_off_basis": power.basis, "min_power_off_s": args.min_off_s,
                "rounds_requested": args.rounds, "rounds": [], "invalid_rounds": [], "started_utc": utc(),
                "challenge_set": "frozen K, raw SHA-256 6743daa6...acd5, read once per family per power-up",
                "read_order": "apuf_lanes then ff_lanes, one read each",
                "raw_response_processing": "save wire bytes 0..3 in lane order; drop the service's XOR byte"}

    for step in [s for s in args.steps.split(",") if s]:
        print("[{}] step {}".format(utc(), step), flush=True)
        if step == "prepare":
            host = world.host_up()
            if not host:
                raise SystemExit("the board is not reachable on %s" % hosts)
            remote = Remote(host, logs)
            remote.run(["mkdir", "-p", "-m", "700", REMOTE_DIR + "/out"])
            remote.put([HERE / "zybo_board.py"], REMOTE_DIR)
            snap = snapshot(remote)
            print(json.dumps({k: snap[k] for k in ("boot_id", "eth0_mac", "fpga_manager_state", "bitstream_sha256",
                                                    "research_firmware_sha256", "research_processes")}, indent=2))
            if not (snap["research_processes"] and snap["fpga_manager_state"] == "operating"):
                raise SystemExit("the board must run the frozen research service before the first power cut")
        elif step == "campaign":
            work.mkdir(parents=True, exist_ok=True)
            while len(meta["rounds"]) < args.rounds:
                round_no = len(meta["rounds"]) + len(meta["invalid_rounds"]) + 1
                host = world.host_up()
                if not host:
                    raise SystemExit("the board is not reachable on %s; fix DHCP or the fallback route first" % hosts)
                remote = Remote(host, logs)
                pre = snapshot(remote)
                rec = {"round": round_no, "boot_id_before": pre["boot_id"],
                       "uptime_days_before": pre["telemetry"]["uptime_s"] / 86400.0}
                try:
                    rec.update(run_round(world, power, round_no, args.min_off_s))
                    remote = Remote(rec["host_after_power_up"], logs)
                    post = snapshot(remote)
                    if post["boot_id"] == pre["boot_id"]:
                        raise Invalid("boot_id did not change: the board was not restarted", rec)
                    rec["boot_id"] = post["boot_id"]
                    rec["uptime_s_shortly_after_power_up"] = post["telemetry"]["uptime_s"]
                    if post["telemetry"]["uptime_s"] > 3600:
                        raise Invalid("uptime of the 'fresh' boot is over an hour", rec)
                    rec["power_off_basis"] = power.basis
                    rec["pl_programmed_by_tool"] = start_research_service(remote)
                    t_started = time.time()               # host wall time of the snapshot below (±0.3 s)
                    started = snapshot(remote)
                    if not frozen_service_running(started):
                        raise Invalid("the research service after power-up is not the frozen one with PL programming", rec)
                    name = "boot_%02d" % round_no
                    remote.run(["python3", REMOTE_DIR + "/zybo_board.py", "read", "--challenges",
                                REMOTE_DIR + "/challenges_K.npy", "--repeats", "1", "--label",
                                "%s_powerup_%02d" % (args.condition, round_no), "--out", REMOTE_DIR + "/out/" + name],
                               log_name="%s.log" % name, timeout=300)
                    dest = fetch_round(remote, name, work)
                    read_meta = json.loads((dest / "meta.json").read_text())
                    if not read_meta["complete"] or read_meta["boot_id_start"] != post["boot_id"]:
                        raise Invalid("the read after power-up did not complete in the same boot", rec)
                    reads = {r["family"]: r for r in read_meta["reads"]}
                    # power-on to read start: (host wall time of the read) - (host wall time when the board was first
                    # seen back); the first sighting is SSH answering or the USB JTAG chip re-enumerating.
                    rec["power_on_to_read_start_s"] = {
                        f: (t_started + reads[f]["read_start_board_monotonic_s"] - started["telemetry"]["board_monotonic_s"])
                        - rec["power_back_host_wall"] for f in reads}
                    rec["power_on_to_read_start_note"] = (
                        "host wall time of the read start minus the host wall time at which the board was first seen "
                        "back (SSH answering, or the USB JTAG chip re-enumerating when it had disappeared); the first "
                        "sighting is at most one poll interval (0.5-1 s) late; board_uptime_at_read_start_s counts "
                        "from kernel start and does not include BootROM/FSBL/U-Boot")
                    rec["board_uptime_at_read_start_s"] = {f: reads[f]["before"]["uptime_s"] for f in reads}
                    rec["xadc_temperature_C"] = {f: [reads[f]["before"]["xadc_die_temperature_C"],
                                                     reads[f]["after"]["xadc_die_temperature_C"]] for f in reads}
                    rec["xadc_vccint_mV"] = {f: reads[f]["before"]["xadc_vccint_mV"] for f in reads}
                    rec["read_duration_s"] = {f: reads[f]["read_duration_s"] for f in reads}
                    rec["bitstream_sha256"] = started["bitstream_sha256"]
                    rec["research_firmware_sha256"] = started["research_firmware_sha256"]
                    rec["research_service_argv"] = started["research_processes"][0]["argv"]
                    rec["research_service_pid"] = started["research_processes"][0]["pid"]
                    rec["eth0_addresses_after_power_up"] = started["eth0_addresses"]
                    rec["directory"] = "rounds/" + name
                    rec["valid"] = True
                    meta["rounds"].append(rec)
                    print("round %d valid: power-off %.1f s, power-on to read %.1f s / %.1f s, %.1f C" % (
                        round_no, rec["power_off_duration_s"],
                        rec["power_on_to_read_start_s"]["apuf_lanes"], rec["power_on_to_read_start_s"]["ff_lanes"],
                        rec["xadc_temperature_C"]["apuf_lanes"][0]), flush=True)
                except Invalid as exc:
                    bad = exc.record
                    bad.update(reason=str(exc), valid=False, utc=utc())
                    meta["invalid_rounds"].append(bad)
                    print("round %d INVALID: %s" % (round_no, exc), flush=True)
                meta["updated_utc"] = utc()
                write_json(meta_path, meta)
        elif step == "rehearsal":
            # Two things can be checked without cutting power: (A) a round in which nothing is switched must
            # be refused; (B) the path every real round takes after the power comes back (start the frozen
            # research service with PL programming, read K once, copy the result).
            rehearsal = {}
            host = world.host_up()
            if not host:
                raise SystemExit("the board is not reachable on %s" % hosts)
            rlog = logs / "rehearsal"
            remote = Remote(host, rlog)
            remote.run(["mkdir", "-p", "-m", "700", REMOTE_DIR + "/out"])
            remote.put([HERE / "zybo_board.py"], REMOTE_DIR)
            pre = snapshot(remote)
            rehearsal["board"] = {"boot_id": pre["boot_id"], "eth0_mac": pre["eth0_mac"], "host": host}
            rehearsal["usb_jtag_serial_present"] = world.usb_present()

            class StubPower(object):
                basis = "rehearsal stub: switches nothing"
                off = on = staticmethod(lambda round_no, min_off: None)
            try:
                pc_rec = run_round(world, StubPower(), 0, args.min_off_s, {"wait_off": 6, "wait_on": 6, "wait_ssh": 6})
                raise SystemExit("rehearsal A failed: a round without any outage was accepted: %r" % pc_rec)
            except Invalid as exc:
                if "not powered off" not in str(exc):
                    raise
                rehearsal["A_round_without_outage"] = {"outcome": "refused", "reason": str(exc),
                                                       "record": exc.record}
            stop = json.loads(remote.get(["python3", REMOTE_DIR + "/zybo_board.py", "snapshot"]))["research_processes"]
            if len(stop) != 1:
                raise SystemExit("expected one research service before the rehearsal")
            remote.run(["sudo", "-n", "kill", "-TERM", str(stop[0]["pid"])], log_name="stop_research.log")
            end = time.monotonic() + 20
            while time.monotonic() < end and ":3040 " in remote.get(["ss", "-ltn"]):
                time.sleep(0.3)
            gone = ":3040 " not in remote.get(["ss", "-ltn"])
            started_by_tool = start_research_service(remote)
            started = snapshot(remote)
            name = "boot_rehearsal"
            remote.run(["python3", REMOTE_DIR + "/zybo_board.py", "read", "--challenges",
                        REMOTE_DIR + "/challenges_K.npy", "--repeats", "1", "--label",
                        "rehearsal_same_boot_not_a_power_cycle", "--out", REMOTE_DIR + "/out/" + name],
                       log_name=name + ".log", timeout=300)
            dest = fetch_round(remote, name, work)
            read_meta = json.loads((dest / "meta.json").read_text())
            rehearsal["B_post_power_up_path"] = {
                "research_listener_gone_before_restart": gone,
                "research_service_started_by_tool": started_by_tool,
                "frozen_service_running": frozen_service_running(started),
                "research_service_argv": started["research_processes"][0]["argv"],
                "research_service_pid": started["research_processes"][0]["pid"],
                "read_complete": read_meta["complete"],
                "same_boot_as_before": read_meta["boot_id_start"] == pre["boot_id"],
                "array_shape": read_meta["array_shape"],
                "xadc_temperature_C": {r["family"]: r["before"]["xadc_die_temperature_C"] for r in read_meta["reads"]},
                "read_duration_s": {r["family"]: r["read_duration_s"] for r in read_meta["reads"]},
                "raw_data": "rounds/" + name,
                "note": ("same boot: the PL was reconfigured from the frozen bitstream by the tool, which is NOT a "
                         "power cycle; this read is a rehearsal of the post-power-up path and is not Z2 data")}
            write_json(rlog / "rehearsal.json", rehearsal)
            print(json.dumps(rehearsal, indent=2, ensure_ascii=False))
        elif step == "assemble":
            if len(meta["rounds"]) < args.rounds:
                raise SystemExit("only %d of %d valid rounds; not assembling" % (len(meta["rounds"]), args.rounds))
            meta["assembled_utc"] = utc()
            assemble(logs, zybo, args.serial, args.condition, meta)
            print("assembled %d rounds into %s" % (args.rounds, zybo / "temperature"))
    write_json(meta_path, meta)


if __name__ == "__main__":
    main()
