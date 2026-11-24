"""Module-power sampling for Jetson Orin Nano.

Reports energy over a measured window so that per-token energy can be computed
with the idle baseline stated separately, as the requirement document asks.

Two sources are tried in order. The INA3221 rails exposed under /sys/class/hwmon
are preferred because they can be polled cheaply from a background thread. If no
rail is found, ``tegrastats`` is parsed instead. The source actually used is
recorded in every result so that numbers from different machines are comparable.

Standalone check:
    python jetson_power.py --probe          # list discovered rails
    python jetson_power.py --idle-seconds 20
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import threading
import time


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def discover_rails():
    """Return [{label, power_uw_reader}] for every readable module-power rail."""
    rails = []
    for hwmon in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = _read(os.path.join(hwmon, "name")) or ""
        if "ina" not in name.lower() and "power" not in name.lower():
            continue
        for idx in range(1, 8):
            label = _read(os.path.join(hwmon, f"in{idx}_label")) \
                or _read(os.path.join(hwmon, f"power{idx}_label")) \
                or _read(os.path.join(hwmon, f"curr{idx}_label"))
            direct = os.path.join(hwmon, f"power{idx}_input")          # microwatts
            volt = os.path.join(hwmon, f"in{idx}_input")               # millivolts
            curr = os.path.join(hwmon, f"curr{idx}_input")             # milliamps
            if os.path.exists(direct):
                rails.append({"label": label or f"{name}:power{idx}", "kind": "power_uw",
                              "paths": [direct]})
            elif os.path.exists(volt) and os.path.exists(curr):
                rails.append({"label": label or f"{name}:in{idx}", "kind": "volt_curr",
                              "paths": [volt, curr]})
    return rails


def _rail_uw(rail):
    if rail["kind"] == "power_uw":
        raw = _read(rail["paths"][0])
        return float(raw) if raw is not None else None
    mv = _read(rail["paths"][0])
    ma = _read(rail["paths"][1])
    if mv is None or ma is None:
        return None
    return float(mv) * float(ma)                                        # mV * mA = uW


def _pick_module_rail(rails):
    """Prefer the whole-module input rail; fall back to the sum of all rails."""
    for wanted in ("VDD_IN", "POM_5V_IN", "VDD_SYS_GPU", "in1"):
        for rail in rails:
            if rail["label"] and wanted.lower() in rail["label"].lower():
                return [rail], f"single rail {rail['label']}"
    return rails, "sum of all discovered rails"


class _TegrastatsSampler:
    """Fallback that parses the POM/VDD field of tegrastats output."""

    PATTERN = re.compile(r"(VDD_IN|POM_5V_IN)\s+(\d+)mW")

    def __init__(self, interval_ms):
        self.interval_ms = interval_ms
        self.proc = None
        self.samples = []
        self._thread = None
        self._stop = threading.Event()

    def _loop(self):
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            m = self.PATTERN.search(line)
            if m:
                self.samples.append((time.perf_counter(), float(m.group(2)) * 1000.0))

    def start(self):
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)


class PowerSampler:
    """Context manager that integrates module power over the enclosed window.

    ``energy_j`` is the integral of sampled power. Subtract an idle baseline
    measured at the same power mode before dividing by generated tokens.
    """

    def __init__(self, interval_ms: int = 50):
        self.interval_ms = interval_ms
        self.rails = discover_rails()
        self.samples: list[tuple[float, float]] = []
        self.source = None
        self.detail = None
        self._selected = []
        self._stop = threading.Event()
        self._thread = None
        self._tegra = None
        if self.rails:
            self._selected, self.detail = _pick_module_rail(self.rails)
            self.source = "sysfs_hwmon"
        elif subprocess.call(["which", "tegrastats"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL) == 0:
            self.source = "tegrastats"
            self.detail = "VDD_IN or POM_5V_IN field"
        else:
            self.source = "unavailable"
            self.detail = "no INA3221 hwmon rail and no tegrastats on PATH"

    def _loop(self):
        while not self._stop.is_set():
            total = 0.0
            ok = False
            for rail in self._selected:
                uw = _rail_uw(rail)
                if uw is not None:
                    total += uw
                    ok = True
            if ok:
                self.samples.append((time.perf_counter(), total))
            self._stop.wait(self.interval_ms / 1000.0)

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        if self.source == "sysfs_hwmon":
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        elif self.source == "tegrastats":
            self._tegra = _TegrastatsSampler(self.interval_ms)
            self._tegra.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._t1 = time.perf_counter()
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=2)
        if self._tegra:
            self._tegra.stop()
            self.samples = self._tegra.samples
        return False

    def result(self) -> dict:
        elapsed = getattr(self, "_t1", time.perf_counter()) - getattr(self, "_t0", time.perf_counter())
        if len(self.samples) < 2:
            return {"source": self.source, "detail": self.detail, "elapsed_s": elapsed,
                    "samples": len(self.samples), "energy_j": None, "mean_power_w": None,
                    "note": "insufficient samples; energy not reported"}
        energy_uj = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            energy_uj += 0.5 * (p0 + p1) * (t1 - t0)
        energy_j = energy_uj / 1e6
        return {"source": self.source, "detail": self.detail, "elapsed_s": elapsed,
                "samples": len(self.samples), "sample_interval_ms": self.interval_ms,
                "energy_j": energy_j, "mean_power_w": energy_j / max(elapsed, 1e-9)}


def measure_idle(seconds: float, interval_ms: int = 50) -> dict:
    with PowerSampler(interval_ms=interval_ms) as s:
        time.sleep(seconds)
    out = s.result()
    out["window"] = "idle baseline"
    return out


def platform_record() -> dict:
    """Environment fields the requirement document asks to be recorded."""
    def cmd(args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "nvpmodel": cmd(["nvpmodel", "-q"]),
        "l4t_release": _read("/etc/nv_tegra_release"),
        "jetson_model": _read("/proc/device-tree/model"),
        "thermal_zones": {os.path.basename(z): _read(os.path.join(z, "temp"))
                          for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*"))},
        "power_rails_discovered": [r["label"] for r in discover_rails()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="list discovered rails and exit")
    ap.add_argument("--idle-seconds", type=float, default=0.0)
    ap.add_argument("--interval-ms", type=int, default=50)
    args = ap.parse_args()
    if args.probe:
        print(json.dumps({"rails": discover_rails(), "platform": platform_record()}, indent=2))
        return
    if args.idle_seconds > 0:
        print(json.dumps(measure_idle(args.idle_seconds, args.interval_ms), indent=2))


if __name__ == "__main__":
    main()
