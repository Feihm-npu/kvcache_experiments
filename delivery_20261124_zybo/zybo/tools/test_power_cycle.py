#!/usr/bin/env python3
"""State-machine tests of the Z2 power-cycle tool with a fake board and a fake clock.

  python3 -m unittest -v test_power_cycle

Nothing here touches a board: the round logic (power_cycle_campaign.run_round) is driven by a
scripted world whose SSH/USB answers and timing are given, and `assemble` is run on synthetic
arrays in a temporary directory.
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import power_cycle_campaign as pc  # noqa: E402

HOST = "xilinx@192.168.1.2"


class FakeWorld(object):
    """up -> down (at off, ssh lingers down_delay seconds) -> booting (at on) -> ssh answers ssh_after later."""

    def __init__(self, usb_tracks_power=True, ssh_after=40.0, never_down=False, never_back=False,
                 early_return_at=None, down_delay=0.0):
        self.t, self.state = 1000.0, "up"
        self.usb_tracks_power, self.ssh_after = usb_tracks_power, ssh_after
        self.never_down, self.never_back, self.early_return_at, self.down_delay = never_down, never_back, early_return_at, down_delay
        self.events, self.cut_at, self.ssh_ready = [], None, None

    def clock(self):
        return self.t

    wall = staticmethod(time.time)

    def sleep(self, seconds):
        self.t += seconds
        if self.state == "down" and self.early_return_at is not None and self.t >= self.early_return_at:
            self.state = "up"

    def power_gone(self):
        return (not self.never_down and self.cut_at is not None and self.t >= self.cut_at + self.down_delay)

    def usb_present(self):
        return not self.usb_tracks_power or not self.power_gone() or self.state == "booting"

    def host_up(self):
        if not self.power_gone():
            return HOST
        if self.state == "booting" and self.t >= self.ssh_ready:
            return HOST
        if self.state == "up":
            return HOST
        return None

    def cut(self):
        self.events.append(("off", self.t))
        self.cut_at = self.t
        if not self.never_down:
            self.state = "down"

    def restore(self):
        self.events.append(("on", self.t))
        if self.never_back:
            self.state = "down"
            return
        self.state = "booting"
        self.ssh_ready = self.t + self.ssh_after


class FakePower(object):
    basis = "fake"

    def __init__(self, world):
        self.world = world

    def off(self, round_no, min_off):
        self.world.cut()

    def on(self, round_no, min_off):
        self.world.restore()


class RoundLogic(unittest.TestCase):
    def test_normal_cycle_is_valid_and_timed(self):
        w = FakeWorld()
        rec = pc.run_round(w, FakePower(w), 1, 60.0)
        self.assertGreaterEqual(rec["power_off_duration_s"], 60.0)
        off_t = [t for e, t in w.events if e == "off"][0]
        on_t = [t for e, t in w.events if e == "on"][0]
        self.assertGreaterEqual(on_t - off_t, 60.0)                    # the tool itself holds the outage
        self.assertAlmostEqual(rec["power_on_request_to_back_s"], 0.0, delta=0.6)   # USB chip is back first
        self.assertAlmostEqual(rec["back_to_ssh_s"], 40.0, delta=1.6)
        self.assertIsNotNone(rec["t_usb_gone"])
        self.assertEqual(rec["host_after_power_up"], HOST)
        self.assertAlmostEqual(rec["power_back_host_wall"], time.time() - (w.t - rec["t_power_back"]), delta=1.0)

    def test_board_back_before_the_minimum_is_invalid(self):
        w = FakeWorld(early_return_at=1020.0)                           # power restored by hand after 20 s
        with self.assertRaises(pc.Invalid) as info:
            pc.run_round(w, FakePower(w), 1, 60.0)
        self.assertIn("before the minimum", str(info.exception))
        self.assertEqual([e for e, t in w.events], ["off"])           # the tool never asked for power back
        self.assertFalse(info.exception.record["valid"])

    def test_board_that_stays_up_is_invalid(self):
        w = FakeWorld(never_down=True)
        with self.assertRaises(pc.Invalid) as info:
            pc.run_round(w, FakePower(w), 1, 60.0, {"wait_off": 10, "wait_on": 10, "wait_ssh": 10})
        self.assertIn("not powered off", str(info.exception))
        self.assertIsNone(info.exception.record["t_ssh_gone"])

    def test_board_that_never_returns_is_invalid(self):
        w = FakeWorld(never_back=True)
        with self.assertRaises(pc.Invalid) as info:
            pc.run_round(w, FakePower(w), 1, 60.0, {"wait_off": 10, "wait_on": 30, "wait_ssh": 10})
        self.assertIn("did not come back", str(info.exception))
        self.assertGreaterEqual(w.t - 1000.0, 60.0)                    # the minimum was still held first

    def test_usb_that_does_not_track_power_is_ignored(self):
        w = FakeWorld(usb_tracks_power=False)
        rec = pc.run_round(w, FakePower(w), 1, 60.0)
        self.assertIsNone(rec["t_usb_gone"])
        self.assertTrue(rec["usb_jtag_present_before"])
        self.assertGreaterEqual(rec["power_on_request_to_back_s"], 40.0)   # SSH was the first sign of life

    def test_minimum_counts_from_the_loss_of_ssh(self):
        w = FakeWorld(down_delay=3.0)                                   # SSH lingers 3 s after the cut
        rec = pc.run_round(w, FakePower(w), 1, 60.0)
        self.assertGreaterEqual(rec["power_off_duration_s"], 60.0)
        self.assertAlmostEqual(rec["t_ssh_gone"] - rec["t_power_off_request"], 3.0, delta=0.6)


class Assemble(unittest.TestCase):
    def test_stacks_valid_rounds_in_order(self):
        tmp = Path(tempfile.mkdtemp())
        work, zybo = tmp / "rounds", tmp / "zybo"
        work.mkdir()
        rows = {"apuf_lanes": [], "ff_lanes": []}
        meta = {"rounds": []}
        for i, round_no in enumerate([1, 3, 4]):                       # round 2 was invalid: absent
            d = work / ("boot_%02d" % round_no)
            d.mkdir()
            for family in rows:
                arr = np.full((1, 4096, 4), round_no, dtype=np.int8)
                np.save(str(d / (family + ".npy")), arr, allow_pickle=False)
            meta["rounds"].append({"round": round_no, "directory": "rounds/" + d.name})
        pc.assemble(tmp, zybo, "SERIAL", "room", meta)               # tmp plays the logs directory
        for family in rows:
            arr = np.load(str(zybo / "temperature" / ("SERIAL_room_%s.npy" % family)), allow_pickle=False)
            self.assertEqual(arr.shape, (3, 4096, 4))
            self.assertEqual(arr.dtype, np.int8)
            self.assertEqual([int(arr[i, 0, 0]) for i in range(3)], [1, 3, 4])
        m = json.loads((zybo / "temperature" / "SERIAL_room_meta.json").read_text())
        self.assertEqual([r["round"] for r in m["rounds"]], [1, 3, 4])
        self.assertFalse((zybo / "temperature" / ".assembling").exists())


class FrozenService(unittest.TestCase):
    def test_service_check(self):
        good = {"fpga_manager_state": "operating",
                "bitstream_sha256": "224cc1fdd3234b070ef7f6568c21fecf1683cbe45508593597e3e2b1bb7b26dd",
                "research_firmware_sha256": "a057df2c6136e72ed9ae10be55dbcbc2ab12f8a16f15d656e7bc1e17c488c1c5",
                "research_processes": [{"argv": ["python3", pc.FROZEN_RESEARCH, "--bit", pc.FROZEN_BIT, "--port", "3040"]}]}
        self.assertTrue(pc.frozen_service_running(good))
        bad = json.loads(json.dumps(good))
        bad["research_processes"][0]["argv"].append("--no-program")
        self.assertFalse(pc.frozen_service_running(bad))
        bad = json.loads(json.dumps(good))
        bad["research_processes"].append(good["research_processes"][0])
        self.assertFalse(pc.frozen_service_running(bad))
        bad = json.loads(json.dumps(good))
        bad["bitstream_sha256"] = "00" * 32
        self.assertFalse(pc.frozen_service_running(bad))


if __name__ == "__main__":
    unittest.main()
