#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment-mode firmware image of the Zybo root module (req_1120 Z4-Z8). Run as root on the board.

  python3 module_serve.py --enrollment enrollment.json --k challenges_K.npy \
      --research puf_server.py --bit system_wrapper_apuf_ff_20260917.bit \
      --iterations C --port 3041 [--secret-store ocm|memory] [--program|--no-program]

The whole protocol is one JSON object per line in, one JSON object per line out:

  {"op": "unlock", "passphrase_hex": ".."}  -> {"success": true, "device_id": "..", "timing": {"acquire_s", "fe_s", "kdf_s"}}
                                               or {"success": false, "device_id": "..", "error": "reconstruction"}
  {"op": "derive", "label_hex": ".."}       -> {"success": true, "device_id": "..", "key_hex": "<64 hex>"}
                                               or {"success": false, "device_id": "..", "error": "locked"}
  {"op": "lock"}                            -> {"success": true, "device_id": ".."}
  anything else                             -> {"success": false, "device_id": "..", "error": "unsupported_request"}

There is no other command. Research-protocol frames (magic "PUF1") are rejected as soon as the
magic arrives. K is read from --k, verified by SHA-256 before the service listens and again before
every evaluation. r, w, w' and k_u are never written anywhere by this file; the audit log on stderr
holds event names, timings and counters only.

Test-only options, never used on a deployed board: --backend replay/sim and --secret-store memory.
"""
import argparse
import hashlib
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import module_core as core  # noqa: E402

MAX_LINE = 16384
MAX_HEX_BYTES = 4096


def now():
    return time.monotonic()


class ModuleServer(object):
    def __init__(self, source, fields, tag_hex, store, iterations, log, idle_timeout=10.0):
        self.source, self.fields, self.tag_hex = source, fields, tag_hex
        self.store, self.iterations, self.log = store, int(iterations), log
        self.idle_timeout = idle_timeout
        self.device_id = fields["device_id"]
        self.bch = core.Bch(fields["t"])
        self.bch.selftest()
        self.ecc_hex, self.seed_hex = fields["ecc_hex"], fields["seed_hex"]
        self.sock = None
        self._stopping = False
        self.counters = {"unlock_ok": 0, "unlock_failed": 0, "derive": 0, "derive_locked": 0, "lock": 0,
                         "unsupported": 0}

    # ---- the three operations -------------------------------------------------------
    def op_unlock(self, passphrase):
        self.store.erase_ku()                      # an unlock attempt always starts from a locked module
        t0 = now()
        w2 = core.measure_word(self.source, self.fields["lanes"])          # step 1: one read of K -> w'
        t1 = now()
        data2 = bytearray(core.pack_bits(w2))
        core.zero([w2])
        data = self.bch.decode_data(data2, bytearray(bytes.fromhex(self.ecc_hex)))   # step 2: BCH(t, 13)
        core.zero([data2])
        ok = False
        if data is not None:
            root = core.toeplitz_root(data, bytes.fromhex(self.seed_hex))
            core.zero([data])                      # w is overwritten with zeros
            self.store.put_r(root)
            core.zero([root])
            ok = self.store.use_r(lambda r: core.tag_matches(r, self.fields, self.tag_hex))
        t2 = now()
        if not ok:                                 # same reply for a failed decode and a failed tag
            self.store.erase_r()
            return {"success": False, "device_id": self.device_id, "error": "reconstruction"}, \
                {"acquire_s": t1 - t0, "fe_s": t2 - t1, "kdf_s": 0.0}
        ku = self.store.use_r(lambda r: core.unlock_key(r, passphrase, self.iterations))   # step 3
        t3 = now()
        self.store.put_ku(ku)
        core.zero([ku])
        self.store.erase_r()                       # step 4: r overwritten; w and w' already zeroed
        core.scrub()
        timing = {"acquire_s": t1 - t0, "fe_s": t2 - t1, "kdf_s": t3 - t2}
        return {"success": True, "device_id": self.device_id, "timing": timing}, timing

    def op_derive(self, label):
        if not self.store.has_ku:
            return {"success": False, "device_id": self.device_id, "error": "locked"}
        key = self.store.use_ku(lambda ku: core.hmac_sha256(ku, label))
        core.zero([label])                         # the label copy; the key is the reply itself
        core.scrub(16)
        return {"success": True, "device_id": self.device_id, "key_hex": key.hex()}

    def op_lock(self):
        self.store.erase_ku()
        self.store.erase_r()
        return {"success": True, "device_id": self.device_id}

    # ---- request validation -----------------------------------------------------------
    @staticmethod
    def _hex_field(req, name):
        value = req[name]
        if not isinstance(value, str) or len(value) % 2 or len(value) > 2 * MAX_HEX_BYTES:
            raise ValueError
        raw = bytearray(bytes.fromhex(value))
        if raw.hex() != value.lower():
            raise ValueError                      # not canonical hex text
        return raw

    def reject(self, reason, op=None):
        """The one answer to everything that is not exactly one of the three requests."""
        self.counters["unsupported"] += 1
        info = {"event": "unsupported_request", "reason": reason, "evaluations": self.source.evaluations}
        if isinstance(op, str) and len(op) <= 24 and all(32 <= ord(ch) < 127 for ch in op):
            info["op"] = op                          # the op name only, never any other request content
        self.log(**info)
        return {"success": False, "device_id": self.device_id, "error": "unsupported_request"}, info

    def handle_line(self, raw):
        """One request line (without the newline) -> (reply dict, log fields)."""
        try:
            req = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return self.reject("not JSON")
        if not isinstance(req, dict) or not isinstance(req.get("op"), str):
            return self.reject("not a JSON object with an op")
        op = req["op"]
        try:
            if op == "unlock" and set(req) == {"op", "passphrase_hex"}:
                passphrase = self._hex_field(req, "passphrase_hex")
            elif op == "derive" and set(req) == {"op", "label_hex"}:
                label = self._hex_field(req, "label_hex")
            elif op == "lock" and set(req) == {"op"}:
                pass
            else:
                return self.reject("unknown op or unexpected fields", op)
        except ValueError:
            return self.reject("malformed field", op)
        before = self.source.evaluations
        if op == "unlock":
            reply, timing = self.op_unlock(passphrase)
            core.zero([passphrase])
            name = "unlock_ok" if reply["success"] else "unlock_failed"
            self.counters[name] += 1
            info = {"event": name, "timing": timing, "evaluations": self.source.evaluations - before}
        elif op == "derive":
            reply = self.op_derive(label)
            name = "derive" if reply["success"] else "derive_locked"
            self.counters[name] += 1
            info = {"event": name, "evaluations": self.source.evaluations - before}
        else:
            reply = self.op_lock()
            self.counters["lock"] += 1
            info = {"event": "lock", "evaluations": self.source.evaluations - before}
        self.log(**info)                          # event name, timings, counters: never request content
        return reply, info

    # ---- connections ------------------------------------------------------------------
    def handle_connection(self, conn, addr):
        buf = bytearray()
        conn.settimeout(self.idle_timeout)
        state = {"lines": 0}
        try:
            while not self._stopping:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                if bytes(buf[:4]) == b"PUF1":      # research-protocol frame: reject at once
                    reply, _ = self.reject("research-protocol frame (magic PUF1)")
                    self._send(conn, reply, state)
                    break
                while b"\n" in buf:
                    line, _, rest = bytes(buf).partition(b"\n")
                    del buf[:]
                    buf.extend(rest)
                    reply, _ = self.handle_line(line.strip(b"\r"))
                    self._send(conn, reply, state)
                if len(buf) > MAX_LINE:
                    reply, _ = self.reject("line longer than %d bytes without a newline" % MAX_LINE)
                    self._send(conn, reply, state)
                    break
        except socket.timeout:
            self.log("connection_idle_timeout", lines=state["lines"])
        except Exception as exc:                 # a bad peer never stops the module; no detail is logged
            self.log("connection_error", error=type(exc).__name__)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _send(self, conn, reply, state):
        state["lines"] += 1
        conn.sendall((json.dumps(reply, separators=(",", ":")) + "\n").encode("utf-8"))

    def bind(self, host, port, backlog=4):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(backlog)
        self.sock = srv

    def serve_forever(self):
        failures = 0
        while not self._stopping:
            try:
                conn, addr = self.sock.accept()
            except OSError as exc:
                if not self._stopping:
                    self.log("accept_error", error=type(exc).__name__)
                    failures += 1
                    if failures > 50:
                        break                     # a broken socket: stop instead of spinning
                    time.sleep(0.1)
                continue
            failures = 0
            try:
                self.handle_connection(conn, addr)
            except Exception as exc:
                self.log("handler_error", error=type(exc).__name__)
                try:
                    conn.close()
                except Exception:
                    pass

    def shutdown(self):
        self._stopping = True
        try:
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass
        self.store.close()


def make_source(args, k):
    if args.backend == "mmio":
        if os.geteuid() != 0:
            raise core.ModuleError("--backend mmio needs root (/dev/mem and FPGA Manager)")
        return core.MmioSource(k, args.research, args.bit, not args.no_program, int(args.base, 0))
    if args.backend == "replay":
        return core.ReplaySource(args.replay_apuf, args.replay_ff, args.replay_start, args.replay_count)
    return core.SimSource(int(args.sim_seed), float(args.sim_flip))


def build(args, log):
    enrollment_path = Path(args.enrollment)
    raw = enrollment_path.read_bytes()
    enrollment = json.loads(raw.decode("utf-8"))
    fields = core.validate_enrollment(enrollment)
    tag_hex = enrollment["tag_hex"]
    if args.device_id and args.device_id != fields["device_id"]:
        raise core.ModuleError("--device-id does not match enrollment.json")
    if args.iterations < 1:
        raise core.ModuleError("--iterations must be positive")
    k = core.load_k(args.k)
    store = (core.open_ocm_store(int(args.ocm_address, 0)) if args.secret_store == "ocm"
             else core.MemorySecretStore())
    try:
        source = make_source(args, k)
    except Exception:
        store.close()
        raise
    server = ModuleServer(source, fields, tag_hex, store, args.iterations, log)
    server.provenance = {
        "enrollment_sha256": hashlib.sha256(raw).hexdigest(),
        "module_serve_sha256": core.sha256_file(str(HERE / "module_serve.py")),
        "module_core_sha256": core.sha256_file(str(HERE / "module_core.py")),
        "k_raw_sha256": core.K_SHA, "k_file_sha256": core.sha256_file(args.k),
        "research_firmware_sha256": core.sha256_file(args.research) if args.research else None,
        "bitstream_sha256": core.sha256_file(args.bit) if args.bit else None,
        "backend": source.backend, "pl_programmed_by_module": getattr(source, "pl_programmed", None),
        "secret_store": store.description, "store_leftover_erased": store.leftover_erased,
        "lanes": fields["lanes"], "t": fields["t"], "m": fields["m"], "iterations": args.iterations,
        "python": sys.version.split()[0], "pid": os.getpid(), "euid": os.geteuid(),
    }
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enrollment", required=True)
    ap.add_argument("--k", required=True)
    ap.add_argument("--iterations", type=int, required=True, help="c: KDF iteration count from calibration.json")
    ap.add_argument("--device-id", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=3041)
    ap.add_argument("--secret-store", choices=("ocm", "memory"), default="ocm")
    ap.add_argument("--ocm-address", default="0xFFFE8000")
    ap.add_argument("--backend", choices=("mmio", "replay", "sim"), default="mmio")
    ap.add_argument("--research", default=None, help="frozen puf_server.py (mmio)")
    ap.add_argument("--bit", default=None, help="frozen bitstream (mmio)")
    ap.add_argument("--base", default="0x40000000")
    ap.add_argument("--no-program", action="store_true",
                    help="reuse the PL that is already configured (default: program it from the frozen bitstream)")
    ap.add_argument("--replay-apuf", default=None)
    ap.add_argument("--replay-ff", default=None)
    ap.add_argument("--replay-start", type=int, default=0)
    ap.add_argument("--replay-count", type=int, default=1)
    ap.add_argument("--sim-seed", default="1")
    ap.add_argument("--sim-flip", default="0.002")
    ap.add_argument("--idle-timeout", type=float, default=10.0)
    args = ap.parse_args()

    def log(event, **fields):
        rec = {"board_monotonic_s": now(), "event": event}
        rec.update(fields)
        sys.stderr.write(json.dumps(rec, sort_keys=True) + "\n")
        sys.stderr.flush()

    server = build(args, log)
    log("startup", **server.provenance)
    server.bind(args.host, args.port, backlog=4)
    log("listening", host=args.host, port=args.port)

    def stop(signum, frame):
        log("signal", signum=signum, **server.counters)
        server.shutdown()
        log("stopped", **server.counters)
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        server.shutdown()
        log("stopped", **server.counters)


if __name__ == "__main__":
    main()
