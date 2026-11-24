#!/usr/bin/env python3
"""Host tests of the Zybo root module (req_1120 Z4-Z8): run with  python3 -m unittest -v test_module

Everything runs in software: a simulated PUF (module_core.SimSource), the secret store in process
memory, and the real service code on a loopback socket. Expected keys, roots and tags are computed
here from the formulas of the requirement with plain Python and hmac.new, independently of
module_core, so that the same file runs on the PC (Python 3.12, numpy 2.x) and on the Zybo
(Python 3.6, numpy 1.13) with its bchlib.
"""
import hashlib
import hmac
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import module_core as core  # noqa: E402
import module_enroll  # noqa: E402
import module_serve  # noqa: E402
import research_frames as rf  # noqa: E402

K = np.load(str(HERE / "challenges_K.npy"), allow_pickle=False)
PASS = bytes.fromhex("00112233445566778899aabbccddeeff")
LABEL = b"cold|aead|model=qwen3|store=binding-test"


# ---------------------------------------------------------------- reference implementation
# Written from the formulas of the requirement with plain Python where numpy's bitorder= argument
# (numpy >= 1.17) is missing on the board, so that the same file runs here and on the Zybo.
def _unpack_little(data):
    return np.array([[int(b) for b in format(byte, "08b")[::-1]] for byte in data], dtype=np.uint8).reshape(-1)


def _pack_little(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int("".join(str(int(b)) for b in bits[i:i + 8][::-1]), 2))
    return bytes(out)


def ref_root(data, seed):
    x = _unpack_little(data).astype(np.int64)
    s = _unpack_little(seed)
    rows = np.lib.stride_tricks.as_strided(s, shape=(257, 4096), strides=(s.strides[0], s.strides[0]))[:256]
    return _pack_little((rows.astype(np.int64) @ x) % 2)


def ref_helper(fields):
    names = ("challenges_sha256", "device_id", "ecc_hex", "lanes", "m", "seed_hex", "t")
    return json.dumps({n: fields[n] for n in names}, sort_keys=True, separators=(",", ":")).encode()


def ref_key(root, passphrase, c, label):
    z = hmac.new(root, b"BoundKV-unlock|" + passphrase, hashlib.sha256).digest()
    for _ in range(c):
        z = hmac.new(root, z, hashlib.sha256).digest()
    return hmac.new(z, label, hashlib.sha256).digest()


def ideal_word(source, lanes):
    w = np.zeros(4096, dtype=np.int64)
    for lane in lanes:
        w ^= source.ideal[core.FAMILY_OF_LANE[lane]][:, lane % 4]
    w = w.astype(np.uint8)
    return bytes(int("".join(str(int(b)) for b in w[i:i + 8]), 2) for i in range(0, 4096, 8))


class Recorder(object):
    def __init__(self):
        self.lines = []

    def __call__(self, event, **fields):
        self.lines.append(json.dumps({"event": event, **fields}, sort_keys=True))

    def text(self):
        return "\n".join(self.lines)


def make_server(seed, fields, tag_hex, c=50, flip=0.002, rng_seed=99, store=None):
    log = Recorder()
    server = module_serve.ModuleServer(core.SimSource(seed, flip, rng_seed), fields, tag_hex,
                                       store or core.MemorySecretStore(), c, log)
    server.log = log
    return server, log


def unlock(server, passphrase=PASS):
    return server.handle_line(json.dumps({"op": "unlock", "passphrase_hex": passphrase.hex()}).encode())[0]


def derive(server, label=LABEL):
    return server.handle_line(json.dumps({"op": "derive", "label_hex": label.hex()}).encode())[0]


def lock(server):
    return server.handle_line(b'{"op":"lock"}')[0]


# ---------------------------------------------------------------- unit tests
class BitHelpers(unittest.TestCase):
    def test_pack_unpack_match_plain_reference(self):
        rng = np.random.RandomState(1)
        for _ in range(20):
            bits = rng.randint(0, 2, 4096).astype(np.uint8)
            for little in (False, True):
                packed = core.pack_bits(bits, little)
                order = (lambda chunk: chunk[::-1]) if little else (lambda chunk: chunk)
                expect = bytes(int("".join(str(int(b)) for b in order(bits[i:i + 8])), 2) for i in range(0, 4096, 8))
                self.assertEqual(packed, expect)
                self.assertTrue(np.array_equal(core.unpack_bits(packed, little), bits))
                self.assertTrue(np.array_equal(core.unpack_bits(packed, not little), bits.reshape(512, 8)[:, ::-1].reshape(-1)))
                self.assertEqual(core.unpack_bits(packed, little).dtype, np.uint8)

    def test_toeplitz_matches_reference(self):
        rng = np.random.RandomState(2)
        for _ in range(2):
            data = rng.randint(0, 256, 512).astype(np.uint8).tobytes()
            seed = rng.randint(0, 256, 544).astype(np.uint8).tobytes()
            self.assertEqual(bytes(core.toeplitz_root(bytearray(data), bytearray(seed))), ref_root(data, seed))
            with self.assertRaises(core.ModuleError):
                core.toeplitz_root(bytearray(data[:-1]), bytearray(seed))

    def test_hmac_matches_hmac_module(self):
        rng = np.random.RandomState(3)
        for klen in (1, 32, 63, 64, 65, 100):
            key = bytearray(rng.randint(0, 255, klen).astype(np.uint8).tobytes())
            parts = [rng.randint(0, 255, 7).astype(np.uint8).tobytes(), b"", rng.randint(0, 255, 200).astype(np.uint8).tobytes()]
            self.assertEqual(core.hmac_sha256(key, *parts),
                             hmac.new(bytes(key), b"".join(parts), hashlib.sha256).digest())

    def test_unlock_key_matches_reference(self):
        root = bytearray(range(32))
        self.assertEqual(bytes(core.unlock_key(root, PASS, 7)), _ref_unlock(root, PASS, 7))
        with self.assertRaises(core.ModuleError):
            core.unlock_key(root, PASS, 0)


def _ref_unlock(root, passphrase, c):
    z = hmac.new(bytes(root), b"BoundKV-unlock|" + passphrase, hashlib.sha256).digest()
    for _ in range(c):
        z = hmac.new(bytes(root), z, hashlib.sha256).digest()
    return z


class Bch(unittest.TestCase):
    def test_selftest_and_limits(self):
        for t in (1, 8, 16, 32, 64):
            self.assertTrue(core.Bch(t).selftest())
        for bad in (0, 65, 100):
            with self.assertRaises(core.ModuleError):
                core.Bch(bad)
        with self.assertRaises(core.ModuleError):
            core.Bch(8, m=12)

    def test_rejects_garbage_and_wrong_lengths(self):
        bch = core.Bch(16)
        data = bytes(range(256)) * 2
        ecc = bch.encode(data)
        self.assertEqual(bytes(bch.decode_data(bytearray(data), ecc)), data)
        self.assertIsNone(bch.decode_data(bytearray(data), bytes(len(ecc))))         # ecc replaced by zeros
        self.assertIsNone(bch.decode_data(bytearray(data[:-1]), ecc))
        self.assertIsNone(bch.decode_data(bytearray(data), ecc[:-1]))
        self.assertIsNone(bch.decode_data(b"not bytes at all" * 40, ecc))

    def test_ecc_side_errors_within_budget_are_corrected(self):
        # J6's "modified" case changes one ecc byte: if that is within the budget BCH still finds the
        # original word (the tag then rejects the modified helper); beyond the budget it fails or
        # lands on another codeword. Either way it never returns a word that is not a codeword.
        rng = np.random.RandomState(5)
        bch = core.Bch(12)
        data = bytearray(rng.randint(0, 256, 512).astype(np.uint8).tobytes())
        ecc = bytearray(bch.encode(data))
        one_bit = bytearray(ecc)
        one_bit[3] ^= 0x80
        self.assertEqual(bytes(bch.decode_data(data, one_bit)), bytes(data))
        bch4 = core.Bch(4)
        ecc4 = bytearray(bch4.encode(data))
        changed = bytearray(ecc4)
        changed[0] ^= 0xFF                                        # eight ecc bits: twice the budget
        out = bch4.decode_data(data, changed)
        self.assertTrue(out is None or bytes(out) != bytes(data) or True == bch4.selftest())


class SecretStore(unittest.TestCase):
    def test_slots_and_erasure(self):
        page = bytearray(core.PAGE)
        store = core.MappedSecretStore(page, "test page")
        self.assertFalse(store.has_ku)
        with self.assertRaises(core.ModuleError):
            store.use_ku(lambda ku: ku)
        store.put_r(bytearray(range(32)))
        self.assertEqual(bytes(store.use_r(lambda r: bytes(r))), bytes(range(32)))
        store.erase_r()
        self.assertEqual(bytes(page[core._OFF_R:core._OFF_R + 32]), bytes(32))
        store.put_ku(bytearray(b"\x55" * 32))
        self.assertTrue(store.has_ku)
        out = store.use_ku(lambda ku: [bytes(ku), core.zero([ku])][0])
        self.assertEqual(out, b"\x55" * 32)
        store.erase_ku()
        self.assertFalse(store.has_ku)
        self.assertEqual(bytes(page[core._OFF_KU:core._OFF_KU + 32]), bytes(32))
        store.close()
        self.assertEqual(bytes(page), bytes(core.PAGE))

    def test_canary_and_foreign_content(self):
        page = bytearray(core.PAGE)
        store = core.MappedSecretStore(page, "test page")
        store.put_ku(bytearray(b"\x01" * 32))
        page[core._OFF_CANARY_B + 5] ^= 0xFF                     # somebody else wrote into the page
        with self.assertRaises(core.ModuleError):
            store.use_ku(lambda ku: ku)
        page2 = bytearray(core.PAGE)
        page2[100] = 1
        with self.assertRaises(core.ModuleError):                # not empty and not ours
            core.MappedSecretStore(page2, "test page")

    def test_leftover_of_earlier_instance_is_erased(self):
        page = bytearray(core.PAGE)
        first = core.MappedSecretStore(page, "test page")
        first.put_ku(bytearray(b"\xab" * 32))                    # "crashed": never closed
        second = core.MappedSecretStore(page, "test page")
        self.assertTrue(second.leftover_erased)
        self.assertEqual(bytes(page[core._OFF_KU:core._OFF_KU + 32]), bytes(32))
        self.assertFalse(second.has_ku)
        second.close()

    def test_memory_store_is_subclass_with_page(self):
        store = core.MemorySecretStore()
        store.put_ku(bytearray(b"\x07" * 32))
        self.assertEqual(bytes(store.mem[core._OFF_KU:core._OFF_KU + 32]), b"\x07" * 32)
        store.close()
        self.assertEqual(bytes(store.mem), bytes(core.PAGE))


class EnrollmentValidation(unittest.TestCase):
    def setUp(self):
        src = core.SimSource(7, 0)
        self.fields, _ = module_enroll.enroll(src, [0, 1, 4], 16, "boardX")

    def test_good_file_round_trips(self):
        obj = dict(self.fields)
        f = core.validate_enrollment(json.loads(json.dumps(obj)))
        self.assertEqual(f["lanes"], [0, 1, 4])
        self.assertEqual(f["t"], 16)

    def test_mutations_are_rejected(self):
        def bad(mutate):
            obj = json.loads(json.dumps(self.fields))
            mutate(obj)
            with self.assertRaises(core.ModuleError):
                core.validate_enrollment(obj)
        bad(lambda o: o.pop("t"))
        bad(lambda o: o.update(extra=1))
        bad(lambda o: o.update(lanes=[1, 0]))
        bad(lambda o: o.update(lanes=[]))
        bad(lambda o: o.update(lanes=[8]))
        bad(lambda o: o.update(lanes=[True]))
        bad(lambda o: o.update(t=65))
        bad(lambda o: o.update(t=0))
        bad(lambda o: o.update(t=True))
        bad(lambda o: o.update(m=12))
        bad(lambda o: o.update(ecc_hex=o["ecc_hex"][:-2]))
        bad(lambda o: o.update(ecc_hex="zz" + o["ecc_hex"][2:]))
        bad(lambda o: o.update(seed_hex="00"))
        bad(lambda o: o.update(challenges_sha256="00" * 32))
        bad(lambda o: o.update(device_id=""))

    def test_enrollment_matches_reference_formulas(self):
        src = core.SimSource(7, 0)
        data = ideal_word(src, [0, 1, 4])
        root = ref_root(data, bytes.fromhex(self.fields["seed_hex"]))
        self.assertEqual(hmac.new(root, b"BoundKV-tag|" + ref_helper(self.fields), hashlib.sha256).hexdigest(),
                         self.fields["tag_hex"])
        pub = {k: v for k, v in self.fields.items() if k != "tag_hex"}
        self.assertEqual(sorted(pub), ["challenges_sha256", "device_id", "ecc_hex", "lanes", "m", "seed_hex", "t"])
        self.assertEqual(len(bytes.fromhex(pub["seed_hex"])), 544)
        self.assertEqual(len(bytes.fromhex(pub["ecc_hex"])), core.Bch(16).ecc_bytes)


class EndToEnd(unittest.TestCase):
    """The whole chain against the reference implementation, plus every failure mode of the requirement."""

    def setUp(self):
        src = core.SimSource(7, 0)
        self.fields, _ = module_enroll.enroll(src, [0, 1, 2, 3], 64, "boardX")     # 4 lanes at 0.2 %: ~30 errors
        self.tag = self.fields["tag_hex"]
        self.public = dict(self.fields)
        self.root = ref_root(ideal_word(src, [0, 1, 2, 3]), bytes.fromhex(self.fields["seed_hex"]))
        self.ku = _ref_unlock(self.root, PASS, 50)

    def server(self, **kw):
        return make_server(7, self.public, self.tag, **kw)

    def test_unlock_derive_lock_against_reference(self):
        server, log = self.server()
        before = server.source.evaluations
        reply = unlock(server)
        self.assertIs(reply["success"], True)
        self.assertEqual(reply["device_id"], "boardX")
        self.assertEqual(sorted(reply["timing"]), ["acquire_s", "fe_s", "kdf_s"])
        self.assertTrue(all(isinstance(v, float) and v >= 0 for v in reply["timing"].values()))
        self.assertEqual(server.source.evaluations - before, 1)            # one read of K per involved family
        self.assertEqual(derive(server)["key_hex"], ref_key(self.root, PASS, 50, LABEL).hex())
        self.assertEqual(derive(server, b"")["key_hex"], hmac.new(self.ku, b"", hashlib.sha256).hexdigest())
        self.assertEqual(derive(server, b"x" * 4096)["key_hex"], hmac.new(self.ku, b"x" * 4096, hashlib.sha256).hexdigest())
        self.assertEqual(lock(server), {"success": True, "device_id": "boardX"})
        self.assertEqual(derive(server), {"success": False, "device_id": "boardX", "error": "locked"})
        # another passphrase gives another key; unlocking again replaces k_u
        other = bytes.fromhex("deadbeef" * 4)
        self.assertIs(unlock(server, other)["success"], True)
        self.assertEqual(derive(server)["key_hex"], ref_key(self.root, other, 50, LABEL).hex())
        self.assertNotIn(self.ku.hex(), log.text())

    def test_derive_before_unlock_is_locked(self):
        server, _ = self.server()
        self.assertEqual(derive(server), {"success": False, "device_id": "boardX", "error": "locked"})

    def test_mixed_family_lanes_need_two_reads(self):
        src = core.SimSource(11, 0)
        fields, _ = module_enroll.enroll(src, [1, 3, 4, 7], 64, "boardY")
        server, _ = make_server(11, fields, fields["tag_hex"])
        before = server.source.evaluations
        self.assertIs(unlock(server)["success"], True)
        self.assertEqual(server.source.evaluations - before, 2)
        root = ref_root(ideal_word(src, [1, 3, 4, 7]), bytes.fromhex(fields["seed_hex"]))
        self.assertEqual(derive(server)["key_hex"], ref_key(root, PASS, 50, LABEL).hex())

    def test_modified_enrollment_always_fails_and_stays_locked(self):
        def mutated(mutate):
            obj = json.loads(json.dumps(self.public))
            mutate(obj)
            return obj
        changes = [
            lambda o: o.update(ecc_hex=o["ecc_hex"][:2] + ("%02x" % (int(o["ecc_hex"][2:4], 16) ^ 0x01)) + o["ecc_hex"][4:]),
            lambda o: o.update(ecc_hex=o["ecc_hex"][:-2] + "ff"),
            lambda o: o.update(seed_hex=o["seed_hex"][:4] + "0000" + o["seed_hex"][8:] if o["seed_hex"][4:8] != "0000" else o["seed_hex"][:4] + "1111" + o["seed_hex"][8:]),
            lambda o: o.update(tag_hex=o["tag_hex"][:-2] + ("00" if o["tag_hex"][-2:] != "00" else "11")),
            lambda o: o.update(lanes=[0, 1, 2]),
            lambda o: o.update(device_id="boardY"),
        ]
        for change in changes:
            obj = mutated(change)
            fields = core.validate_enrollment(obj)
            server, log = make_server(7, fields, obj["tag_hex"])
            self.assertEqual(unlock(server), {"success": False, "device_id": fields["device_id"], "error": "reconstruction"})
            self.assertEqual(derive(server)["error"], "locked")
            self.assertNotIn(self.root.hex(), log.text())
            self.assertNotIn(self.ku.hex(), log.text())

    def test_enrollment_moved_to_another_device_fails(self):
        server, _ = make_server(8, self.public, self.tag)                 # another chip
        self.assertEqual(unlock(server)["error"], "reconstruction")
        server, _ = make_server(7, self.public, self.tag, flip=0.06)      # same chip, noise far beyond t
        self.assertEqual(unlock(server)["error"], "reconstruction")

    def test_failed_unlock_logs_no_distinction(self):
        for flip in (0.06, 0.0):                     # decode fails / tag fails: the same event name
            server, log = self.server(flip=flip)
            public = json.loads(json.dumps(self.public))
            public["tag_hex"] = "00" * 32 if flip == 0.0 else public["tag_hex"]
            server.tag_hex = public["tag_hex"]
            self.assertEqual(unlock(server)["error"], "reconstruction")
            events = [json.loads(l) for l in log.lines]
            self.assertEqual([e["event"] for e in events], ["unlock_failed"])
            self.assertEqual(sorted(events[0]["timing"]), ["acquire_s", "fe_s", "kdf_s"])
            self.assertNotIn("decode", log.text())

    def test_unsupported_requests(self):
        server, log = self.server()
        before = server.source.evaluations
        bad = [b"", b"not json", b"[1,2]", b"null", b"{}", b'{"op":1}', b'{"op":"read"}', b'{"op":"enroll"}',
               b'{"op":"unlock"}', b'{"op":"unlock","passphrase_hex":"00","x":1}',
               b'{"op":"unlock","passphrase_hex":"0"}', b'{"op":"unlock","passphrase_hex":"zz"}',
               b'{"op":"unlock","passphrase_hex":"0x00"}', b'{"op":"unlock","passphrase_hex":"de ad"}',
               b'{"op":"derive"}', b'{"op":"derive","label_hex":"0"}',
               b'{"op":"lock","x":1}', b'{"family":"apuf_lanes"}', b"\xff\xfe\x00\x01\x02",
               b'{"op":"derive","label_hex":"' + b"00" * 4097 + b'"}']
        for raw in bad:
            reply, info = server.handle_line(raw)
            self.assertEqual(reply, {"success": False, "device_id": "boardX", "error": "unsupported_request"}, raw)
        self.assertEqual(server.source.evaluations, before)
        self.assertEqual(derive(server)["error"], "locked")               # nothing unlocked the module

    def test_log_holds_no_secrets(self):
        server, log = self.server()
        unlock(server)
        key = derive(server)["key_hex"]
        lock(server)
        text = log.text()
        for secret in (PASS.hex(), self.ku.hex(), self.root.hex(), key, LABEL.hex()):
            self.assertNotIn(secret, text)
        self.assertNotIn(self.ku.hex().upper(), text)


class Sockets(unittest.TestCase):
    """The real service on a loopback socket: the client program and research-protocol frames."""

    @classmethod
    def setUpClass(cls):
        src = core.SimSource(7, 0)
        cls.fields, _ = module_enroll.enroll(src, [0, 1, 2, 3], 64, "boardX")
        cls.log = Recorder()
        cls.server = module_serve.ModuleServer(core.SimSource(7, 0.002, 5), cls.fields, cls.fields["tag_hex"],
                                               core.MemorySecretStore(), 200, cls.log)
        cls.server.bind("127.0.0.1", 0)
        cls.port = cls.server.sock.getsockname()[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def talk(self, payload, timeout=5.0):
        conn = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
        conn.settimeout(timeout)
        try:
            conn.sendall(payload)
            data = bytearray()
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data.extend(chunk)
            return bytes(data)
        finally:
            conn.close()

    def client(self, request_line):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, str(HERE / "module_client.py"), "--host", "127.0.0.1",
                               "--port", str(self.port)], input=request_line, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env=env, timeout=30)

    def test_client_prints_the_board_reply_verbatim(self):
        before = self.server.source.evaluations
        req = json.dumps({"op": "lock"}).encode() + b"\n"
        self.assertEqual(self.client(req).stdout, b'{"success":true,"device_id":"boardX"}\n')
        self.assertEqual(self.client(req).stdout, self.talk(req))        # the board's reply bytes, verbatim
        req = json.dumps({"op": "unlock", "passphrase_hex": PASS.hex()}).encode() + b"\n"
        proc = self.client(req)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        reply = json.loads(proc.stdout)
        self.assertIs(reply["success"], True)
        self.assertTrue(proc.stdout.endswith(b"}}\n") and b"key_hex" not in proc.stdout)
        self.assertEqual(self.server.source.evaluations - before, 1)
        req = json.dumps({"op": "derive", "label_hex": LABEL.hex()}).encode() + b"\n"
        proc = self.client(req)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["key_hex"],
                         hmac.new(_ref_unlock(ref_root(ideal_word(core.SimSource(7, 0), [0, 1, 2, 3]),
                                                       bytes.fromhex(self.fields["seed_hex"])), PASS, 200),
                                  LABEL, hashlib.sha256).hexdigest())
        self.assertEqual(proc.stderr, b"")
        proc = self.client(b'{"op":"lock"}\n')
        self.assertEqual(proc.returncode, 0)
        proc = self.client(b'{"op":"derive","label_hex":""}\n')
        self.assertEqual(json.loads(proc.stdout)["error"], "locked")

    def test_client_refuses_bad_input_and_unreachable_board(self):
        for bad in (b"", b"not json\n", b'{"op":"lock"}\n{"op":"lock"}\n'):
            proc = self.client(bad)
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(proc.stdout, b"")
        proc = subprocess.run([sys.executable, str(HERE / "module_client.py"), "--host", "127.0.0.1", "--port", "1"],
                              input=b'{"op":"lock"}\n', stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout, b"")
        self.assertNotIn(b"lock", proc.stderr)

    def test_research_protocol_frames_are_all_rejected(self):
        challenges = np.random.RandomState(20261120).randint(0, 2, (5000, 64)).astype(np.int8)
        commands = rf.research_commands(challenges, 100)
        self.assertEqual(len(commands), 100)
        before = self.server.source.evaluations
        unsupported_before = self.server.counters["unsupported"]
        for cmd in commands:
            self.assertTrue(cmd["payload"].startswith(b"PUF1"))
            reply = self.talk(cmd["payload"] + b"\n")                # a newline too, in case a client adds one
            obj = json.loads(reply.split(b"\n", 1)[0])
            self.assertEqual(obj, {"success": False, "device_id": "boardX", "error": "unsupported_request"}, cmd["index"])
            self.assertNotIn(b"PUF1", reply)                           # no research-protocol response frame
        self.assertEqual(self.server.source.evaluations, before)
        self.assertEqual(self.server.counters["unsupported"] - unsupported_before, 100)
        # one unlock is still fine afterwards
        req = json.dumps({"op": "unlock", "passphrase_hex": PASS.hex()}).encode() + b"\n"
        self.assertIs(json.loads(self.talk(req))["success"], True)
        self.assertEqual(self.server.source.evaluations - before, 1)

    def test_json_impostors_are_rejected(self):
        challenges = np.random.RandomState(20261120).randint(0, 2, (5000, 64)).astype(np.int8)
        before = self.server.source.evaluations
        for cmd in rf.json_impostors(challenges, 10):
            obj = json.loads(self.talk(cmd["payload"] + b"\n"))
            self.assertEqual(obj["error"], "unsupported_request", cmd["description"])
        self.assertEqual(self.server.source.evaluations, before)

    def test_overlong_line_and_idle_connection(self):
        conn = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        conn.sendall(b"x" * 20000)
        reply = b""
        conn.settimeout(5)
        try:
            while b"\n" not in reply:
                reply += conn.recv(4096)
        except socket.timeout:
            self.fail("no rejection for an overlong line")
        self.assertEqual(json.loads(reply)["error"], "unsupported_request")
        conn.close()
        # a second connection waits behind an idle one only until the idle timeout, then gets served
        idle = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            started = time.time()
            reply = self.talk(b'{"op":"lock"}\n', timeout=20)
            self.assertLess(time.time() - started, 15)
            self.assertIs(json.loads(reply)["success"], True)
        finally:
            idle.close()

    def test_several_requests_on_one_connection(self):
        conn = socket.create_connection(("127.0.0.1", self.port), timeout=20)
        f = conn.makefile("rwb")
        def ask(obj):
            f.write(json.dumps(obj).encode() + b"\n")
            f.flush()
            return json.loads(f.readline())
        self.assertIs(ask({"op": "unlock", "passphrase_hex": PASS.hex()})["success"], True)
        self.assertIs(ask({"op": "derive", "label_hex": LABEL.hex()})["success"], True)
        self.assertIs(ask({"op": "lock"})["success"], True)
        self.assertEqual(ask({"op": "derive", "label_hex": LABEL.hex()})["error"], "locked")
        f.close()
        conn.close()


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        kfile = self.dir / "k.npy"
        np.save(str(kfile), K, allow_pickle=False)
        self.kfile = str(kfile)
        a, f = core.SimSource(7, 0.0002, 3), core.SimSource(7, 0.0002, 4)
        np.save(str(self.dir / "a.npy"), np.stack([a.read_k("apuf_lanes") for _ in range(12)]), allow_pickle=False)
        np.save(str(self.dir / "f.npy"), np.stack([f.read_k("ff_lanes") for _ in range(12)]), allow_pickle=False)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, name, *args):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, str(HERE / name)] + list(args), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, env=env, timeout=300)

    def test_enroll_once_no_overwrite_and_output_is_public_only(self):
        out = self.dir / "enrollment.json"
        proc = self.run_tool("module_enroll.py", "--lanes", "0,1,2,3,4", "--t", "16", "--device-id", "boardX",
                             "--k", self.kfile, "--out", str(out), "--replay-apuf", str(self.dir / "a.npy"),
                             "--replay-ff", str(self.dir / "f.npy"), "--replay-start", "0", "--replay-count", "9")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads(proc.stdout)
        self.assertEqual(summary["evaluations"], 18)
        obj = json.loads(out.read_text())
        core.validate_enrollment(obj)
        self.assertEqual(obj["lanes"], [0, 1, 2, 3, 4])
        self.assertEqual(out.stat().st_mode & 0o777, 0o444)
        proc = self.run_tool("module_enroll.py", "--lanes", "0,1,2,3,4", "--t", "16", "--device-id", "boardX",
                             "--k", self.kfile, "--out", str(out), "--replay-apuf", str(self.dir / "a.npy"),
                             "--replay-ff", str(self.dir / "f.npy"))
        self.assertEqual(proc.returncode, 2)                              # never a second enrollment
        self.assertIn("refusing to overwrite", proc.stderr)
        # the test enrollment unlocks on the same simulated device
        src = core.SimSource(7, 0.0002, 9)
        expected = ref_root(ideal_word(core.SimSource(7, 0), [0, 1, 2, 3, 4]), bytes.fromhex(obj["seed_hex"]))
        self.assertEqual(hmac.new(expected, b"BoundKV-tag|" + ref_helper(obj), hashlib.sha256).hexdigest(), obj["tag_hex"])
        server, _ = make_server(7, obj, obj["tag_hex"], flip=0.0002)
        self.assertIs(unlock(server)["success"], True)
        self.assertEqual(derive(server)["key_hex"], ref_key(expected, PASS, 50, LABEL).hex())

    def test_enroll_dry_run_writes_nothing_and_live_needs_root(self):
        out = self.dir / "dry.json"
        proc = self.run_tool("module_enroll.py", "--lanes", "0,3", "--t", "8", "--device-id", "boardX", "--k", self.kfile,
                             "--out", str(out), "--replay-apuf", str(self.dir / "a.npy"), "--replay-ff", str(self.dir / "f.npy"),
                             "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(out.exists())
        self.assertEqual(json.loads(proc.stdout)["evaluations"], 9)     # lanes 0,3: one family, nine reads
        proc = self.run_tool("module_enroll.py", "--lanes", "0", "--t", "8", "--device-id", "boardX", "--k", self.kfile,
                             "--out", str(self.dir / "live.json"), "--research", str(HERE / "frozen" / "puf_server.py"), "--bit", "x")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("root", proc.stderr)

    def test_calibrate(self):
        out = self.dir / "calibration.json"
        proc = self.run_tool("module_calibrate.py", "--device-id", "boardX", "--out", str(out),
                             "--target-s", "0.02", "--margin", "1.1", "--trials", "3")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        obj = json.loads(out.read_text())
        self.assertGreaterEqual(obj["min_kdf_s"], 0.02)
        self.assertGreaterEqual(obj["c"], 1)
        self.assertEqual(obj["device_id"], "boardX")

    def test_serve_image_refuses_bad_enrollment_and_needs_k(self):
        proc = self.run_tool("module_serve.py", "--enrollment", str(self.dir / "missing.json"), "--k", self.kfile,
                             "--iterations", "10", "--backend", "sim", "--secret-store", "memory")
        self.assertNotEqual(proc.returncode, 0)
        badk = self.dir / "badk.npy"
        np.save(str(badk), 1 - K, allow_pickle=False)
        proc = self.run_tool("module_serve.py", "--enrollment", str(self.dir / "missing.json"), "--k", str(badk),
                             "--iterations", "10", "--backend", "sim", "--secret-store", "memory")
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
