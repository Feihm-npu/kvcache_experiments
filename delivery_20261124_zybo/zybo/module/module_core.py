#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core of the Zybo root module for req_1120 part Z (Z4-Z8).

Works on Python 3.6 / numpy 1.13 (the board) and on current Python / numpy (tests).
Nothing here writes a file, prints or logs. The two firmware images decide what may
leave the module: module_enroll.py (enrollment mode, run once per board) and
module_serve.py (deployment mode: unlock / derive / lock only).

Lane numbering used by `lanes` everywhere: 0-3 = apuf_lanes 0-3, 4-7 = ff_lanes 0-3.
"""
import hashlib
import hmac
import importlib.util
import mmap
import os
import secrets
import struct
import sys
import time

import numpy as np

try:
    import bchlib
except ImportError:            # on the board bchlib lives in the xilinx user site; root needs PYTHONPATH
    raise ModuleError("bchlib is not importable (pip install bchlib; as root set PYTHONPATH to xilinx's site-packages)")


sys.dont_write_bytecode = True

DATA_BITS = 4096
DATA_BYTES = DATA_BITS // 8
K_SHA = "6743daa679eaf86c3cba25c028728774bcf4a80f214b17d9b9e2e31cec55acd5"
BIT_SHA = "224cc1fdd3234b070ef7f6568c21fecf1683cbe45508593597e3e2b1bb7b26dd"
RESEARCH_SHA = "a057df2c6136e72ed9ae10be55dbcbc2ab12f8a16f15d656e7bc1e17c488c1c5"
TAG_DOMAIN = b"BoundKV-tag|"
UNLOCK_DOMAIN = b"BoundKV-unlock|"
HELPER_FIELDS = ("challenges_sha256", "device_id", "ecc_hex", "lanes", "m", "seed_hex", "t")
MAX_T = 64                      # bchlib 2.1.3 rejects BCH(t, m=13) for t > 64
FAMILY_OF_LANE = dict((lane, "apuf_lanes" if lane < 4 else "ff_lanes") for lane in range(8))
ZERO32 = bytes(32)


class ModuleError(Exception):
    """Anything that must stop the module without leaking detail."""


# ---------------------------------------------------------------- bits and bytes
_WEIGHTS = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)
_SHIFTS = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.uint8)


def pack_bits(bits, little=False):
    """0/1 array -> bytes; equals np.packbits(bits, bitorder="little" if little else "big")."""
    b = np.asarray(bits, dtype=np.uint8).reshape(-1, 8)
    if little:
        b = b[:, ::-1]
    return (b.astype(np.uint16) * _WEIGHTS).sum(axis=1).astype(np.uint8).tobytes()


def unpack_bits(data, little=False):
    """bytes -> 0/1 uint8 array; equals np.unpackbits(data, bitorder=...)."""
    b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 1)
    bits = ((b >> _SHIFTS) & 1).astype(np.uint8)
    if little:
        bits = bits[:, ::-1]
    return bits.reshape(-1)


def zero(buf):
    """Overwrite every mutable buffer given (bytearray, ndarray); never fails."""
    for b in buf:
        try:
            if b is None:
                continue
            if isinstance(b, np.ndarray):
                b.fill(0)
            else:
                b[:] = bytes(len(b))
        except Exception:
            pass


def scrub(count=128):
    """Best effort: reuse and zero small heap blocks so that freed 32-byte digests get overwritten."""
    for _ in range(count):
        t = bytearray(32)
        t[:] = ZERO32
        del t


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- K
def check_k(k):
    """Returns the raw SHA-256 of K if shape/dtype/values are right, raises otherwise."""
    if k.shape != (4096, 64) or k.dtype != np.int8 or not bool(((k == 0) | (k == 1)).all()):
        raise ModuleError("K shape, dtype or values are wrong")
    digest = hashlib.sha256(np.ascontiguousarray(k).tobytes()).hexdigest()
    if digest != K_SHA:
        raise ModuleError("K SHA-256 mismatch")
    return digest


def load_k(path):
    """Read-only load of the frozen K from a .npy file, verified before any use."""
    k = np.ascontiguousarray(np.load(path, allow_pickle=False))
    check_k(k)
    return k


def k_wire(k):
    """K rows as the research protocol's challenge words: row bit i -> word bit i, uint64 little-endian."""
    pw = np.uint64(1) << np.arange(64, dtype=np.uint64)
    words = (k.astype(np.uint64) * pw).sum(axis=1)
    for i in (0, 1, 4095):
        if int(words[i]) != sum(int(bit) << j for j, bit in enumerate(k[i])):
            raise ModuleError("K wire packing self-check failed")
    return words.astype("<u8").tobytes()


# ---------------------------------------------------------------- BCH(t, m=13)
def _ecc_mask(ecc_bits):
    """Byte mask of the meaningful ecc bits (big-endian bit order inside the ecc bytes)."""
    mask = [0xFF] * (ecc_bits // 8)
    rem = ecc_bits % 8
    if rem:
        mask.append(0xFF << (8 - rem) & 0xFF)
    return bytes(mask)


class Bch(object):
    """Systematic BCH over 512 data bytes with bchlib. decode_data() never raises."""

    def __init__(self, t, m=13):
        if not 1 <= t <= MAX_T or m != 13:
            raise ModuleError("t must be in 1..%d and m must be 13" % MAX_T)
        self.t, self.m = t, m
        try:
            self.bch = bchlib.BCH(t, m=m)
        except Exception:
            raise ModuleError("bchlib could not build BCH(t=%d, m=13)" % t)
        self.ecc_bits = int(self.bch.ecc_bits)
        self.ecc_bytes = int(self.bch.ecc_bytes)
        self._mask = np.frombuffer(_ecc_mask(self.ecc_bits), dtype=np.uint8)

    def encode(self, data):
        return bytearray(self.bch.encode(bytes(data)))

    def _same_ecc(self, a, b):
        x = np.frombuffer(bytes(a), dtype=np.uint8)
        y = np.frombuffer(bytes(b), dtype=np.uint8)
        return bool(np.array_equal(x & self._mask, y & self._mask))

    def decode_data(self, data, ecc):
        """Corrected 512-byte data as a new bytearray, or None when the word does not decode."""
        try:
            d, e = bytearray(data), bytearray(ecc)
            if len(d) != DATA_BYTES or len(e) != self.ecc_bytes:
                return None
            n = self.bch.decode(d, e)
            if isinstance(n, tuple):            # bchlib 1.x returned (positions, ecc_err)
                return None                     # unsupported API; refuse rather than guess
            if n < 0 or n > self.t:
                return None
            self.bch.correct(d, e)
            if not self._same_ecc(self.bch.encode(bytes(d)), e):
                return None                     # not a codeword after correction (miscorrection)
            return d
        except Exception:
            return None

    def selftest(self):
        """Checks the installed bchlib on this machine: corrects up to t errors, never returns the
        original for t+1 errors, tolerates ecc-side errors. Raises ModuleError otherwise."""
        rng = np.random.RandomState(20261120)
        for trial in range(6):
            data = bytearray(rng.randint(0, 256, DATA_BYTES).astype(np.uint8).tobytes())
            ecc = self.encode(data)
            nerr = [0, self.t // 3, self.t][trial % 3]
            bad = bytearray(data)
            for j in rng.choice(DATA_BITS, nerr, replace=False):
                bad[j >> 3] ^= 0x80 >> (j & 7)
            if bytes(self.decode_data(bad, ecc) or b"") != bytes(data):
                raise ModuleError("bchlib self-test failed: %d errors not corrected" % nerr)
            if self.t + 1 <= DATA_BITS:
                bad = bytearray(data)
                for j in range(self.t + 1):
                    bad[j >> 3] ^= 0x80 >> (j & 7)
                if bytes(self.decode_data(bad, ecc) or b"") == bytes(data):
                    raise ModuleError("bchlib self-test failed: t+1 errors returned the original")
            bad_ecc = bytearray(ecc)
            bad_ecc[0] ^= 0x80                  # one error on the ecc side
            if bytes(self.decode_data(data, bad_ecc) or b"") != bytes(data):
                raise ModuleError("bchlib self-test failed: ecc-side error")
        return True


# ---------------------------------------------------------------- Toeplitz hashing, HMAC
def toeplitz_root(data, seed):
    """r (32 bytes) of the requirement, step 4: r = packbits((rows @ x) % 2, little), rows = 256
    consecutive 4096-bit windows of s = unpackbits(seed, little), x = unpackbits(data, little)."""
    if len(data) != DATA_BYTES or len(seed) != 544:
        raise ModuleError("toeplitz_root: wrong input length")
    x = unpack_bits(data, little=True).astype(np.int64)
    s = np.ascontiguousarray(unpack_bits(seed, little=True))
    rows = np.lib.stride_tricks.as_strided(s, shape=(257, DATA_BITS), strides=(s.strides[0], s.strides[0]))[:256]
    bits = (rows.astype(np.int64) @ x) % 2
    return bytearray(pack_bits(bits, little=True))


def hmac_sha256(key, *msgs):
    """HMAC-SHA256 for keys up to 64 bytes, equal to hmac.new(key, b"".join(msgs), sha256). The
    message pieces are hashed one by one (no concatenated copy) and every key-derived buffer is a
    bytearray that is zeroed before returning."""
    if len(key) > 64:
        key = bytearray(hashlib.sha256(bytes(key)).digest())
    k = bytearray(64)
    k[:len(key)] = key
    ipad, opad = bytearray(64), bytearray(64)
    for i in range(64):
        ipad[i] = k[i] ^ 0x36
        opad[i] = k[i] ^ 0x5C
    zero([k])
    inner = hashlib.sha256(ipad)
    for piece in msgs:
        inner.update(piece)
    outer = hashlib.sha256(opad)
    outer.update(inner.digest())
    zero([ipad, opad])
    return outer.digest()


def unlock_key(root, passphrase, iterations):
    """z = HMAC(r, "BoundKV-unlock|" + passphrase); c times z = HMAC(r, z); k_u = z."""
    if iterations < 1:
        raise ModuleError("iterations must be positive")
    z = bytearray(hmac_sha256(root, UNLOCK_DOMAIN, passphrase))
    for _ in range(iterations):
        z[:] = hmac_sha256(root, z)
    scrub(8)
    return z


def helper_bytes(fields):
    """json.dumps(the seven public enrollment fields, sort_keys=True, separators=(",", ":"))."""
    import json
    return json.dumps(dict((name, fields[name]) for name in HELPER_FIELDS),
                      sort_keys=True, separators=(",", ":")).encode()


def tag_matches(root, fields, tag_hex):
    """Constant-time comparison of the recomputed tag with tag_hex; the recomputed tag is zeroed."""
    expected = bytearray(bytes.fromhex(tag_hex))
    tag = bytearray(hmac_sha256(root, TAG_DOMAIN, helper_bytes(fields)))
    try:
        return hmac.compare_digest(tag, expected)
    finally:
        zero([tag, expected])


# ---------------------------------------------------------------- where r and k_u live
OCM_MAGIC = b"BOUNDKV1"
PAGE = 4096
_OFF_R, _OFF_KU, _OFF_CANARY_A, _OFF_CANARY_B = 0x40, 0x60, 0x008, 0x7C0


class MappedSecretStore(object):
    """r and k_u slots inside one memory page that the caller maps (OCM on the board).

    The page must be empty when the store is created, unless it holds the leftovers of an earlier
    instance of this module (same magic and canary in both places), which are then erased. Two
    random canaries are checked before every access; if either changed, the module refuses to run.
    """

    def __init__(self, mem, description):
        self.mem, self.description = mem, description
        self._r = self._ku = False
        self.leftover_erased = False
        page = bytes(mem[0:PAGE])
        if page.strip(b"\0"):
            if page[:8] == OCM_MAGIC and page[8:40] == page[_OFF_CANARY_B:_OFF_CANARY_B + 32]:
                mem[0:PAGE] = bytes(PAGE)
                self.leftover_erased = True
            else:
                raise ModuleError("secret page is not empty; refusing to touch it")
        self.canary = secrets.token_bytes(32)
        mem[0:8] = OCM_MAGIC
        mem[_OFF_CANARY_A:_OFF_CANARY_A + 32] = self.canary
        mem[_OFF_CANARY_B:_OFF_CANARY_B + 32] = self.canary

    def _intact(self):
        return (bytes(self.mem[0:8]) == OCM_MAGIC
                and bytes(self.mem[_OFF_CANARY_A:_OFF_CANARY_A + 32]) == self.canary
                and bytes(self.mem[_OFF_CANARY_B:_OFF_CANARY_B + 32]) == self.canary)

    def _check(self):
        if not self._intact():
            raise ModuleError("secret page canary changed")

    def _put(self, slot, buf):
        self._check()
        self.mem[slot:slot + 32] = bytes(buf)

    def _use(self, slot, valid, fn):
        """Runs fn with a zeroable copy of the slot; the copy is zeroed whatever fn does."""
        self._check()
        if not valid:
            raise ModuleError("slot is empty")
        tmp = bytearray(self.mem[slot:slot + 32])
        try:
            return fn(tmp)
        finally:
            zero([tmp])

    def _erase(self, slot):
        self._check()
        self.mem[slot:slot + 32] = ZERO32

    def put_r(self, buf):
        self._put(_OFF_R, buf)
        self._r = True

    def use_r(self, fn):
        return self._use(_OFF_R, self._r, fn)

    def erase_r(self):
        self._erase(_OFF_R)
        self._r = False

    def put_ku(self, buf):
        self._put(_OFF_KU, buf)
        self._ku = True

    def use_ku(self, fn):
        return self._use(_OFF_KU, self._ku, fn)

    def erase_ku(self):
        self._erase(_OFF_KU)
        self._ku = False

    @property
    def has_ku(self):
        return self._ku

    def close(self):
        """Zero the whole page, magic and canaries included."""
        try:
            self.mem[0:PAGE] = bytes(PAGE)
        except Exception:
            pass
        self._r = self._ku = False


class MemorySecretStore(MappedSecretStore):
    """Fallback: the same layout in a private bytearray of the process."""

    def __init__(self):
        MappedSecretStore.__init__(self, bytearray(PAGE), "process memory (private bytearray)")


def ocm_page_is_usable(address=0xFFFE8000):
    """True when the kernel's resource tree shows the page as OCM only (not System RAM)."""
    line_match = None
    with open("/proc/iomem") as f:
        for line in f:
            span = line.split(":", 1)
            parts = span[0].strip().split("-")
            if len(parts) != 2:
                continue
            start, end = int(parts[0], 16), int(parts[1], 16)
            if start <= address and address + PAGE - 1 <= end:
                line_match = (span[1].strip(), start, end)
    return bool(line_match and line_match[0].endswith("ocmc")), line_match


def open_ocm_store(address=0xFFFE8000):
    usable, found = ocm_page_is_usable(address)
    if not usable:
        raise ModuleError("0x%08X is not shown as OCM in /proc/iomem (%r)" % (address, found))
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    try:
        mem = mmap.mmap(fd, PAGE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=address)
    finally:
        os.close(fd)
    return MappedSecretStore(mem, "on-chip memory (OCM) at 0x%08X, one 4 KiB page" % address)


# ---------------------------------------------------------------- PUF sources
def combine(family_lanes, lanes):
    """XOR of the selected lanes: family_lanes maps family -> (4096, 4) 0/1 array; returns uint8 (4096,)."""
    out = np.zeros(4096, dtype=np.uint8)
    for lane in lanes:
        out ^= family_lanes[FAMILY_OF_LANE[lane]][:, lane % 4].astype(np.uint8)
    return out


def families_of(lanes):
    seen = []
    for lane in lanes:
        if FAMILY_OF_LANE[lane] not in seen:
            seen.append(FAMILY_OF_LANE[lane])
    return sorted(seen)


def _check_lanes(lanes):
    if (not isinstance(lanes, list) or not lanes or lanes != sorted(set(lanes))
            or any(not isinstance(l, int) or isinstance(l, bool) or l < 0 or l > 7 for l in lanes)):
        raise ModuleError("lanes must be a sorted list of distinct lane numbers 0-7")
    return list(lanes)


class MmioSource(object):
    """Live reads through the frozen research firmware, imported by path after its SHA-256 is checked.
    The measurement functions are the research service's own; nothing is copied or re-implemented."""

    backend = "mmio"
    needs_root = True

    def __init__(self, k, research_path, bit_path, program, base=0x40000000):
        self.k_bytes = np.ascontiguousarray(k).tobytes()
        self.words = struct.unpack("<4096Q", k_wire(k))
        if sha256_file(research_path) != RESEARCH_SHA or sha256_file(bit_path) != BIT_SHA:
            raise ModuleError("frozen research firmware or bitstream SHA-256 mismatch")
        spec = importlib.util.spec_from_file_location("frozen_research", research_path)
        research = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(research)
        self.research = research
        if program:
            research.load_bitstream(bit_path)
            self.pl_programmed = True
        else:
            if open("/sys/class/fpga_manager/fpga0/state").read().strip() != "operating":
                raise ModuleError("PL is not configured and programming was not requested")
            self.pl_programmed = False
        self.mmio = research.Mmio(base, 0x1000)
        self.measure = {"apuf_lanes": research.measure_apuf_lanes, "ff_lanes": research.measure_ff_lanes}
        self.evaluations = 0

    def _verify_k(self):
        if hashlib.sha256(self.k_bytes).hexdigest() != K_SHA:
            raise ModuleError("K changed under the module")

    def read_k(self, family):
        self._verify_k()
        out = np.empty((4096, 4), dtype=np.int8)
        measure = self.measure[family]
        for i, word in enumerate(self.words):
            out[i] = measure(self.mmio, word)
        self.evaluations += 1
        return out


class ReplaySource(object):
    """Test only: recorded real lane reads of K (int8 (R, 4096, 4) per family), served row by row."""

    backend = "replay"
    needs_root = False

    def __init__(self, apuf_npy, ff_npy, start, count):
        self.data = {"apuf_lanes": np.load(apuf_npy, allow_pickle=False),
                     "ff_lanes": np.load(ff_npy, allow_pickle=False)}
        for family, arr in self.data.items():
            if arr.ndim != 3 or arr.shape[1:] != (4096, 4) or arr.dtype != np.int8:
                raise ModuleError("replay file for %s must be int8 (R, 4096, 4)" % family)
            if not bool(((arr == 0) | (arr == 1)).all()):
                raise ModuleError("replay file for %s is not binary" % family)
        self.start, self.count = int(start), int(count)
        self.rows = min(a.shape[0] for a in self.data.values())
        if self.count <= 0 or self.start + self.count > self.rows:
            raise ModuleError("replay range %d+%d outside the %d recorded reads" % (start, count, self.rows))
        self.cursor = {"apuf_lanes": 0, "ff_lanes": 0}
        self.evaluations = 0

    def read_k(self, family):
        row = self.start + self.cursor[family] % self.count
        self.cursor[family] += 1
        self.evaluations += 1
        return np.array(self.data[family][row], dtype=np.int8, copy=True)


class SimSource(object):
    """Test only: an ideal response per (family, lane, challenge) from a seeded PRNG, plus flips."""

    backend = "sim"
    needs_root = False

    def __init__(self, device_seed, flip_probability=0.002, rng_seed=None):
        rng = np.random.RandomState(device_seed)
        self.ideal = dict((f, rng.randint(0, 2, (4096, 4)).astype(np.int8)) for f in ("apuf_lanes", "ff_lanes"))
        self.flip = np.random.RandomState(rng_seed if rng_seed is not None else device_seed + 1)
        self.p = float(flip_probability)
        self.evaluations = 0

    def read_k(self, family):
        flips = self.flip.random_sample((4096, 4)) < self.p
        self.evaluations += 1
        return np.array(self.ideal[family] ^ flips.astype(np.int8), dtype=np.int8)


# ---------------------------------------------------------------- enrollment data
def enrollment_fields(device_id, lanes, t, ecc, seed):
    return {"device_id": device_id, "lanes": list(lanes), "m": 13, "t": int(t),
            "ecc_hex": bytes(ecc).hex(), "seed_hex": bytes(seed).hex(), "challenges_sha256": K_SHA}


def validate_enrollment(obj):
    """Returns the seven public fields with strict types, or raises ModuleError."""
    if not isinstance(obj, dict) or set(obj) != set(HELPER_FIELDS) | {"tag_hex"}:
        raise ModuleError("enrollment.json must hold exactly the specified fields")
    f = dict((name, obj[name]) for name in HELPER_FIELDS)
    if not isinstance(f["device_id"], str) or not f["device_id"] or len(f["device_id"]) > 64:
        raise ModuleError("device_id must be a short non-empty string")
    f["lanes"] = _check_lanes(f["lanes"])
    for name in ("m", "t"):
        if not isinstance(f[name], int) or isinstance(f[name], bool):
            raise ModuleError("%s must be an integer" % name)
    if f["m"] != 13 or not 1 <= f["t"] <= MAX_T:
        raise ModuleError("m must be 13 and t in 1..%d" % MAX_T)
    try:
        ecc = bytearray(bytes.fromhex(f["ecc_hex"]))
        seed = bytearray(bytes.fromhex(f["seed_hex"]))
        tag = bytearray(bytes.fromhex(obj["tag_hex"]))
    except (ValueError, TypeError, AttributeError):
        raise ModuleError("hex field is malformed")
    if len(seed) != 544 or len(tag) != 32 or len(ecc) != Bch(f["t"]).ecc_bytes:
        raise ModuleError("hex field has the wrong length")
    if f["challenges_sha256"] != K_SHA:
        raise ModuleError("enrollment was not made with the frozen K")
    zero([ecc, seed, tag])
    return f


def read_k_families(source, families):
    """One read of K per family involved; returns {family: (4096, 4) int8}."""
    return dict((family, source.read_k(family)) for family in families)


def measure_word(source, lanes):
    """Deployment step 1: one read of K, combined lane by lane into w' (uint8 (4096,)).
    The lane arrays are zeroed before returning; the caller owns w'."""
    read = read_k_families(source, families_of(lanes))
    w2 = combine(read, lanes)
    zero(list(read.values()))
    return w2
