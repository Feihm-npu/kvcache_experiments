#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PUF-COLP v1 服务端 (Zynq PYNQ 2.5.6, PS 侧), 自包含无 pynq 依赖.

通过 Linux FPGA Manager (/sys/class/fpga_manager/fpga0) 把 bitstream 配进 PL,
用 /dev/mem mmap 驱动 FLAM PUF AXI4-Lite 外设, 对外实现 docs/protocol.md 协议,
供 pc/collect.py --backend tcp 调用.

依赖: Python 3.6+, 板上 root (写 /lib/firmware + /sys/.../firmware + /dev/mem).
用法 (板上 root):
    sudo python3 puf_server.py --bit /home/xilinx/puf/system_wrapper.bit \
        --port 3040 --base 0x40000000
"""
import argparse
import itertools
import random
import json
import mmap
import os
import socket
import struct
import sys
import threading
import time
from typing import Optional, Tuple

# ---------------- 协议常量 (与 pc/puf_collector/protocol.py 对齐) ----------------
MAGIC = b"PUF1"
HEADER_FMT = "<4sHI"           # magic(4) + cmd(uint16) + length(uint32) = 10 B
HEADER_SIZE = struct.calcsize(HEADER_FMT)

CMD_HELLO = 0x01
CMD_ECHO = 0x02
CMD_CONFIGURE = 0x10
CMD_RESET = 0x11
CMD_SEND_CHALLENGES = 0x20
CMD_GET_RESPONSES = 0x21
CMD_GET_OIT = 0x30
CMD_GET_PARAMS = 0x31
CMD_ERROR = 0xFF

# FLAM PUF 寄存器偏移 (puf_flam_top.v, 字节地址, AXI4-Lite 32-bit)
REG_CHAL_LO = 0x00
REG_CHAL_HI = 0x04
REG_CTRL = 0x08            # W [0]=start ; R [0]=busy
REG_RESP = 0x0C            # R [0]=resp [1]=done
REG_CFG = 0x10
REG_DBG = 0x14
REG_PUF4X4_C0 = 0x18
REG_PUF4X4_C1 = 0x1C
REG_PUF4X4_C2 = 0x20
REG_PUF4X4_C3 = 0x24
REG_PUF4X4_CTRL = 0x28       # W [0]=start ; R [0]=busy
REG_PUF4X4_RESP = 0x2C       # R [5:0]=pair bits [6]=xor [7]=done
REG_OI_CTRL = 0x30            # W [0]=start [3:2]=lane/core ; R [0]=busy
REG_OI_RESP = 0x34            # R [0]=selected resp [1]=done [5:2]=all core resp
REG_TPDOA_CTRL = 0x38         # W [0]=start ; R [0]=busy
REG_TPDOA_RESP = 0x3C         # R [0]=resp [1]=done [2]=dut_dbg [3]=ref_dbg
REG_FF_CTRL = 0x40            # W [0]=start ; R [0]=busy
REG_FF_RESP = 0x44            # R [3:0]=lane bits [4]=xor [5]=done [9:6]=feed bits

# 硬件实际支持的族。
# 说明: 当前设计保留 64-stage APUF 路径供 FLAM/CT/OI 使用，新增 native
# 4-line/24-stage 4x4 RTL 路径供 puf4x4 使用；OI 使用 4 个独立 APUF core。
# 族内偏差均写入 puf_params/hw_note。
HW_FAMILIES = ("ct", "flam", "puf4x4", "oi", "oi_lanes", "tpdoa",
               "apuf_lanes", "ff_lanes")

DEVICE_INFO = {
    "board": "Zybo Z7-10",
    "part": "xc7z010clg400-1",
    "clock_MHz": 100,
    "fw_version": "pynq-ps-apuf64-multifamily-1.2-apuf-xor-ff",
    "families": list(HW_FAMILIES),
    "physical_core": "shared APUF, four independent APUFs, four feed-forward APUFs, 24-stage 4x4 PUF, and TP-DOA race core",
}

MASK64 = (1 << 64) - 1
EVEN_MASK64 = 0x5555555555555555  # bit0,2,...,62
ODD_MASK64 = 0xAAAAAAAAAAAAAAAA   # bit1,3,...,63
FLAM_G_WORD = 0xD800000000000000
TPDOA_MV_WORD = 0xA6D3C5E9B17F402D
TPDOA_BIT_SEL = 5
CT_BRANCH_B_XOR = 0xA5A5A5A5A5A5A5A5
CT_BRANCH_C_XOR = 0x3C3C3C3C3C3C3C3C
OI_SEED = 0x01C0FFEE
PERMS24 = list(itertools.permutations(range(4)))
PAIR_LIST = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
OI_LANE_MASKS = [
    0x1111222233334444,
    0x5555666677778888,
    0x9999AAAABBBBCCCC,
    0xDDDDEEEEFFFF0000,
]


def _bits_from_word(word, n=64):
    return [int((word >> i) & 1) for i in range(n)]


def _parity64(word):
    word &= MASK64
    word ^= word >> 32
    word ^= word >> 16
    word ^= word >> 8
    word ^= word >> 4
    word &= 0xF
    return (0x6996 >> word) & 1


def _gen_oit_perms(seed=OI_SEED):
    rng = random.Random(seed)
    u0, u1 = [], []
    for _ in range(64):
        p0 = [0, 1, 2, 3]
        p1 = [0, 1, 2, 3]
        rng.shuffle(p0)
        rng.shuffle(p1)
        u0.append(tuple(p0))
        u1.append(tuple(p1))
    return u0, u1


OI_U0_PERMS, OI_U1_PERMS = _gen_oit_perms()


def _oit_bytes():
    out = bytearray()
    for table in (OI_U0_PERMS, OI_U1_PERMS):
        for p in table:
            for src in range(4):
                for dst in range(4):
                    out.append(1 if p[src] == dst else 0)
    return bytes(out)


FLAM_PARAMS = {
    "K": 64,
    "N_round": 1,
    "n_round": 1,
    "tap_vector_g": _bits_from_word(FLAM_G_WORD),
    "feedback_pos": 63,
    "challenge_transform": "roll_right(C) XOR (MSB(C) * g)",
    "physical_core": "64-stage LUT-preserved APUF plus native 4x4 permutation RTL",
    "hw_note": "single-instance FLAM surrogate: 1-round LFSR transform measured by physical APUF",
}

CT_PARAMS = {
    "K": 64,
    "branch_select": "if V_odd==0: A; elif V_even==0: B; else: C",
    "branch_A": "APUF(C & EVEN_MASK64)",
    "branch_B": "APUF(roll_left(C,1) XOR CT_BRANCH_B_XOR)",
    "branch_C": "APUF(bit_reverse(C) XOR CT_BRANCH_C_XOR)",
    "physical_core": "shared 64-stage LUT-preserved APUF",
    "hw_note": "CT surrogate using one physical APUF with branch-specific challenge transforms; branch ratios are computed from delivered challenges",
}

PUF4X4_PARAMS = {
    "S": 24,
    "permutation_encoding": "reference 4x4-apuf lexicographic table; challenge byte value 0..23 directly selects each stage mux sources",
    "response_pairs": [[0,1],[1,2],[2,3],[0,2],[1,3],[0,3]],
    "permutation_challenge_packing": "24 values in 0..23 packed as 24 little-endian 5-bit fields into AXI PUF4X4_C0/C1/C2/C3",
    "physical_core": "native 4-line, 24-stage permutation delay network with six pairwise arbiters",
    "hw_note": "4x4 redesign RTL path aligned to 4x4-apuf stage count, direct permutation encoding, and mutual-order pair order; responses_1bit is XOR",
}

OI_PARAMS = {
    "n": 64,
    "k": 4,
    "oit_seed": OI_SEED,
    "oit_encoding": "U table shape (64,4,4), rows are source lanes and columns are destination lanes",
    "physical_core": "4 independent 64-stage LUT-preserved APUFs, one per OI lane",
    "challenge_transform": "OIT-derived per-lane challenge words measured on independent APUF cores",
    "dont_touch": "all stage LUTs and arbiter FFs in each APUF core",
    "placement": "not yet manually constrained; cores are separate RTL instances and must be checked post-implementation",
    "hw_note": "OI redesign option A: fixed OIT route transform derives four lane challenge words; each lane is measured by a separate physical APUF core; final response XORs the four lane responses",
}

TPDOA_PARAMS = {
    "bit_sel": TPDOA_BIT_SEL,
    "MV": _bits_from_word(TPDOA_MV_WORD),
    "mv_word_hex": "0x%016X" % TPDOA_MV_WORD,
    "physical_core": "PL multiplier-selector delay race core",
    "delay_measurement_method": "race_arbiter_ref_trim42",
    "dont_touch": "partial-product selector LUTs, delay mux network, reference chain, arbiter FF",
    "hw_note": "TP-DOA redesign path: challenge drives multiplier-style partial-product selectors in PL; selected delay path races a calibrated reference chain (q42 debug tap selected by firmware from TPDOA_RESP[2]). This replaces the previous PS integer product function and must pass BER pilot before full collection.",
}

APUF_LANES_PARAMS = {
    "n": 64,
    "lanes": 4,
    "physical_core": "four independent 64-stage LUT-preserved APUFs",
    "challenge_mapping": "the same unmodified 64-bit challenge drives every lane",
    "outputs": "four raw lane responses followed by their XOR",
}

FF_LANES_PARAMS = {
    "n": 64,
    "lanes": 4,
    "feed_tap": 15,
    "feed_stage": 32,
    "physical_core": "four independent 64-stage feed-forward APUFs",
    "challenge_mapping": "the intermediate arbiter after stage 15 controls switch stage 32",
    "outputs": "four raw lane responses followed by their XOR",
}

PARAMS_BY_FAMILY = {
    "flam": FLAM_PARAMS,
    "ct": CT_PARAMS,
    "puf4x4": PUF4X4_PARAMS,
    "oi": OI_PARAMS,
    "oi_lanes": OI_PARAMS,
    "tpdoa": TPDOA_PARAMS,
    "apuf_lanes": APUF_LANES_PARAMS,
    "ff_lanes": FF_LANES_PARAMS,
}

FPGA_MAN = "/sys/class/fpga_manager/fpga0/firmware"
FPGA_MAN_FLAGS = "/sys/class/fpga_manager/fpga0/flags"


class ProtocolError(RuntimeError):
    pass


# ---------------- bitstream 加载 (FPGA Manager) ----------------
def parse_bit_data(bitfile):
    # type: (str) -> bytes
    """解析 Xilinx .bit 头, 返回 tag 0x65 后的裸配置数据 (大端字序)."""
    with open(bitfile, "rb") as f:
        c = f.read()
    off = 0
    # 第一字段: 2 字节 BE 长度 + data
    n = struct.unpack(">H", c[off:off + 2])[0]
    off += 2 + n
    # 2 字节未知字段
    off += 2
    while True:
        if off >= len(c):
            raise ProtocolError("bit header parse: no 0x65 field")
        desc = c[off]
        off += 1
        if desc == 0x65:
            length = struct.unpack(">I", c[off:off + 4])[0]
            off += 4
            return c[off:off + length]
        # 字段 a/b/c/d: 2 字节长度 + 字符串
        n = struct.unpack(">H", c[off:off + 2])[0]
        off += 2 + n


def bit_to_bin_bytes(bitfile):
    # type: (str) -> bytes
    """.bit -> fpga_manager 需要的 .bin: 每 32 位字字节序交换 (对齐 pynq 2.5.6)."""
    data = parse_bit_data(bitfile)
    pad = (-len(data)) % 4
    if pad:
        data = data + b"\x00" * pad
    arr = bytearray(data)
    for i in range(0, len(arr), 4):
        arr[i], arr[i + 1], arr[i + 2], arr[i + 3] = \
            arr[i + 3], arr[i + 2], arr[i + 1], arr[i]
    return bytes(arr)


def load_bitstream(bitfile, tag="puf_flam"):
    # type: (str, str) -> None
    """把 bitstream 配进 PL: 转 .bin -> /lib/firmware/<tag>.bin -> echo 到 sysfs."""
    if os.geteuid() != 0:
        raise EnvironmentError("need root to program PL (fpga_manager + /dev/mem)")
    binpath = "/lib/firmware/{}.bin".format(tag)
    bdata = bit_to_bin_bytes(bitfile)
    with open(binpath, "wb") as f:
        f.write(bdata)
    # 全 bitstream: flags='0' (部分重配才用 '1')
    with open(FPGA_MAN_FLAGS, "w") as f:
        f.write("0")
    with open(FPGA_MAN, "w") as f:
        f.write("{}.bin".format(tag))
    # fpga_manager 写入即触发, 等几拍让 PL 重配稳定
    time.sleep(0.2)


# ---------------- MMIO (/dev/mem) ----------------
class Mmio(object):
    def __init__(self, base, length=0x1000):
        # type: (int, int) -> None
        pagesize = mmap.PAGESIZE
        base_page = base & ~(pagesize - 1)
        self._off = base - base_page
        fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self._mem = mmap.mmap(fd, length + self._off, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE, offset=base_page)
        os.close(fd)

    def read(self, off):
        # type: (int) -> int
        return struct.unpack_from("<I", self._mem, self._off + off)[0]

    def write(self, off, val):
        # type: (int, int) -> None
        struct.pack_into("<I", self._mem, self._off + off, val & 0xFFFFFFFF)


# ---------------- FLAM 一次测量 ----------------
def measure_one(mmio, word64, timeout=200000):
    # type: (Mmio, int, int) -> int
    """驱动 FSM 测一个 64 位挑战, 返回 1 位响应.

    word64 的 bit i -> challenge[i]; 低 32 位写 CHAL_LO, 高 32 位写 CHAL_HI.
    """
    mmio.write(REG_CHAL_LO, word64 & 0xFFFFFFFF)
    mmio.write(REG_CHAL_HI, (word64 >> 32) & 0xFFFFFFFF)
    mmio.write(REG_CTRL, 1)          # start (start_pulse 自清零, 同周期清 done)
    mmio.read(REG_CTRL)              # 占位读: 确保 busy 已拉高, 消除竞争
    busy = 1
    for _ in range(timeout):
        busy = mmio.read(REG_CTRL) & 1
        if not busy:
            break
    if busy:
        raise ProtocolError("PUF measurement timeout (busy stuck); PL 未配置?")
    return mmio.read(REG_RESP) & 1


# ---------------- 帧 I/O ----------------
def _read_exactly(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise ProtocolError("connection closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(f):
    # type: (object) -> Tuple[int, bytes]
    head = _read_exactly(f, HEADER_SIZE)
    magic, cmd, length = struct.unpack(HEADER_FMT, head)
    if magic != MAGIC:
        raise ProtocolError("bad magic {!r}".format(magic))
    payload = _read_exactly(f, length) if length else b""
    return cmd, payload


def pack_frame(cmd, payload=b""):
    # type: (int, bytes) -> bytes
    return struct.pack(HEADER_FMT, MAGIC, cmd, len(payload)) + payload


def unpack_challenges_binary(data, n):
    # type: (bytes, int) -> list
    if len(data) < n * 8:
        raise ProtocolError("binary challenge payload short: {} < {}".format(len(data), n * 8))
    return list(struct.unpack_from("<%dQ" % n, data, 0))


def unpack_challenges_puf4x4(data, n):
    # type: (bytes, int) -> list
    if len(data) < n * 24:
        raise ProtocolError("puf4x4 challenge payload short: {} < {}".format(len(data), n * 24))
    rows = []
    for i in range(n):
        row = tuple(bytearray(data[i * 24:(i + 1) * 24]))
        rows.append(row)
    return rows


def _roll_right64(word):
    return ((word >> 1) | ((word & 1) << 63)) & MASK64


def _roll_left64(word, n=1):
    n &= 63
    return ((word << n) | (word >> (64 - n))) & MASK64


def _reverse64(word):
    out = 0
    for i in range(64):
        out = (out << 1) | ((word >> i) & 1)
    return out & MASK64


def _flam_word(word):
    out = _roll_right64(word)
    if (word >> 63) & 1:
        out ^= FLAM_G_WORD
    return out & MASK64


def _ct_word_and_branch(word):
    v_even = _parity64(word & EVEN_MASK64)
    v_odd = _parity64(word & ODD_MASK64)
    if v_odd == 0:
        return word & EVEN_MASK64, 0
    if v_even == 0:
        return (_roll_left64(word, 1) ^ CT_BRANCH_B_XOR) & MASK64, 1
    return (_reverse64(word) ^ CT_BRANCH_C_XOR) & MASK64, 2


def _oi_lane_words(word):
    lanes = [0, 1, 2, 3]
    lane_words = [0, 0, 0, 0]
    for stage in range(64):
        bit = (word >> stage) & 1
        perm = OI_U1_PERMS[stage] if bit else OI_U0_PERMS[stage]
        new_lanes = [0, 0, 0, 0]
        for src, dst in enumerate(perm):
            new_lanes[dst] = lanes[src]
        lanes = new_lanes
        for lane in range(4):
            v = bit
            lane_words[lane] |= int(v) << stage
    return [(lane_words[i] ^ OI_LANE_MASKS[i]) & MASK64 for i in range(4)]



def _pack_puf4x4_word120(row):
    word = 0
    for stage, idx in enumerate(row):
        word |= (int(idx) % 24) << (stage * 5)
    return word


def measure_puf4x4(mmio, row, timeout=200000):
    word = _pack_puf4x4_word120(row)
    mmio.write(REG_PUF4X4_C0, word & 0xFFFFFFFF)
    mmio.write(REG_PUF4X4_C1, (word >> 32) & 0xFFFFFFFF)
    mmio.write(REG_PUF4X4_C2, (word >> 64) & 0xFFFFFFFF)
    mmio.write(REG_PUF4X4_C3, (word >> 96) & 0xFFFFFF)
    mmio.write(REG_PUF4X4_CTRL, 1)
    mmio.read(REG_PUF4X4_CTRL)
    busy = 1
    for _ in range(timeout):
        busy = mmio.read(REG_PUF4X4_CTRL) & 1
        if not busy:
            break
    if busy:
        raise ProtocolError("PUF4x4 measurement timeout (busy stuck); PL lacks audit-fix RTL?")
    val = mmio.read(REG_PUF4X4_RESP)
    r6 = [(val >> k) & 1 for k in range(6)]
    r1 = (val >> 6) & 1
    return r6, r1


def measure_oi_lane(mmio, word64, lane, timeout=200000):
    mmio.write(REG_CHAL_LO, word64 & 0xFFFFFFFF)
    mmio.write(REG_CHAL_HI, (word64 >> 32) & 0xFFFFFFFF)
    mmio.write(REG_OI_CTRL, ((int(lane) & 3) << 2) | 1)
    mmio.read(REG_OI_CTRL)
    busy = 1
    for _ in range(timeout):
        busy = mmio.read(REG_OI_CTRL) & 1
        if not busy:
            break
    if busy:
        raise ProtocolError("OI lane measurement timeout (busy stuck); PL lacks OI independent APUF RTL?")
    val = mmio.read(REG_OI_RESP)
    return val & 1


def measure_apuf_lanes(mmio, word64, timeout=200000):
    mmio.write(REG_CHAL_LO, word64 & 0xFFFFFFFF)
    mmio.write(REG_CHAL_HI, (word64 >> 32) & 0xFFFFFFFF)
    mmio.write(REG_OI_CTRL, 1)
    mmio.read(REG_OI_CTRL)
    busy = 1
    for _ in range(timeout):
        busy = mmio.read(REG_OI_CTRL) & 1
        if not busy:
            break
    if busy:
        raise ProtocolError("APUF lane measurement timeout")
    raw = mmio.read(REG_OI_RESP)
    return [(raw >> (2 + lane)) & 1 for lane in range(4)]


def measure_ff_lanes(mmio, word64, timeout=200000):
    mmio.write(REG_CHAL_LO, word64 & 0xFFFFFFFF)
    mmio.write(REG_CHAL_HI, (word64 >> 32) & 0xFFFFFFFF)
    mmio.write(REG_FF_CTRL, 1)
    mmio.read(REG_FF_CTRL)
    busy = 1
    for _ in range(timeout):
        busy = mmio.read(REG_FF_CTRL) & 1
        if not busy:
            break
    if busy:
        raise ProtocolError("feed-forward APUF measurement timeout")
    raw = mmio.read(REG_FF_RESP)
    return [(raw >> lane) & 1 for lane in range(4)]


def measure_tpdoa(mmio, word64, timeout=200000):
    mmio.write(REG_CHAL_LO, word64 & 0xFFFFFFFF)
    mmio.write(REG_CHAL_HI, (word64 >> 32) & 0xFFFFFFFF)
    mmio.write(REG_TPDOA_CTRL, 1)
    mmio.read(REG_TPDOA_CTRL)
    busy = 1
    for _ in range(timeout):
        busy = mmio.read(REG_TPDOA_CTRL) & 1
        if not busy:
            break
    if busy:
        raise ProtocolError("TP-DOA measurement timeout (busy stuck); PL lacks TPDOA race RTL?")
    raw = mmio.read(REG_TPDOA_RESP)
    return (raw >> 2) & 1


def _tpdoa_response(word):
    return int(((int(word) * int(TPDOA_MV_WORD)) >> TPDOA_BIT_SEL) & 1)


# ---------------- 连接处理 ----------------
def handle_conn(conn, addr, mmio):
    print("[srv] connection from {}".format(addr), file=sys.stderr)
    f = conn.makefile("rwb")
    pending_chal = None   # type: Optional[bytes]
    cur_family = None     # type: Optional[str]
    cur_seed = 0
    try:
        while True:
            try:
                cmd, payload = recv_frame(f)
            except ProtocolError as e:
                print("[srv] {} frame error: {}".format(addr, e), file=sys.stderr)
                break
            if cmd == CMD_HELLO:
                f.write(pack_frame(CMD_HELLO, json.dumps(DEVICE_INFO).encode()))
            elif cmd == CMD_ECHO:
                f.write(pack_frame(CMD_ECHO, payload))
            elif cmd == CMD_CONFIGURE:
                req = json.loads(payload.decode())
                fam = req.get("family")
                if fam not in HW_FAMILIES:
                    f.write(pack_frame(CMD_ERROR,
                        ("family {!r} not implemented in HW service (only {})".format(fam, HW_FAMILIES)).encode()))
                else:
                    cur_family = fam
                    cur_seed = int(req.get("seed", 0))
                    f.write(pack_frame(CMD_CONFIGURE,
                        json.dumps({"ok": True, "instance_id": req.get("instance_id", 0), "seed": cur_seed}).encode()))
            elif cmd == CMD_RESET:
                cur_family = None
                pending_chal = None
                f.write(pack_frame(CMD_RESET, json.dumps({"ok": True}).encode()))
            elif cmd == CMD_GET_PARAMS:
                if cur_family in PARAMS_BY_FAMILY:
                    f.write(pack_frame(CMD_GET_PARAMS, json.dumps(PARAMS_BY_FAMILY[cur_family]).encode()))
                else:
                    f.write(pack_frame(CMD_ERROR,
                        ("GET_PARAMS: no family configured or unsupported ({})".format(cur_family)).encode()))
            elif cmd == CMD_GET_OIT:
                if cur_family in ("oi", "oi_lanes"):
                    f.write(pack_frame(CMD_GET_OIT, _oit_bytes()))
                else:
                    f.write(pack_frame(CMD_ERROR, b"GET_OIT: configured family is not OI"))
            elif cmd == CMD_SEND_CHALLENGES:
                pending_chal = payload       # 无单独响应
            elif cmd == CMD_GET_RESPONSES:
                resp = _do_collect(mmio, cur_family, pending_chal, payload)
                f.write(pack_frame(CMD_GET_RESPONSES, resp))
                pending_chal = None
            else:
                f.write(pack_frame(CMD_ERROR, "unknown cmd 0x{:02X}".format(cmd).encode()))
            f.flush()
    except (BrokenPipeError, ConnectionResetError) as e:
        print("[srv] {} gone: {!r}".format(addr, e), file=sys.stderr)
    except Exception as e:
        print("[srv] {} handler error: {!r}".format(addr, e), file=sys.stderr)
        try:
            f.write(pack_frame(CMD_ERROR, str(e).encode()))
            f.flush()
        except Exception:
            pass
    finally:
        try:
            f.close()
            conn.close()
        except Exception:
            pass
        print("[srv] {} closed".format(addr), file=sys.stderr)


def _do_collect(mmio, family, chal_bytes, n_payload):
    # type: (Mmio, Optional[str], Optional[bytes], bytes) -> bytes
    if family not in HW_FAMILIES:
        raise ProtocolError("collect: family={!r} not configured/supported".format(family))
    if chal_bytes is None:
        raise ProtocolError("collect: no pending challenges (SEND_CHALLENGES first)")
    if len(n_payload) < 4:
        raise ProtocolError("GET_RESPONSES payload need uint32 n")
    n = struct.unpack_from("<I", n_payload, 0)[0]

    if family == "puf4x4":
        rows = unpack_challenges_puf4x4(chal_bytes, n)
        out = bytearray(n * 7)
        for i, row in enumerate(rows):
            r6, r1 = measure_puf4x4(mmio, row)
            base = i * 7
            for k in range(6):
                out[base + k] = r6[k] & 0xFF
            out[base + 6] = r1 & 0xFF
        return bytes(out)

    words = unpack_challenges_binary(chal_bytes, n)
    out = bytearray(n)

    if family == "flam":
        for i, w in enumerate(words):
            out[i] = measure_one(mmio, _flam_word(w)) & 0xFF
    elif family == "ct":
        for i, w in enumerate(words):
            tw, _branch = _ct_word_and_branch(w)
            out[i] = measure_one(mmio, tw) & 0xFF
    elif family == "oi":
        for i, w in enumerate(words):
            bit = 0
            for lane, tw in enumerate(_oi_lane_words(w)):
                bit ^= (measure_oi_lane(mmio, tw, lane) & 1)
            out[i] = bit & 0xFF
    elif family == "oi_lanes":
        out = bytearray(n * 5)
        for i, w in enumerate(words):
            bit = 0
            base = i * 5
            for lane, tw in enumerate(_oi_lane_words(w)):
                lane_bit = measure_oi_lane(mmio, tw, lane) & 1
                out[base + lane] = lane_bit
                bit ^= lane_bit
            out[base + 4] = bit & 0xFF
    elif family in ("apuf_lanes", "ff_lanes"):
        out = bytearray(n * 5)
        measure = measure_apuf_lanes if family == "apuf_lanes" else measure_ff_lanes
        for i, w in enumerate(words):
            lanes = measure(mmio, w)
            base = i * 5
            bit = 0
            for lane, lane_bit in enumerate(lanes):
                out[base + lane] = lane_bit & 1
                bit ^= lane_bit
            out[base + 4] = bit & 1
    elif family == "tpdoa":
        for i, w in enumerate(words):
            out[i] = measure_tpdoa(mmio, w) & 0xFF
    else:
        raise ProtocolError("collect: unexpected family {!r}".format(family))
    return bytes(out)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="PUF-COLP v1 TCP server (PYNQ)")
    ap.add_argument("--bit", required=True, help="PL bitstream (.bit) path")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=3040)
    ap.add_argument("--base", default="0x40000000", help="PUF AXI4-Lite base addr")
    ap.add_argument("--no-program", action="store_true",
                    help="skip PL programming (assume already loaded)")
    args = ap.parse_args()

    base = int(args.base, 0)
    if not args.no_program:
        print("[init] programming PL from {}".format(args.bit), file=sys.stderr)
        load_bitstream(args.bit)
        print("[init] PL programmed", file=sys.stderr)
    else:
        print("[init] skipping PL programming", file=sys.stderr)

    mmio = Mmio(base, 0x1000)
    dbg = mmio.read(REG_DBG)
    print("[init] base={} DBG=0x{:08X}".format(hex(base), dbg), file=sys.stderr)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(4)
    print("[srv] listening on {}:{}".format(args.host, args.port), file=sys.stderr)
    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=handle_conn, args=(conn, addr, mmio), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[srv] shutting down", file=sys.stderr)
    finally:
        srv.close()


if __name__ == "__main__":
    main()
