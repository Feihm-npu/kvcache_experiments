"""Hot-tier ring masking: int16 cache, ChaCha8 pad, de-masked in register.

Implements the frozen design of ``docs/DESIGN_FREEZE_TWO_TIER.md``. The cache a
decode step reads from memory holds only $\\widetilde X = X_q + M \\bmod 2^{16}$;
the pad $M$ is regenerated inside the attention kernel from a counter and never
occupies memory. Unlike the Gaussian mask it replaces, a ring pad has no
averaging mechanism: differencing two residencies yields a uniform ring element.

What a resident cache holds in device memory, and nothing else:

* ``kt``, ``vt``: int16, every content row padded, reserved rows zero;
* ``skt``, ``svt``: the per-channel quantization scales as fp32 bit patterns
  XORed with keystream from their own counter range. A scale is the per-channel
  maximum of the content, so in plaintext it would be a fingerprint of it;
* a clamp counter.

The key is not stored on the device: it is passed to each launch as kernel
arguments. Plaintext K, V and scales exist only while a cache enters residency
(prefill), which is the window the design scopes out.

Kept separate from ``puf_attention`` on purpose. That wrapper carries a dozen
interacting options and a set of audited results; this path is new and is wired
in through its own installer so neither can break the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ROUNDS_DEFAULT = 8
BLOCK_N_DEFAULT = 32          # frozen: mask tile dominates register pressure
INT16_MAX = 32767
HEADROOM_DEFAULT = 2.0
"""Scales are fixed when a page enters residency, but tokens decoded later can
exceed the range the prefill established: measured ratios reach 1.9x per channel
on Qwen3-0.6B. Without headroom those channels wrap silently in int16 and the
affected positions become noise, so the scale reserves a factor and a clamp
backstops anything beyond it."""
SCALE_BASE = 0x7F000000
"""First ChaCha block of the scale counter range. K and V use [blk_k, ...) and
[blk_v, ...); ``check_counter_ranges`` keeps all three disjoint and inside i32."""

# Integer arguments that change per step or per residency. Triton would
# otherwise specialise on their values (divisible by 16, equal to 1), and compile
# inside the decode loop whenever the length crossed a multiple of 16 or a new
# residency drew a new key.
_KEY_ARGS = ["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "n0", "n1", "n2"]


# --------------------------------------------------------------------------
# ChaCha, RFC 8439 state layout
# --------------------------------------------------------------------------
@triton.jit
def _rotl(x, n: tl.constexpr):
    return (x << n) | (x >> (32 - n))


@triton.jit
def _qr(a, b, c, d):
    a += b; d ^= a; d = _rotl(d, 16)
    c += d; b ^= c; b = _rotl(b, 12)
    a += b; d ^= a; d = _rotl(d, 8)
    c += d; b ^= c; b = _rotl(b, 7)
    return a, b, c, d


@triton.jit
def _keystream(ctr, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
               ROUNDS: tl.constexpr):
    zu = ctr * 0
    i0 = zu + 0x61707865; i1 = zu + 0x3320646E
    i2 = zu + 0x79622D32; i3 = zu + 0x6B206574
    i4 = zu + k0.to(tl.uint32); i5 = zu + k1.to(tl.uint32)
    i6 = zu + k2.to(tl.uint32); i7 = zu + k3.to(tl.uint32)
    i8 = zu + k4.to(tl.uint32); i9 = zu + k5.to(tl.uint32)
    i10 = zu + k6.to(tl.uint32); i11 = zu + k7.to(tl.uint32)
    i12 = ctr
    i13 = zu + n0.to(tl.uint32); i14 = zu + n1.to(tl.uint32)
    i15 = zu + n2.to(tl.uint32)
    s0, s1, s2, s3 = i0, i1, i2, i3
    s4, s5, s6, s7 = i4, i5, i6, i7
    s8, s9, s10, s11 = i8, i9, i10, i11
    s12, s13, s14, s15 = i12, i13, i14, i15
    for _ in range(ROUNDS // 2):
        s0, s4, s8, s12 = _qr(s0, s4, s8, s12)
        s1, s5, s9, s13 = _qr(s1, s5, s9, s13)
        s2, s6, s10, s14 = _qr(s2, s6, s10, s14)
        s3, s7, s11, s15 = _qr(s3, s7, s11, s15)
        s0, s5, s10, s15 = _qr(s0, s5, s10, s15)
        s1, s6, s11, s12 = _qr(s1, s6, s11, s12)
        s2, s7, s8, s13 = _qr(s2, s7, s8, s13)
        s3, s4, s9, s14 = _qr(s3, s4, s9, s14)
    return (s0 + i0, s1 + i1, s2 + i2, s3 + i3, s4 + i4, s5 + i5, s6 + i6,
            s7 + i7, s8 + i8, s9 + i9, s10 + i10, s11 + i11, s12 + i12,
            s13 + i13, s14 + i14, s15 + i15)


@triton.jit
def _pad_tile(blk, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
              ROUNDS: tl.constexpr, BN: tl.constexpr, C: tl.constexpr):
    """[BN, C] counters -> [BN, 32*C] pad values, every cipher bit used.

    The join tree's channel permutation is part of the Protect/Attend contract;
    both sides go through this function, so they agree by construction.
    """
    w = _keystream(blk, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2, ROUNDS)
    p0 = tl.join((w[0] & 0xFFFF).to(tl.int32), ((w[0] >> 16) & 0xFFFF).to(tl.int32))
    p1 = tl.join((w[1] & 0xFFFF).to(tl.int32), ((w[1] >> 16) & 0xFFFF).to(tl.int32))
    p2 = tl.join((w[2] & 0xFFFF).to(tl.int32), ((w[2] >> 16) & 0xFFFF).to(tl.int32))
    p3 = tl.join((w[3] & 0xFFFF).to(tl.int32), ((w[3] >> 16) & 0xFFFF).to(tl.int32))
    p4 = tl.join((w[4] & 0xFFFF).to(tl.int32), ((w[4] >> 16) & 0xFFFF).to(tl.int32))
    p5 = tl.join((w[5] & 0xFFFF).to(tl.int32), ((w[5] >> 16) & 0xFFFF).to(tl.int32))
    p6 = tl.join((w[6] & 0xFFFF).to(tl.int32), ((w[6] >> 16) & 0xFFFF).to(tl.int32))
    p7 = tl.join((w[7] & 0xFFFF).to(tl.int32), ((w[7] >> 16) & 0xFFFF).to(tl.int32))
    p8 = tl.join((w[8] & 0xFFFF).to(tl.int32), ((w[8] >> 16) & 0xFFFF).to(tl.int32))
    p9 = tl.join((w[9] & 0xFFFF).to(tl.int32), ((w[9] >> 16) & 0xFFFF).to(tl.int32))
    pa = tl.join((w[10] & 0xFFFF).to(tl.int32), ((w[10] >> 16) & 0xFFFF).to(tl.int32))
    pb = tl.join((w[11] & 0xFFFF).to(tl.int32), ((w[11] >> 16) & 0xFFFF).to(tl.int32))
    pc = tl.join((w[12] & 0xFFFF).to(tl.int32), ((w[12] >> 16) & 0xFFFF).to(tl.int32))
    pd = tl.join((w[13] & 0xFFFF).to(tl.int32), ((w[13] >> 16) & 0xFFFF).to(tl.int32))
    pe = tl.join((w[14] & 0xFFFF).to(tl.int32), ((w[14] >> 16) & 0xFFFF).to(tl.int32))
    pf = tl.join((w[15] & 0xFFFF).to(tl.int32), ((w[15] >> 16) & 0xFFFF).to(tl.int32))
    a0 = tl.reshape(tl.join(p0, p1), (BN, C, 4))
    a1 = tl.reshape(tl.join(p2, p3), (BN, C, 4))
    a2 = tl.reshape(tl.join(p4, p5), (BN, C, 4))
    a3 = tl.reshape(tl.join(p6, p7), (BN, C, 4))
    a4 = tl.reshape(tl.join(p8, p9), (BN, C, 4))
    a5 = tl.reshape(tl.join(pa, pb), (BN, C, 4))
    a6 = tl.reshape(tl.join(pc, pd), (BN, C, 4))
    a7 = tl.reshape(tl.join(pe, pf), (BN, C, 4))
    b0 = tl.reshape(tl.join(a0, a1), (BN, C, 8))
    b1 = tl.reshape(tl.join(a2, a3), (BN, C, 8))
    b2 = tl.reshape(tl.join(a4, a5), (BN, C, 8))
    b3 = tl.reshape(tl.join(a6, a7), (BN, C, 8))
    c0 = tl.reshape(tl.join(b0, b1), (BN, C, 16))
    c1 = tl.reshape(tl.join(b2, b3), (BN, C, 16))
    t = tl.reshape(tl.join(c0, c1), (BN, C, 32))
    return tl.reshape(t, (BN, C * 32))


@triton.jit
def _word_tile(blk, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
               ROUNDS: tl.constexpr, C: tl.constexpr):
    """[C] counters -> [16*C] uint32 keystream words, for the scale XOR."""
    w = _keystream(blk, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2, ROUNDS)
    a0 = tl.join(w[0], w[1]); a1 = tl.join(w[2], w[3])
    a2 = tl.join(w[4], w[5]); a3 = tl.join(w[6], w[7])
    a4 = tl.join(w[8], w[9]); a5 = tl.join(w[10], w[11])
    a6 = tl.join(w[12], w[13]); a7 = tl.join(w[14], w[15])
    b0 = tl.reshape(tl.join(a0, a1), (C, 4))
    b1 = tl.reshape(tl.join(a2, a3), (C, 4))
    b2 = tl.reshape(tl.join(a4, a5), (C, 4))
    b3 = tl.reshape(tl.join(a6, a7), (C, 4))
    c0 = tl.reshape(tl.join(b0, b1), (C, 8))
    c1 = tl.reshape(tl.join(b2, b3), (C, 8))
    return tl.reshape(tl.join(c0, c1), (C * 16,))


@triton.jit
def _scale(S, h, blk_s, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
           MASKED: tl.constexpr, ROUNDS: tl.constexpr, D: tl.constexpr):
    """Load one head's D scales, removing the XOR in register when masked."""
    cs = tl.arange(0, D)
    w = tl.load(S + h * D + cs).to(tl.uint32, bitcast=True)
    if MASKED == 1:
        ctr = (blk_s + h * (D // 16) + tl.arange(0, D // 16)).to(tl.uint32)
        w = w ^ _word_tile(ctr, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
                           ROUNDS, D // 16)
    return w.to(tl.float32, bitcast=True)


@triton.jit
def _wrap16(x):
    y = x & 0xFFFF
    return tl.where(y >= 32768, y - 65536, y)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n_rows", "cap", "pos_base", "blk_base"]
            + _KEY_ARGS)
def _protect(Xq, Xt, n_rows, cap, pos_base, stride_h, stride_n, blk_base,
             k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
             ROUNDS: tl.constexpr, D: tl.constexpr, BN: tl.constexpr,
             SIGN: tl.constexpr = 1):
    """Mask rows [pos_base, pos_base+n_rows) of a cache whose capacity is ``cap``.

    Counters stride by ``cap``, not by the current length, so appending a row
    never shifts the counter of any existing row. ``SIGN=-1`` removes the pad,
    which eviction uses to hand the quantized rows to the cold tier.
    """
    h = tl.program_id(0); sp = tl.program_id(1)
    cs = tl.arange(0, D)
    row = sp * BN + tl.arange(0, BN)
    valid = row < n_rows
    pos = pos_base + row
    off = h * stride_h + pos[:, None] * stride_n + cs[None, :]
    xq = tl.load(Xq + off, mask=valid[:, None], other=0).to(tl.int32)
    cch = tl.arange(0, D // 32)
    base = blk_base + (h * cap + pos[:, None]) * (D // 32) + cch[None, :]
    m = _pad_tile(base.to(tl.uint32), k0, k1, k2, k3, k4, k5, k6, k7,
                  n0, n1, n2, ROUNDS, BN, D // 32)
    tl.store(Xt + off, _wrap16(xq + SIGN * m).to(tl.int16), mask=valid[:, None])


@triton.jit(do_not_specialize=["blk_sk", "blk_sv"] + _KEY_ARGS)
def _xor_scales(SK, SV, OK_, OV, blk_sk, blk_sv,
                k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
                ROUNDS: tl.constexpr, D: tl.constexpr):
    """Encrypt (or decrypt: XOR is its own inverse) one head's K and V scales."""
    h = tl.program_id(0); which = tl.program_id(1)
    src = SK if which == 0 else SV
    dst = OK_ if which == 0 else OV
    blk = blk_sk if which == 0 else blk_sv
    cs = tl.arange(0, D)
    w = tl.load(src + h * D + cs).to(tl.uint32, bitcast=True)
    ctr = (blk + h * (D // 16) + tl.arange(0, D // 16)).to(tl.uint32)
    w = w ^ _word_tile(ctr, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
                       ROUNDS, D // 16)
    tl.store(dst + h * D + cs, w.to(tl.int32, bitcast=True))


@triton.jit(do_not_specialize=["cap", "pos", "blk_k", "blk_v", "blk_sk",
                               "blk_sv"] + _KEY_ARGS)
def _append_row(KN, VN, K, V, SK, SV, CLAMP, cap, pos, stride_h, stride_n,
                blk_k, blk_v, blk_sk, blk_sv,
                k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
                MASKED: tl.constexpr, ROUNDS: tl.constexpr, D: tl.constexpr):
    """Quantize, count clamps, pad and store one decoded row of K or V.

    One launch covers both tensors (program_id(1) selects K or V) and replaces
    the dozen host-side tensor ops a decode step used to spend per layer. The
    scale is decrypted in register, so no plaintext scale is ever in memory.
    """
    h = tl.program_id(0); which = tl.program_id(1)
    src = KN if which == 0 else VN
    dst = K if which == 0 else V
    S = SK if which == 0 else SV
    blk = blk_k if which == 0 else blk_v
    blk_s = blk_sk if which == 0 else blk_sv
    cs = tl.arange(0, D)
    s = _scale(S, h, blk_s, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
               MASKED, ROUNDS, D)
    y = tl.math.div_rn(tl.load(src + h * D + cs), s)
    tl.atomic_add(CLAMP, tl.sum((tl.abs(y) > 32767.0).to(tl.int32)))
    # clamp-then-round equals round-then-clamp at these bounds, and rint is the
    # round-half-to-even that torch.round uses at residency entry
    q = libdevice.rint(tl.minimum(tl.maximum(y, -32767.0), 32767.0)).to(tl.int32)
    if MASKED == 1:
        ctr = (blk + (h * cap + pos) * (D // 32)
               + tl.arange(0, D // 32)).to(tl.uint32)
        m = _pad_tile(ctr[None, :], k0, k1, k2, k3, k4, k5, k6, k7,
                      n0, n1, n2, ROUNDS, 1, D // 32)
        q = _wrap16(q + tl.reshape(m, (D,)))
    tl.store(dst + h * stride_h + pos * stride_n + cs, q.to(tl.int16))


@triton.jit(do_not_specialize=["N", "CAP", "blk_k", "blk_v", "blk_sk",
                               "blk_sv"] + _KEY_ARGS)
def _decode_split(Q, K, V, SK, SV, PartO, PartM, PartL, N, CAP, GQ,
                  stride_qh, stride_kh, stride_kn, blk_k, blk_v, blk_sk, blk_sv,
                  k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
                  MASKED: tl.constexpr, ROUNDS: tl.constexpr,
                  D: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr):
    """Partial attention for one (query head, split). GQA: qh -> qh // GQ."""
    qh = tl.program_id(0); sp = tl.program_id(1)
    h = qh // GQ
    cs = tl.arange(0, D)
    q = tl.load(Q + qh * stride_qh + cs).to(tl.float32)
    sk = _scale(SK, h, blk_sk, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
                MASKED, ROUNDS, D)
    sv = _scale(SV, h, blk_sv, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
                MASKED, ROUNDS, D)
    qs = q * sk

    m_i = tl.full((1,), -float("inf"), tl.float32)
    l_i = tl.zeros((1,), tl.float32)
    acc = tl.zeros((D,), tl.float32)

    lo = sp * SPLIT
    hi = tl.minimum(lo + SPLIT, N)
    for start in range(lo, hi, BN):
        pos = start + tl.arange(0, BN)
        valid = pos < hi
        off = h * stride_kh + pos[:, None] * stride_kn + cs[None, :]
        kt = tl.load(K + off, mask=valid[:, None], other=0).to(tl.int32)
        vt = tl.load(V + off, mask=valid[:, None], other=0).to(tl.int32)
        if MASKED == 1:
            cch = tl.arange(0, D // 32)
            base = (h * CAP + pos[:, None]) * (D // 32) + cch[None, :]
            mk = _pad_tile((base + blk_k).to(tl.uint32), k0, k1, k2, k3, k4,
                           k5, k6, k7, n0, n1, n2, ROUNDS, BN, D // 32)
            mv = _pad_tile((base + blk_v).to(tl.uint32), k0, k1, k2, k3, k4,
                           k5, k6, k7, n0, n1, n2, ROUNDS, BN, D // 32)
            kt = _wrap16(kt - mk)
            vt = _wrap16(vt - mv)
        kf = kt.to(tl.float32)
        vf = vt.to(tl.float32) * sv[None, :]

        sc = tl.sum(kf * qs[None, :], axis=1) * (1.0 / tl.sqrt(float(D)))
        sc = tl.where(valid, sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc))
        alpha = tl.exp(m_i - m_new)
        pr = tl.exp(sc - m_new)
        l_i = l_i * alpha + tl.sum(pr)
        acc = acc * alpha + tl.sum(pr[:, None] * vf, axis=0)
        m_i = m_new

    ns = tl.num_programs(1)
    tl.store(PartO + (qh * ns + sp) * D + cs, acc)
    tl.store(PartM + qh * ns + sp, tl.sum(m_i))
    tl.store(PartL + qh * ns + sp, tl.sum(l_i))


@triton.jit(do_not_specialize=["N", "CAP", "blk_k", "blk_v", "blk_sk",
                               "blk_sv"] + _KEY_ARGS)
def _decode_split_gqa(Q, K, V, SK, SV, PartO, PartM, PartL, N, CAP,
                      stride_qh, stride_kh, stride_kn, blk_k, blk_v, blk_sk, blk_sv,
                      k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
                      MASKED: tl.constexpr, ROUNDS: tl.constexpr,
                      D: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr,
                      GQ: tl.constexpr, GQP: tl.constexpr):
    """Partial attention for one (KV head, split) and all GQ query heads of it.

    Each K/V tile is loaded, and its pad regenerated, once per KV head rather than
    once per query head: under grouped-query attention the per-query-head kernel
    (`_decode_split`) ran the cipher GQ times over the same elements. The query
    heads of the group are rows of a [GQP, D] block (GQP = GQ rounded up to a
    power of two); each row is scored and accumulated with exactly the
    per-query-head arithmetic, so the result is the same as `_decode_split`'s.
    """
    h = tl.program_id(0); sp = tl.program_id(1)
    cs = tl.arange(0, D)
    gs = tl.arange(0, GQP)
    gval = gs < GQ
    qh = h * GQ + gs
    q = tl.load(Q + qh[:, None] * stride_qh + cs[None, :], mask=gval[:, None],
                other=0.0).to(tl.float32)
    sk = _scale(SK, h, blk_sk, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
                MASKED, ROUNDS, D)
    sv = _scale(SV, h, blk_sv, k0, k1, k2, k3, k4, k5, k6, k7, n0, n1, n2,
                MASKED, ROUNDS, D)
    qs = q * sk[None, :]

    m_i = tl.full((GQP,), -float("inf"), tl.float32)
    l_i = tl.zeros((GQP,), tl.float32)
    acc = tl.zeros((GQP, D), tl.float32)

    lo = sp * SPLIT
    hi = tl.minimum(lo + SPLIT, N)
    for start in range(lo, hi, BN):
        pos = start + tl.arange(0, BN)
        valid = pos < hi
        off = h * stride_kh + pos[:, None] * stride_kn + cs[None, :]
        kt = tl.load(K + off, mask=valid[:, None], other=0).to(tl.int32)
        vt = tl.load(V + off, mask=valid[:, None], other=0).to(tl.int32)
        if MASKED == 1:
            cch = tl.arange(0, D // 32)
            base = (h * CAP + pos[:, None]) * (D // 32) + cch[None, :]
            mk = _pad_tile((base + blk_k).to(tl.uint32), k0, k1, k2, k3, k4,
                           k5, k6, k7, n0, n1, n2, ROUNDS, BN, D // 32)
            mv = _pad_tile((base + blk_v).to(tl.uint32), k0, k1, k2, k3, k4,
                           k5, k6, k7, n0, n1, n2, ROUNDS, BN, D // 32)
            kt = _wrap16(kt - mk)
            vt = _wrap16(vt - mv)
        kf = kt.to(tl.float32)
        vf = vt.to(tl.float32) * sv[None, :]

        sc = tl.zeros((GQP, BN), tl.float32)
        for g in tl.static_range(GQP):
            qg = tl.sum(tl.where(gs[:, None] == g, qs, 0.0), axis=0)
            sg = tl.sum(kf * qg[None, :], axis=1) * (1.0 / tl.sqrt(float(D)))
            sc = tl.where(gs[:, None] == g, sg[None, :], sc)
        sc = tl.where(valid[None, :], sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.exp(m_i - m_new)
        pr = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pr, axis=1)
        pv = tl.zeros((GQP, D), tl.float32)
        for g in tl.static_range(GQP):
            pg = tl.sum(tl.where(gs[:, None] == g, pr, 0.0), axis=0)
            og = tl.sum(pg[:, None] * vf, axis=0)
            pv = tl.where(gs[:, None] == g, og[None, :], pv)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    ns = tl.num_programs(1)
    rows = qh * ns + sp
    tl.store(PartO + rows[:, None] * D + cs[None, :], acc, mask=gval[:, None])
    tl.store(PartM + rows, m_i, mask=gval)
    tl.store(PartL + rows, l_i, mask=gval)


@triton.jit
def _reduce(PartO, PartM, PartL, Out, NS, D: tl.constexpr, S: tl.constexpr):
    qh = tl.program_id(0)
    cs = tl.arange(0, D); sp = tl.arange(0, S)
    ok = sp < NS
    m = tl.load(PartM + qh * NS + sp, mask=ok, other=-float("inf"))
    l = tl.load(PartL + qh * NS + sp, mask=ok, other=0.0)
    w = tl.where(ok, tl.exp(m - tl.max(m)), 0.0)
    po = tl.load(PartO + (qh * NS + sp)[:, None] * D + cs[None, :],
                 mask=ok[:, None], other=0.0)
    tl.store(Out + qh * D + cs,
             tl.sum(po * w[:, None], axis=0) / tl.sum(l * w))


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------
def _q16(x: torch.Tensor) -> torch.Tensor:
    """Round into int16, clamping instead of wrapping.

    ``Tensor.to(torch.int16)`` wraps out-of-range values without warning, which
    turns one out-of-scale channel into noise for that whole position.
    """
    return torch.round(x).clamp_(-INT16_MAX, INT16_MAX).to(torch.int16)


def key_args(key, nonce) -> tuple:
    """The 8 key and 3 nonce words as signed 32-bit kernel arguments.

    Triton types a scalar argument by its value, so a word at or above 2^31
    would be typed i64 and compile a different kernel from a word below it.
    Mapping every word into the signed 32-bit range keeps one type; the kernel
    reinterprets it as uint32, so the bits are unchanged.
    """
    words = [int(x) & 0xFFFFFFFF for x in (*key, *nonce)]
    if len(words) != 11:
        raise ValueError(f"expected 8 key + 3 nonce words, got {len(words)}")
    return tuple(w - (1 << 32) if w >= 1 << 31 else w for w in words)


@dataclass
class RingCache:
    """One layer's hot-tier state. See the module docstring for what is in memory.

    ``split`` and ``n_split`` are fixed from the capacity, not from ``n``. They
    are kernel constexprs, so deriving them from the current length would change
    the specialisation on every appended token and put Triton compilation inside
    the decode loop.
    """
    kt: torch.Tensor                      # [HKV, cap, D] int16
    vt: torch.Tensor
    skt: torch.Tensor                     # [HKV, D] int32: fp32 bits ^ keystream
    svt: torch.Tensor
    n: int
    blk_k: int
    blk_v: int
    masked: bool
    kargs: tuple = field(default_factory=tuple, repr=False)
    """Key words for the launches. Host memory only: never copied to the device
    except as per-launch kernel arguments."""
    blk_sk: int = SCALE_BASE
    blk_sv: int = SCALE_BASE
    split: int = 0
    n_split: int = 0
    buf: dict = field(default_factory=dict)
    clamped: torch.Tensor | None = None
    """[1] int32 count of elements the clamp caught, so range loss is visible,
    not silent. Kept on the device; read with ``int()`` only when reported."""
    residency_id: str = ""
    """Public identifier of the residency whose key produced this pad. Every entry
    into the hot tier must use a fresh one; see ``ring_install.new_residency_id``."""

    def plan(self, splits: int, bn: int) -> None:
        cap = self.kt.shape[1]
        self.split = max(bn, triton.next_power_of_2(max(1, -(-cap // splits))))
        self.n_split = triton.cdiv(cap, self.split)

    def buffers(self, hq: int, d: int):
        """Partial-result buffers, allocated once per residency."""
        if not self.buf:
            dev = self.kt.device
            self.buf = {
                "po": torch.empty(hq * self.n_split * d, device=dev,
                                  dtype=torch.float32),
                "pm": torch.empty(hq * self.n_split, device=dev,
                                  dtype=torch.float32),
                "pl": torch.empty(hq * self.n_split, device=dev,
                                  dtype=torch.float32),
                "out": torch.empty(hq, d, device=dev, dtype=torch.float32),
            }
        return self.buf


def check_counter_ranges(h: int, cap: int, d: int, blk_k: int,
                         blk_v: int, blk_s: int = SCALE_BASE) -> None:
    """K, V and scale counters must be disjoint and stay inside signed 32 bits.

    One key serves every pad of a residency, so the counter ranges are what keep
    those pads independent. An overlap would reuse keystream between them. The
    kernels compute counters in i32 before reinterpreting them as the uint32
    ChaCha block counter, so every range must end below 2^31.
    """
    span = h * cap * (d // 32)
    s_span = 2 * h * (d // 16)
    ranges = sorted([(blk_k, blk_k + span, "K"), (blk_v, blk_v + span, "V"),
                     (blk_s, blk_s + s_span, "scale")])
    for (_, hi1, a), (lo2, _, b) in zip(ranges, ranges[1:]):
        if hi1 > lo2:
            raise ValueError(f"counter ranges overlap: {a} ends at {hi1}, "
                             f"{b} starts at {lo2}")
    if ranges[-1][1] > 1 << 31:
        raise ValueError(f"counter range of {ranges[-1][2]} ends at "
                         f"{ranges[-1][1]}, which overflows signed 32 bits")


def quantize_and_protect(k: torch.Tensor, v: torch.Tensor, key, nonce,
                         blk_k: int, blk_v: int, masked: bool = True,
                         rounds: int = ROUNDS_DEFAULT,
                         bn: int = BLOCK_N_DEFAULT,
                         headroom: float = HEADROOM_DEFAULT,
                         n_valid: int | None = None,
                         residency_id: str = "") -> RingCache:
    """Entry into the hot tier: per-channel quantize, then pad.

    ``k``, ``v``: [HKV, cap, D] fp32, of which the first ``n_valid`` rows hold
    content and the rest are reserved for decoded positions. Scales are per
    (head, channel), computed from the content entering residency, which is the
    static-quantization choice Gate 1b measured at 6.5e-5 output error.

    Only the ``n_valid`` content rows are padded. The reserved rows are left at
    zero rather than padded: zero is known plaintext, so padding it would put
    the pad for every future position into memory, and the decode step that
    later writes real data at that position would reuse the same pad. Two
    snapshots of the cache would then recover the decoded K and V exactly.
    """
    H, cap, D = k.shape
    n = cap if n_valid is None else int(n_valid)
    if not 0 < n <= cap:
        raise ValueError(f"n_valid={n_valid} outside (0, {cap}]")
    blk_sk, blk_sv = SCALE_BASE, SCALE_BASE + H * (D // 16)
    check_counter_ranges(H, cap, D, blk_k, blk_v, SCALE_BASE)
    kargs = key_args(key, nonce)
    lim = INT16_MAX
    sk = (k[:, :n].abs().amax(1) * headroom / lim).clamp_min(1e-12)   # [H, D]
    sv = (v[:, :n].abs().amax(1) * headroom / lim).clamp_min(1e-12)
    kq = _q16(k / sk[:, None, :]).contiguous()
    vq = _q16(v / sv[:, None, :]).contiguous()
    kq[:, n:] = 0
    vq[:, n:] = 0
    sk_bits = sk.contiguous().view(torch.int32)
    sv_bits = sv.contiguous().view(torch.int32)
    clamped = torch.zeros(1, dtype=torch.int32, device=k.device)
    common = dict(n=n, blk_k=blk_k, blk_v=blk_v, masked=masked, kargs=kargs,
                  blk_sk=blk_sk, blk_sv=blk_sv, clamped=clamped,
                  residency_id=residency_id)
    if not masked:
        return RingCache(kq, vq, sk_bits.clone(), sv_bits.clone(), **common)
    kt = torch.zeros_like(kq); vt = torch.zeros_like(vq)
    grid = (H, triton.cdiv(n, bn))
    _protect[grid](kq, kt, n, cap, 0, cap * D, D, blk_k, *kargs,
                   ROUNDS=rounds, D=D, BN=bn)
    _protect[grid](vq, vt, n, cap, 0, cap * D, D, blk_v, *kargs,
                   ROUNDS=rounds, D=D, BN=bn)
    skt = torch.empty_like(sk_bits); svt = torch.empty_like(sv_bits)
    _xor_scales[(H, 2)](sk_bits, sv_bits, skt, svt, blk_sk, blk_sv, *kargs,
                        ROUNDS=rounds, D=D)
    # the plaintext staging copies are this layer's only native state; clear
    # them rather than leave them in the allocator's free pool
    for t in (kq, vq, sk, sv):
        t.zero_()
    return RingCache(kt, vt, skt, svt, **common)


def reveal_scales(cache: RingCache, rounds: int = ROUNDS_DEFAULT):
    """Decrypt the scales. For tests and diagnostics only, never on a decode path."""
    H, D = cache.skt.shape
    if not cache.masked:
        return cache.skt.view(torch.float32), cache.svt.view(torch.float32)
    sk = torch.empty_like(cache.skt); sv = torch.empty_like(cache.svt)
    _xor_scales[(H, 2)](cache.skt, cache.svt, sk, sv, cache.blk_sk,
                        cache.blk_sv, *cache.kargs, ROUNDS=rounds, D=D)
    return sk.view(torch.float32), sv.view(torch.float32)


def ring_decode(q: torch.Tensor, cache: RingCache, *, masked: bool | None = None,
                rounds: int = ROUNDS_DEFAULT, bn: int = BLOCK_N_DEFAULT,
                splits: int = 32, kernel: str = "gqa") -> torch.Tensor:
    """One decode step. ``q``: [HQ, D] fp32 (already RoPE'd). Returns [HQ, D].

    Splits past the current length contribute nothing: their running maximum
    stays at -inf, so the reduction weights them to zero. ``kernel="gqa"`` runs
    one program per (KV head, split) for all query heads of the group;
    ``"per_head"`` is the earlier one-program-per-query-head kernel, kept for
    comparison.
    """
    masked = cache.masked if masked is None else masked
    if masked != cache.masked:
        raise ValueError("decode mode does not match how the cache was built")
    HQ, D = q.shape
    HKV, CAP, _ = cache.kt.shape
    GQ = HQ // HKV
    if not cache.n_split:
        cache.plan(splits, bn)
    S, SPLIT = cache.n_split, cache.split
    b = cache.buffers(HQ, D)
    # the query carries the K scale, so the kernel applies it only once.
    # Without grouping (GQ = 1) there is nothing to share, and the per-head
    # kernel is the one whose masked and unmasked builds agree bit for bit.
    if kernel == "gqa" and GQ > 1:
        _decode_split_gqa[(HKV, S)](
            q.contiguous(), cache.kt, cache.vt, cache.skt, cache.svt,
            b["po"], b["pm"], b["pl"],
            cache.n, CAP, D, CAP * D, D, cache.blk_k, cache.blk_v,
            cache.blk_sk, cache.blk_sv, *cache.kargs,
            MASKED=1 if masked else 0, ROUNDS=rounds, D=D, BN=bn, SPLIT=SPLIT,
            GQ=GQ, GQP=triton.next_power_of_2(GQ))
    elif kernel in ("gqa", "per_head"):
        _decode_split[(HQ, S)](
            q.contiguous(), cache.kt, cache.vt, cache.skt, cache.svt,
            b["po"], b["pm"], b["pl"],
            cache.n, CAP, GQ, D, CAP * D, D, cache.blk_k, cache.blk_v,
            cache.blk_sk, cache.blk_sv, *cache.kargs,
            MASKED=1 if masked else 0, ROUNDS=rounds, D=D, BN=bn, SPLIT=SPLIT)
    else:
        raise ValueError(f"kernel must be gqa or per_head (got {kernel})")
    _reduce[(HQ,)](b["po"], b["pm"], b["pl"], b["out"], S, D=D,
                   S=triton.next_power_of_2(S))
    return b["out"]


def append(cache: RingCache, k_new: torch.Tensor, v_new: torch.Tensor, *,
           masked: bool | None = None, rounds: int = ROUNDS_DEFAULT,
           bn: int = BLOCK_N_DEFAULT) -> None:
    """Append one decoded position, reusing the residency's scales and counters.

    Each position's counter is used exactly once per residency: the row at ``n``
    has never been padded before (``quantize_and_protect`` leaves reserved rows at
    zero) and ``n`` only increases. Rewriting an earlier position would reuse its
    pad, so it requires a new residency rather than a call here.
    """
    masked = cache.masked if masked is None else masked
    if masked != cache.masked:
        raise ValueError("append mode does not match how the cache was built")
    H, D = k_new.shape
    n = cache.n
    cap = cache.kt.shape[1]
    if n >= cap:
        raise RuntimeError(f"ring cache full: n={n} cap={cap}")
    _append_row[(H, 2)](k_new.contiguous(), v_new.contiguous(), cache.kt,
                        cache.vt, cache.skt, cache.svt, cache.clamped, cap, n,
                        cap * D, D, cache.blk_k, cache.blk_v, cache.blk_sk,
                        cache.blk_sv, *cache.kargs,
                        MASKED=1 if masked else 0, ROUNDS=rounds, D=D)
    cache.n = n + 1


def scrub(*tensors) -> None:
    """Overwrite tensors that held plaintext before they return to the allocator.

    Best effort: it covers buffers this package owns. It is not a guarantee
    about copies made by other libraries or by the CUDA caching allocator.
    """
    for t in tensors:
        if t is not None:
            t.zero_()


def export_quantized(cache: RingCache, rounds: int = ROUNDS_DEFAULT,
                     bn: int = BLOCK_N_DEFAULT):
    """Eviction: the residency's quantized rows and scales, pad removed.

    Returns ``(kq, vq, sk, sv, n)`` with ``kq``/``vq`` int16 [H, n, D] and the
    scales fp32 [H, D]. The result is plaintext and is meant to go straight into
    the cold tier, which seals it; the caller scrubs it after sealing. The
    full-capacity staging buffers used here are scrubbed before returning.
    """
    H, cap, D = cache.kt.shape
    n = cache.n
    sk, sv = reveal_scales(cache, rounds)
    if not cache.masked:
        return (cache.kt[:, :n].clone(), cache.vt[:, :n].clone(), sk.clone(),
                sv.clone(), n)
    kq = torch.empty_like(cache.kt); vq = torch.empty_like(cache.vt)
    grid = (H, triton.cdiv(n, bn))
    _protect[grid](cache.kt, kq, n, cap, 0, cap * D, D, cache.blk_k,
                   *cache.kargs, ROUNDS=rounds, D=D, BN=bn, SIGN=-1)
    _protect[grid](cache.vt, vq, n, cap, 0, cap * D, D, cache.blk_v,
                   *cache.kargs, ROUNDS=rounds, D=D, BN=bn, SIGN=-1)
    out = (kq[:, :n].clone(), vq[:, :n].clone(), sk.clone(), sv.clone(), n)
    scrub(kq, vq, sk, sv)
    return out


def restore_quantized(kq: torch.Tensor, vq: torch.Tensor, sk: torch.Tensor,
                      sv: torch.Tensor, cap: int, key, nonce, blk_k: int,
                      blk_v: int, rounds: int = ROUNDS_DEFAULT,
                      bn: int = BLOCK_N_DEFAULT,
                      residency_id: str = "") -> RingCache:
    """Load: enter a fresh residency from quantized rows, without requantizing.

    Reusing the evicted scales keeps the values bit-identical across the move,
    so decode after a reload computes exactly what it would have computed had
    the cache never left. The pad is new: a fresh key means fresh counters.
    """
    H, n, D = kq.shape
    if not 0 < n <= cap:
        raise ValueError(f"n={n} outside (0, {cap}]")
    blk_sk, blk_sv = SCALE_BASE, SCALE_BASE + H * (D // 16)
    check_counter_ranges(H, cap, D, blk_k, blk_v, SCALE_BASE)
    kargs = key_args(key, nonce)
    stage_k = torch.zeros(H, cap, D, dtype=torch.int16, device=kq.device)
    stage_v = torch.zeros_like(stage_k)
    stage_k[:, :n] = kq; stage_v[:, :n] = vq
    kt = torch.zeros_like(stage_k); vt = torch.zeros_like(stage_v)
    grid = (H, triton.cdiv(n, bn))
    _protect[grid](stage_k, kt, n, cap, 0, cap * D, D, blk_k, *kargs,
                   ROUNDS=rounds, D=D, BN=bn)
    _protect[grid](stage_v, vt, n, cap, 0, cap * D, D, blk_v, *kargs,
                   ROUNDS=rounds, D=D, BN=bn)
    sk_bits = sk.float().contiguous().view(torch.int32)
    sv_bits = sv.float().contiguous().view(torch.int32)
    skt = torch.empty_like(sk_bits); svt = torch.empty_like(sv_bits)
    _xor_scales[(H, 2)](sk_bits, sv_bits, skt, svt, blk_sk, blk_sv, *kargs,
                        ROUNDS=rounds, D=D)
    stage_k.zero_(); stage_v.zero_()
    return RingCache(kt, vt, skt, svt, n=n, blk_k=blk_k, blk_v=blk_v,
                     masked=True, kargs=kargs, blk_sk=blk_sk, blk_sv=blk_sv,
                     clamped=torch.zeros(1, dtype=torch.int32, device=kq.device),
                     residency_id=residency_id)
