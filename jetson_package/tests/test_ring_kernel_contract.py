"""Kernel contract of the ring hot tier: the pad is ChaCha8, and nothing recompiles.

The key reaches the kernels as an int32 device tensor. Words at or above 2^31 are
stored as negative int32, so the conformance test uses keys whose words set the
top bit, a case the RFC 8439 test key (all words below 2^31) never exercises.

The compile test guards the property the end-to-end harness depends on: a new
residency (fresh random key) and a growing length must reuse compiled kernels.
Its negative control shows the guard does see value specialisation, which is
what put compilation inside the decode loop before the fix.
"""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from puf4secure_kvcache import ring_attn as ra
from puf4secure_kvcache.harness_guard import KernelGuard

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CONSTANTS = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)
K_BASE, V_BASE = 0, 0x40000000


# ------------------------------------------------------- NumPy reference
def _rotl32(x, n):
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def _chacha_block(key, counter, nonce, rounds):
    s = list(CONSTANTS) + list(key) + [counter & 0xFFFFFFFF] + list(nonce)
    w = list(s)

    def qr(a, b, c, d):
        w[a] = (w[a] + w[b]) & 0xFFFFFFFF; w[d] = _rotl32(w[d] ^ w[a], 16)
        w[c] = (w[c] + w[d]) & 0xFFFFFFFF; w[b] = _rotl32(w[b] ^ w[c], 12)
        w[a] = (w[a] + w[b]) & 0xFFFFFFFF; w[d] = _rotl32(w[d] ^ w[a], 8)
        w[c] = (w[c] + w[d]) & 0xFFFFFFFF; w[b] = _rotl32(w[b] ^ w[c], 7)

    for _ in range(rounds // 2):
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
    return [(w[i] + s[i]) & 0xFFFFFFFF for i in range(16)]


def _ref_rows(key, nonce, rounds, d, first_counter, n_rows):
    """Pad rows as int16, mirroring the join tree of ``_pad_tile``."""
    c = d // 32
    words = np.array([_chacha_block(key, first_counter + b, nonce, rounds)
                      for b in range(n_rows * c)], dtype=np.uint64)
    words = words.reshape(n_rows, c, 16)
    lo = (words & 0xFFFF).astype(np.int64)
    hi = ((words >> 16) & 0xFFFF).astype(np.int64)
    lvl = [np.stack([lo[:, :, i], hi[:, :, i]], axis=-1) for i in range(16)]
    width = 2
    while len(lvl) > 1:
        lvl = [np.stack([lvl[j], lvl[j + 1]], axis=-1).reshape(n_rows, c, width * 2)
               for j in range(0, len(lvl), 2)]
        width *= 2
    out = lvl[0].reshape(n_rows, d)
    return np.where(out >= 32768, out - 65536, out).astype(np.int16)


def _key(top_bit: bool):
    words = list(struct.unpack("<11I", os.urandom(44)))
    if top_bit:
        words = [w | 0x80000000 for w in words]
    return words[:8], words[8:]


# ------------------------------------------------------------ conformance
@cuda
@pytest.mark.parametrize("top_bit", [True, False])
def test_pad_is_chacha8_with_any_key_words(top_bit):
    """Zero plaintext, so the stored cache is the pad itself."""
    key, nonce = _key(top_bit)
    h, cap, n, d = 2, 6, 4, 128
    z = torch.zeros(h, cap, d, device="cuda")
    c = ra.quantize_and_protect(z, z.clone(), key, nonce, K_BASE, V_BASE,
                                masked=True, n_valid=n)
    for base, got in ((K_BASE, c.kt), (V_BASE, c.vt)):
        for hh in range(h):
            want = _ref_rows(key, nonce, 8, d, base + hh * cap * (d // 32), n)
            assert np.array_equal(got[hh, :n].cpu().numpy(), want)
    # the appended row uses the counter of its own position
    ra.append(c, torch.zeros(h, d, device="cuda"),
              torch.zeros(h, d, device="cuda"), masked=True)
    for base, got in ((K_BASE, c.kt), (V_BASE, c.vt)):
        for hh in range(h):
            want = _ref_rows(key, nonce, 8, d,
                             base + (hh * cap + n) * (d // 32), 1)
            assert np.array_equal(got[hh, n:n + 1].cpu().numpy(), want)


def test_key_args_round_trip_every_word_as_signed_i32():
    key, nonce = _key(True)
    ka = ra.key_args(key, nonce)
    assert all(-(1 << 31) <= x < (1 << 31) for x in ka)
    assert [x & 0xFFFFFFFF for x in ka] == key + nonce


def _ref_words(key, nonce, rounds, first_counter, n_blocks):
    """Keystream words in the order of ``_word_tile``'s join tree."""
    w = np.array([_chacha_block(key, first_counter + b, nonce, rounds)
                  for b in range(n_blocks)], dtype=np.uint64)       # [C, 16]
    lvl, width = [w[:, i] for i in range(16)], 1
    while len(lvl) > 1:
        lvl = [np.stack([lvl[j], lvl[j + 1]], axis=-1).reshape(n_blocks, width * 2)
               for j in range(0, len(lvl), 2)]
        width *= 2
    return lvl[0].reshape(-1).astype(np.uint32)


@cuda
def test_scales_are_encrypted_with_their_own_keystream():
    """Memory holds scale bits XOR keystream from the scale counter range."""
    key, nonce = _key(True)
    h, cap, n, d = 2, 8, 6, 128
    g = torch.Generator(device="cpu").manual_seed(7)
    k = (torch.randn(h, cap, d, generator=g) * 3).cuda()
    v = (torch.randn(h, cap, d, generator=g) * 3).cuda()
    c = ra.quantize_and_protect(k, v, key, nonce, K_BASE, V_BASE,
                                masked=True, n_valid=n)
    sk, sv = ra.reveal_scales(c)
    assert torch.equal(sk, (k[:, :n].abs().amax(1) * ra.HEADROOM_DEFAULT
                            / ra.INT16_MAX).clamp_min(1e-12))
    for blk, enc, plain in ((c.blk_sk, c.skt, sk), (c.blk_sv, c.svt, sv)):
        for hh in range(h):
            ks = _ref_words(key, nonce, 8, blk + hh * (d // 16), d // 16)
            bits = plain[hh].cpu().view(torch.int32).numpy().view(np.uint32)
            got = enc[hh].cpu().numpy().view(np.uint32)
            assert np.array_equal(got, bits ^ ks)


@cuda
def test_fused_append_quantizes_exactly_like_residency_entry():
    """The in-kernel quantizer must equal the torch one, clamp included."""
    h, cap, n, d = 4, 40, 32, 128
    g = torch.Generator(device="cpu").manual_seed(8)
    k = torch.randn(h, cap, d, generator=g).cuda()
    v = torch.randn(h, cap, d, generator=g).cuda()
    key, nonce = _key(True)
    c = ra.quantize_and_protect(k, v, key, nonce, K_BASE, V_BASE,
                                masked=False, n_valid=n)
    sk, sv = ra.reveal_scales(c)
    big_k = k[:, n].clone(); big_k[0, :5] = 1e6        # five clamps in K
    ra.append(c, big_k, v[:, n])
    assert torch.equal(c.kt[:, n], ra._q16(big_k / sk))
    assert torch.equal(c.vt[:, n], ra._q16(v[:, n] / sv))
    assert int(c.clamped) == 5


# ------------------------------------------------- no compile in the loop
def _residency(cap, n, h=2, d=128):
    g = torch.Generator(device="cpu").manual_seed(n)
    k = torch.randn(h, cap, d, generator=g).cuda()
    v = torch.randn(h, cap, d, generator=g).cuda()
    key, nonce = _key(top_bit=bool(n % 2))
    return ra.quantize_and_protect(k, v, key, nonce, K_BASE, V_BASE,
                                   masked=True, n_valid=n), k, v


@cuda
def test_new_residency_and_growing_length_do_not_recompile():
    cap, hq = 160, 4
    q = torch.randn(hq, 128, device="cuda")
    c, k, v = _residency(cap, 100)                # warm every kernel once
    ra.ring_decode(q, c)
    ra.append(c, k[:, 100], v[:, 100])
    ra.ring_decode(q, c)
    with KernelGuard() as g:
        for n0 in (97, 112, 120):                 # fresh keys, both key parities
            c, k, v = _residency(cap, n0)
            for i in range(20):                   # crosses a multiple of 16
                ra.append(c, k[:, n0 + i], v[:, n0 + i])
                ra.ring_decode(q, c)
    assert g.compiles == 0, g.compile_sigs
    assert g.launch_names["_append_row"] == 60    # one launch per appended row
    assert g.launches > 0


@triton.jit
def _probe(X, n):
    tl.store(X, n)


@cuda
def test_negative_control_guard_sees_value_specialisation():
    """Scalar ints specialise on value: the two triggers the fix removed."""
    x = torch.zeros(1, device="cuda", dtype=torch.int64)
    _probe[(1,)](x, 17)
    with KernelGuard() as g:
        _probe[(1,)](x, 32)                       # divisible by 16
        _probe[(1,)](x, 0x9E3779B9)               # top bit set: typed i64
    assert g.compiles == 2, g.compile_sigs


# ------------------------------------------- head_dim 64 (Llama-3.2-1B)
@cuda
@pytest.mark.parametrize("d", [64, 128])
def test_masked_decode_bitexact_and_pad_conformant_at_head_dim(d):
    h, hq, cap, n = 2, 8, 70, 48
    key, nonce = _key(True)
    g = torch.Generator(device="cpu").manual_seed(d)
    k = torch.randn(h, cap, d, generator=g).cuda()
    v = torch.randn(h, cap, d, generator=g).cuda()
    q = torch.randn(hq, d, device="cuda")
    cm = ra.quantize_and_protect(k, v, key, nonce, K_BASE, V_BASE,
                                 masked=True, n_valid=n)
    cu = ra.quantize_and_protect(k, v, key, nonce, K_BASE, V_BASE,
                                 masked=False, n_valid=n)
    for i in range(6):
        assert torch.equal(ra.ring_decode(q, cm), ra.ring_decode(q, cu))
        ra.append(cm, k[:, n + i], v[:, n + i])
        ra.append(cu, k[:, n + i], v[:, n + i])
    z = torch.zeros(h, cap, d, device="cuda")
    cz = ra.quantize_and_protect(z, z.clone(), key, nonce, K_BASE, V_BASE,
                                 masked=True, n_valid=4)
    want = _ref_rows(key, nonce, 8, d, K_BASE + 1 * cap * (d // 32), 4)
    assert np.array_equal(cz.kt[1, :4].cpu().numpy(), want)


# ------------------------------------------- hot -> cold -> hot round trip
@cuda
def test_evict_seal_load_is_exact_and_repads_freshly():
    """A grown residency, evicted through the cold tier and reloaded under a new
    key, decodes bit-identically to the residency that never left. Its scales
    travel sealed with the pages, and the plaintext export is scrubbed once
    sealed, so the reload has only what the cold tier returns."""
    from puf4secure_kvcache import cold_tier as ct
    from puf4secure_kvcache.root_module import SimulatedRootModule
    h, hq, cap, n, d = 2, 8, 96, 40, 128
    g = torch.Generator(device="cpu").manual_seed(11)
    k = torch.randn(h, cap, d, generator=g).cuda()
    v = torch.randn(h, cap, d, generator=g).cuda()
    q = torch.randn(hq, d, device="cuda")
    key1, nonce1 = _key(True)
    c1 = ra.quantize_and_protect(k, v, key1, nonce1, K_BASE, V_BASE,
                                 masked=True, n_valid=n)
    for i in range(7):                                   # grow during decode
        ra.append(c1, k[:, n + i], v[:, n + i])
    kq, vq, sk, sv, m = ra.export_quantized(c1)
    counter = ct.MonotonicCounter()
    module = SimulatedRootModule("device_A", iterations=8)
    module.unlock(b"owner passphrase")
    sealed = ct.seal_cache([(kq, vq)], scales=[(sk, sv)], module=module,
                           model_id="m", store_id="s", counter=counter,
                           page_tokens=16)
    ra.scrub(kq, vq, sk, sv)
    [(kq2, vq2)], [(sk2, sv2)] = ct.open_cache(
        sealed, module=module, model_id="m", store_id="s",
        counter=counter, device="cuda", return_scales=True)
    key2, nonce2 = _key(False)
    c2 = ra.restore_quantized(kq2, vq2, sk2, sv2, cap, key2, nonce2, K_BASE, V_BASE)
    assert torch.equal(ra.ring_decode(q, c1), ra.ring_decode(q, c2))
    ra.append(c1, k[:, m], v[:, m]); ra.append(c2, k[:, m], v[:, m])
    assert torch.equal(ra.ring_decode(q, c1), ra.ring_decode(q, c2))
    same = (c1.kt[:, :m] == c2.kt[:, :m]).float().mean().item()
    assert same < 20 / 65536, "reload must draw a fresh pad"
