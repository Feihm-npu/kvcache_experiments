"""Pad uniqueness for the ring hot tier.

Two defects reused the one-time pad, and each test below is paired with a negative
control that reproduces the defect, so a pass shows the test can see the failure
rather than merely that nothing went wrong.

1. Across residencies: the key depended only on (session, layer), so every cache
   entering the hot tier in one session received the same pad. Subtracting two
   masked caches returned the difference of their plaintexts. Keys are now drawn
   from the CSPRNG per residency and never derived from the device root.
2. Within a residency: the whole capacity was padded, including rows reserved for
   decoded positions. Those rows held zero, so memory contained the pad for every
   future position, and the decode step that later wrote there reused it.
"""

from __future__ import annotations

import pytest
import torch

from puf4secure_kvcache import ring_attn as ra
from puf4secure_kvcache import ring_install as ri

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, D = 4, 128
CHANCE = 1.0 / 65536


def _kv(cap, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(H, cap, D, generator=g) * 2
    v = torch.randn(H, cap, D, generator=g) * 2
    return k.cuda(), v.cuda()


def _protect(k, v, key=None, n_valid=None):
    """Enter residency under ``key``, or under a fresh residency key."""
    key = key or ri.residency_key()
    return ra.quantize_and_protect(k, v, *key, ri.K_BASE, ri.V_BASE,
                                   masked=True, n_valid=n_valid,
                                   residency_id=ri.new_residency_id())


def _mod16(x):
    return torch.remainder(x.to(torch.int64), 65536)


# ------------------------------------------------- residency ids and keys
def test_residency_ids_never_repeat():
    ids = {ri.new_residency_id() for _ in range(2000)}
    assert len(ids) == 2000


def test_residency_ids_survive_counter_reset():
    """A restarted process resets the counter; ids must still differ."""
    saved = ri._EPOCH
    first = ri.new_residency_id()
    ri._EPOCH = saved                                 # simulate a fresh process
    second = ri.new_residency_id()
    assert first.split(".")[0] == second.split(".")[0]   # same epoch number...
    assert first != second                                # ...different id


def test_residency_keys_are_fresh_and_full_width():
    keys = [ri.residency_key() for _ in range(256)]
    assert len({tuple(k) + tuple(n) for k, n in keys}) == 256
    words = [w for k, n in keys for w in k + n]
    assert all(0 <= w < 2 ** 32 for w in words)
    assert any(w >= 2 ** 31 for w in words)          # top bit is exercised


# ----------------------------------------------- 1. across residencies
@cuda
def test_same_content_gets_independent_pads_across_residencies():
    k, v = _kv(256, seed=1)
    c1 = _protect(k, v)
    c2 = _protect(k, v)
    same = (c1.kt == c2.kt).float().mean().item()
    assert same < 20 * CHANCE, f"masked copies agree at rate {same}"


@cuda
def test_no_two_time_pad_across_residencies():
    ka, va = _kv(256, seed=2)
    kb, vb = _kv(256, seed=3)
    ca = _protect(ka, va)
    cb = _protect(kb, vb)
    qa = ra.quantize_and_protect(ka, va, [0] * 8, [0] * 3, ri.K_BASE, ri.V_BASE,
                                 masked=False)
    qb = ra.quantize_and_protect(kb, vb, [0] * 8, [0] * 3, ri.K_BASE, ri.V_BASE,
                                 masked=False)
    leak = (_mod16(ca.kt) - _mod16(cb.kt)) % 65536 == (
        _mod16(qa.kt) - _mod16(qb.kt)) % 65536
    assert leak.float().mean().item() < 20 * CHANCE


@cuda
def test_negative_control_shared_residency_is_a_two_time_pad():
    """The old behaviour: one key for both caches. Differencing must fully leak."""
    ka, va = _kv(256, seed=2)
    kb, vb = _kv(256, seed=3)
    shared = ri.residency_key()
    ca = _protect(ka, va, key=shared)
    cb = _protect(kb, vb, key=shared)
    qa = ra.quantize_and_protect(ka, va, [0] * 8, [0] * 3, ri.K_BASE, ri.V_BASE,
                                 masked=False)
    qb = ra.quantize_and_protect(kb, vb, [0] * 8, [0] * 3, ri.K_BASE, ri.V_BASE,
                                 masked=False)
    leak = (_mod16(ca.kt) - _mod16(cb.kt)) % 65536 == (
        _mod16(qa.kt) - _mod16(qb.kt)) % 65536
    assert leak.all(), "the control should reproduce the two-time pad"


# ------------------------------------------------ 2. within a residency
@cuda
def test_reserved_rows_hold_no_pad_and_append_is_not_recoverable():
    cap, n = 288, 256
    k, v = _kv(cap, seed=4)
    c = _protect(k, v, n_valid=n)
    assert torch.count_nonzero(c.kt[:, n:]) == 0
    assert torch.count_nonzero(c.vt[:, n:]) == 0
    before = c.kt[:, n].clone()                       # snapshot 1
    ra.append(c, k[:, n], v[:, n], masked=True)
    after = c.kt[:, n].clone()                        # snapshot 2
    kq_new = ra._q16(k[:, n] / ra.reveal_scales(c)[0])
    recovered = _mod16(after) - _mod16(before)
    hit = (recovered % 65536 == _mod16(kq_new)).float().mean().item()
    assert hit < 20 * CHANCE, f"two snapshots recover the appended row at {hit}"


@cuda
def test_negative_control_padding_reserved_rows_leaks_the_appended_row():
    """The old behaviour: pad the whole capacity, then append into a padded row."""
    cap, n = 288, 256
    k, v = _kv(cap, seed=4)
    kz, vz = k.clone(), v.clone()
    kz[:, n:] = 0
    vz[:, n:] = 0
    c = _protect(kz, vz, n_valid=cap)
    c.n = n                                           # reserved rows were padded
    before = c.kt[:, n].clone()
    ra.append(c, k[:, n], v[:, n], masked=True)
    after = c.kt[:, n].clone()
    kq_new = ra._q16(k[:, n] / ra.reveal_scales(c)[0])
    recovered = (_mod16(after) - _mod16(before)) % 65536
    assert torch.equal(recovered, _mod16(kq_new)), \
        "the control should recover the appended row exactly"


# --------------------------------------------------- correctness intact
@cuda
def test_masked_decode_still_bitexact_with_unmasked():
    cap, n, hq = 300, 260, 8
    k, v = _kv(cap, seed=5)
    q = torch.randn(hq, D, device="cuda")
    rid = ri.new_residency_id()
    key, nonce = ri.residency_key()
    cm = ra.quantize_and_protect(k, v, key, nonce, ri.K_BASE, ri.V_BASE,
                                 masked=True, n_valid=n, residency_id=rid)
    cu = ra.quantize_and_protect(k, v, key, nonce, ri.K_BASE, ri.V_BASE,
                                 masked=False, n_valid=n)
    assert torch.equal(ra.ring_decode(q, cm, masked=True),
                       ra.ring_decode(q, cu, masked=False))
    for i in range(5):
        ra.append(cm, k[:, n + i], v[:, n + i], masked=True)
        ra.append(cu, k[:, n + i], v[:, n + i], masked=False)
        assert torch.equal(ra.ring_decode(q, cm, masked=True),
                           ra.ring_decode(q, cu, masked=False))


def test_counter_ranges_must_not_overlap_or_overflow():
    ra.check_counter_ranges(8, 16_384, 128, ri.K_BASE, ri.V_BASE)
    with pytest.raises(ValueError, match="overlap"):
        ra.check_counter_ranges(8, 16_384, 128, 0, 1000)
    with pytest.raises(ValueError, match="overlap"):            # V into scales
        ra.check_counter_ranges(8, 1 << 25, 128, ri.K_BASE, ri.V_BASE)
    with pytest.raises(ValueError, match="overflow"):
        ra.check_counter_ranges(8, 16_384, 128, 0, (1 << 32) - 1000)
