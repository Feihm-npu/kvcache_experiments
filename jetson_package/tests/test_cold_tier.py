"""Cold-tier integrity matrix: every tampering the persistent-copy adversary can
do is rejected, and the untampered store opens bit-exactly.

The adversary owns the stored bytes, so each test edits ``SealedCache`` freely and
then opens with honest coordinates.
"""

from __future__ import annotations

import pytest
import torch

from puf4secure_kvcache import cold_tier as ct
from puf4secure_kvcache.puf_sim import ReconstructionError
from puf4secure_kvcache.root_module import SimulatedRootModule

MODEL, STORE, DEV = "qwen3-0.6b", "session-42/prefix", "device_A"
PASS = b"owner passphrase"
_PUF_KEYS = {"mode", "ber", "noise_seed", "correction_capacity", "impostor_device_id"}


def _cache(layers=3, h=2, s=600, d=16, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(h, s, d, generator=g).to(dtype),
             torch.randn(h, s, d, generator=g).to(dtype)) for _ in range(layers)]


def _evicted(layers=3, h=2, s=600, d=16, seed=0):
    """What eviction exports: int16 rows and fp32 per-channel scales [h, d]."""
    kv = _cache(layers, h, s, d, seed=seed)
    sc = [(k.abs().amax(1) * 2 / 32767, v.abs().amax(1) * 2 / 32767) for k, v in kv]
    q = [((k / a[:, None]).round().to(torch.int16),
          (v / b[:, None]).round().to(torch.int16)) for (k, v), (a, b) in zip(kv, sc)]
    return q, sc


def _module(device=DEV, passphrase=PASS, **puf_kwargs):
    """An unlocked simulated root module; unlocking raises on a failed root."""
    m = SimulatedRootModule(device, iterations=8, **puf_kwargs)
    m.unlock(passphrase)
    return m


def _seal(kv, counter, store=STORE, device=DEV, model=MODEL, **kw):
    return ct.seal_cache(kv, module=_module(device), model_id=model, store_id=store,
                         counter=counter, page_tokens=256, **kw)


def _open(sealed, counter, store=STORE, device=DEV, model=MODEL, passphrase=PASS,
          **kw):
    puf = {k: kw.pop(k) for k in list(kw) if k in _PUF_KEYS}
    return ct.open_cache(sealed, module=_module(device, passphrase, **puf),
                         model_id=model, store_id=store, counter=counter, **kw)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_round_trip_is_bit_exact(dtype):
    c = ct.MonotonicCounter()
    kv = _cache(dtype=dtype)
    got = _open(_seal(kv, c), c)
    for (k, v), (k2, v2) in zip(kv, got):
        assert torch.equal(k, k2) and torch.equal(v, v2)


def test_ciphertext_carries_no_plaintext_bytes():
    c = ct.MonotonicCounter()
    kv = _cache()
    sealed = _seal(kv, c)
    plain = kv[0][0][:, :256].contiguous().view(torch.uint8).numpy().tobytes()
    assert plain[:64] not in sealed.pages[(0, 0, 0)]


def test_bit_flip_is_rejected():
    c = ct.MonotonicCounter()
    s = _seal(_cache(), c)
    b = bytearray(s.pages[(1, 1, 0)]); b[100] ^= 0x01
    s.pages[(1, 1, 0)] = bytes(b)
    with pytest.raises(ct.IntegrityError):
        _open(s, c)


@pytest.mark.parametrize("a,b", [((0, 0, 0), (0, 1, 0)),   # within a layer
                                 ((0, 0, 0), (2, 0, 0)),   # across layers
                                 ((0, 0, 0), (0, 0, 1))])  # K for V
def test_page_swap_is_rejected(a, b):
    c = ct.MonotonicCounter()
    s = _seal(_cache(s=512), c)                          # equal-length pages
    s.pages[a], s.pages[b] = s.pages[b], s.pages[a]
    with pytest.raises(ct.IntegrityError):
        _open(s, c)


def test_page_from_another_store_is_rejected():
    c = ct.MonotonicCounter()
    a = _seal(_cache(seed=1), c, store="store-A")
    b = _seal(_cache(seed=2), c, store="store-B")      # same epoch number, 1
    a.pages[(0, 0, 0)] = b.pages[(0, 0, 0)]
    with pytest.raises(ct.IntegrityError):
        _open(a, c, store="store-A")


def test_truncation_is_rejected():
    c = ct.MonotonicCounter()
    s = _seal(_cache(), c)
    del s.pages[(2, 2, 1)]
    with pytest.raises(ct.IntegrityError):
        _open(s, c)


def test_rollback_to_an_older_valid_version_is_rejected():
    c = ct.MonotonicCounter()
    old = _seal(_cache(seed=1), c)                     # epoch 1
    _seal(_cache(seed=2), c)                           # epoch 2 supersedes it
    with pytest.raises(ct.IntegrityError):
        _open(old, c)


def test_rollback_control_without_trusted_counter_is_accepted():
    """Without the trusted counter, an old valid version is indistinguishable:
    the property comes from platform state, not from the cipher."""
    c = ct.MonotonicCounter()
    old = _seal(_cache(seed=1), c)
    snapshot = ct.MonotonicCounter()                   # adversary resets state
    snapshot._v = dict(c._v)
    _seal(_cache(seed=2), c)
    _open(old, snapshot)                               # opens: rollback undetected


@pytest.mark.parametrize("kw", [{"device": "device_B"}])
def test_wrong_device_cannot_open(kw):
    c = ct.MonotonicCounter()
    s = _seal(_cache(), c)
    with pytest.raises(ct.IntegrityError):
        _open(s, c, **kw)


def test_impostor_root_cannot_open():
    """This enrollment on another chip fails the tag check: a cache miss."""
    c = ct.MonotonicCounter()
    s = _seal(_cache(), c)
    with pytest.raises(ReconstructionError):
        _open(s, c, mode="wrong_device")


def test_wrong_passphrase_cannot_open():
    c = ct.MonotonicCounter()
    s = _seal(_cache(), c)
    with pytest.raises(ct.IntegrityError):
        _open(s, c, passphrase=b"a guess")


def test_wrong_model_cannot_open():
    c = ct.MonotonicCounter()
    s = _seal(_cache(), c)
    with pytest.raises(ct.IntegrityError):
        _open(s, c, model="llama-3.2-1b")


def test_unauthenticated_header_is_not_trusted():
    c = ct.MonotonicCounter()
    kv = _cache()
    s = _seal(kv, c)
    s.header["epoch"] = 999
    s.epoch = 999
    got = _open(s, c)
    assert torch.equal(got[0][0], kv[0][0])


def test_noisy_root_within_correction_budget_opens_and_beyond_does_not():
    c = ct.MonotonicCounter()
    kv = _cache()
    s = _seal(kv, c)
    got = _open(s, c, mode="noisy", ber=0.02, noise_seed=3)
    assert torch.equal(got[1][1], kv[1][1])
    with pytest.raises(ReconstructionError):
        _open(s, c, mode="noisy", ber=0.45, noise_seed=3)


# ------------------------------------------------------------------ scales
# The scales an evicted residency needs to reload identify the content in
# plaintext, so they are sealed beside the pages and get the same matrix.
def test_store_without_scales_returns_as_before():
    c = ct.MonotonicCounter()
    kv = _cache()
    s = _seal(kv, c)
    assert s.scales == {}
    got = _open(s, c)
    assert isinstance(got, list) and torch.equal(got[2][1], kv[2][1])
    got, sc = _open(s, c, return_scales=True)
    assert sc is None and torch.equal(got[0][0], kv[0][0])


def test_round_trip_with_scales_is_bit_exact():
    c = ct.MonotonicCounter()
    kv, sc = _evicted()
    got, got_sc = _open(_seal(kv, c, scales=sc), c, return_scales=True)
    for (k, v), (k2, v2) in zip(kv, got):
        assert torch.equal(k, k2) and torch.equal(v, v2)
    for (a, b), (a2, b2) in zip(sc, got_sc):
        assert a2.dtype == torch.float32 and torch.equal(a, a2) and torch.equal(b, b2)


def test_seal_leaves_the_callers_tensors_intact():
    """With one head the host rows are a view of the caller's CPU tensor, as is a
    contiguous CPU scale; only copies the cold tier made itself are scrubbed."""
    c = ct.MonotonicCounter()
    kv, sc = _evicted(h=1)
    want = [[t.clone() for t in pair] for pair in kv + sc]
    _seal(kv, c, scales=sc)
    assert all(torch.equal(t, w) for pair, wp in zip(kv + sc, want)
               for t, w in zip(pair, wp))


@pytest.mark.parametrize("ret", [True, False])
def test_scale_bit_flip_is_rejected(ret):
    """Rejected even when the scales are not asked for: a store opens whole."""
    c = ct.MonotonicCounter()
    kv, sc = _evicted()
    s = _seal(kv, c, scales=sc)
    b = bytearray(s.scales[(1, 3)]); b[40] ^= 0x01
    s.scales[(1, 3)] = bytes(b)
    with pytest.raises(ct.IntegrityError, match="authentication"):
        _open(s, c, return_scales=ret)


@pytest.mark.parametrize("a,b", [((0, 2), (0, 3)),   # K-scales for V-scales
                                 ((0, 2), (2, 2)),   # across layers
                                 ((1, 3), (2, 2))])  # both at once
def test_scale_swap_is_rejected(a, b):
    c = ct.MonotonicCounter()
    kv, sc = _evicted()
    s = _seal(kv, c, scales=sc)
    s.scales[a], s.scales[b] = s.scales[b], s.scales[a]
    with pytest.raises(ct.IntegrityError, match="authentication"):
        _open(s, c, return_scales=True)


def test_scale_and_page_messages_do_not_interchange():
    """Two int16 tokens and an fp32 scale are both 128 bytes here, so only the
    type in the associated data tells the two messages apart."""
    c = ct.MonotonicCounter()
    kv, sc = _evicted(s=258)                       # last page holds 2 tokens
    s = _seal(kv, c, scales=sc)
    assert len(s.pages[(0, 1, 0)]) == len(s.scales[(0, 2)])
    s.pages[(0, 1, 0)], s.scales[(0, 2)] = s.scales[(0, 2)], s.pages[(0, 1, 0)]
    with pytest.raises(ct.IntegrityError, match="authentication"):
        _open(s, c, return_scales=True)


@pytest.mark.parametrize("edit", ["drop_one", "drop_all", "extra_layer", "graft"])
def test_missing_or_extra_scale_message_is_rejected(edit):
    """``drop_all`` makes the store look sealed without scales and ``graft`` does
    the reverse; the manifest records which it was."""
    c = ct.MonotonicCounter()
    kv, sc = _evicted()
    s = _seal(kv, c, scales=sc)
    if edit == "drop_one":
        del s.scales[(2, 3)]
    elif edit == "drop_all":
        s.scales.clear()
    elif edit == "extra_layer":
        s.scales[(3, 2)] = s.scales[(0, 2)]
    else:                                          # onto a later seal without
        old, s = s, _seal(kv, c)
        s.scales = dict(old.scales)
    with pytest.raises(ct.IntegrityError, match="scale set"):
        _open(s, c, return_scales=True)


def test_scales_from_another_store_are_rejected():
    c = ct.MonotonicCounter()
    kv_a, sc_a = _evicted(seed=1)
    kv_b, sc_b = _evicted(seed=2)
    a = _seal(kv_a, c, store="store-A", scales=sc_a)
    b = _seal(kv_b, c, store="store-B", scales=sc_b)  # same epoch number, 1
    _open(a, c, store="store-A", return_scales=True)  # control: opens as sealed
    a.scales = dict(b.scales)                         # a complete, valid set
    with pytest.raises(ct.IntegrityError, match="authentication"):
        _open(a, c, store="store-A", return_scales=True)


def test_scales_from_an_older_version_are_rejected():
    """Same store and key, and the counter is retained, so only the epoch in the
    associated data separates the old set from the current one."""
    c = ct.MonotonicCounter()
    kv1, sc1 = _evicted(seed=1)
    kv2, sc2 = _evicted(seed=2)
    old = _seal(kv1, c, scales=sc1)                   # epoch 1
    new = _seal(kv2, c, scales=sc2)                   # epoch 2 supersedes it
    _open(new, c, return_scales=True)                 # control: opens as sealed
    new.scales = dict(old.scales)
    with pytest.raises(ct.IntegrityError, match="authentication"):
        _open(new, c, return_scales=True)


def test_malformed_scales_are_refused_before_an_epoch_is_spent():
    c = ct.MonotonicCounter()
    kv, sc = _evicted()
    s = _seal(kv, c, scales=sc)
    for bad in (sc[:2], [sc[0], sc[1], (sc[2][0], sc[2][1][:1])]):
        with pytest.raises(ValueError):
            _seal(kv, c, scales=bad)
    _open(s, c, return_scales=True)                   # still the current version
