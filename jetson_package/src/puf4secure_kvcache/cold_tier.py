"""Cold tier: AEAD-sealed KV pages, keyed by the device root.

Whatever leaves the device -- a cache persisted for reuse, swapped to host or
disk, or migrated -- is sealed here, and its cost is paid once per movement, not
once per decode step. The hot tier (``ring_attn``) takes over when a cache is
loaded back: open once, enter residency under a fresh pad, decode many steps.

Binding, and what each defends:

* **key**: ``HMAC(k_u, "cold|aead|" ‖ model ‖ store)``, derived inside the root
  module (``root_module``) from the unlock key, which the module derives from the
  device root and the owner's passphrase. The host receives this key and never
  the root, and nothing that the device stores determines it: a copy of the
  store is useless away from the module, and on the device each passphrase
  guess costs one unlock. This is the property a key saved beside the cache, or
  in the device's fuses, cannot have.
* **associated data**: rebuilt by the *opener* from the coordinates it asks for
  (model, store, layer, page, epoch, shape), never read from the stored header.
  A page moved to another slot, another layer or another store fails its tag.
* **manifest**: the page count and token count, sealed like a page, so dropping
  or appending pages is detected.
* **scales**: the per-channel quantization scales an evicted residency needs to
  reload, one message per layer for K and one for V, typed apart from pages in
  the associated data. A scale is a per-channel maximum of the content, so in
  plaintext it is a fingerprint of it; the manifest records their shape and
  dtype, so dropping or grafting scale messages is detected like a missing page.
* **epoch**: compared with a monotonic counter the device holds. Rollback to an
  older, validly sealed version cannot be detected by cryptography alone; it
  needs this trusted state, which is an assumption on the platform.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

VERSION = 1
PAGE_TOKENS_DEFAULT = 256
_DTYPES = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2,
           torch.int16: 3}
_DTYPES_INV = {v: k for k, v in _DTYPES.items()}
SCALE_K, SCALE_V = 2, 3   # tensor-type codes, after the pages' K=0 and V=1


class IntegrityError(RuntimeError):
    """A sealed object failed authentication or a structural check."""


class MonotonicCounter:
    """Trusted per-store version counter (simulated platform state).

    Stands in for a TPM NV counter or a counter inside the root module. The
    cold tier can only reject rollback if this state is outside the adversary's
    reach; that is stated as an assumption, not provided by the cipher.
    """

    def __init__(self):
        self._v: dict[str, int] = {}
        self._lock = threading.Lock()

    def bump(self, store_id: str) -> int:
        with self._lock:
            self._v[store_id] = self._v.get(store_id, 0) + 1
            return self._v[store_id]

    def current(self, store_id: str) -> int:
        return self._v.get(store_id, 0)


def _h(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()[:16]


def cold_key(module, model_id: str, store_id: str) -> bytes:
    """The sealing key, requested from the unlocked root module on every use."""
    return module.derive(f"cold|aead|model={model_id}|store={store_id}".encode())


def _aad(model_id, store_id, epoch, layer, page, kv, n_tok, dtype, h, d) -> bytes:
    return (b"KVCOLD" + struct.pack(">H", VERSION) + _h(model_id) + _h(store_id)
            + struct.pack(">QiiBiBii", epoch, layer, page, kv, n_tok, dtype, h, d))


def _scale_aad(model_id, store_id, epoch, layer, kind, dtype, n_bytes,
               shape) -> bytes:
    return (b"KVSCAL" + struct.pack(">H", VERSION) + _h(model_id) + _h(store_id)
            + struct.pack(f">QiBBIB{len(shape)}i", epoch, layer, kind, dtype,
                          n_bytes, len(shape), *shape))


def _manifest_aad(model_id, store_id, epoch) -> bytes:
    return b"KVMANI" + struct.pack(">H", VERSION) + _h(model_id) + _h(store_id) \
        + struct.pack(">Q", epoch)


@dataclass
class SealedCache:
    """What is persisted. Every field may be read and modified by the adversary."""
    model_id: str
    store_id: str
    epoch: int
    manifest: bytes                           # nonce ‖ ciphertext ‖ tag
    pages: dict = field(default_factory=dict)  # (layer, page, 0=K|1=V) -> bytes
    header: dict = field(default_factory=dict) # informational, unauthenticated
    scales: dict = field(default_factory=dict) # (layer, 2=K|3=V) -> bytes

    def nbytes(self) -> int:
        return len(self.manifest) + sum(len(p) for p in self.pages.values()) \
            + sum(len(p) for p in self.scales.values())


def _seal(aead, plain: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(12)          # random 96-bit nonce; one key per store
    return nonce + aead.encrypt(nonce, plain, aad)


def _open(aead, blob: bytes, aad: bytes) -> bytes:
    try:
        return aead.decrypt(blob[:12], blob[12:], aad)
    except InvalidTag as e:
        raise IntegrityError("authentication failed") from e


def _to_host_rows(x: torch.Tensor) -> torch.Tensor:
    """[H, S, d] -> host uint8 [S, H*d*elt]: one GPU transpose and one copy,
    after which every page is a contiguous byte range. The transposed device
    copy holds plaintext, so it is scrubbed once it reaches the host."""
    g = x.detach().permute(1, 0, 2).contiguous()
    h = g.cpu().view(torch.uint8).reshape(x.shape[1], -1)
    if _ptr(g) not in (_ptr(x), _ptr(h)):          # a copy that nobody returns
        g.zero_()
    return h


def _to_host_bytes(x: torch.Tensor) -> torch.Tensor:
    """Any tensor -> flat host uint8; a device copy is scrubbed as above."""
    g = x.detach().contiguous()
    h = g.cpu().view(torch.uint8).reshape(-1)
    if _ptr(g) not in (_ptr(x), _ptr(h)):
        g.zero_()
    return h


def _ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def _scrub_copy(copy: torch.Tensor, orig: torch.Tensor) -> None:
    """Zero a host plaintext copy, unless it is a view of the caller's own
    tensor: a CPU input that needed no copy (one head, or a contiguous scale)."""
    if _ptr(copy) != _ptr(orig):
        copy.numpy().fill(0)


def _scale_layout(scales, n_layers: int):
    """The shape and dtype every scale tensor shares, or ValueError."""
    ts = [t for pair in scales for t in pair]
    if len(scales) != n_layers or len(ts) != 2 * n_layers:
        raise ValueError("scales must be one (K, V) pair per layer")
    shape, dt = tuple(ts[0].shape), ts[0].dtype
    if dt not in _DTYPES or any(tuple(t.shape) != shape or t.dtype != dt
                                for t in ts):
        raise ValueError("scales must share one shape and a supported dtype")
    return shape, dt


def seal_cache(kv, *, module, model_id: str, store_id: str,
               counter: MonotonicCounter, page_tokens: int = PAGE_TOKENS_DEFAULT,
               threads: int = 1, scales=None) -> SealedCache:
    """Seal a cache: ``kv`` is a list over layers of (K, V), each [H, S, d].

    ``module`` is the device's unlocked root module (``root_module``).
    A new epoch is taken from the counter, so every seal of a store supersedes
    the previous one. K and V of a page are separate messages whose associated
    data says which is which. ``scales``, a list over layers of (K-scales,
    V-scales) of one shape and dtype (``ring_attn.export_quantized`` gives fp32
    [H, d]), adds a message per tensor, typed 2 and 3 where pages are 0 and 1.
    ``threads`` > 1 helps only with a binding that releases the GIL; the
    ``cryptography`` binding does not (measured).
    """
    if scales is not None:              # before the epoch: a bad call spends none
        scales = [tuple(p) for p in scales]
        s_shape, s_dt = _scale_layout(scales, len(kv))
    epoch = counter.bump(store_id)
    aead = AESGCM(cold_key(module, model_id, store_id))
    h, s, d = kv[0][0].shape
    dt = kv[0][0].dtype
    n_pages = -(-s // page_tokens)
    staged = [(_to_host_rows(k), _to_host_rows(v)) for k, v in kv]
    host = [(rk.numpy(), rv.numpy()) for rk, rv in staged]

    def one(job):
        layer, p, which = job
        lo, hi = p * page_tokens, min(s, (p + 1) * page_tokens)
        rows = memoryview(host[layer][which][lo:hi]).cast("B")
        return job, _seal(aead, rows, _aad(model_id, store_id, epoch, layer, p,
                                           which, hi - lo, _DTYPES[dt], h, d))

    jobs = [(l, p, w) for l in range(len(kv)) for p in range(n_pages)
            for w in (0, 1)]
    if threads > 1:
        with ThreadPoolExecutor(threads) as ex:
            pages = dict(ex.map(one, jobs))
    else:
        pages = dict(map(one, jobs))
    for (k, v), (rk, rv) in zip(kv, staged):  # plaintext rows, now sealed
        _scrub_copy(rk, k)
        _scrub_copy(rv, v)
    man = {"layers": len(kv), "tokens": s, "page_tokens": page_tokens,
           "pages": n_pages, "dtype": _DTYPES[dt], "h": h, "d": d}
    sealed_scales = {}
    if scales is not None:
        man["scales"] = {"dtype": _DTYPES[s_dt], "shape": list(s_shape)}
        for layer, pair in enumerate(scales):
            for kind, t in zip((SCALE_K, SCALE_V), pair):
                hb = _to_host_bytes(t)
                sealed_scales[(layer, kind)] = _seal(aead, memoryview(hb.numpy()),
                    _scale_aad(model_id, store_id, epoch, layer, kind,
                               _DTYPES[s_dt], hb.numel(), s_shape))
                _scrub_copy(hb, t)
    man = json.dumps(man).encode()
    return SealedCache(model_id, store_id, epoch,
                       _seal(aead, man, _manifest_aad(model_id, store_id, epoch)),
                       pages, header={"epoch": epoch, "store_id": store_id},
                       scales=sealed_scales)


def open_cache(sealed: SealedCache, *, module, model_id: str,
               store_id: str, counter: MonotonicCounter, device="cpu",
               threads: int = 1, return_scales: bool = False):
    """Authenticate and decrypt a sealed cache; returns the list of (K, V).

    The opener supplies every coordinate itself. Nothing in ``sealed`` other than
    ciphertext is trusted: the epoch comes from the counter, the layer, page and
    K/V slot from the position being filled, the store and model from the caller.
    ``return_scales=True`` returns ``(kv, scales)``: the list over layers of
    (K-scales, V-scales) on ``device``, or None if the store was sealed without
    them. Scales the manifest records are authenticated either way, so a store
    opens whole or not at all.
    """
    epoch = counter.current(store_id)
    aead = AESGCM(cold_key(module, model_id, store_id))
    man = json.loads(_open(aead, sealed.manifest,
                           _manifest_aad(model_id, store_id, epoch)))
    L, s, pt, n_pages = man["layers"], man["tokens"], man["page_tokens"], man["pages"]
    h, d, dt = man["h"], man["d"], _DTYPES_INV[man["dtype"]]
    want = {(l, p, w) for l in range(L) for p in range(n_pages) for w in (0, 1)}
    if set(sealed.pages) != want:
        raise IntegrityError("page set does not match the manifest")
    sm = man.get("scales")                     # absent: sealed without scales
    want_s, s_bytes = set(), 0
    if sm is not None:
        s_shape, s_dt = tuple(sm["shape"]), _DTYPES_INV[sm["dtype"]]
        s_bytes = math.prod(s_shape) * torch.tensor([], dtype=s_dt).element_size()
        want_s = {(l, w) for l in range(L) for w in (SCALE_K, SCALE_V)}
    if set(sealed.scales) != want_s:
        raise IntegrityError("scale set does not match the manifest")
    row_bytes = h * d * torch.tensor([], dtype=dt).element_size()
    host = [[torch.empty(s, row_bytes, dtype=torch.uint8) for _ in (0, 1)]
            for _ in range(L)]
    host_s = {job: torch.empty(s_bytes, dtype=torch.uint8) for job in want_s}

    def one(job):
        layer, p, which = job
        lo, hi = p * pt, min(s, (p + 1) * pt)
        plain = _open(aead, sealed.pages[job],
                      _aad(model_id, store_id, epoch, layer, p, which, hi - lo,
                           man["dtype"], h, d))
        if len(plain) != (hi - lo) * row_bytes:
            raise IntegrityError("page length does not match its coordinates")
        # The binding returns immutable bytes, which Python cannot overwrite; a
        # native implementation would decrypt in place. The mutable copy is ours.
        buf = bytearray(plain)
        host[layer][which][lo:hi] = torch.frombuffer(buf, dtype=torch.uint8) \
            .view(hi - lo, row_bytes)
        buf[:] = bytes(len(buf))

    def one_scale(job):
        layer, kind = job
        plain = _open(aead, sealed.scales[job],
                      _scale_aad(model_id, store_id, epoch, layer, kind,
                                 sm["dtype"], s_bytes, s_shape))
        if len(plain) != s_bytes:
            raise IntegrityError("scale length does not match the manifest")
        buf = bytearray(plain)
        host_s[job].copy_(torch.frombuffer(buf, dtype=torch.uint8))
        buf[:] = bytes(len(buf))

    try:
        if threads > 1:
            with ThreadPoolExecutor(threads) as ex:
                list(ex.map(one, sorted(want)))
        else:
            for job in sorted(want):
                one(job)
        for job in sorted(want_s):
            one_scale(job)
    except BaseException:                   # nothing is returned: scrub it all
        for buf_t in [*(x for pair in host for x in pair), *host_s.values()]:
            buf_t.zero_()
        raise
    out = []
    for layer in range(L):
        kv = []
        for w in (0, 1):
            src = host[layer][w]
            g = src.to(device)
            o = g.view(dt).view(s, h, d).permute(1, 0, 2).contiguous()
            kv.append(o)
            if _ptr(g) not in (_ptr(o), _ptr(src)):   # device copy, not returned
                g.zero_()
            if _ptr(src) != _ptr(o):                   # host plaintext, not returned
                src.zero_()
        out.append((kv[0], kv[1]))
    if not return_scales:
        for src in host_s.values():                    # authenticated, not asked for
            src.zero_()
        return out
    if sm is None:
        return out, None
    scales = []
    for layer in range(L):
        pair = []
        for kind in (SCALE_K, SCALE_V):
            src = host_s[(layer, kind)]
            o = src.to(device).view(s_dt).view(s_shape)  # views: no device copy
            pair.append(o)
            if _ptr(src) != _ptr(o):
                src.zero_()
        scales.append((pair[0], pair[1]))
    return out, scales
