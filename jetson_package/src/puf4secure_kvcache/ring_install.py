"""Install the ring hot tier on a HuggingFace decoder, for end-to-end measurement.

Deliberately separate from ``install_puf_attention``: that wrapper carries a dozen
interacting options and a set of audited results, and this path manages its own
cache rather than HuggingFace's, so merging them would put both at risk.

Lifecycle, following ``docs/DESIGN_FREEZE_TWO_TIER.md``:

* **prefill** runs the model's own attention in plaintext and records post-RoPE
  K/V per layer. This is the "page is decrypted once on load" step;
* **entering residency** quantizes per channel and applies the ChaCha8 pad, after
  which only the masked int16 form is in memory;
* **each decode step** regenerates the pad inside the attention kernel, appends
  the new position, and never writes a plaintext cache.

The pad key of a residency is drawn from the OS CSPRNG, one per layer, and is
never derived from the device root. A resident cache never outlives its
process (eviction goes through the cold tier, which the root keys), so nothing
needs to re-derive the key, and a later compromise of the root reveals nothing
about residencies that already ended.

Three modes share the path so the comparison is controlled: ``plain`` leaves the
model untouched, ``quant`` uses the same int16 cache with no pad, and ``ring``
is the design. ``quant`` isolates quantization from masking.
"""

from __future__ import annotations

import os
import struct
import threading

import torch

from . import ring_attn as ra

K_BASE = 0
V_BASE = 0x40000000

_EPOCH = 0
_EPOCH_LOCK = threading.Lock()


def new_residency_id() -> str:
    """A fresh identifier for one entry of one cache into the hot tier.

    Bookkeeping only: it names the residency in logs and metadata and carries no
    key material. A process-wide counter joined with 64 random bits keeps it
    unique across reinstalled wrappers and restarted processes.
    """
    global _EPOCH
    with _EPOCH_LOCK:
        epoch = _EPOCH
        _EPOCH += 1
    return f"{epoch}.{os.urandom(8).hex()}"


def residency_key():
    """A fresh ChaCha8 key (256 bits) and nonce (96 bits) for one layer of one
    residency, from the OS CSPRNG.

    Keys must never repeat across residencies: an earlier version keyed on
    (session, layer), so every cache in a session shared one pad and subtracting
    two masked caches returned the difference of their plaintexts. Independent
    256-bit keys collide with probability T^2 / 2^257 over T residencies.
    """
    raw = os.urandom(44)
    return (list(struct.unpack("<8I", raw[:32])),
            list(struct.unpack("<3I", raw[32:44])))


def _rope(q, k, cos, sin):
    def rot(x):
        a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-b, a), dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


def _forward(self, hidden_states, position_embeddings, attention_mask,
             past_key_values=None, cache_position=None, **kwargs):
    st = self._ring
    # This path keeps its own cache, so HuggingFace cannot derive the decode
    # position from a populated Cache object. A caller that does not supply
    # cache_position would silently rotate every decoded token to position 0,
    # so the mismatch is made loud instead.
    # Every layer receives the same cache_position, so the check runs in layer 0
    # only: reading the position back is a GPU synchronisation, and doing it in
    # every layer stalled the launch pipeline once per layer per step.
    n_new = hidden_states.shape[-2]
    if n_new == 1 and st["cache"] is not None and self.layer_idx == 0:
        want = st["cache"].n
        got = None if cache_position is None else int(cache_position[0])
        if got != want:
            raise RuntimeError(
                f"ring path needs an explicit cache_position: expected {want}, "
                f"got {got}. Without it the rotary angle stays at position 0.")
    shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    q = self.q_proj(hidden_states).view(shape)
    k = self.k_proj(hidden_states).view(shape)
    if hasattr(self, "q_norm"):
        q = self.q_norm(q)
    if hasattr(self, "k_norm"):
        k = self.k_norm(k)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = _rope(q, k, cos, sin)

    if q.shape[2] > 1:
        # prefill: plaintext attention, then enter residency
        hq, hkv = q.shape[1], k.shape[1]
        rep = hq // hkv
        kk = k.repeat_interleave(rep, 1).float()
        vv = v.repeat_interleave(rep, 1).float()
        sc = torch.matmul(q.float(), kk.transpose(2, 3)) * self.scaling
        n = q.shape[2]
        causal = torch.full((n, n), float("-inf"), device=q.device).triu(1)
        a = torch.softmax(sc + causal, dim=-1)
        out = torch.matmul(a, vv)

        cap = n + st["max_new"] + 2
        kbuf = torch.zeros(hkv, cap, self.head_dim, device=k.device,
                           dtype=torch.float32)
        vbuf = torch.zeros_like(kbuf)
        kbuf[:, :n] = k[0].float()
        vbuf[:, :n] = v[0].float()
        rid = new_residency_id()
        key, nonce = residency_key()
        st["cache"] = ra.quantize_and_protect(
            kbuf, vbuf, key, nonce, K_BASE, V_BASE,
            masked=(st["mode"] == "ring"), n_valid=n, residency_id=rid)
        # the fp32 staging copies are this layer's plaintext; overwrite them
        # rather than return them to the allocator's free pool
        kbuf.zero_()
        vbuf.zero_()
    else:
        cache = st["cache"]
        ra.append(cache, k[0, :, 0, :].float(), v[0, :, 0, :].float(),
                  masked=(st["mode"] == "ring"))
        o = ra.ring_decode(q[0, :, 0, :].float(), cache,
                           masked=(st["mode"] == "ring"),
                           splits=st["splits"])
        out = o[None, :, None, :]

    out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
    return self.o_proj(out.to(hidden_states.dtype)), None


def install_ring_attention(model, *, mode: str = "ring",
                           max_new: int = 128, splits: int = 32):
    """Patch every decoder layer's attention onto the ring path.

    ``mode``: ``ring`` (int16 + ChaCha8 pad), ``quant`` (int16, no pad), or
    ``plain`` to leave the model alone. ``max_new`` sizes the residency buffer.
    The hot tier needs no device root: residency keys come from the CSPRNG.
    """
    if mode == "plain":
        return model
    if mode not in ("ring", "quant"):
        raise ValueError(f"mode must be ring, quant or plain (got {mode})")
    layers = model.model.layers
    for lyr in layers:
        attn = lyr.self_attn
        attn._ring = {"mode": mode, "max_new": max_new,
                      "splits": splits, "cache": None}
        attn._ring_orig_forward = attn.forward
        attn.forward = _forward.__get__(attn, type(attn))
    model._ring_installed = True
    return model


def uninstall_ring_attention(model):
    for lyr in model.model.layers:
        attn = lyr.self_attn
        if hasattr(attn, "_ring_orig_forward"):
            attn.forward = attn._ring_orig_forward
            del attn._ring_orig_forward
        if hasattr(attn, "_ring"):
            del attn._ring
    model._ring_installed = False
    return model
