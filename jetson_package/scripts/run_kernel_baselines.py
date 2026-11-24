#!/usr/bin/env python3
"""RQ3 (contribution C2): what the in-register pad costs at kernel level, against
the alternatives a reviewer would ask about.

One launch carries one decode step of attention work for a model: all layers'
KV heads side by side (e.g. Qwen3-0.6B: 28 x 8 = 224 KV heads, GQA 2, head dim
128), a single query per head, a cache of N tokens. Variants:

  ring            production split-K kernel (`ring_attn.ring_decode`), pad
                  regenerated and removed in registers, one program per KV head
                  and split for all query heads of the group
  ring_per_head   the earlier kernel, one program per query head, which
                  regenerates each pad once per query head of the group
  quant           the same kernel on the same int16 cache with no pad: the
                  control that isolates the cipher
  decrypt+attend  remove the pad in a separate pass that writes the plaintext
                  int16 cache back to HBM, then run `quant` on it: what a
                  decrypt-before-attention design costs on the GPU, and the
                  plaintext it leaves in memory every step
  bf16_splitk     a bf16 kernel of the same split-K structure (no scale, no pad)
  flash_bf16      FlashAttention-2 decode (`flash_attn_with_kvcache`) on a bf16
                  cache: the production kernel a serving system would use

Harness discipline (memory: harness-before-mechanism): every variant and every N
is compiled and warmed before timing; N is never a specialisation key; variants
are timed in interleaved rounds with CUDA events and reported as the median
over rounds; ring, quant and decrypt+attend must agree bit for bit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import torch
import triton
import triton.language as tl
try:                                   # absent on platforms without a FA-2 build
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

from puf4secure_kvcache import ring_attn as ra
from puf4secure_kvcache import ring_install as ri


@triton.jit(do_not_specialize=["N"])
def _decode_split_bf16(Q, K, V, PartO, PartM, PartL, N, GQ,
                       stride_qh, stride_kh, stride_kn,
                       D: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr):
    """`ring_attn._decode_split` with a bf16 cache: same tiles, no scale, no pad."""
    qh = tl.program_id(0); sp = tl.program_id(1)
    h = qh // GQ
    cs = tl.arange(0, D)
    q = tl.load(Q + qh * stride_qh + cs).to(tl.float32)
    m_i = tl.full((1,), -float("inf"), tl.float32)
    l_i = tl.zeros((1,), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    lo = sp * SPLIT
    hi = tl.minimum(lo + SPLIT, N)
    for start in range(lo, hi, BN):
        pos = start + tl.arange(0, BN)
        valid = pos < hi
        off = h * stride_kh + pos[:, None] * stride_kn + cs[None, :]
        kf = tl.load(K + off, mask=valid[:, None], other=0).to(tl.float32)
        vf = tl.load(V + off, mask=valid[:, None], other=0).to(tl.float32)
        sc = tl.sum(kf * q[None, :], axis=1) * (1.0 / tl.sqrt(float(D)))
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


def median_ms(fn, iters: int) -> float:
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-heads", type=int, default=224)
    ap.add_argument("--gqa", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--splits", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tag", default="qwen3step")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    H, D, GQ = args.kv_heads, args.d, args.gqa
    HQ = H * GQ
    bn = ra.BLOCK_N_DEFAULT
    torch.manual_seed(0)

    setups = {}
    for N in args.contexts:
        k = torch.randn(H, N, D, device=dev) * 2
        v = torch.randn(H, N, D, device=dev) * 2
        q = torch.randn(HQ, D, device=dev)
        key, nonce = ri.residency_key()
        ring = ra.quantize_and_protect(k, v, key, nonce, ri.K_BASE, ri.V_BASE,
                                       masked=True, n_valid=N)
        quant = ra.quantize_and_protect(k, v, key, nonce, ri.K_BASE, ri.V_BASE,
                                        masked=False, n_valid=N)
        dec = ra.quantize_and_protect(k, v, key, nonce, ri.K_BASE, ri.V_BASE,
                                      masked=False, n_valid=N)   # scratch target
        kb, vb = k.to(torch.bfloat16), v.to(torch.bfloat16)
        k_fl = kb.transpose(0, 1).unsqueeze(0).contiguous()     # [1, N, H, D]
        v_fl = vb.transpose(0, 1).unsqueeze(0).contiguous()
        q_fl = q.to(torch.bfloat16).view(1, 1, HQ, D)
        seqlens = torch.full((1,), N, dtype=torch.int32, device=dev)
        ring.plan(args.splits, bn)
        quant.plan(args.splits, bn)
        dec.plan(args.splits, bn)
        split, n_split = ring.split, ring.n_split
        part = {"po": torch.empty(HQ * n_split * D, device=dev),
                "pm": torch.empty(HQ * n_split, device=dev),
                "pl": torch.empty(HQ * n_split, device=dev),
                "out": torch.empty(HQ, D, device=dev)}
        del k, v

        def run_ring(c=ring, q=q):
            return ra.ring_decode(q, c, masked=True, splits=args.splits)

        def run_ring_ph(c=ring, q=q):
            return ra.ring_decode(q, c, masked=True, splits=args.splits,
                                  kernel="per_head")

        def run_quant(c=quant, q=q):
            return ra.ring_decode(q, c, masked=False, splits=args.splits)

        def run_decrypt_attend(r=ring, c=dec, q=q, N=N):
            grid = (H, triton.cdiv(N, bn))
            ra._protect[grid](r.kt, c.kt, N, N, 0, N * D, D, r.blk_k, *r.kargs,
                              ROUNDS=ra.ROUNDS_DEFAULT, D=D, BN=bn, SIGN=-1)
            ra._protect[grid](r.vt, c.vt, N, N, 0, N * D, D, r.blk_v, *r.kargs,
                              ROUNDS=ra.ROUNDS_DEFAULT, D=D, BN=bn, SIGN=-1)
            return ra.ring_decode(q, c, masked=False, splits=args.splits)

        def run_bf16(kb=kb, vb=vb, q=q, N=N, split=split, n_split=n_split, p=part):
            _decode_split_bf16[(HQ, n_split)](
                q, kb, vb, p["po"], p["pm"], p["pl"], N, GQ, D, N * D, D,
                D=D, BN=bn, SPLIT=split)
            ra._reduce[(HQ,)](p["po"], p["pm"], p["pl"], p["out"], n_split, D=D,
                              S=triton.next_power_of_2(n_split))
            return p["out"]

        def run_flash(q_fl=q_fl, k_fl=k_fl, v_fl=v_fl, seqlens=seqlens):
            return flash_attn_with_kvcache(q_fl, k_fl, v_fl, cache_seqlens=seqlens)

        setups[N] = {"ring": run_ring, "ring_per_head": run_ring_ph,
                     "quant": run_quant,
                     "decrypt+attend": run_decrypt_attend,
                     "bf16_splitk": run_bf16}
        if flash_attn_with_kvcache is not None:
            setups[N]["flash_bf16"] = run_flash

    # correctness, and compilation of every variant at every N before timing
    checks = {}
    for N, vs in setups.items():
        o_ring = vs["ring"]().clone()
        o_ph = vs["ring_per_head"]().clone()
        o_quant = vs["quant"]().clone()
        o_dec = vs["decrypt+attend"]().clone()
        o_bf = vs["bf16_splitk"]().clone()
        checks[N] = {"ring_equals_quant": bool(torch.equal(o_ring, o_quant)),
                     "ring_equals_per_head": bool(torch.equal(o_ring, o_ph)),
                     "decrypt_attend_equals_quant": bool(torch.equal(o_dec, o_quant))}
        if "flash_bf16" in vs:
            o_fl = vs["flash_bf16"]().float().view(HQ, D)
            checks[N]["bf16_splitk_vs_flash_max_abs"] = float((o_bf - o_fl).abs().max())
            checks[N]["ring_vs_flash_max_abs"] = float((o_ring - o_fl).abs().max())
        for fn in vs.values():
            for _ in range(3):
                fn()
    torch.cuda.synchronize()

    times = {N: {k: [] for k in vs} for N, vs in setups.items()}
    for _ in range(args.rounds):
        for N, vs in setups.items():
            for k, fn in vs.items():
                times[N][k].append(median_ms(fn, args.iters))

    res = {"device": torch.cuda.get_device_name(0), "kv_heads": H, "q_heads": HQ,
           "head_dim": D, "block_n": bn, "splits": args.splits,
           "rounds": args.rounds, "iters": args.iters,
           "geometry": f"{args.tag}: {H} KV heads x GQA {GQ}, head dim {D}",
           "per_context": {}}
    for N in setups:
        med = {k: statistics.median(v) for k, v in times[N].items()}
        res["per_context"][str(N)] = {
            "median_ms": med,
            "spread": {k: (max(v) - min(v)) / statistics.median(v) for k, v in times[N].items()},
            "ring_over_quant": med["ring"] / med["quant"],
            "per_head_over_quant": med["ring_per_head"] / med["quant"],
            "ring_over_bf16_splitk": med["ring"] / med["bf16_splitk"],
            "ring_over_flash_bf16": (med["ring"] / med["flash_bf16"]
                                     if "flash_bf16" in med else None),
            "decrypt_attend_over_ring": med["decrypt+attend"] / med["ring"],
            "checks": checks[N]}
        r = res["per_context"][str(N)]
        print(f"N={N:6d}  " + "  ".join(f"{k} {m:.3f}" for k, m in med.items())
              + f"  | ring/quant {r['ring_over_quant']:.2f}  ring/bf16 {r['ring_over_bf16_splitk']:.2f}"
              f"  ring/flash {r['ring_over_flash_bf16'] or float('nan'):.2f}"
              f"  dec/ring {r['decrypt_attend_over_ring']:.2f}"
              f"  {checks[N]}", flush=True)
    out = args.out or pathlib.Path(f"experiments/runs/kernel_baselines_{args.tag}.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
