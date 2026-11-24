#!/usr/bin/env python3
"""GPU time per decode step, beside the wall time the e2e harness reports.

At batch 1 with HuggingFace eager, decode is dispatch-bound: the GPU is idle for
most of each step while Python issues kernels, so wall-clock ratios between
variants measure launch counts and hide any work done on the GPU. This script
reports the other side: kernels per step and the GPU time they take, and the
attention-related kernels within it. GPU time is the step time a variant would
approach if the host side were removed (CUDA graphs, batching), so the ratio of
GPU times bounds the cost the cipher would show in a GPU-bound regime.

Wall times measured here include profiler overhead and are reported only for the
GPU-busy fraction; the e2e harness is the source of wall-clock numbers.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM

from puf4secure_kvcache.model_utils import resolve_snapshot_path
from puf4secure_kvcache.ring_install import (install_ring_attention,
                                             uninstall_ring_attention)

RING_KERNELS = ("_decode_split", "_reduce", "_append_row")


@torch.no_grad()
def decode(model, ids, steps, prof_ctx=None):
    P = ids.shape[1]
    pos = torch.arange(P, device=ids.device)
    out = model(input_ids=ids, use_cache=True, cache_position=pos,
                position_ids=pos[None])
    lg, past = out.logits[:, -1], out.past_key_values
    torch.cuda.synchronize()
    if prof_ctx:
        prof_ctx.__enter__()
    t0 = time.perf_counter()
    for i in range(steps):
        nxt = lg.argmax(-1, keepdim=True)
        cp = torch.tensor([P + i], device=ids.device)
        out = model(input_ids=nxt, past_key_values=past, use_cache=True,
                    cache_position=cp, position_ids=cp[None])
        lg, past = out.logits[:, -1], out.past_key_values
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / steps * 1e3
    if prof_ctx:
        prof_ctx.__exit__(None, None, None)
    return wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", default="/home/feihm/.cache/huggingface/"
                    "hub/models--Qwen--Qwen3-0.6B")
    ap.add_argument("--tag", default="qwen3")
    ap.add_argument("--prefills", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384])
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    mp = resolve_snapshot_path(args.model_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        mp, local_files_only=True, trust_remote_code=True, dtype=torch.float32,
        device_map="cuda").eval()
    model.config._attn_implementation = "eager"
    res = {"model": args.tag, "steps": args.steps, "per_prefill": {}}
    for P in args.prefills:
        g = torch.Generator().manual_seed(P)
        ids = torch.randint(100, 1000, (1, P), generator=g).to("cuda")
        row = {}
        for mode in ("plain", "quant", "ring"):
            uninstall_ring_attention(model)
            if mode != "plain":
                install_ring_attention(model, mode=mode,
                                       max_new=args.steps + 8)
            decode(model, ids, args.steps)                     # warm, compile
            ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            wall = decode(model, ids, args.steps, ctx)
            ev = [e for e in ctx.key_averages() if e.device_type.name == "CUDA"]
            n = sum(e.count for e in ev) / args.steps
            gpu = sum(e.self_device_time_total for e in ev) / args.steps / 1e3
            ring_k = {k: sum(e.self_device_time_total for e in ev if k in e.key)
                      / args.steps / 1e3 for k in RING_KERNELS}
            row[mode] = {"kernels_per_step": n, "gpu_ms_per_step": gpu,
                         "gpu_busy_fraction": gpu / wall,
                         "ring_kernel_ms": ring_k}
            print(f"P={P:6d} {mode:5s} kernels/step {n:7.1f}  GPU {gpu:6.2f} ms"
                  f"  busy {gpu / wall:5.1%}  "
                  + " ".join(f"{k}={v:.3f}" for k, v in ring_k.items() if v),
                  flush=True)
        uninstall_ring_attention(model)
        for mode in ("quant", "ring"):
            row[f"{mode}_gpu_ratio_vs_plain"] = (row[mode]["gpu_ms_per_step"]
                                                 / row["plain"]["gpu_ms_per_step"])
        row["ring_gpu_ratio_vs_quant"] = (row["ring"]["gpu_ms_per_step"]
                                          / row["quant"]["gpu_ms_per_step"])
        res["per_prefill"][str(P)] = row
        print(f"         GPU-time ratios: quant/plain "
              f"{row['quant_gpu_ratio_vs_plain']:.3f}  ring/plain "
              f"{row['ring_gpu_ratio_vs_plain']:.3f}  ring/quant "
              f"{row['ring_gpu_ratio_vs_quant']:.3f}", flush=True)
    out = args.out or pathlib.Path(
        f"experiments/runs/ring_decode_profile_{args.tag}_v2.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({"out": str(out)}))


if __name__ == "__main__":
    main()
