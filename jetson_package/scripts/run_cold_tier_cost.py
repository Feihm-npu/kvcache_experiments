#!/usr/bin/env python3
"""RQ4 (cold tier): what AEAD costs when it is paid per movement, not per step.

For a real Qwen3-0.6B cache at each context length:

* seal (device -> host, AEAD per page) and open (AEAD per page -> device), at
  several thread counts, and the sealed-size overhead;
* entry into the hot tier (quantize + pad every layer under a fresh residency);
* recomputing the cache instead (the prefill a load replaces);
* the naive alternative the two-tier design exists to avoid: no hot tier, so the
  whole cache is opened on every decode step. Its per-step cost is the open time.

The decode step time is read from the verified e2e artifact, not re-measured.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

from puf4secure_kvcache import cold_tier as ct
from puf4secure_kvcache import ring_attn as ra
from puf4secure_kvcache import ring_install as ri
from puf4secure_kvcache.model_utils import kv_to_list, resolve_snapshot_path
from puf4secure_kvcache.root_module import SimulatedRootModule


def timed(fn, reps):
    fn()                                  # warm: Triton compile, allocator
    out, ts = None, []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return out, statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", default="/home/feihm/.cache/huggingface/"
                    "hub/models--Qwen--Qwen3-0.6B")
    ap.add_argument("--tag", default="qwen3")
    ap.add_argument("--prefills", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384])
    # the cryptography binding holds the GIL, so threads do not scale
    # (measured: 1.11 GiB/s at 1, 8 and 32 threads); keep the knob for a
    # binding that releases it
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--dtypes", nargs="+", default=["fp32", "bf16"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--e2e", type=pathlib.Path, default=pathlib.Path(
        "experiments/runs/ring_e2e_verified_qwen3_v2.json"))
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    mp = resolve_snapshot_path(args.model_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        mp, local_files_only=True, trust_remote_code=True, dtype=torch.float32,
        device_map="cuda").eval()
    model.config._attn_implementation = "eager"
    # the decode step time, if a latency run exists, converts the open time into
    # the per-step slowdown of decrypting before every step; without it that
    # column is simply not reported
    e2e = (json.loads(args.e2e.read_text())["per_prefill"]
           if args.e2e and args.e2e.is_file() else {})
    dts = {"fp32": torch.float32, "bf16": torch.bfloat16}
    # one unlock per session, outside every timed seal and open
    module = SimulatedRootModule("device_A")
    module.unlock(b"owner passphrase")
    res = {"model": args.tag, "page_tokens": ct.PAGE_TOKENS_DEFAULT,
           "aead": "AES-256-GCM (OpenSSL via cryptography), random 96-bit nonce",
           "per_prefill": {}}
    import os
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    a, buf = AESGCM(AESGCM.generate_key(256)), os.urandom(256 << 20)
    t = time.perf_counter(); a.encrypt(b"\0" * 12, buf, b""); dt = time.perf_counter() - t
    res["raw_aes_gcm_single_core_GBps"] = len(buf) / dt / 1e9
    print(f"raw AES-256-GCM single core: {len(buf) / dt / 1e9:.2f} GB/s", flush=True)

    for P in args.prefills:
        g = torch.Generator().manual_seed(P)
        ids = torch.randint(100, 1000, (1, P), generator=g).to("cuda")
        pos = torch.arange(P, device="cuda")

        def prefill():
            with torch.no_grad():
                return model(input_ids=ids, use_cache=True, cache_position=pos,
                             position_ids=pos[None]).past_key_values
        past, t_prefill = timed(prefill, args.reps)
        kv32 = [(k[0].contiguous(), v[0].contiguous())
                for k, v in kv_to_list(past)]
        del past
        row = {"recompute_prefill_s": t_prefill}
        step_ms = e2e.get(str(P), {}).get("ring", {}).get("step_ms")
        row["ring_decode_step_ms_from_e2e"] = step_ms

        for dn in args.dtypes:
            kv = [(k.to(dts[dn]), v.to(dts[dn])) for k, v in kv32]
            plain_bytes = sum(k.numel() * k.element_size() * 2 for k, _ in kv)
            r = {"plaintext_bytes": plain_bytes, "seal_s": {}, "open_s": {}}
            for th in args.threads:
                ctr = ct.MonotonicCounter()
                sealed, t_seal = timed(lambda: ct.seal_cache(
                    kv, module=module, model_id=args.tag, store_id="s",
                    counter=ctr, threads=th), args.reps)
                got, t_open = timed(lambda: ct.open_cache(
                    sealed, module=module, model_id=args.tag,
                    store_id="s", counter=ctr, device="cuda", threads=th),
                    args.reps)
                r["seal_s"][str(th)] = t_seal
                r["open_s"][str(th)] = t_open
                r["sealed_bytes"] = sealed.nbytes()
                assert all(torch.equal(a[0], b[0].to(a[0].device))
                           for a, b in zip(kv, got))
                del got
            r["size_overhead"] = r["sealed_bytes"] / plain_bytes - 1
            best_open = min(r["open_s"].values())
            r["open_GBps_best"] = plain_bytes / best_open / 1e9
            r["seal_GBps_best"] = plain_bytes / min(r["seal_s"].values()) / 1e9

            # hot-tier entry: every layer under a fresh residency
            def entry():
                out = []
                for li, (k, v) in enumerate(kv):
                    rid = ri.new_residency_id()
                    key, nonce = ri.residency_key()
                    out.append(ra.quantize_and_protect(
                        k.float(), v.float(), key, nonce, ri.K_BASE, ri.V_BASE,
                        masked=True, residency_id=rid))
                return out
            _, t_entry = timed(entry, args.reps)
            r["hot_entry_s"] = t_entry
            r["load_total_s"] = best_open + t_entry
            r["load_vs_recompute"] = r["load_total_s"] / t_prefill
            if step_ms:
                # no hot tier: the whole cache is opened before every step
                r["per_step_aead_slowdown"] = (best_open * 1e3 + step_ms) / step_ms
                # steps after which paying once beats paying per step
                r["two_tier_breakeven_steps"] = r["load_total_s"] / best_open
            row[dn] = r
            print(f"P={P:6d} {dn}: {plain_bytes / 2**30:.2f} GiB  seal "
                  + " ".join(f"{t}t:{s:.2f}s" for t, s in r["seal_s"].items())
                  + "  open " + " ".join(f"{t}t:{s:.2f}s"
                                         for t, s in r["open_s"].items())
                  + f"  entry {t_entry * 1e3:.0f}ms  recompute {t_prefill:.2f}s"
                  + f"  load/recompute {r['load_vs_recompute']:.2f}"
                  + (f"  per-step-AEAD slowdown x{r['per_step_aead_slowdown']:.1f}"
                     if step_ms else ""), flush=True)
            del kv
        res["per_prefill"][str(P)] = row
        del kv32
        torch.cuda.empty_cache()

    out = args.out or pathlib.Path(f"experiments/runs/cold_tier_cost_{args.tag}.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({"out": str(out)}))


if __name__ == "__main__":
    main()
