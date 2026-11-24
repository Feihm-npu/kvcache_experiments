#!/usr/bin/env python3
"""C3: energy per decoded token for plain, quant (int16, no pad) and ring.

Reuses the mode switch of run_ring_e2e_verified.py, so the three variants run
exactly the code the latency measurements run. Every round decodes once with
each variant in rotated order; the prefill runs outside the power window and
only the decode steps run under the module-power sampler of jetson_power.py.
The idle baseline is measured separately and reported beside, never subtracted
silently.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from puf4secure_kvcache.model_utils import resolve_snapshot_path
from puf4secure_kvcache.ring_install import uninstall_ring_attention

HERE = pathlib.Path(__file__).parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e2e = _load("e2e", "run_ring_e2e_verified.py")
power = _load("power", "jetson_power.py")


@torch.no_grad()
def decode_energy(model, ids, n_new: int):
    """Prefill outside the power window, then decode n_new tokens inside it."""
    dev = ids.device
    P = ids.shape[1]
    pos = torch.arange(P, device=dev)
    out = model(input_ids=ids, use_cache=True, cache_position=pos,
                position_ids=pos[None, :])
    torch.cuda.synchronize()
    logits, past = out.logits[:, -1, :], out.past_key_values
    with power.PowerSampler() as ps:
        for i in range(n_new):
            nxt = logits.argmax(-1, keepdim=True)
            cp = torch.tensor([P + i], device=dev)
            out = model(input_ids=nxt, past_key_values=past, use_cache=True,
                        cache_position=cp, position_ids=cp[None, :])
            logits, past = out.logits[:, -1, :], out.past_key_values
        torch.cuda.synchronize()
    rec = ps.result()
    rec["tokens"] = n_new
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", required=True)
    ap.add_argument("--tag", default="qwen3")
    ap.add_argument("--prefills", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--splits", type=int, default=32)
    ap.add_argument("--idle-seconds", type=float, default=20.0)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    mp = resolve_snapshot_path(args.model_cache_dir)
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mp, local_files_only=True, trust_remote_code=True, dtype=torch.float32,
        device_map="cuda").eval()
    model.config._attn_implementation = "eager"

    res = {"model": args.tag, "new_tokens": args.new_tokens, "rounds": args.rounds,
           "platform": power.platform_record(),
           "idle": power.measure_idle(args.idle_seconds), "per_prefill": {}}
    for P in args.prefills:
        g = torch.Generator().manual_seed(P)
        ids = torch.randint(100, max(101, len(tok) - 1), (1, P), generator=g).to("cuda")
        for mode in e2e.MODES:                       # warm every variant first
            e2e.set_mode(model, mode, args.new_tokens, args.splits)
            e2e.decode_run(model, ids, args.new_tokens)
        rows = {m: [] for m in e2e.MODES}
        for r in range(args.rounds):
            order = list(e2e.MODES)
            for mode in order[r % 3:] + order[:r % 3]:
                e2e.set_mode(model, mode, args.new_tokens, args.splits)
                rows[mode].append(decode_energy(model, ids, args.new_tokens))
        uninstall_ring_attention(model)
        res["per_prefill"][str(P)] = {
            m: {"rounds": v,
                "median_joules_per_token": statistics.median(
                    x["energy_j"] / x["tokens"] for x in v if x.get("energy_j") is not None)
                if any(x.get("energy_j") is not None for x in v) else None,
                "median_tokens_per_s": statistics.median(x["tokens"] / x["elapsed_s"] for x in v)}
            for m, v in rows.items()}
        print(P, {m: (r["median_joules_per_token"], r["median_tokens_per_s"])
                  for m, r in res["per_prefill"][str(P)].items()}, flush=True)
    out = args.out or pathlib.Path(f"jetson_energy_{args.tag}.json")
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
