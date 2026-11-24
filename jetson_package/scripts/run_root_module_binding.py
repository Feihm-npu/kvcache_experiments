#!/usr/bin/env python3
"""RQ4 (root module): a sealed cache opens only on its own board and under the
owner's passphrase, and one unlock costs a measured time.

Seals a Qwen3-0.6B cache with the first board's root module, then opens it
``--trials`` times in each setting that has a config:

* ``owner``:    first board, owner's passphrase     -> opens bit-exactly
* ``wrong``:    first board, a fresh wrong passphrase -> fails authentication
* ``other``:    second board, its own enrollment     -> fails authentication
* ``moved``:    second board started with the first board's enrollment
* ``modified``: first board started with a modified helper
                                                      -> both fail the tag

Each config is a module config for ``root_module.module_from_config``. Unlock
and derive are timed on the host; a module that reports phase timings in its
unlock reply has them recorded as well. The output holds no key, root or
passphrase; the owner's passphrase is read from a file, never from argv.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

from puf4secure_kvcache import cold_tier as ct
from puf4secure_kvcache.model_utils import kv_to_list, resolve_snapshot_path
from puf4secure_kvcache.puf_sim import ReconstructionError
from puf4secure_kvcache.root_module import module_from_config

STORE = "binding-test"


def outcome_of(cfg, passphrase, sealed, counter, model_id, reference):
    """One unlock and one open with a fresh module client; returns the outcome
    class and the host-side unlock time."""
    module = module_from_config(cfg)
    t = time.perf_counter()
    try:
        timing = module.unlock(passphrase) or {}
    except ReconstructionError:
        return "tag_or_reconstruction_failed", time.perf_counter() - t, {}
    t_unlock = time.perf_counter() - t
    try:
        got = ct.open_cache(sealed, module=module, model_id=model_id,
                            store_id=STORE, counter=counter)
    except ct.IntegrityError:
        return "authentication_failed", t_unlock, timing
    finally:
        module.lock()
    same = all(torch.equal(a, b) and torch.equal(c, d)
               for (a, c), (b, d) in zip(reference, got))
    return ("opened_bit_exact" if same else "opened_mismatch"), t_unlock, timing


def summary(xs):
    return {"median": statistics.median(xs), "max": max(xs)} if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", type=pathlib.Path, required=True)
    for name in ("other", "moved", "modified"):
        ap.add_argument(f"--{name}", type=pathlib.Path, default=None)
    ap.add_argument("--passphrase-file", type=pathlib.Path, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--model-cache-dir", default="/home/feihm/.cache/huggingface/"
                    "hub/models--Qwen--Qwen3-0.6B")
    ap.add_argument("--tag", default="qwen3")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    passphrase = args.passphrase_file.read_bytes().rstrip(b"\r\n")

    model = AutoModelForCausalLM.from_pretrained(
        resolve_snapshot_path(args.model_cache_dir), local_files_only=True,
        trust_remote_code=True, dtype=torch.float32, device_map="cuda").eval()
    model.config._attn_implementation = "eager"
    g = torch.Generator().manual_seed(args.context)
    ids = torch.randint(100, 1000, (1, args.context), generator=g).to("cuda")
    pos = torch.arange(args.context, device="cuda")
    with torch.no_grad():
        past = model(input_ids=ids, use_cache=True, cache_position=pos,
                     position_ids=pos[None]).past_key_values
    reference = [(k[0].contiguous().cpu(), v[0].contiguous().cpu())
                 for k, v in kv_to_list(past)]
    del past, model

    counter = ct.MonotonicCounter()
    owner = module_from_config(args.owner)
    owner.unlock(passphrase)
    sealed = ct.seal_cache(reference, module=owner, model_id=args.tag,
                           store_id=STORE, counter=counter)
    label = f"cold|aead|model={args.tag}|store={STORE}".encode()
    derive_s = []
    for _ in range(args.trials):
        t = time.perf_counter()
        owner.derive(label)
        derive_s.append(time.perf_counter() - t)
    owner.lock()

    settings = [("owner", args.owner, False), ("wrong", args.owner, True)]
    settings += [(n, getattr(args, n), False) for n in ("other", "moved", "modified")
                 if getattr(args, n) is not None]
    res = {"model": args.tag, "context": args.context, "trials": args.trials,
           "host": platform.node(), "torch": torch.__version__,
           "derive_s": summary(derive_s), "settings": {}}
    for name, cfg, wrong in settings:
        counts, unlock_s, phases = {}, [], {}
        for _ in range(args.trials):
            pw = os.urandom(16).hex().encode() if wrong else passphrase
            kind, t_unlock, timing = outcome_of(cfg, pw, sealed, counter,
                                                args.tag, reference)
            counts[kind] = counts.get(kind, 0) + 1
            unlock_s.append(t_unlock)
            for k, v in timing.items():
                if isinstance(v, (int, float)):
                    phases.setdefault(k, []).append(v)
        res["settings"][name] = {
            "outcomes": counts, "unlock_s": summary(unlock_s),
            "module_reported_s": {k: summary(v) for k, v in phases.items()}}
        print(name, counts, flush=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({"out": str(args.out)}))


if __name__ == "__main__":
    main()
