#!/usr/bin/env python3
"""End-to-end ring decode, with the harness itself under test.

No timing is reported until the harness has shown it was not measuring itself.
Per prefill length the run proves, and records:

1. **no JIT in the timed region** -- a Triton compile counter must read zero over
   every timed repeat, not over a sample window, so compilation is never
   charged to the mechanism. An earlier version checked a 6-step window only;
   Triton specialised the length argument on divisibility by 16, so step 15 of
   the first repeat compiled (7 s for ``quant``, 29 s for ``ring``) outside the
   window, and only the median kept it out of the headline;
2. **a stable launch count per decode step** -- drift would mean the work per
   step is not the geometry under test;
3. **matching cache geometry** between ``ring`` and its ``quant`` control, so the
   difference between them is the cipher and nothing else;
4. **a reproducible median** -- variants run interleaved, not in blocks, so clock
   and thermal drift cannot be mistaken for an effect;
5. **a negative control** -- ``quant`` exercises the whole integration path with
   the cipher switched off, which separates infrastructure cost from mechanism
   cost. In an earlier run this control was itself 15x slow, which is what
   revealed the harness fault rather than a design fault.

Reported per prefill: tok/s, per-token latency, TTFT, attention-kernel share and
end-to-end step time, plus the ratios that the design actually turns on,
``T_ring / T_plain`` and ``tok/s_ring / tok/s_plain``, against context length.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from puf4secure_kvcache.harness_guard import (KernelGuard, cache_geometry,
                                              geometry_matches)
from puf4secure_kvcache.model_utils import resolve_snapshot_path
from puf4secure_kvcache.ring_install import (install_ring_attention,
                                             uninstall_ring_attention)

MODES = ("plain", "quant", "ring")


@torch.no_grad()
def decode_run(model, ids, n_new: int, guard: KernelGuard | None = None):
    """Greedy decode with explicit positions. Returns tokens, TTFT, step times."""
    dev = ids.device
    P = ids.shape[1]
    pos = torch.arange(P, device=dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True, cache_position=pos,
                position_ids=pos[None, :])
    torch.cuda.synchronize(); ttft = time.perf_counter() - t0
    logits, past = out.logits[:, -1, :], out.past_key_values
    toks, steps = [], []
    for i in range(n_new):
        nxt = logits.argmax(-1, keepdim=True)
        toks.append(int(nxt))
        cp = torch.tensor([P + i], device=dev)
        torch.cuda.synchronize(); t = time.perf_counter()
        out = model(input_ids=nxt, past_key_values=past, use_cache=True,
                    cache_position=cp, position_ids=cp[None, :])
        torch.cuda.synchronize(); steps.append(time.perf_counter() - t)
        if guard is not None:
            guard.mark(f"step{i}")
        logits, past = out.logits[:, -1, :], out.past_key_values
    return toks, ttft, steps


def set_mode(model, mode, n_new, splits):
    uninstall_ring_attention(model)
    if mode != "plain":
        install_ring_attention(model, mode=mode, max_new=n_new + 4,
                               splits=splits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", type=pathlib.Path, default=pathlib.Path(
        "/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
    ap.add_argument("--tag", default="qwen3")
    ap.add_argument("--prefills", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384])
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--splits", type=int, default=32)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    mp = resolve_snapshot_path(args.model_cache_dir)
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True,
                                        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mp, local_files_only=True, trust_remote_code=True,
        dtype=torch.float32, device_map="cuda").eval()
    model.config._attn_implementation = "eager"

    res = {"model": args.tag, "new_tokens": args.new_tokens,
           "repeats": args.repeats, "dtype": "fp32",
           "method": ("variants interleaved; compile counter must be zero in "
                      "the timed region; launch count per step must be stable"),
           "per_prefill": {}}

    for P in args.prefills:
        g = torch.Generator().manual_seed(P)
        ids = torch.randint(100, max(101, len(tok) - 1), (1, P),
                            generator=g).to("cuda")
        checks, geo, samples = {}, {}, {m: [] for m in MODES}
        toks_by_mode = {}

        # --- warm up every variant and capture geometry + launch profile ---
        for mode in MODES:
            set_mode(model, mode, args.new_tokens, args.splits)
            # every length the timed run will reach, so any length-dependent
            # specialisation is compiled here and not inside a timed step
            decode_run(model, ids, args.new_tokens)
            with KernelGuard() as kg:
                _, _, _ = decode_run(model, ids, 6, guard=kg)
            rep = kg.report()
            checks[mode] = {
                "compiles_in_timed_region": rep["compiles_in_region"],
                "compile_signatures": rep["compile_signatures"],
                "launches_per_step": rep["launches_per_step"],
                "launches_per_step_stable": rep["launches_per_step_stable"],
                "launches_by_kernel": rep["launches_by_kernel"],
            }
            geo[mode] = cache_geometry(model)
            kg.assert_no_compiles(f"{mode} timed region at P={P}")
            if not rep["launches_per_step_stable"]:
                raise RuntimeError(
                    f"{mode} at P={P}: launch count per step drifts "
                    f"{rep['launches_per_step']}")

        ok, diffs = geometry_matches(geo["quant"], geo["ring"])
        checks["ring_quant_geometry_match"] = ok
        checks["ring_quant_geometry_diffs"] = diffs
        if not ok:
            raise RuntimeError(f"ring and its quant control differ in geometry "
                               f"at P={P}: {diffs}")

        # --- interleaved timing: one repeat of each variant per round ---
        # Every timed repeat runs under the guard; it only counts, so the cost
        # is a counter increment per launch.
        order = list(MODES)
        timed = {m: {"compiles": 0, "launch_profiles": set()} for m in MODES}
        for r in range(args.repeats):
            rot = order[r % len(order):] + order[:r % len(order)]
            for mode in rot:
                set_mode(model, mode, args.new_tokens, args.splits)
                with KernelGuard() as kg:
                    t, ttft, steps = decode_run(model, ids, args.new_tokens,
                                                guard=kg)
                timed[mode]["compiles"] += kg.compiles
                timed[mode]["launch_profiles"].add(
                    tuple(kg.launches_per_mark[1:]))
                kg.assert_no_compiles(f"{mode} timed repeat {r} at P={P}")
                ms = sorted(x * 1e3 for x in steps)
                samples[mode].append({
                    "ttft_s": ttft,
                    # per-step median is the robust statistic: a single stalled
                    # step distorts len/sum(steps) badly, which is what produced
                    # a 98% spread in the first run of this harness.
                    "latency_ms": ms[len(ms) // 2],
                    "p90_ms": ms[int(0.9 * (len(ms) - 1))],
                    "max_ms": ms[-1],
                    "tokens_per_s": len(steps) / sum(steps),
                    "tps_from_median": 1e3 / ms[len(ms) // 2],
                    "step_ms": ms[len(ms) // 2],
                    "steps_ms": ms,
                    # keep the order too: sorting alone hides which step stalled
                    "steps_ms_in_order": [x * 1e3 for x in steps],
                    "argmax_step": max(range(len(steps)),
                                       key=lambda i: steps[i]),
                })
                toks_by_mode[mode] = t
        uninstall_ring_attention(model)
        for mode in MODES:
            profs = timed[mode]["launch_profiles"]
            flat = {x for prof in profs for x in prof}
            checks[mode]["compiles_in_timed_repeats"] = timed[mode]["compiles"]
            checks[mode]["timed_launches_per_step"] = sorted(flat)
            checks[mode]["timed_launches_stable"] = len(flat) <= 1
            if len(flat) > 1:
                raise RuntimeError(f"{mode} at P={P}: launch count per step "
                                   f"drifts across timed repeats: {sorted(flat)}")

        row = {"checks": checks, "geometry": geo["ring"]}
        for mode in MODES:
            s = samples[mode]
            lat = [x["latency_ms"] for x in s]
            tpsm = [x["tps_from_median"] for x in s]
            row[mode] = {
                # headline uses the median step, with the tail reported beside it
                "latency_ms": statistics.median(lat),
                "latency_rel_spread": (max(lat) - min(lat)) / statistics.median(lat),
                "tokens_per_s": statistics.median(tpsm),
                "tokens_per_s_rel_spread": (max(tpsm) - min(tpsm))
                / statistics.median(tpsm),
                "p90_ms": statistics.median([x["p90_ms"] for x in s]),
                "max_ms": max(x["max_ms"] for x in s),
                "tail_ratio_max_over_median": max(x["max_ms"] for x in s)
                / statistics.median(lat),
                "tps_sum_based": statistics.median([x["tokens_per_s"] for x in s]),
                "ttft_s": statistics.median([x["ttft_s"] for x in s]),
                "step_ms": statistics.median(lat),
                "raw_steps_ms": [x["steps_ms"] for x in s],
                "steps_in_order": [x["steps_ms_in_order"] for x in s],
                "slowest_step_index": [x["argmax_step"] for x in s],
            }
        base = row["plain"]
        for mode in ("quant", "ring"):
            row[f"{mode}_tps_ratio"] = row[mode]["tokens_per_s"] / base["tokens_per_s"]
            row[f"{mode}_time_ratio"] = row[mode]["step_ms"] / base["step_ms"]
        row["ring_over_quant_time_ratio"] = row["ring"]["step_ms"] / row["quant"]["step_ms"]
        row["token_agreement"] = {
            m: sum(int(a == b) for a, b in zip(toks_by_mode[m],
                                               toks_by_mode["plain"]))
            / len(toks_by_mode["plain"]) for m in MODES}
        row["ring_equals_quant_tokens"] = (toks_by_mode["ring"]
                                           == toks_by_mode["quant"])
        # the median must be reproducible before any ratio is believed
        worst = max(row[m]["latency_rel_spread"] for m in MODES)
        checks["latency_rel_spread_worst"] = worst
        checks["median_reproducible"] = worst <= 0.05
        if not checks["median_reproducible"]:
            print(f"  WARNING P={P}: median step time varies by "
                  f"{worst:.1%} across repeats; ratios below are not reliable",
                  flush=True)
        res["per_prefill"][str(P)] = row

        print(f"P={P:6d} | checks: compiles(probe)={[checks[m]['compiles_in_timed_region'] for m in MODES]} "
              f"compiles(timed)={[checks[m]['compiles_in_timed_repeats'] for m in MODES]} "
              f"launch/step={[checks[m]['launches_per_step'][1] for m in MODES]} "
              f"geo_match={checks['ring_quant_geometry_match']}")
        print(f"         plain {base['tokens_per_s']:7.2f}  "
              f"quant {row['quant']['tokens_per_s']:7.2f} (x{row['quant_tps_ratio']:.3f})  "
              f"ring {row['ring']['tokens_per_s']:7.2f} (x{row['ring_tps_ratio']:.3f})  tok/s | "
              f"T_ring/T_plain={row['ring_time_ratio']:.3f}  "
              f"spread(med)={worst:.3f}  "
              f"tail max/med ring={row['ring']['tail_ratio_max_over_median']:.2f}  "
              f"ring==quant tokens: {row['ring_equals_quant_tokens']}", flush=True)

    res["scaling"] = {
        "context": [int(p) for p in res["per_prefill"]],
        "ring_time_ratio": [res["per_prefill"][p]["ring_time_ratio"]
                            for p in res["per_prefill"]],
        "ring_tps_ratio": [res["per_prefill"][p]["ring_tps_ratio"]
                           for p in res["per_prefill"]],
        "quant_time_ratio": [res["per_prefill"][p]["quant_time_ratio"]
                             for p in res["per_prefill"]],
    }
    out = args.out or pathlib.Path(
        f"experiments/runs/ring_e2e_verified_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True))
    print("\nT_ring/T_plain vs context:",
          dict(zip(res["scaling"]["context"],
                   [round(x, 3) for x in res["scaling"]["ring_time_ratio"]])))
    print(json.dumps({"out": str(out)}))


if __name__ == "__main__":
    main()
