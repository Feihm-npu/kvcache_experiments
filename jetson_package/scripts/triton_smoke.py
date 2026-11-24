#!/usr/bin/env python3
"""Gate C0: do the ring kernels compile and run bit-exactly on this GPU?

Writes triton_smoke.json with versions, the device, and for two head dimensions
and two grouped-query ratios whether the padded decode equals the unpadded one
bit for bit, and how far both are from an fp64 reference over the same
quantized cache. Run this before anything else; if status is not "passed",
stop and send the JSON back.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import traceback

import torch

out = {"python": sys.version.split()[0], "torch": torch.__version__}
try:
    import triton
    out["triton"] = triton.__version__
    out["cuda"] = torch.version.cuda
    out["device"] = torch.cuda.get_device_name(0)
    out["capability"] = list(torch.cuda.get_device_capability(0))
    from puf4secure_kvcache import ring_attn as ra
    from puf4secure_kvcache import ring_install as ri

    checks = {}
    torch.manual_seed(0)
    for d, hkv, gq in ((64, 2, 4), (128, 2, 2)):
        cap, n = 300, 260
        k = torch.randn(hkv, cap, d, device="cuda") * 2
        v = torch.randn(hkv, cap, d, device="cuda") * 2
        q = torch.randn(hkv * gq, d, device="cuda")
        key, nonce = ri.residency_key()
        cm = ra.quantize_and_protect(k, v, key, nonce, ri.K_BASE, ri.V_BASE,
                                     masked=True, n_valid=n)
        cu = ra.quantize_and_protect(k, v, key, nonce, ri.K_BASE, ri.V_BASE,
                                     masked=False, n_valid=n)
        t0 = time.perf_counter()
        om = ra.ring_decode(q, cm, masked=True)
        torch.cuda.synchronize()
        first_call = time.perf_counter() - t0
        ou = ra.ring_decode(q, cu, masked=False)
        # fp64 reference over the dequantized int16 cache
        sk, sv = ra.reveal_scales(cu)
        kd = cu.kt[:, :n].double() * sk[:, None, :].double()
        vd = cu.vt[:, :n].double() * sv[:, None, :].double()
        kr = kd.repeat_interleave(gq, 0)
        vr = vd.repeat_interleave(gq, 0)
        att = torch.softmax(torch.einsum("hd,hnd->hn", q.double(), kr) / d ** 0.5, -1)
        ref = torch.einsum("hn,hnd->hd", att, vr)
        checks[f"d{d}_gqa{gq}"] = {
            "ring_equals_quant": bool(torch.equal(om, ou)),
            "max_abs_vs_fp64": float((om.double() - ref).abs().max()),
            "first_call_s": first_call,
        }
    out["checks"] = checks
    ok = all(c["ring_equals_quant"] and c["max_abs_vs_fp64"] < 1e-3 for c in checks.values())
    out["status"] = "passed" if ok else "failed"
except Exception:                       # report, do not hide, any failure
    out["status"] = "error"
    out["error"] = traceback.format_exc()
path = pathlib.Path("triton_smoke.json")
path.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
