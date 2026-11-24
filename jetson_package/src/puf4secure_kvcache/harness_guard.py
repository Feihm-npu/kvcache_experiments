"""Instrumentation that makes a GPU timing harness a verified object.

Three measurements in this project were wrong because the harness, not the
mechanism, was at fault: a buffer that fitted in L2 so no HBM traffic was timed,
a single-program grid so nothing overlapped, and constexprs derived from a
growing cache so Triton recompiled inside the timed loop. Each produced a
plausible number that would have supported the wrong decision.

So the rule here is that a performance number is only reported once the harness
has proved it was not measuring itself:

* ``compiles`` must be zero inside the timed region. A non-zero count means JIT
  time was charged to the mechanism;
* ``launches`` per decode step must be identical across steps. Drift means the
  work per step is not what the geometry says it is;
* geometry must match between the variant under test and its control, or the
  comparison is not of the mechanism;
* and a negative control that exercises everything except the mechanism must be
  present, so infrastructure cost can be subtracted rather than attributed.

Usage::

    with KernelGuard() as g:
        ...timed region...
    g.assert_no_compiles()
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from triton.runtime.jit import JITFunction

_orig_do_compile = JITFunction._do_compile
_orig_run = JITFunction.run
_active: list["KernelGuard"] = []


def _patched_do_compile(self, key, signature, device, constexprs, options,
                        attrs, warmup):
    for g in _active:
        g.compiles += 1
        g.compile_sigs[f"{self.fn.__name__}|{constexprs}"] += 1
    return _orig_do_compile(self, key, signature, device, constexprs, options,
                            attrs, warmup)


def _patched_run(self, *args, grid, warmup, **kwargs):
    for g in _active:
        g.launches += 1
        g.launch_names[self.fn.__name__] += 1
    return _orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)


JITFunction._do_compile = _patched_do_compile
JITFunction.run = _patched_run


@dataclass
class KernelGuard:
    """Counts Triton compilations and launches inside a region."""
    compiles: int = 0
    launches: int = 0
    compile_sigs: Counter = field(default_factory=Counter)
    launch_names: Counter = field(default_factory=Counter)
    marks: list = field(default_factory=list)

    def __enter__(self):
        _active.append(self)
        return self

    def __exit__(self, *exc):
        _active.remove(self)
        return False

    def mark(self, label: str) -> None:
        """Record the launch count at a step boundary."""
        self.marks.append((label, self.launches))

    @property
    def launches_per_mark(self) -> list[int]:
        out, prev = [], 0
        for _, n in self.marks:
            out.append(n - prev)
            prev = n
        return out

    def assert_no_compiles(self, where: str = "timed region") -> None:
        if self.compiles:
            raise RuntimeError(
                f"{self.compiles} Triton specialization(s) compiled inside the "
                f"{where}; JIT time would be charged to the mechanism. "
                f"Signatures: {dict(self.compile_sigs)}")

    def launch_count_is_stable(self, skip_first: int = 1) -> bool:
        """True when every step after the first issues the same launch count."""
        per = self.launches_per_mark[skip_first:]
        return len(set(per)) <= 1 if per else True

    def report(self) -> dict:
        per = self.launches_per_mark
        return {
            "compiles_in_region": self.compiles,
            "compile_signatures": dict(self.compile_sigs),
            "launches_total": self.launches,
            "launches_per_step": per,
            "launches_per_step_stable": self.launch_count_is_stable(),
            "launches_by_kernel": dict(self.launch_names),
        }


def cache_geometry(model) -> dict:
    """Geometry a variant and its control must share for the comparison to hold."""
    out = {}
    for i, lyr in enumerate(getattr(model.model, "layers", [])):
        st = getattr(lyr.self_attn, "_ring", None)
        if st is None or st.get("cache") is None:
            continue
        c = st["cache"]
        out[i] = {"cap": int(c.kt.shape[1]), "n": int(c.n),
                  "split": int(c.split), "n_split": int(c.n_split),
                  "dtype": str(c.kt.dtype), "shape": list(c.kt.shape),
                  "clamped": int(c.clamped)}
    return out


def geometry_matches(a: dict, b: dict, ignore=("clamped",)) -> tuple[bool, list]:
    """Compare two geometry dicts, returning the differing fields."""
    diffs = []
    for k in sorted(set(a) | set(b)):
        ga, gb = a.get(k), b.get(k)
        if ga is None or gb is None:
            diffs.append((k, ga, gb)); continue
        for f in ga:
            if f in ignore:
                continue
            if ga[f] != gb[f]:
                diffs.append((f"layer{k}.{f}", ga[f], gb[f]))
    return (not diffs), diffs
