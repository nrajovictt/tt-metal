#!/usr/bin/env python3
"""Check that two SDPA roofline implementations agree: ours and the polaris model of record.

Offline, no card, no venv. Reads nothing but the two modules.

    python3 analysis/campaigns/roofline_compat.py                    # default paths
    python3 analysis/campaigns/roofline_compat.py A.py B.py          # explicit

There are two SDPA rooflines in this project. `$TTM/analysis/roofline.py` is the internal one that
`zone_decomp.py` imports first; `$POLARIS/ttsim/perf/roofline_sdpa.py` is the model of record
(runbook section 9) and is what the reducer falls back to. Because the fallback is silent, a
divergence between them shows up as decomposition columns that quietly change meaning depending on
which checkout a colleague happens to have. This script makes that visible.

It checks three things, in increasing strictness:

1. The reducer contract. `zone_decomp.py` constructs SdpaConfig with ten keywords and reads four
   result attributes. If either module fails that, the reducer is broken with it and nothing else
   matters. Exit status is non-zero only for this.
2. The API surface. Which SdpaConfig fields and which result attributes each side offers, and what
   is missing from one relative to the other. A field present on one side only is not necessarily a
   bug, but it is the thing that makes results incomparable.
3. Numeric agreement. Every shared, numeric result attribute over a config sweep spanning the T2.1
   grid, the fidelities, head dims, batch, dtypes and the non-prefill regimes. Reports the worst
   relative divergence per attribute and the config that produced it.

Absent the second module the script still runs and prints the surface of the one it has, which
doubles as documentation of the contract a replacement has to satisfy.
"""

import importlib.util
import math
import os
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

# Path contract (PORTABLE_CONTRACT.md).
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
WORK = Path(os.environ.get("SDPA_WORK", _REPO.parent if _REPO else _SD.parents[3]))
TTM = Path(os.environ.get("TTM", _REPO or WORK / "tt-metal"))
POLARIS = Path(os.environ.get("POLARIS", WORK / "polaris"))

DEFAULT_OURS = TTM / "analysis" / "roofline.py"
DEFAULT_POLARIS = POLARIS / "ttsim" / "perf" / "roofline_sdpa.py"

# What zone_decomp.py actually needs. Keep in sync with its model_floor() and decompose().
CONTRACT_KWARGS = ["S", "head_dim", "q_chunk", "k_chunk", "num_heads", "num_kv_heads",
                   "num_cores", "is_causal", "fidelity", "exp_approx_mode"]
CONTRACT_ATTRS = ["math_active_cycles", "inner_iters", "fpu_cycles", "sfpu_cycles"]

BASE = dict(S=4096, head_dim=128, q_chunk=128, k_chunk=128, num_heads=32, num_kv_heads=8,
            num_cores=110, is_causal=True, fidelity="HiFi2", exp_approx_mode=True)

# Spans the shapes the campaign measured plus the regimes the model routes differently.
SWEEP = (
    [dict(q_chunk=q, k_chunk=k, is_causal=c)
     for c in (True, False) for q, k in [(64, 128), (128, 128), (128, 256), (128, 512),
                                         (256, 128), (512, 128), (512, 512)]]
    + [dict(fidelity=f) for f in ("LoFi", "HiFi3", "HiFi4")]
    + [dict(head_dim=hd) for hd in (64, 96)]
    + [dict(S=s) for s in (1024, 2048, 8192)]
    + [dict(batch=b) for b in (2, 8)]
    + [dict(num_kv_heads=n) for n in (1, 32)]
    + [dict(num_cores=n) for n in (16, 64)]
    + [dict(exp_approx_mode=False), dict(fp32_dest_acc=True, accum_dtype="float32"),
       dict(input_dtype="bfloat16"), dict(kv_input_dtype="bfloat16"),
       dict(sliding_window=1024), dict(has_attn_mask=True, is_causal=False),
       dict(is_chunked=True, chunk_start_idx=1024), dict(kv_seq=8192, is_causal=False),
       dict(v_head_dim=64), dict(attention_sink=True)]
)


def load(path, name):
    """Import a module by file path so both can be live at once, whatever they call themselves."""
    path = Path(path)
    if not path.exists():
        return None, f"absent: {path}"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    except Exception as exc:                                   # noqa: BLE001 - report, never raise
        return None, f"import failed: {type(exc).__name__}: {exc}"
    missing = [n for n in ("SdpaConfig", "predict") if not hasattr(mod, n)]
    if missing:
        return None, f"does not export {', '.join(missing)}"
    return mod, "ok"


def config_fields(mod):
    try:
        return {f.name for f in dc_fields(mod.SdpaConfig)}
    except TypeError:                                          # not a dataclass; fall back to a probe
        return set(getattr(mod.SdpaConfig(**BASE), "__dict__", {}))


def result_attrs(mod):
    r = mod.predict(mod.SdpaConfig(**{k: BASE[k] for k in CONTRACT_KWARGS if k in config_fields(mod)}))
    return {n for n in dir(r) if not n.startswith("_") and not callable(getattr(r, n))}, r


def check_contract(mod, label):
    """The ten keywords and four attributes zone_decomp.py depends on. Returns True if usable."""
    have = config_fields(mod)
    missing_kw = [k for k in CONTRACT_KWARGS if k not in have]
    try:
        r = mod.predict(mod.SdpaConfig(**BASE))
    except Exception as exc:                                   # noqa: BLE001
        print(f"  {label:10s} FAIL  predict() raised on the reducer's own config: {exc}")
        return False
    missing_at = [a for a in CONTRACT_ATTRS if not hasattr(r, a)]
    bad = [a for a in CONTRACT_ATTRS if hasattr(r, a)
           and not isinstance(getattr(r, a), (int, float))]
    ok = not (missing_kw or missing_at or bad)
    print(f"  {label:10s} {'PASS' if ok else 'FAIL'}  "
          + (f"missing kwargs {missing_kw}; " if missing_kw else "")
          + (f"missing attrs {missing_at}; " if missing_at else "")
          + (f"non-numeric {bad}; " if bad else "")
          + (f"floor={getattr(r, 'math_active_cycles', '?')}" if ok else ""))
    return ok


def rel(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if a == b else float("inf")
    # components and config_echo are dicts of named terms; a whole-object compare would only ever
    # say "different", so recurse and report the worst term. Same for the reason lists.
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max((rel(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((rel(x, y) for x, y in zip(a, b)), default=0.0)
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return 0.0 if a == b else float("inf")
    if any(isinstance(v, float) and math.isnan(v) for v in (a, b)):
        return 0.0 if all(isinstance(v, float) and math.isnan(v) for v in (a, b)) else float("inf")
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom if denom else float("inf")


def compare(mod_a, mod_b, label_a, label_b):
    fields_a, fields_b = config_fields(mod_a), config_fields(mod_b)
    only_a, only_b = sorted(fields_a - fields_b), sorted(fields_b - fields_a)
    print(f"\nSdpaConfig fields: {len(fields_a)} in {label_a}, {len(fields_b)} in {label_b}")
    if only_a:
        print(f"  only in {label_a}: {', '.join(only_a)}")
    if only_b:
        print(f"  only in {label_b}: {', '.join(only_b)}")
    if not only_a and not only_b:
        print("  identical")

    attrs_a, _ = result_attrs(mod_a)
    attrs_b, _ = result_attrs(mod_b)
    oa, ob = sorted(attrs_a - attrs_b), sorted(attrs_b - attrs_a)
    print(f"\nresult attributes: {len(attrs_a)} in {label_a}, {len(attrs_b)} in {label_b}")
    if oa:
        print(f"  only in {label_a}: {', '.join(oa)}")
    if ob:
        print(f"  only in {label_b}: {', '.join(ob)}")
    if not oa and not ob:
        print("  identical")

    shared_fields = fields_a & fields_b
    shared_attrs = sorted(attrs_a & attrs_b)
    worst = {a: (0.0, None) for a in shared_attrs}
    skipped = []
    ran = 0
    for delta in SWEEP:
        if any(k not in shared_fields for k in delta):
            skipped.append(delta)
            continue
        kw = {**{k: v for k, v in BASE.items() if k in shared_fields}, **delta}
        try:
            ra = mod_a.predict(mod_a.SdpaConfig(**kw))
            rb = mod_b.predict(mod_b.SdpaConfig(**kw))
        except Exception as exc:                               # noqa: BLE001
            print(f"  predict() raised on {delta}: {exc}")
            continue
        ran += 1
        for at in shared_attrs:
            d = rel(getattr(ra, at), getattr(rb, at))
            if d > worst[at][0]:
                worst[at] = (d, delta)

    print(f"\nnumeric sweep: {ran} configs compared"
          + (f", {len(skipped)} skipped for fields one side lacks" if skipped else ""))
    agree = [a for a in shared_attrs if worst[a][0] == 0.0]
    differ = sorted((a for a in shared_attrs if worst[a][0] > 0.0),
                    key=lambda a: -worst[a][0])
    print(f"  {len(agree)} of {len(shared_attrs)} shared attributes agree exactly on every config")
    if differ:
        print(f"  {'attribute':34s}{'worst rel. diff':>16s}   worst config")
        for a in differ:
            d, cfg = worst[a]
            shown = "inf (type or NaN mismatch)" if math.isinf(d) else f"{100 * d:.4f}%"
            print(f"  {a:34s}{shown:>16s}   {cfg}")
    return differ


def surface_only(mod, label):
    print(f"\nOnly {label} is present. Its surface, which a replacement has to match:")
    print(f"  SdpaConfig fields ({len(config_fields(mod))}): {', '.join(sorted(config_fields(mod)))}")
    attrs, _ = result_attrs(mod)
    print(f"\n  result attributes ({len(attrs)}): {', '.join(sorted(attrs))}")


def main():
    args = sys.argv[1:]
    p_ours = Path(args[0]) if len(args) > 0 else DEFAULT_OURS
    p_pol = Path(args[1]) if len(args) > 1 else DEFAULT_POLARIS
    ours, why_ours = load(p_ours, "_rf_ours")
    pol, why_pol = load(p_pol, "_rf_polaris")

    print("modules")
    print(f"  ours     {p_ours}  [{why_ours}]")
    print(f"  polaris  {p_pol}  [{why_pol}]")

    print("\nzone_decomp.py contract (10 keywords, 4 result attributes)")
    ok = True
    for mod, label in ((ours, "ours"), (pol, "polaris")):
        if mod is not None:
            ok &= check_contract(mod, label)
        else:
            print(f"  {label:10s} skipped, module not loaded")

    if ours is not None and pol is not None:
        differ = compare(ours, pol, "ours", "polaris")
        print("\nverdict: "
              + ("the two agree on every shared attribute" if not differ
                 else f"{len(differ)} shared attribute(s) diverge; the reducer's columns depend on "
                      "which module is importable, so pin one or reconcile them"))
    elif ours is not None or pol is not None:
        surface_only(ours or pol, "ours" if ours else "polaris")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
