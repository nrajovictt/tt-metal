#!/usr/bin/env python3
"""Hypothetical SDPA schedule with separate FPU and SFPU issue threads (Quasar), from Blackhole-measured stage costs.

Tasks per K-chunk step j (wall cycles from sdpa_unit_split.py, 256-token chunks):
  FPU thread : QK_j, RED_j (row-max reduce = MAX minus its SFPU part), PV_j, CF_j (CORR minus its SFPU part)
  SFPU thread: MS_j (eltwise max with m_{j-1}), CE_j (exp(m_{j-1} - m_j)), EX_j (SUBEXP incl. its FPU subtract)
True dependencies: RED_j<-QK_j; MS_j<-RED_j,MS_{j-1}; CE_j,EX_j<-MS_j; PV_j<-EX_j; CF_j<-PV_j,CE_j,CF_{j-1}.
Resource dependency: QK_j needs a free scores buffer; with 2 buffers QK_j <- PV_{j-2}.
Each thread runs its own queue in order: FPU = QK_{j+1}, RED_{j+1}, PV_j, CF_j ...; SFPU = MS_j, CE_j, EX_j ...
Blackhole baseline = the same stages back to back on one math thread.

usage: python tools/tracy/prefill_study/quasar_model.py OUT.json SPLIT.json LAYER.json
"""

import json
import sys


def simulate(cost, n=12):
    F = [("QK", 0), ("RED", 0)] + [t for j in range(n) for t in (("QK", j + 1), ("RED", j + 1), ("PV", j), ("CF", j))]
    S = [t for j in range(n + 1) for t in (("MS", j), ("CE", j), ("EX", j))]
    deps = {
        "QK": lambda j: [("PV", j - 2)],
        "RED": lambda j: [("QK", j)],
        "MS": lambda j: [("RED", j), ("MS", j - 1)],
        "CE": lambda j: [("MS", j)],
        "EX": lambda j: [("MS", j)],
        "PV": lambda j: [("EX", j)],
        "CF": lambda j: [("PV", j), ("CE", j), ("CF", j - 1)],
    }
    end, sched, free, idx = {}, [], {"F": 0, "S": 0}, {"F": 0, "S": 0}
    q = {"F": F, "S": S}
    while idx["F"] < len(F) or idx["S"] < len(S):
        moved = False
        for t in ("F", "S"):
            if idx[t] >= len(q[t]):
                continue
            name, j = q[t][idx[t]]
            dd = [d for d in deps[name](j) if d[1] >= 0]
            if all(d in end for d in dd):
                st = max([free[t]] + [end[d] for d in dd])
                end[(name, j)] = free[t] = st + cost[name]
                sched.append((t, name, j, st, st + cost[name]))
                idx[t] += 1
                moved = True
        assert moved, "deadlock"
    ex = [s for s in sched if s[1] == "EX"]
    return ex[-2][3] - ex[-3][3], sched


def main():
    out, split_json, layer_json = sys.argv[1:4]
    sp = json.load(open(split_json))["split_256"]
    cost = {
        "QK": sp["QK"]["cycles"],
        "RED": sp["MAX"]["cycles"] - sp["MAX"]["sfpu"],
        "PV": sp["PV"]["cycles"],
        "CF": sp["CORR"]["cycles"] - sp["CORR"]["sfpu"],
        "MS": sp["MAX"]["sfpu"],
        "CE": sp["CORR"]["sfpu"],
        "EX": sp["SUBEXP"]["cycles"],
    }
    period, sched = simulate(cost)
    bh = sum(v["cycles"] for v in sp.values())
    fpu_work = cost["QK"] + cost["RED"] + cost["PV"] + cost["CF"]
    r = period / bh
    print(
        f"Blackhole step {bh} cycles, Quasar model {period} cycles -> {bh / period:.2f}x; "
        f"SFPU thread busy {(cost['MS'] + cost['CE'] + cost['EX']) / period * 100:.0f}%, FPU thread {fpu_work / period * 100:.0f}%"
    )
    rows = []
    for e in json.load(open(layer_json)):
        if e["seq"] < 2048:
            continue  # 64-token chunks: stage timings not reliable enough to model
        L = e["layer"]
        sq = L["sdpa_us"] * r
        lq = L["kernel_us"] - L["sdpa_us"] + sq
        rows.append(
            {
                "seq": e["seq"],
                "sdpa_us": L["sdpa_us"],
                "sdpa_model_us": round(sq, 1),
                "layer_us": L["kernel_us"],
                "layer_model_us": round(lq, 1),
                "layer_change_pct": round((lq / L["kernel_us"] - 1) * 100, 1),
            }
        )
        print(
            f"  {e['seq']:>6}: SDPA {L['sdpa_us']:10.1f} -> {sq:10.1f} us   layer {L['kernel_us']:10.1f} -> {lq:10.1f} us ({rows[-1]['layer_change_pct']}%)"
        )
    json.dump(
        {
            "cost": cost,
            "blackhole_step": bh,
            "model_step": period,
            "ratio": r,
            "rows": rows,
            "schedule": [list(s) for s in sched],
        },
        open(out, "w"),
        indent=1,
    )
    print(f"[json] {out}")


if __name__ == "__main__":
    main()
