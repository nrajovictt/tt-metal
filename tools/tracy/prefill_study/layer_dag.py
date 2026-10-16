#!/usr/bin/env python3
"""Longest dependency chain of one decoder layer (prefill), from device kernel times.

The ops run one at a time on the device; this asks how long the layer would take if only true data dependencies
mattered. Edges come from the model code (attention.py forward_prefill, mlp.py forward, decoder.py forward); the
profiler records no tensor IDs. Notably SDPA reads the type-cast Q/K/V directly, not the KV cache, so the two
PagedFillCache ops are off the chain.

usage: python tools/tracy/prefill_study/layer_dag.py generated/profiler/prefill_4k_1dev [FIRST_ROW=1113]
FIRST_ROW = row of the layer's Embeddings op in the ops csv (1113 for the default 2k..32k runs; 658 for the
TT_PREFILL_MIN_PADDING runs). The 23 rows from there are expected in this program order:
"""

import glob
import sys

import pandas as pd

ORDER = [
    "Emb",
    "SliceC",
    "SliceS",
    "LN1",
    "QKV",
    "Heads",
    "RotQ",
    "RotK",
    "TcK",
    "TcV",
    "FillK",
    "FillV",
    "TcQ",
    "SDPA",
    "Concat",
    "Wo",
    "Add1",
    "LN2",
    "W1",
    "W3",
    "Mul",
    "W2",
    "Add2",
]
CODES = [
    "Embeddings",
    "Slice",
    "Slice",
    "LayerNorm",
    "MinimalMatmul",
    "NlpCreateHeads",
    "RotaryEmbeddingLlama",
    "RotaryEmbeddingLlama",
    "Typecast",
    "Typecast",
    "PagedFillCache",
    "PagedFillCache",
    "Typecast",
    "SDPA",
    "NLPConcatHeads",
    "Matmul",
    "BinaryNg",
    "LayerNorm",
    "Matmul",
    "Matmul",
    "BinaryNg",
    "MinimalMatmul",
    "BinaryNg",
]
EDGES = [
    ("Emb", "LN1"),
    ("LN1", "QKV"),
    ("QKV", "Heads"),
    ("Heads", "RotQ"),
    ("Heads", "RotK"),
    ("Heads", "TcV"),
    ("SliceC", "RotQ"),
    ("SliceS", "RotQ"),
    ("SliceC", "RotK"),
    ("SliceS", "RotK"),
    ("RotQ", "TcQ"),
    ("RotK", "TcK"),
    ("TcK", "FillK"),
    ("TcV", "FillV"),
    ("TcQ", "SDPA"),
    ("TcK", "SDPA"),
    ("TcV", "SDPA"),
    ("SDPA", "Concat"),
    ("Concat", "Wo"),
    ("Wo", "Add1"),
    ("Emb", "Add1"),
    ("Add1", "LN2"),
    ("LN2", "W1"),
    ("LN2", "W3"),
    ("W1", "Mul"),
    ("W3", "Mul"),
    ("Mul", "W2"),
    ("W2", "Add2"),
    ("Add1", "Add2"),
]


def main():
    run = sys.argv[1]
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 1113
    rep = sorted(glob.glob(f"{run}/reports/*/"))[-1]
    ops = pd.read_csv(glob.glob(rep + "ops_perf_results_*.csv")[0], low_memory=False)
    rows = ops.iloc[first : first + len(ORDER)]
    codes = rows["OP CODE"].str.replace(r"(DeviceOperation|Operation)$", "", regex=True).tolist()
    if any(not c.startswith(e) and not c.endswith(e) for c, e in zip(codes, CODES)):
        sys.exit(
            f"[error] rows {first}.. do not match the expected op order:\n{codes}\n(at 3072 attention adds a Pad and a Slice; adjust ORDER)"
        )
    us = dict(zip(ORDER, rows["DEVICE KERNEL DURATION [ns]"].astype(float) / 1e3))
    fin, prev = {}, {}
    for n in ORDER:
        p = [a for a, b in EDGES if b == n]
        fin[n] = max([fin[x] for x in p], default=0) + us[n]
        prev[n] = max(p, key=lambda x: fin[x]) if p else None
    path = [max(fin, key=fin.get)]
    while prev[path[-1]]:
        path.append(prev[path[-1]])
    path.reverse()
    total = sum(us.values())
    print(
        f"layer (sum of op kernel times) {total:.1f} us; longest dependency chain {fin[path[-1]]:.1f} us ({fin[path[-1]] / total * 100:.1f}%)"
    )
    print("chain: " + " -> ".join(path))
    print("off the chain: " + ", ".join(f"{n} {us[n]:.1f}" for n in ORDER if n not in path))
    print("op times (us): " + ", ".join(f"{n} {us[n]:.1f}" for n in ORDER))


if __name__ == "__main__":
    main()
