#!/usr/bin/env python3
"""Join the T2.1 grid-1 zones-off walls measured on this tree to the published campaign walls.

Offline, no arguments, reads $DD and writes beside the captures, like the other reducers. The
published column is read out of the WALLS table of model/refit_r2_fit.py, which is the only place
in this checkout that still carries the campaign's measured numbers; the file is parsed, never
imported, because importing it needs the polaris model.

The wall of a run is the mean of invocations 1 and 2, invocation 0 being a warmup, matching the
convention the campaign reduced with.
"""

import ast
import csv
import os
import re
import subprocess
from pathlib import Path

SD = Path(__file__).resolve().parent
DD = Path(os.environ.get("DD", SD))
REFIT = SD / "model" / "refit_r2_fit.py"
OUT = DD / "rebaseline_grid1.csv"

LABEL = re.compile(r"^(non-causal|causal) q(\d+) k(\d+)")


def published_t21():
    """{(mode, q_chunk, k_chunk): cycles} for the T2.1 block of the WALLS table."""
    tree = ast.parse(REFIT.read_text())
    walls = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "WALLS" for t in node.targets)
    )
    out = {}
    for entry in walls.elts:
        label, _group, block, _set, _cfg, measured = entry.elts
        if block.value != "T2.1":
            continue
        m = LABEL.match(label.value)
        if not m:
            continue
        mode = "noncausal" if m.group(1) == "non-causal" else "causal"
        out[(mode, int(m.group(2)), int(m.group(3)))] = float(measured.value)
    return out


def measured_wall(tag):
    """Mean wall_dev_cycles over run_idx 1 and 2 of a <tag>_runs.csv, or None if absent."""
    path = DD / f"{tag}_runs.csv"
    if not path.exists():
        return None
    walls = [
        float(row["wall_dev_cycles"])
        for row in csv.DictReader(path.open())
        if int(row["run_idx"]) in (1, 2)
    ]
    return sum(walls) / len(walls) if walls else None


def card():
    for tag in sorted(DD.glob("t21_*_zoff.csv")):
        head = tag.open().readline()
        if head.startswith("PROVENANCE:"):
            return head[len("PROVENANCE:") :].split(";")[0].strip()
    return "unknown-card"


def main():
    published = published_t21()
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=SD
    ).stdout.strip()
    rows = []
    for (mode, q, k), ref in sorted(published.items()):
        tag = f"t21_{mode}_q{q}k{k}_zoff"
        got = measured_wall(tag)
        rows.append(
            {
                "config": f"{mode} q{q} k{k}",
                "published_p100a_cycles": round(ref, 1),
                "measured_cycles": round(got, 1) if got else "",
                "delta_pct": round(100 * (got / ref - 1), 2) if got else "",
                "tag": tag,
            }
        )

    with OUT.open("w", newline="") as fh:
        fh.write(
            f"# PROVENANCE: {card()}; tt-metal {sha}; zones off, mean of invocations 1 and 2; "
            f"published column from {REFIT.relative_to(SD.parent.parent)} WALLS block T2.1\n"
        )
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    width = max(len(r["config"]) for r in rows)
    print(f"{'config':<{width}}  {'published':>12}  {'measured':>12}  {'delta':>8}")
    for r in rows:
        got = f"{r['measured_cycles']:,.0f}" if r["measured_cycles"] != "" else "-"
        delta = f"{r['delta_pct']:+.2f}%" if r["delta_pct"] != "" else "-"
        print(f"{r['config']:<{width}}  {r['published_p100a_cycles']:>12,.0f}  {got:>12}  {delta:>8}")
    done = [r["delta_pct"] for r in rows if r["delta_pct"] != ""]
    if done:
        print(f"\n{len(done)}/{len(rows)} points measured, mean delta {sum(done) / len(done):+.2f}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
