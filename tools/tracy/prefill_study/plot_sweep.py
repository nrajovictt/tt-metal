#!/usr/bin/env python3
"""Static PNG of the full sweep: kernel time per category (log), share of kernel time, FPU/SFPU utilization.

usage: python tools/tracy/prefill_study/plot_sweep.py LAYER.json OUT.png      (LAYER.json from layer_table.py)
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INK, INK2, INK3, RULE = "#15191e", "#48515b", "#737c87", "#e4e7eb"
MM, SDPA, OTHER = "#2a78d6", "#eb6834", "#c4c9cf"


def main():
    rows = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": RULE,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
        }
    )
    L = [r["layer"] for r in rows]
    x = list(range(len(rows)))
    labels = [f"{r['seq'] // 1024}k" if r["seq"] >= 1024 else str(r["seq"]) for r in rows]
    t = [l["kernel_us"] / 1e3 for l in L]
    mm = [l["matmul_us"] / 1e3 for l in L]
    sd = [l["sdpa_us"] / 1e3 for l in L]
    ot = [a - b - c for a, b, c in zip(t, mm, sd)]

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 13.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 0.9, 1.0], "hspace": 0.32}
    )
    fig.subplots_adjust(left=0.08, right=0.80, top=0.875, bottom=0.095)
    fig.suptitle(
        "Llama 3.1 8B prefill on Blackhole P150 (1 chip)",
        x=0.08,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
        y=0.975,
    )
    fig.text(
        0.08,
        0.94,
        "Embedding + one decoder layer, device kernel time from the profiler. Every prompt length measured at its real size.",
        fontsize=10.5,
        color=INK2,
    )

    def title(ax, a, b):
        ax.set_title(a, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22)
        ax.text(0, 1.02, b, transform=ax.transAxes, fontsize=9, color=INK3, va="bottom")

    def end(ax, y, text, dy=0):
        ax.annotate(
            text,
            xy=(x[-1], y),
            xytext=(10, dy),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            color=INK2,
            annotation_clip=False,
        )

    ax = axes[0]
    title(ax, "Kernel time", "milliseconds, log scale")
    for ys, c, n, lw in (
        (t, INK, "Total", 2.4),
        (mm, MM, "Matmul", 2),
        (sd, SDPA, "SDPA", 2),
        (ot, INK3, "Other ops", 1.5),
    ):
        ax.plot(x, ys, color=c, lw=lw, marker="o", ms=5.5, mec="white", mew=1.2, label=n, zorder=3)
        end(ax, ys[-1], f"{n} {ys[-1]:.0f} ms", {"Total": 7, "SDPA": -5}.get(n, 0))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("ms")
    ax.legend(loc="upper left", frameon=False, ncol=4, fontsize=9.5, handlelength=3)

    ax = axes[1]
    title(ax, "Share of kernel time", "% of the layer's device kernel time")
    mp, sp = [l["matmul_pct"] for l in L], [l["sdpa_pct"] for l in L]
    op = [100 - a - b for a, b in zip(mp, sp)]
    for vals, bottom, c, n in (
        (mp, [0] * len(x), MM, "Matmul"),
        (sp, mp, SDPA, "SDPA"),
        (op, [a + b for a, b in zip(mp, sp)], OTHER, "Other ops"),
    ):
        ax.bar(x, vals, 0.68, bottom=bottom, color=c, label=n, edgecolor="white", linewidth=1.5, zorder=3)
    for i in x:
        ax.text(
            i,
            mp[i] / 2,
            f"{mp[i]:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            zorder=4,
        )
        if sp[i] >= 6:
            ax.text(
                i,
                mp[i] + sp[i] / 2,
                f"{sp[i]:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=INK,
                fontweight="bold",
                zorder=4,
            )
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9.5)

    ax = axes[2]
    title(
        ax,
        "Math engine utilization",
        "FPU in matmul ops, SFPU in SDPA ops, weighted by op kernel time. Solid: median over active cores. Dashed: whole worker grid.",
    )
    ends = []
    for k, c, ls, n in (
        ("fpu_matmul_act", MM, "-", "FPU, active core"),
        ("fpu_matmul_full", MM, "--", "FPU, full grid"),
        ("sfpu_sdpa_act", SDPA, "-", "SFPU, active core"),
        ("sfpu_sdpa_full", SDPA, "--", "SFPU, full grid"),
    ):
        ys = [l[k] for l in L]
        ax.plot(
            x,
            ys,
            color=c,
            ls=ls,
            lw=2,
            marker="o",
            ms=5,
            mfc=c if ls == "-" else "white",
            mec=c if ls == "--" else "white",
            mew=1.3,
            zorder=3,
            label=n,
        )
        ends.append([ys[-1], f"{n} {ys[-1]:.1f}%"])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 5)
    for y, txt in ends:
        end(ax, y, txt)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=9, handlelength=3.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_xlim(-0.6, len(x) - 0.4)
    fig.text(
        0.08,
        0.025,
        "Scope: Embeddings, attention and MLP of one layer (final norm and LM head excluded). Source: tools/tracy/prefill_study/layer_table.py",
        fontsize=8.5,
        color=INK3,
        va="bottom",
    )
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
