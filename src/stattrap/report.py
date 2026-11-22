"""Paper figures (matplotlib, static PNG). Conventions: one hue per message, colour-blind-safe categorical
slots (blue/orange/aqua validated all-pairs), thin marks, recessive grid, direct labels, a table (CSV) exists for
every figure. Models are shown as small multiples, never as more than three colours."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def fig_trap_heatmap(fam: pd.DataFrame, path, style="direct"):
    d = fam[fam["style"] == style]
    if d.empty:
        return
    t = d.pivot(index="model", columns="family", values="trap_rate")
    fig, ax = plt.subplots(figsize=(1.0 * t.shape[1] + 2.4, 0.55 * t.shape[0] + 1.6))
    im = ax.imshow(t.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto"); ax.grid(False)
    ax.set_xticks(range(t.shape[1]), t.columns, rotation=40, ha="right"); ax.set_yticks(range(t.shape[0]), t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iloc[i, j]; ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v > 0.55 else INK, fontsize=8)
    ax.set_title(f"Trap rate by model and family ({style} prompting); chance = 25%", loc="left", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03, label="share choosing the trap")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_cot_gain(cot: pd.DataFrame, path):
    if cot.empty:
        return
    c = cot.sort_values("cot_gain"); y = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(c) + 1.4))
    ax.hlines(y, c.gain_lo, c.gain_hi, color=BLUE, lw=1.6); ax.plot(c.cot_gain, y, "o", color=BLUE, ms=6)
    ax.axvline(0, color=MUTED, lw=1); ax.set_yticks(y, c.model); ax.grid(axis="y", visible=False)
    for yi, (g, p) in enumerate(zip(c.cot_gain, c.p_holm)):
        ax.text(c.gain_hi.max() + 0.01, yi, f"{g:+.1%}  (Holm p={p:.3f})", va="center", fontsize=8, color=INK)
    ax.set_xlabel("accuracy gain from step-by-step prompting (95% CI, cluster bootstrap)")
    ax.set_xlim(min(c.gain_lo.min() - 0.02, -0.05), c.gain_hi.max() + 0.22)
    ax.set_title("Does chain-of-thought help?", loc="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_position_bias(pos: pd.DataFrame, path, style="direct"):
    d = pos[pos["style"] == style].reset_index(drop=True)
    if d.empty:
        return
    n = len(d); cols = min(n, 4); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.1 * rows), sharey=True, squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), d.iterrows()):
        v = [r[f"freq_chosen_{l}"] for l in "ABCD"]
        ax.bar(list("ABCD"), v, color=BLUE, width=0.55); ax.axhline(0.25, color=MUTED, lw=1, ls="--")
        ax.set_title(r["model"], fontsize=8, loc="left"); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Chosen letter frequency ({style}); dashed = 25% expected", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_scaling(sc: pd.DataFrame, path):
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    d = sc.dropna(subset=["params_b"])
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    fam_colors = dict(zip(sorted(d.family.unique())[:3], [BLUE, ORANGE, AQUA]))
    sizes = sorted(d.params_b.unique())
    for ax, col, title in zip(axes, ("acc", "trap_rate"), ("Accuracy", "Trap rate")):
        for (fam, style), g in d.groupby(["family", "style"]):
            g = g.sort_values("params_b")
            ax.plot(g.params_b, g[col], marker="o" if style == "direct" else "s", color=fam_colors.get(fam, MUTED),
                    lw=1.4, ms=6, ls="-" if style == "direct" else "--", label=f"{fam} ({style})")
        ax.axhline(0.25, color=MUTED, lw=0.8, ls=":"); ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(sizes)); ax.xaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in sizes]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("parameters (billions, log scale)"); ax.set_title(title + "  (dotted = 25% chance)", loc="left", fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
