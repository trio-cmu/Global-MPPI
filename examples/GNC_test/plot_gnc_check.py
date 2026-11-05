"""Kernel-surrogate GNC on two 1D curves: the demo's three equal-width valleys, and a
counter-example with a wide shallow valley next to a narrow deep one.

Both use 101 fixed samples and --stages bandwidths, geometric from 0.18 to 0.012 (3 stages
gives about the demo's 0.18 -> 0.065 -> 0.012). Each stage's
minimizer is found by continuation (local descent from the previous stage's minimizer),
as GNC does; stage 1 takes the global minimizer of the smoothest surrogate.

With --kernel wendland the weights are the C^2 Wendland kernel (1 - r)^4 (4 r + 1),
r = |u - u_i| / support, and the support radii run from 0.5 to 0.035 (about 2.8x the
Gaussian bandwidths, as in demo_black_curve_gnc.py).

    python plot_gnc_check.py [out.png] [--stages 5] [--kernel wendland]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("out", nargs="?", default=None)
parser.add_argument("--stages", type=int, default=3)
parser.add_argument("--kernel", choices=["gaussian", "wendland"], default="gaussian")
args = parser.parse_args()
suffix = "" if args.kernel == "gaussian" else "_wendland"
out = Path(args.out) if args.out else Path(__file__).with_name(f"gnc_check_{args.stages}stages{suffix}.png")
WENDLAND = args.kernel == "wendland"
H = "support" if WENDLAND else "h"
x = np.linspace(0, 1, 6001)
xs = np.linspace(0, 1, 101)
BANDWIDTHS = [round(h, 4) for h in np.geomspace(*((0.5, 0.035) if WENDLAND else (0.18, 0.012)), args.stages)]
COLORS = ["#53769A", "#6a4fd3", "#CF902B", "#e87ba4", "#148F84", "#eb6834", "#2a78d6"][: args.stages]
if args.stages == 3:
    COLORS = ["#53769A", "#CF902B", "#148F84"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def tri(c, w):
    return np.maximum(1 - np.abs(x - c) / w, 0)


CASES = [
    ("Demo curve: three valleys of the same width",
     gaussian_filter1d(1 - 0.78 * tri(0.266, 0.057) - 0.38 * tri(0.520, 0.058) - 1.02 * tri(0.776, 0.058),
                       5.5, mode="nearest")),
    ("Counter-example: wide shallow vs narrow deep",
     1 - 0.5 * tri(0.3, 0.20) - 1.0 * tri(0.75, 0.02)),
]


def nw(ys, h):
    if WENDLAND:
        r = np.minimum(np.abs(x[:, None] - xs[None]) / h, 1.0)
        w = (1 - r) ** 4 * (4 * r + 1)
    else:
        w = np.exp(-0.5 * (x[:, None] - xs[None]) ** 2 / h ** 2)
    return (w @ ys) / w.sum(1)


def descend(c, u):
    i = int(np.searchsorted(x, u))
    while 0 < i < len(x) - 1 and min(c[i - 1], c[i + 1]) < c[i]:
        i = i - 1 if c[i - 1] < c[i + 1] else i + 1
    return i


n_col = len(BANDWIDTHS) + 1
fig, axes = plt.subplots(2, n_col, figsize=(4.75 * n_col, 7.6), dpi=150, sharey="row")
fig.patch.set_facecolor(SURFACE)
for row, (title, y) in enumerate(CASES):
    ys = np.interp(xs, x, y)
    ax = axes[row, 0]
    ax.plot(x, y, color=INK, lw=2.2, label="true cost J(u)")
    ax.scatter(xs, ys, s=10, color="#3E78B2", zorder=3, label="101 fixed samples")
    g = int(np.argmin(y))
    ax.scatter(x[g], y[g], marker="*", s=180, color="#d43d51", zorder=5, label=f"true minimum u = {x[g]:.3f}")
    ax.set_title(chr(ord("A") + row * n_col) + "   " + title, loc="left", fontsize=10, color=INK, weight="bold")
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    u = None
    for k, (h, col) in enumerate(zip(BANDWIDTHS, COLORS), start=1):
        c = nw(ys, h)
        i = int(np.argmin(c)) if u is None else descend(c, u)
        u = x[i]
        ax = axes[row, k]
        ax.plot(x, y, color=INK, lw=1.4, alpha=0.45, label="true cost")
        ax.plot(x, c, color=col, lw=2.4, label=f"smoothed surrogate, {H} = {h:g}")
        ax.scatter(xs, ys, s=6, color="grey", alpha=0.3)
        ax.axvline(u, color=col, lw=1, ls=(0, (3, 3)))
        ax.scatter(u, c[i], marker="*", s=170, color=col, edgecolor="white", zorder=5,
                   label=f"GNC point u = {u:.3f}, true cost {np.interp(u, x, y):.2f}")
        ax.scatter(x[g], y[g], marker="*", s=110, color="#d43d51", alpha=0.6, zorder=4)
        ok = abs(u - x[g]) < 0.02
        letter = chr(ord("A") + row * n_col + k)
        ax.set_title(f"{letter}   stage {k}: {H} = {h:g}   " + ("✓ true valley" if ok else "✗ wrong valley"),
                     loc="left", fontsize=10, color="#148F84" if ok else "#d43d51", weight="bold")
        ax.legend(loc="lower left", fontsize=7.5, frameon=False)
for ax in axes.flat:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, 1)
for ax in axes[1]:
    ax.set_xlabel("control u", color=INK_2)
axes[0, 0].set_ylabel("cost", color=INK_2)
axes[1, 0].set_ylabel("cost", color=INK_2)
fig.suptitle(f"{'Wendland' if WENDLAND else 'Gaussian'} kernel-surrogate GNC, {len(BANDWIDTHS)} stages: same 101 "
             f"samples at every stage, {'support radius' if WENDLAND else 'bandwidth'} "
             + " → ".join(f"{h:g}" for h in BANDWIDTHS),
             x=0.04, ha="left", fontsize=13, color=INK, weight="bold")
fig.text(0.04, 0.925, "Top: the demo's curve; smoothing finds the deepest valley because all valleys have the same "
         "width. Bottom: the smoothed average scores a valley by its area, so the wide shallow valley wins and "
         "the continuation stays there. Red star = true minimum.", fontsize=9, color=INK_2)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(out, facecolor=fig.get_facecolor())
fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())
print(out)
