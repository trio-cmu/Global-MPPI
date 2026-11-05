"""One-row comparison of LSE smoothing on one plane, from the grids saved by cube_lse_plane.py (no rollouts):
top row 3-D surfaces, bottom row the same planes seen from above, one column per cost, one shared color scale.

    python plot_lse_plane.py                                  # plane 6: sigma 0.05 ... 0.001, then plain
    python plot_lse_plane.py --plane 6 --sigmas 0.01 0.1 0.3 --out my.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LogNorm

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="LSE smoothing of one cube cost plane, one figure")
parser.add_argument("--plane", type=int, default=6)
parser.add_argument("--sigmas", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.025, 0.05],
                    help="LSE sigmas, shown from the largest to the smallest, then the plain cost")
parser.add_argument("--out", default=None, help="default cube_lse_plane<k>_sigma<lo>-<hi>.png")
args = parser.parse_args()
stem = f"cube_lse_plane{args.plane}"
out = Path(args.out) if args.out else here / f"{stem}_sigma{min(args.sigmas):g}-{max(args.sigmas):g}.png"

cases = []  # (title, grid J, center cost, half-width)
for tag, title in [(f"lse{s:g}", f"LSE σ = {s:g}") for s in sorted(args.sigmas, reverse=True)] + [("plain", "plain cost")]:
    d = json.loads((here / f"{stem}_{tag}.json").read_text())
    p = d["planes"][0]
    cases.append((title, np.array(p["J"], float), p["center_cost"], p["half"]))
lam = json.loads((here / f"{stem}_lse{args.sigmas[0]:g}.json").read_text())["label"]

zmin = min(J.min() for _, J, _, _ in cases)
zmax = max(J.max() for _, J, _, _ in cases)
norm = LogNorm(zmin, zmax)
zfloor = 0.97 * zmin
cmap = plt.get_cmap("coolwarm")  # ParaView's cool to warm, as Li et al. (2018) and SAM, Figure 1
n = len(cases)
fig = plt.figure(figsize=(4.2 * n, 8.6), layout="constrained")
gs = fig.add_gridspec(2, n, height_ratios=[1.15, 1])
for i, (title, J, c0, h) in enumerate(cases):
    g = np.linspace(-h, h, J.shape[0])
    A, B = np.meshgrid(g, g)
    ax = fig.add_subplot(gs[0, i], projection="3d")
    ax.plot_surface(A, B, J, facecolors=cmap(norm(J)), rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=True, lightsource=LightSource(azdeg=315, altdeg=50))
    ax.plot([0, 0], [0, 0], [zfloor, c0], color="k", lw=2)  # stick at the center
    ax.set_zlim(zfloor, zmax)
    ax.set_xlabel("a (d1)", labelpad=-2)
    ax.set_ylabel("b (d2)", labelpad=-2)
    ax.set_zlabel("cost", labelpad=0)
    ax.tick_params(labelsize=8, pad=0)
    ax.view_init(elev=35, azim=-15)  # -60 + 45
    ax.set_title(f"{title}\ncenter cost {c0:.3f}", fontsize=13, weight="bold" if title == "plain cost" else "normal")
    ax = fig.add_subplot(gs[1, i])
    im = ax.pcolormesh(A, B, J, norm=norm, cmap=cmap, shading="auto")
    ax.contour(A, B, J, levels=np.geomspace(zmin, zmax, 14), colors="k", linewidths=0.35, alpha=0.5)
    ax.plot(0, 0, "*", ms=15, mfc="gold", mec="k", mew=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("a (d1)")
    if i == 0:
        ax.set_ylabel("b (d2)")
    ax.tick_params(labelsize=9)
    ax.set_title(f"from above: min {J.min():.3f}, max {J.max():.3f}", fontsize=10.5)
fig.colorbar(im, ax=fig.axes, shrink=0.55, label="cost (log)", pad=0.03)
lam_txt = lam.split("σ = ")[1].split(", ", 1)[1]  # "λ = 0.1, 100 samples"
fig.suptitle(f"In-hand cube rotation: LSE smoothing of one slice of the 64-D planning cost "
             f"(best interior sample, random plane {args.plane}; {lam_txt}; one shared color scale)", fontsize=14, y=1.04)
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("RESULT saved", out)
