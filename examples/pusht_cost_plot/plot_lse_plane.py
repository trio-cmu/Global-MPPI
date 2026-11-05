"""One-row comparison of LSE smoothing on one PushT cost plane, from the grids saved by pusht_cost_surface.py
(pusht_cost_surface.json and pusht_cost_surface_lse<sigma>.json, no rollouts): the best interior sample's random
plane --plane. Columns from the largest sigma to the smallest, then the plain cost; top row 3-D surfaces,
bottom row the same planes seen from above, one shared color scale. Same figure as
examples/In-hand_manipulation_cost_plot/plot_lse_plane.py for the cube.

    python plot_lse_plane.py                                  # plane 2: sigma 0.05 ... 0.001, then plain
    python plot_lse_plane.py --plane 2 --sigmas 0.01 0.1 0.25 --out my.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LogNorm

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="LSE smoothing of one PushT cost plane, one figure")
parser.add_argument("--plane", type=int, default=2, help="random plane through the best interior sample (1-3)")
parser.add_argument("--sigmas", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.025, 0.05],
                    help="LSE sigmas, shown from the largest to the smallest, then the plain cost")
parser.add_argument("--out", default=None, help="default pusht_lse_plane<k>_sigma<lo>-<hi>.png")
args = parser.parse_args()
out = Path(args.out) if args.out else here / f"pusht_lse_plane{args.plane}_sigma{min(args.sigmas):g}-{max(args.sigmas):g}.png"


def best_interior_plane(path):
    d = json.loads(path.read_text())
    p = next(p for p in d["planes"] if p["center"].startswith("best") and p["plane"] == args.plane)
    J = np.array([[np.nan if v is None else v for v in row] for row in p["J"]], float)
    return d.get("label", "plain cost"), J, p["center_cost"], p["half"]


cases, labels = [], []  # (title, grid J, center cost, half-width)
for s in sorted(args.sigmas, reverse=True):
    label, J, c0, h = best_interior_plane(here / f"pusht_cost_surface_lse{s:g}.json")
    cases.append((f"LSE σ = {s:g}", J, c0, h))
    labels.append(label)
_, J, c0, h = best_interior_plane(here / "pusht_cost_surface.json")
cases.append(("plain cost", J, c0, h))
settings = {lab.split("σ = ")[1].split(", ", 1)[1] for lab in labels}  # e.g. {"λ = 0.1, 100 samples"}
assert len(settings) == 1, f"the sigmas use different LSE settings: {settings}"

zmin = min(np.nanmin(J) for _, J, _, _ in cases)
zmax = max(np.nanmax(J) for _, J, _, _ in cases)
norm = LogNorm(zmin, zmax)
zfloor = 0.9 * zmin
cmap = plt.get_cmap("coolwarm")  # ParaView's cool to warm, as Li et al. (2018) and SAM, Figure 1
n = len(cases)
fig = plt.figure(figsize=(4.2 * n, 8.6), layout="constrained")
gs = fig.add_gridspec(2, n, height_ratios=[1.15, 1])
for i, (title, J, c0, h) in enumerate(cases):
    g = np.linspace(-h, h, J.shape[0])
    A, B = np.meshgrid(g, g)
    ax = fig.add_subplot(gs[0, i], projection="3d")
    ax.plot_surface(A, B, np.ma.masked_invalid(J), facecolors=cmap(norm(np.nan_to_num(J, nan=zmax))), rstride=1,
                    cstride=1, linewidth=0, antialiased=False, shade=True,
                    lightsource=LightSource(azdeg=315, altdeg=50))
    ax.plot([0, 0], [0, 0], [zfloor, c0], color="k", lw=2)  # stick at the center
    ax.set_zlim(zfloor, zmax)
    ax.set_xlabel("a (d1)", labelpad=-2)
    ax.set_ylabel("b (d2)", labelpad=-2)
    ax.set_zlabel("cost", labelpad=0)
    ax.tick_params(labelsize=8, pad=0)
    ax.view_init(elev=5, azim=95)  # pitch 35 - 30, yaw -15 + 90 + 20
    ax.set_title(f"{title}\ncenter cost {c0:.2f}", fontsize=13, weight="bold" if title == "plain cost" else "normal")
    ax = fig.add_subplot(gs[1, i])
    im = ax.pcolormesh(A, B, J, norm=norm, cmap=cmap, shading="auto")
    ax.contour(A, B, J, levels=np.geomspace(zmin, zmax, 14), colors="k", linewidths=0.35, alpha=0.5)
    ax.plot(0, 0, "*", ms=15, mfc="gold", mec="k", mew=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("a (d1)")
    if i == 0:
        ax.set_ylabel("b (d2)")
    ax.tick_params(labelsize=9)
    ax.set_title(f"from above: min {np.nanmin(J):.2f}, max {np.nanmax(J):.2f}", fontsize=10.5)
fig.colorbar(im, ax=fig.axes, shrink=0.55, label="cost (log)", pad=0.03)
fig.suptitle(f"PushT: LSE smoothing of one slice of the 12-D planning cost "
             f"(best interior sample, random plane {args.plane}; {settings.pop()}; one shared color scale)",
             fontsize=14, y=1.04)
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("RESULT saved", out)
