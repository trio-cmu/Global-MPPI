"""How LSE smoothing reshapes one slice of the 64-D cube rotation cost: the best interior sample's random
plane --plane (default 6) from cube_cost_surface.py (same --seed, so the same center and the same d1, d2),
for the plain cost and every --sigmas.

    J_lse(u) = -lambda log mean_l exp(-J(clip(u + sigma eps_l)) / lambda)

with the same --lse-samples perturbations eps_l at every grid point (common random numbers) and sigma in
joint-target units, as GlobalMPPI.lse_smoothing_rollout. All panels share one color scale.

Outputs (next to this file):
    cube_lse_plane<k>.png                     3-D surface + top-down map per sigma, and the slice b = 0
    cube_lse_plane<k>_<plain|lse<sigma>>.json  grids for make_surface_viewer.py
    cube_lse_plane<k>.html                    interactive viewer, toggle between the sigmas

Example:
    python cube_lse_plane.py                       # plane 6, sigmas 0.001 ... 0.3
    python cube_lse_plane.py --plane 3 --sigmas 0.01 0.1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.colors import LightSource, LogNorm
from mujoco import mjx

from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.cube import CubeRotation

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="LSE smoothing of one random plane of the cube rotation cost")
parser.add_argument("--plane", type=int, default=6, help="random plane through the best interior sample (1-based)")
parser.add_argument("--sigmas", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.3])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n", type=int, default=1000)
parser.add_argument("--hold-planes", type=int, default=3, help="planes drawn for the hold-still row first (as cube_cost_surface.py)")
parser.add_argument("--grid", type=int, default=101)
parser.add_argument("--wide", type=float, default=1.0)
parser.add_argument("--local", type=float, default=0.5)
parser.add_argument("--interior", type=float, default=0.6)
parser.add_argument("--chunk", type=int, default=2048)
parser.add_argument("--lse-samples", type=int, default=100)
parser.add_argument("--lse-lambda", type=float, default=0.1)
args = parser.parse_args()
stem = f"cube_lse_plane{args.plane}"

# ---------------------------------------------------------------- the same setup as cube_cost_surface.py
task = CubeRotation()
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mj_data.qvel[:] = 0.0
mujoco.mj_forward(mj_model, mj_data)
state = mjx.put_data(mj_model, mj_data)
state = state.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)
ctrl = GlobalMPPI(task, num_samples=128, noise_level=0.1, temperature=0.1, spline_type="zero",
                  plan_horizon=0.25, num_knots=4)
nu, num_knots = task.model.nu, ctrl.num_knots
dim = num_knots * nu
tk = jnp.linspace(0.0, ctrl.plan_horizon, num_knots) + state.time
u_min = np.tile(np.asarray(task.u_min, np.float64), num_knots)
u_max = np.tile(np.asarray(task.u_max, np.float64), num_knots)
mid, half_box = (u_min + u_max) / 2, (u_max - u_min) / 2
to_u = lambda z: mid + half_box * z


@jax.jit
def _costs(knots):
    return ctrl.rollout(state, tk, knots).costs.sum(axis=-1)


def rollout(points):
    points = np.atleast_2d(points)
    out = []
    for k in range(0, len(points), args.chunk):
        P = np.pad(points[k: k + args.chunk], ((0, max(0, k + args.chunk - len(points))), (0, 0)))
        out.append(np.asarray(_costs(jnp.asarray(P.reshape(-1, num_knots, nu), jnp.float32)))[: len(points) - k])
    return np.concatenate(out).astype(np.float64)


eps = np.random.default_rng(args.seed + 1).standard_normal((args.lse_samples, dim))


def cost(points, sigma):
    if sigma <= 0:
        return rollout(points)
    points, L, lam = np.atleast_2d(points), args.lse_samples, args.lse_lambda
    step = max(1, args.chunk // L)
    out = []
    for k in range(0, len(points), step):
        Q = np.clip(points[k: k + step, None] + sigma * eps[None], u_min, u_max)
        J = rollout(Q.reshape(-1, dim)).reshape(-1, L)
        m = J.min(axis=1)
        out.append(m - lam * np.log(np.mean(np.exp(-(J - m[:, None]) / lam), axis=1)))
    return np.concatenate(out)


# replay cube_cost_surface.py's random draws to get its center and plane
rng = np.random.default_rng(args.seed)
rng.uniform(-1, 1, (args.n, dim))  # the full-box samples (not needed here)
Zi = rng.uniform(-args.interior, args.interior, (args.n, dim))
center = Zi[np.argmin(rollout(to_u(Zi)))]
for _ in range(args.hold_planes):
    rng.standard_normal((dim, 2))  # the hold-still planes
for _ in range(args.plane):
    d1, d2 = np.linalg.qr(rng.standard_normal((dim, 2)))[0].T
a = np.linspace(-1, 1, args.grid)
A, B = np.meshgrid(a, a)
P = center + args.local * (A.reshape(-1, 1) * d1 + B.reshape(-1, 1) * d2)
assert np.all(np.abs(P) <= 1), "plane leaves the box"
U = to_u(P)

title = f"best of {args.n} random samples, |z| ≤ {args.interior:g}"
results, jsons = [], []
for sigma in [0.0] + list(args.sigmas):
    J = cost(U, sigma).reshape(A.shape)
    c0 = float(cost(to_u(center), sigma)[0])
    label = (f"LSE smoothed, σ = {sigma:g}, λ = {args.lse_lambda:g}, {args.lse_samples} samples"
             if sigma > 0 else "plain cost")
    results.append((sigma, label, J, c0))
    path = here / f"{stem}_{f'lse{sigma:g}' if sigma > 0 else 'plain'}.json"
    path.write_text(json.dumps(dict(
        label=label, median=float(np.median(J)), hold_cost=float("nan"), grid=args.grid, dim=dim, seed=args.seed,
        planes=[dict(center=title, plane=args.plane, half=args.local, center_cost=c0,
                     J=[[round(float(v), 4) for v in row] for row in J])])).replace("NaN", "null"))
    jsons.append(str(path))
    print(f"{label}: center {c0:.4f}, min {J.min():.4f}, max {J.max():.4f}, "
          f"roughness {np.abs(np.diff(J, axis=1)).mean():.2e}", flush=True)

# ---------------------------------------------------------------- figure: one panel per sigma + the slice b = 0
zmin = min(r[2].min() for r in results)
zmax = max(r[2].max() for r in results)
norm = LogNorm(zmin, zmax)
zfloor = 0.97 * zmin
cmap = plt.get_cmap("coolwarm")
ncol = 3
nrow = -(-len(results) // ncol)
fig = plt.figure(figsize=(6 * ncol, 10 * nrow + 5), layout="constrained")
gs = fig.add_gridspec(2 * nrow + 1, ncol, height_ratios=[1.15, 1] * nrow + [0.9])
h = args.local
for i, (sigma, label, J, c0) in enumerate(results):
    r, col = divmod(i, ncol)
    ax = fig.add_subplot(gs[2 * r, col], projection="3d")
    ax.plot_surface(h * A, h * B, J, facecolors=cmap(norm(J)), rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=True, lightsource=LightSource(azdeg=315, altdeg=50))
    ax.plot([0, 0], [0, 0], [zfloor, c0], color="k", lw=2)
    ax.set_zlim(zfloor, zmax)
    ax.set_xlabel("a (d1)")
    ax.set_ylabel("b (d2)")
    ax.set_zlabel("cost")
    ax.view_init(elev=35, azim=-60)
    ax.set_title(f"{'plain cost' if sigma == 0 else f'LSE σ = {sigma:g}'}   (center cost {c0:.3f})", fontsize=12)
    ax = fig.add_subplot(gs[2 * r + 1, col])
    im = ax.pcolormesh(h * A, h * B, J, norm=norm, cmap=cmap, shading="auto")
    ax.contour(h * A, h * B, J, levels=np.geomspace(zmin, zmax, 14), colors="k", linewidths=0.35, alpha=0.5)
    ax.plot(0, 0, "*", ms=16, mfc="gold", mec="k", mew=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("a (d1)")
    ax.set_ylabel("b (d2)")
    ax.set_title(f"from above: min {J.min():.3f}, max {J.max():.3f}", fontsize=10)
fig.colorbar(im, ax=fig.axes, shrink=min(0.4, 1.6 / nrow), label="cost (log)", pad=0.01)

ax = fig.add_subplot(gs[2 * nrow, :])  # the slice b = 0 through the center, all sigmas
colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(results) - 1))
for i, (sigma, label, J, c0) in enumerate(results):
    row = J[args.grid // 2]
    if sigma == 0:
        ax.plot(h * a, row, color="k", lw=1.4, label="plain cost")
    else:
        ax.plot(h * a, row, color=colors[i - 1], lw=2, label=f"σ = {sigma:g}")
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("a (d1), slice b = 0 through the center")
ax.set_ylabel("cost")
ax.set_title("the same slice for every σ: smoothing removes the cliffs and the contact noise, and lifts the minimum",
             fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(ncol=5, fontsize=10, loc="upper left")
fig.suptitle(f"In-hand cube rotation: LSE smoothing of one slice of the 64-D planning cost\n"
             f"{title}, random plane {args.plane}, |a|, |b| ≤ {h:g};  λ = {args.lse_lambda:g}, "
             f"{args.lse_samples} perturbations per point, one shared color scale", fontsize=14)
out = here / f"{stem}.png"
fig.savefig(out, dpi=120)
print("RESULT saved", out)

subprocess.run([sys.executable, str(here / "make_surface_viewer.py"), "--json", *jsons,
                "--out", str(here / f"{stem}.html")], check=True)
