"""3-D surfaces of the 12-D PushT planning cost on random planes, after Li et al. (2018),
"Visualizing the loss landscape of neural nets" (the method behind Figure 1 of SAM).

The cost J(x) is one 1 s rollout of the 6-knot x 2-control spline x in [-1, 1]^12 from one
start state. No planner is involved: each panel is J(c + a d1 + b d2) on a grid of (a, b),
with d1, d2 two random orthonormal directions and c
  top     the box center x = 0 (the pusher holds still), |a|, |b| <= --wide
  bottom  the cheapest of --n uniform random samples in the inner box [-interior, interior]^12,
          so that its planes stay mostly inside [-1, 1]^12, |a|, |b| <= --local
Each center gets a row of 3-D surfaces and below it the same planes seen from above (log
color in ParaView's cool-to-warm map, as Li et al. and SAM, Figure 1; center starred). Grid points outside the box [-1, 1]^12 are left out
(white / holes), not clipped: a clipped point lies on a box face, not on the plane.

--lse-sigma > 0 draws the LSE-smoothed cost of GlobalMPPI instead,
    J_lse(x) = -lambda log mean_l exp(-J(clip(x + sigma eps_l)) / lambda),
with the same --lse-samples perturbations eps_l at every grid point (common random numbers,
so the surface has no Monte Carlo noise from point to point). The centers and planes are
the same as without smoothing (same --seed); the centers are still picked by the plain cost.

Example:
    python pusht_cost_surface.py --grid 101 --planes 3
    python pusht_cost_surface.py --lse-sigma 0.3   # -> pusht_cost_surface_lse0.3.png / .json
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.colors import LightSource, LogNorm
from mujoco import mjx

from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.pusht import PushT

parser = argparse.ArgumentParser(description="3-D surfaces of the PushT planning cost on random planes")
parser.add_argument("--pusher", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--block", type=float, nargs=3, default=[0.1, 0.1, 1.3], metavar=("X", "Y", "THETA"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n", type=int, default=1000, help="random samples to find the row-2 center")
parser.add_argument("--planes", type=int, default=3, help="random planes per row")
parser.add_argument("--grid", type=int, default=101, help="grid points per axis")
parser.add_argument("--wide", type=float, default=2.0, help="half-width of the row-1 planes")
parser.add_argument("--local", type=float, default=0.5, help="half-width of the bottom planes")
parser.add_argument("--interior", type=float, default=0.6,
                    help="the bottom center is the best random sample in [-interior, interior]^12")
parser.add_argument("--chunk", type=int, default=1024, help="rollouts per GPU call")
parser.add_argument("--lse-sigma", type=float, default=0.0, help="LSE perturbation scale (0 = plain cost)")
parser.add_argument("--lse-samples", type=int, default=100, help="LSE perturbations per point (GlobalMPPI: 100)")
parser.add_argument("--lse-lambda", type=float, default=0.1, help="LSE temperature (GlobalMPPI: 0.1)")
parser.add_argument("--out", default=None, help="default pusht_cost_surface[_lse<sigma>].png")
parser.add_argument("--json", default=None,
                    help="the grids for the interactive viewer (make_surface_viewer.py), "
                         "default pusht_cost_surface[_lse<sigma>].json")
args = parser.parse_args()
stem = "pusht_cost_surface" + (f"_lse{args.lse_sigma:g}" if args.lse_sigma > 0 else "")
args.out = args.out or str(Path(__file__).with_name(stem + ".png"))
args.json = args.json or str(Path(__file__).with_name(stem + ".json"))

# ---------------------------------------------------------------- PushT, as in examples/pusht.py
task = PushT()
mj_model = task.mj_model
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = [*args.block, *args.pusher]  # [block x, y, theta, pusher x, y]
mj_data.qvel[:] = 0.0
mujoco.mj_forward(mj_model, mj_data)
state = mjx.put_data(mj_model, mj_data)
state = state.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)

ctrl = GlobalMPPI(task, num_samples=256, noise_level=0.1, temperature=0.1, spline_type="cubic",
                  plan_horizon=1.0, num_knots=6)  # only for its rollout
nu = task.model.nu
num_knots = ctrl.num_knots
dim = num_knots * nu
tk = jnp.linspace(0.0, ctrl.plan_horizon, num_knots) + state.time
u_min = np.tile(np.asarray(task.u_min), num_knots)
u_max = np.tile(np.asarray(task.u_max), num_knots)


@jax.jit
def _costs(knots):
    return ctrl.rollout(state, tk, knots).costs.sum(axis=-1)


def rollout(points):
    """Total rollout cost of flat knot vectors (m, dim) inside the box, in fixed-size chunks."""
    points = np.atleast_2d(points)
    out = []
    for k in range(0, len(points), args.chunk):
        P = np.pad(points[k: k + args.chunk], ((0, max(0, k + args.chunk - len(points))), (0, 0)))
        out.append(np.asarray(_costs(jnp.asarray(P.reshape(-1, num_knots, nu), jnp.float32)))[: len(points) - k])
    return np.concatenate(out).astype(np.float64)


eps = np.random.default_rng(args.seed + 1).standard_normal((args.lse_samples, dim))  # shared by all points


def cost(points):
    """The plotted cost: the plain rollout cost, or with --lse-sigma its LSE smoothing (perturbations
    clipped into the box, as GlobalMPPI.lse_smoothing_rollout)."""
    if args.lse_sigma <= 0:
        return rollout(points)
    points, L, lam = np.atleast_2d(points), args.lse_samples, args.lse_lambda
    out = []
    for k in range(0, len(points), max(1, args.chunk // L)):
        Q = np.clip(points[k: k + max(1, args.chunk // L), None] + args.lse_sigma * eps[None], u_min, u_max)
        J = rollout(Q.reshape(-1, dim)).reshape(-1, L)
        m = J.min(axis=1)
        out.append(m - lam * np.log(np.mean(np.exp(-(J - m[:, None]) / lam), axis=1)))
    return np.concatenate(out)


rng = np.random.default_rng(args.seed)
X = rng.uniform(u_min, u_max, (args.n, dim))
f = rollout(X)
plateau = float(np.median(f))
print(f"{args.n} random samples: min {f.min():.3f}, median (plateau) {plateau:.3f}, max {f.max():.3f}, "
      f"on the plateau {np.mean(np.isclose(f, plateau, atol=1e-3)):.0%}")

Xi = rng.uniform(args.interior * u_min, args.interior * u_max, (args.n, dim))
fi = rollout(Xi)
print(f"{args.n} random samples in [-{args.interior:g}, {args.interior:g}]^{dim}: min {fi.min():.3f}")
rows = [("x = 0 (box center)", np.zeros(dim), args.wide),
        (f"best of {args.n} random samples in [−{args.interior:g}, {args.interior:g}]^{dim}", Xi[np.argmin(fi)],
         args.local)]
a = np.linspace(-1, 1, args.grid)
A, B = np.meshgrid(a, a)
surfaces = []
for title, center, half in rows:
    for p in range(args.planes):
        d1, d2 = np.linalg.qr(rng.standard_normal((dim, 2)))[0].T  # random orthonormal directions
        P = center + half * (A.reshape(-1, 1) * d1 + B.reshape(-1, 1) * d2)
        inside = np.all((P >= u_min) & (P <= u_max), axis=1)
        J = np.full(len(P), np.nan)
        J[inside] = cost(P[inside])
        surfaces.append((title, p, half, J.reshape(A.shape), float(cost(center)[0])))
        print(f"{title}, plane {p + 1}: min {np.nanmin(J):.3f}, max {np.nanmax(J):.3f}, "
              f"outside the box {1 - inside.mean():.0%}")

label = (f"LSE smoothed, σ = {args.lse_sigma:g}, λ = {args.lse_lambda:g}, {args.lse_samples} samples"
         if args.lse_sigma > 0 else "plain cost")
Path(args.json).write_text(json.dumps(dict(
    label=label, plateau=plateau, grid=args.grid, block=args.block, pusher=args.pusher, seed=args.seed,
    planes=[dict(center=title, plane=p + 1, half=half, center_cost=c0,
                 J=[[None if not np.isfinite(v) else round(float(v), 2) for v in row] for row in J])
            for title, p, half, J, c0 in surfaces])))
print("RESULT grids saved", args.json)

# ---------------------------------------------------------------- plot
zmax = max(np.nanmax(s[3]) for s in surfaces)
zmin = min(np.nanmin(s[3]) for s in surfaces)
norm = LogNorm(zmin, zmax)
cmap = plt.get_cmap("coolwarm").copy()  # ParaView's cool to warm, the colors of Li et al. / SAM
cmap.set_bad("white")
fig = plt.figure(figsize=(6 * args.planes, 20), layout="constrained")
gs = fig.add_gridspec(2 * len(rows), args.planes, height_ratios=[1.15, 1] * len(rows))
for i, (title, p, half, J, c0) in enumerate(surfaces):
    r = i // args.planes
    on_plateau = np.isclose(J, plateau, atol=1e-3)
    # 3-D surface: holes where the plane leaves the box
    ax = fig.add_subplot(gs[2 * r, p], projection="3d")
    face = cmap(norm(np.where(np.isfinite(J), J, zmax)))
    ax.plot_surface(half * A, half * B, np.ma.masked_invalid(J), facecolors=face, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=True, lightsource=LightSource(azdeg=315, altdeg=50))
    ax.plot([0, 0], [0, 0], [0, c0], color="k", lw=2)  # a stick at the center, visible through the surface
    ax.set_zlim(0, zmax)
    ax.set_xlabel("a (d1)")
    ax.set_ylabel("b (d2)")
    ax.set_zlabel("cost")
    ax.view_init(elev=35, azim=-60)
    ax.set_title(f"{title}\nrandom plane {p + 1}, |a|, |b| ≤ {half:g}   (center cost {c0:.2f})", fontsize=10)
    # the same plane from above
    ax = fig.add_subplot(gs[2 * r + 1, p])
    im = ax.pcolormesh(half * A, half * B, J, norm=norm, cmap=cmap, shading="auto")
    ax.contour(half * A, half * B, J, levels=np.geomspace(zmin, zmax, 12), colors="k", linewidths=0.35, alpha=0.5)
    ax.plot(0, 0, "*", ms=16, mfc="gold", mec="k", mew=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("a (d1)")
    ax.set_ylabel("b (d2)")
    ax.set_title(f"from above: plateau {plateau:.2f} on {on_plateau.mean():.0%}, white = outside the box "
                 f"({np.mean(~np.isfinite(J)):.0%})", fontsize=9)
fig.colorbar(im, ax=fig.axes, shrink=0.4, label="rollout cost (log)", pad=0.01)
fig.suptitle(f"PushT planning cost ({label}) J(c + a d1 + b d2) on random planes, x ∈ [−1, 1]^{dim}\n"
             f"(start: block {tuple(args.block)}, pusher {tuple(args.pusher)}; plateau = {plateau:.2f}, "
             f"pusher misses the block)", fontsize=13)
fig.savefig(args.out, dpi=120)
print("RESULT saved", args.out)
