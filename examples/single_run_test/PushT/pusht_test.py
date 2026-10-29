"""One KSOS solve on PushT, outside the control loop.

Given the pusher and block positions: sample control knots in the box
[u_min, u_max]^(num_knots * nu), roll each one out from that state for its
cost, RCM-order the samples, build the Wendland kernel features and solve
with ADMM-sparse. Then roll out the solution and compare it with the best
sample.

Sampling:
  (samplers: global_mppi/sampling.py)
  sobol      n Sobol points over the whole box (SobolSampler, on GPU).
  two-level  ksos-tools' clustered design: n / cluster-size Sobol cluster
             centers, cluster-size Sobol points in a ball of 0.5 sigma around
             each, sigma = 0.5 x the centers' mean nearest-neighbour distance.
             Points in one cluster are all within sigma of each other, so
             each cluster is a clique of K (10 x 10 -> cliques of 10). CPU.
  two-level-gpu  the same two-level Sobol design on the GPU (TwoLevelSampler):
             the Sobol centers and each cluster's Sobol ball points are made
             once (CPU), then each call shifts the centers randomly (mod 1),
             scales them into the box, sets sigma from their nearest-neighbour
             distances and places each cluster's ball points in the 0.5
             sigma ball -- all in one jitted call.

Example:
    python pusht_test.py --pusher 0.0 0.0 --block 0.1 0.1 1.3 --n 100 --sigma 1.75
    python pusht_test.py --sampling two-level --n 100 --cluster-size 10
    python pusht_test.py --sampling two-level-gpu --n 1000 --cluster-size 10
"""

import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from global_mppi.algs import GlobalMPPI
from global_mppi.sampling import SobolSampler, TwoLevelSampler, cluster_report, two_level_cpu
from global_mppi.tasks.pusht import PushT
from ksos_tools.problems.features import build_problem
from ksos_tools.problems.kernels import wendland
from ksos_tools.problems.sampling import rcm_order
from ksos_tools.solvers import ADMM

parser = argparse.ArgumentParser(description="Sobol -> RCM -> ADMM-sparse on PushT")
parser.add_argument("--pusher", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--block", type=float, nargs=3, default=[0.1, 0.1, 1.3], metavar=("X", "Y", "THETA"))
parser.add_argument("--n", type=int, default=100, help="number of samples")
parser.add_argument("--sampling", choices=("sobol", "two-level", "two-level-gpu"), default="sobol")
parser.add_argument("--cluster-size", type=int, default=10, help="two-level: points per cluster")
parser.add_argument("--cluster-radius", type=float, default=0.5,
                    help="two-level-gpu: ball radius in units of sigma (default 0.5: the largest "
                         "that keeps every cluster a clique; smaller couples the points more)")
parser.add_argument("--sigma", type=float, default=None,
                    help="Wendland kernel radius (default: 1.75 for sobol, "
                         "0.5 x mean nearest-center distance for two-level)")
parser.add_argument("--close-weight", type=float, default=0.0,
                    help="PushT close_to_block_weight (pusher-to-block distance term; 0 = original task)")
parser.add_argument("--lambd", type=float, default=1e-5)
parser.add_argument("--time-limit", type=float, default=30.0, help="ADMM time limit (s)")
parser.add_argument("--admm-backend", choices=("cpu", "cuda"), default="cpu",
                    help="ADMM iterations on the CPU (ADMM-sparse) or on the GPU with "
                         "cuKernelSOS (ADMM-cuda); the setup stays on the CPU either way")
parser.add_argument("--anderson-mem", type=int, default=None,
                    help="cuda only: Anderson acceleration memory (0 = off; default: "
                         "cuKernelSOS's choice, 5 for samples of dimension >= 3)")
parser.add_argument("--anderson-reg", type=float, default=1e-10,
                    help="cuda only: Anderson regularization (default 1e-10, cuKernelSOS's own; "
                         "1e-6 / 1e-4 can rescue a solve where the extrapolation blows up)")
parser.add_argument("--cuda-option", action="append", default=[], metavar="KEY=VALUE",
                    help="cuda only: any other cukernelsos.ksos.iterate option, e.g. "
                         "anderson_reg=1e-6, anderson_safeguard=0.9, ruiz=true (repeatable)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save", default=None, help="write the samples, costs and alpha to this CSV")
parser.add_argument("--plot-paths", default=None,
                    help="roll out every sample, save the pusher/block paths to <name>.npz "
                         "and draw them to this PNG")
args = parser.parse_args()
if args.admm_backend == "cuda":
    # cuKernelSOS's Python binding: the first cuKernelSOS/python found in a
    # parent folder (the checkout next to Global-MPPI), wherever this file sits
    for parent in Path(__file__).resolve().parents:
        if (parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
            os.environ.setdefault("CUKERNELSOS_PYTHON", str(parent / "cuKernelSOS" / "python"))
            break

# ---------------------------------------------------------------- state
task = PushT(close_to_block_weight=args.close_weight)
mj_model = task.mj_model
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
# qpos = [block x, block y, block theta, pusher x, pusher y]
mj_data.qpos = [*args.block, *args.pusher]
mj_data.qvel[:] = 0.0
mujoco.mj_forward(mj_model, mj_data)
state = mjx.put_data(mj_model, mj_data)
state = state.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)

ctrl = GlobalMPPI(
    task,
    num_samples=256,
    noise_level=0.1,
    temperature=0.1,
    spline_type="cubic",
    plan_horizon=1.0,
    num_knots=6,
)
nu = task.model.nu
dim = ctrl.num_knots * nu
tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots) + state.time


@jax.jit
def rollout_costs(knots):
    """Total cost of each knot sequence, (n, num_knots, nu) -> (n,)."""
    return ctrl.rollout(state, tk, knots).costs.sum(axis=-1)


# the original task cost (position + orientation only), to judge any --close-weight run
ctrl_orig = GlobalMPPI(PushT(), num_samples=256, noise_level=0.1, temperature=0.1,
                       spline_type="cubic", plan_horizon=1.0, num_knots=6)


@jax.jit
def original_costs(knots):
    return ctrl_orig.rollout(state, tk, knots).costs.sum(axis=-1)


u_min = np.tile(np.asarray(task.u_min), ctrl.num_knots)
u_max = np.tile(np.asarray(task.u_max), ctrl.num_knots)


# ---------------------------------------------------------------- 1. sampling
# samplers from global_mppi.sampling: flat (n, dim) samples in the box
# [u_min, u_max]; only their Sobol unit points are made on the CPU, once
lower, upper = jnp.asarray(u_min, jnp.float32), jnp.asarray(u_max, jnp.float32)
rng = jax.random.PRNGKey(args.seed)
t_setup, outside, cluster = 0.0, 0.0, None
if args.sampling in ("sobol", "two-level-gpu"):
    t0 = time.perf_counter()
    if args.sampling == "sobol":
        sampler = SobolSampler(args.n, dim, seed=args.seed)
        sample = jax.jit(lambda key: (sampler.sample(key, lower, upper), None, 0.0))
    else:
        sampler = TwoLevelSampler(args.n, dim, cluster_size=args.cluster_size, seed=args.seed,
                                  cluster_radius=args.cluster_radius)
        sample = jax.jit(lambda key: sampler.sample(key, lower, upper))
        cluster = sampler.cluster
    t_setup = time.perf_counter() - t0  # CPU, once: the Sobol unit points
    jax.block_until_ready(sample(rng))  # compile
    t0 = time.perf_counter()
    points, sigma, outside = sample(rng)  # stays on the GPU
    points.block_until_ready()
    t_sample = time.perf_counter() - t0
    if args.sigma is not None or sigma is None:
        sigma = 1.75 if args.sigma is None else args.sigma
    sigma, outside = float(sigma), float(outside)
else:
    # ksos-tools' fixed-seed design on the CPU, on PushT's box [-1, 1]^dim
    t0 = time.perf_counter()
    two_level, sigma, cluster, outside = two_level_cpu(args.n, dim, args.cluster_size, float(u_max[0]), args.sigma)
    points = jnp.asarray(two_level, dtype=jnp.float32)
    points.block_until_ready()
    t_sample = time.perf_counter() - t0
knots = points.reshape(args.n, ctrl.num_knots, nu)

print(f"block={args.block}  pusher={args.pusher}  n={args.n}  dim={dim}  "
      f"sampling={args.sampling}  sigma={sigma:.4f}")

# ---------------------------------------------------------------- 2. rollout
rollout_costs(knots).block_until_ready()  # compile
t0 = time.perf_counter()
f = rollout_costs(knots).block_until_ready()
t_rollout = time.perf_counter() - t0
# to the host for RCM / kernel / ADMM (CPU)
samples = np.asarray(knots, dtype=np.float64).reshape(args.n, dim)
f = np.asarray(f, dtype=np.float64)
print(f"devices: knots on {knots.devices()}, costs on {rollout_costs(knots).devices()}")
print(f"rollout costs: min={f.min():.4f} median={np.median(f):.4f} max={f.max():.4f}")

if cluster is not None:
    print(cluster_report(samples, f, cluster, sigma))
    # how strongly the points of one cluster are coupled by the kernel
    full = np.bincount(cluster) == args.cluster_size
    X = samples[np.isin(cluster, np.flatnonzero(full))].reshape(-1, args.cluster_size, dim)
    r = np.linalg.norm(X[:, :, None] - X[:, None], axis=-1) / sigma
    k = wendland(r, dim)[:, ~np.eye(args.cluster_size, dtype=bool)]
    print(f"  within-cluster kernel values: median {np.median(k):.2e}, mean {k.mean():.2e}, "
          f"max {k.max():.2e}; median distance {np.median(r[:, ~np.eye(args.cluster_size, dtype=bool)]):.3f} sigma")
    print(f"  knots clipped into [-1, 1]: {outside:.1%}")

# ---------------------------------------------------------------- 3. RCM
t0 = time.perf_counter()
perm = rcm_order(samples, sigma)
samples, f = samples[perm], f[perm]
t_rcm = time.perf_counter() - t0

# ---------------------------------------------------------------- 4. kernel
t0 = time.perf_counter()
problem = build_problem(samples, sigma, lambda _: f, lambd=args.lambd)
t_kernel = time.perf_counter() - t0
print(f"kernel: band={problem.bandwidth}  nnz(Phi)={problem.Phi.nnz}")

# ---------------------------------------------------------------- 5. ADMM
cuda_options = {"anderson_reg": args.anderson_reg}
if args.anderson_mem is not None:
    cuda_options["anderson_mem"] = args.anderson_mem
for option in args.cuda_option:
    key, value = option.split("=", 1)
    if value.lower() in ("true", "false"):
        cuda_options[key] = value.lower() == "true"
    else:
        cuda_options[key] = int(value) if value.lstrip("-").isdigit() else float(value)
t0 = time.perf_counter()
x, info = ADMM.solve_ksos(
    problem.Phi,
    problem.f,
    problem.lambd,
    samples=problem.samples,
    time_limit=args.time_limit,
    backend=args.admm_backend,
    cuda_options=cuda_options if args.admm_backend == "cuda" else None,
    record_history=False,
)
t_admm = time.perf_counter() - t0
stats = info["admm_stats"]
print(
    f"cliques: num={stats['num_cliques']}  max={stats['max_clique']}  "
    f"mean={stats['mean_clique']:.1f}  fill_edges={stats['fill_edges']}"
)
print(
    f"ADMM: status={info['status']}  iters={info['iterations']}  "
    f"{'anderson accepted/rejected=' + str(stats.get('anderson_accepted')) + '/' + str(stats.get('anderson_rejected')) + '  ' if 'anderson_accepted' in stats else ''}"
    f"c={info['cost']:.4f}  sum(alpha)={info['alpha_sum']:.4f}"
)

# ---------------------------------------------------------------- 6. check
alpha = np.asarray(info["alpha"])
top = np.argsort(alpha)[::-1][:5]
print("largest alpha (weight, sample cost):", [(round(float(alpha[i]), 3), round(float(f[i]), 3)) for i in top])
# how many samples carry weight: effective count 1 / sum(alpha^2) (1 = one sample)
big = np.abs(alpha) > 1e-2
print(f"alpha spread: effective samples {1.0 / np.sum(alpha ** 2):.2f}, |alpha| > 0.01: {big.sum()} "
      f"(positive {np.sum(alpha > 1e-2)}, negative {np.sum(alpha < -1e-2)}), "
      f"sum of negative alpha {alpha[alpha < 0].sum():+.4f}"
      + (f", clusters {np.unique(cluster[perm][big]).tolist()}" if cluster is not None else ""))
x_knots = jnp.asarray(np.clip(x, u_min, u_max).reshape(1, ctrl.num_knots, nu), dtype=jnp.float32)
x_cost = float(rollout_costs(x_knots)[0])
print(f"rollout cost at ADMM x: {x_cost:.4f}   best sample: {f.min():.4f}   "
      f"||x - best sample|| = {np.linalg.norm(np.asarray(x) - samples[np.argmin(f)]):.4f}   "
      f"alpha: max {alpha.max():.4f}, sum of the rest {alpha.sum() - alpha.max():+.4f}, "
      f"min {alpha.min():+.2e}")
print("x knots:\n", np.asarray(x).reshape(ctrl.num_knots, nu))
if args.close_weight:
    best_knots = jnp.asarray(samples[np.argmin(f)].reshape(1, ctrl.num_knots, nu), dtype=jnp.float32)
    f_orig = np.asarray(original_costs(jnp.asarray(samples.reshape(-1, ctrl.num_knots, nu), jnp.float32)))
    print(f"original task cost (weight 0): x {float(original_costs(x_knots)[0]):.4f}, best sample by this "
          f"cost {float(original_costs(best_knots)[0]):.4f}, best original over all samples {f_orig.min():.4f}; "
          f"samples on the original plateau (52.288) {np.mean(np.isclose(f_orig, 52.2881, atol=1e-3)):.0%}, "
          f"on a plateau of this cost {np.mean(np.isclose(f, np.median(f), atol=1e-3)):.0%}")

if args.save:
    # one row per sample, in RCM order; u{k}_{j} is knot k, control j
    header = "rcm_idx,sobol_idx,cost,alpha," + ",".join(
        f"u{k}_{j}" for k in range(ctrl.num_knots) for j in range(nu)
    )
    table = np.column_stack([np.arange(args.n), perm, f, alpha, samples])
    np.savetxt(args.save, table, delimiter=",", header=header, comments="",
               fmt=["%d", "%d"] + ["%.6f"] * (table.shape[1] - 2))
    print(f"saved {args.n} samples to {args.save}")


# ---------------------------------------------------------------- 7. paths
@jax.jit
def rollout_qpos(knots):
    """qpos along each rollout, (n, num_knots, nu) -> (n, H + 1, 5), with the
    initial state first. qpos = [block x, y, theta, pusher x, y]."""
    tq = jnp.linspace(tk[0], tk[-1], ctrl.ctrl_steps)
    controls = ctrl.interp_func(tq, tk, knots)
    states, _ = ctrl.eval_rollouts(ctrl.model, state, controls, knots)
    q0 = jnp.broadcast_to(state.qpos, (knots.shape[0], 1, state.qpos.shape[0]))
    return jnp.concatenate([q0, states.qpos], axis=1)


def draw_T(ax, x, y, theta, **kw):
    """The T block (pusht.xml: stem 0.02 x 0.08, bar 0.06 x 0.02 at +0.03 y)."""
    from matplotlib.patches import Polygon

    c, s_ = np.cos(theta), np.sin(theta)
    for cx, cy, hx, hy in ((0.0, 0.0, 0.01, 0.04), (0.0, 0.03, 0.03, 0.01)):
        corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) + [cx, cy]
        world = corners @ np.array([[c, s_], [-s_, c]]) + [x, y]
        ax.add_patch(Polygon(world, closed=True, **kw))


if args.plot_paths:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    sample_knots = jnp.asarray(samples.reshape(args.n, ctrl.num_knots, nu), dtype=jnp.float32)
    qpos = np.asarray(rollout_qpos(sample_knots))  # RCM order, like samples / f
    pusher = qpos[:, :, 3:5] + [0.0, 0.1]  # pusher body sits at y = 0.1 in pusht.xml
    block = qpos[:, :, 0:3]
    npz = args.plot_paths.rsplit(".", 1)[0] + ".npz"
    np.savez(npz, samples=samples, cost=f, alpha=alpha, sobol_idx=perm,
             pusher_xy=pusher, block_xytheta=block, dt=ctrl.dt)
    print(f"saved paths to {npz}")

    INK, MUTED, GRID, SURF, GRAY, ORANGE, GREEN = (
        "#2b2b29", "#6b6a65", "#e4e3df", "#fcfcfb", "#c9c8c3", "#e8743b", "#2f9e6e")
    plateau = np.isclose(f, np.median(f), atol=1e-3)
    best_i = int(np.argmax(alpha))
    cmap = LinearSegmentedColormap.from_list("blue", ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"])
    norm = Normalize(f[~plateau].min(), f[~plateau].max())
    color = lambda i: GRAY if plateau[i] else cmap(norm(f[i]))
    order = np.concatenate([np.flatnonzero(plateau), np.flatnonzero(~plateau)[np.argsort(-f[~plateau])]])
    order = order[order != best_i]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlecolor": INK,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), constrained_layout=True)
    allxy = np.concatenate([pusher.reshape(-1, 2), block[:, :, :2].reshape(-1, 2), [[0, 0]]])
    lo, hi = allxy.min(0) - 0.06, allxy.max(0) + 0.06

    def frame(ax, title, pts=None):
        """Equal-aspect top view; zoomed to `pts` (plus goal and starts) if given."""
        a, b = lo, hi
        if pts is not None:
            pts = np.concatenate([pts, [[0, 0], pusher[0, 0], block[0, 0, :2]]])
            a, b = pts.min(0) - 0.05, pts.max(0) + 0.05
        ax.set_aspect("equal"); ax.set_xlim(a[0], b[0]); ax.set_ylim(a[1], b[1])
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title(title)
        draw_T(ax, 0.0, 0.0, 0.0, fc="none", ec=GREEN, lw=1.6, ls="--", zorder=2)  # goal
        draw_T(ax, *block[0, 0], fc="#ebe9e4", ec=INK, lw=1.0, zorder=2)  # start
        ax.scatter(*pusher[0, 0], s=70, color="#d1453b", edgecolor=SURF, lw=1.5, zorder=8)

    # (a) pusher paths of every sample
    ax = axes[0]
    frame(ax, f"(a) pusher path of all {args.n} samples")
    for i in order:
        ax.plot(*pusher[i].T, color=color(i), lw=0.8 if plateau[i] else 1.1,
                alpha=0.55 if plateau[i] else 0.9, zorder=3)
    ax.plot(*pusher[best_i].T, color=ORANGE, lw=2.6, zorder=6)
    ax.scatter(*pusher[best_i, -1], s=40, color=ORANGE, edgecolor=SURF, zorder=7)

    # (b) where the block ends up (non-plateau samples)
    ax = axes[1]
    frame(ax, f"(b) final block pose: {np.sum(~plateau)} samples that move it (zoomed)",
          block[~plateau, :, :2].reshape(-1, 2))
    for i in order:
        if plateau[i]:
            continue
        ax.plot(*block[i, :, :2].T, color=color(i), lw=1.0, zorder=3)
        draw_T(ax, *block[i, -1], fc="none", ec=color(i), lw=0.9, alpha=0.8, zorder=3)
    ax.plot(*block[best_i, :, :2].T, color=ORANGE, lw=2.4, zorder=6)
    draw_T(ax, *block[best_i, -1], fc=ORANGE, ec=INK, lw=0.8, alpha=0.85, zorder=6)

    # (c) the best sample over time
    ax = axes[2]
    frame(ax, f"(c) best sample (cost {f[best_i]:.2f}, alpha {alpha[best_i]:.2f}) over 1 s (zoomed)",
          np.concatenate([pusher[best_i], block[best_i, :, :2]]))
    H = block.shape[1] - 1
    for k, t in enumerate(np.linspace(0, H, 6).astype(int)):
        a = 0.25 + 0.75 * k / 5
        draw_T(ax, *block[best_i, t], fc=ORANGE, ec=INK, lw=0.6, alpha=a * 0.7, zorder=4 + k)
        ax.scatter(*pusher[best_i, t], s=55, color="#d1453b", alpha=a, edgecolor=SURF, zorder=10)
        ax.annotate(f"{t * ctrl.dt:.1f}s", pusher[best_i, t], (8, -14 if k % 2 else 8),
                    textcoords="offset points",
                    fontsize=8, color=INK, zorder=11)
    ax.plot(*pusher[best_i].T, color="#d1453b", lw=1.2, ls=":", zorder=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes.tolist(), shrink=0.7, pad=0.01, aspect=30)
    cb.set_label("rollout cost (dark = better)\ngray = plateau, orange = best", color=INK)
    fig.suptitle(f"PushT rollouts of {args.n} Sobol samples   start: block {tuple(args.block)}, "
                 f"pusher {tuple(args.pusher)}   red dot = pusher start, gray T = block start, "
                 f"green dashed T = goal", color=INK, fontsize=11, fontweight="bold")
    fig.savefig(args.plot_paths, dpi=150, facecolor=SURF)
    print(f"saved plot to {args.plot_paths}")

print(
    f"time: sampling({'cpu' if args.sampling == 'two-level' else 'gpu'})={t_sample*1e3:.1f}ms"
    f"{f' (+ once: Sobol unit points {t_setup*1e3:.1f}ms cpu)' if t_setup else ''}  rollout(gpu)={t_rollout*1e3:.1f}ms  "
    f"rcm={t_rcm*1e3:.1f}ms  kernel={t_kernel*1e3:.1f}ms  admm({args.admm_backend})={t_admm:.3f}s "
    f"(setup={info['setup_time']:.3f}s solve={info['solve_time']:.3f}s)"
)
