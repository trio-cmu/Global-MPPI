"""Does LSE smoothing help KSOS on PushT? Two restart chains, same schedule.

GlobalMPPI's restart scheme: restart i samples a trust region of radius
decay^i * u_max centered on the previous restart's KSOS solution (0 at
restart 0), solves KSOS there, and passes its x on as the next center.

  original chain  KSOS on the rollout cost f.
  lse chain       KSOS on the LSE surrogate with a shrinking scale s_i
                  (GlobalMPPI's lse_smoothing_sigma 0.3, 0.2, 0.1, 0.05,
                  0.001): f_lse(x) = -lam log mean_l exp(-(f(x + s_i eps_l)
                  - m) / lam) + m, eps_l ~ N(0, I), m = min_l f(x + s_i eps_l).
                  Large s_i at first smooths the plateaus and contact jumps;
                  the last s_i ~ 0 recovers f.

Samples: two-level Sobol (TwoLevelSampler), n per restart in both chains, in
each trust region. Every x is judged by its ORIGINAL cost (a fresh rollout);
restart 0 sees the same samples in both chains.

Example:
    python pusht_smoothing_test.py --n 100 --seed 0 --lse-samples 200
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
from global_mppi.sampling import TwoLevelSampler
from global_mppi.tasks.pusht import PushT
from ksos_tools.problems.features import build_problem
from ksos_tools.problems.sampling import rcm_order
from ksos_tools.solvers import ADMM

parser = argparse.ArgumentParser(description="KSOS restart chains on PushT with and without LSE smoothing")
parser.add_argument("--pusher", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--block", type=float, nargs=3, default=[0.1, 0.1, 1.3], metavar=("X", "Y", "THETA"))
parser.add_argument("--n", type=int, default=100, help="samples per restart (both chains)")
parser.add_argument("--cluster-size", type=int, default=10, help="two-level: points per cluster")
parser.add_argument("--cluster-radius", type=float, default=0.5, help="two-level: ball radius / sigma")
parser.add_argument("--decay", type=float, default=0.8, help="trust region radius = decay^i * u_max")
parser.add_argument("--lse-sigmas", type=float, nargs="+", default=[0.3, 0.2, 0.1, 0.05, 0.001],
                    help="LSE perturbation scale per restart (also sets the number of restarts)")
parser.add_argument("--lse-samples", type=int, default=200, help="perturbations per sample")
parser.add_argument("--lse-lambda", type=float, default=0.1, help="LSE temperature (GlobalMPPI: 0.1)")
parser.add_argument("--clip-region", action="store_true",
                    help="keep everything in the restart's trust region: clip the LSE perturbations "
                         "and the KSOS solution x into [center - radius, center + radius]")
parser.add_argument("--common-noise", action="store_true",
                    help="one set of perturbations per restart, shared by every sample (common random "
                         "numbers): f_lse is then a fixed, continuous function of x")
parser.add_argument("--chunk", type=int, default=64, help="samples per LSE rollout batch (GPU memory)")
parser.add_argument("--lambd", type=float, default=1e-5)
parser.add_argument("--time-limit", type=float, default=60.0, help="ADMM time limit (s)")
parser.add_argument("--admm-backend", choices=("cpu", "cuda"), default="cuda")
parser.add_argument("--anderson-reg", type=float, default=1e-10, help="cuda only")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if args.admm_backend == "cuda":
    # cuKernelSOS's Python binding: the first cuKernelSOS/python in a parent folder
    for parent in Path(__file__).resolve().parents:
        if (parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
            os.environ.setdefault("CUKERNELSOS_PYTHON", str(parent / "cuKernelSOS" / "python"))
            break

# ---------------------------------------------------------------- state
task = PushT()
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

ctrl = GlobalMPPI(task, num_samples=256, noise_level=0.1, temperature=0.1,
                  spline_type="cubic", plan_horizon=1.0, num_knots=6)
nu = task.model.nu
dim = ctrl.num_knots * nu
tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots) + state.time
u_min = np.tile(np.asarray(task.u_min), ctrl.num_knots)
u_max = np.tile(np.asarray(task.u_max), ctrl.num_knots)


@jax.jit
def rollout_costs(knots):
    """Total cost of each knot sequence, (n, num_knots, nu) -> (n,)."""
    return ctrl.rollout(state, tk, knots).costs.sum(axis=-1)


@jax.jit
def lse_costs(knots, key, scale, lo, hi):
    """GlobalMPPI's LSE-smoothed cost of each knot sequence, (n, num_knots, nu) -> (n,):
    a soft minimum (temperature lse_lambda) of the costs of lse_samples Gaussian
    perturbations of this scale (not clipped, as in GlobalMPPI)."""
    if args.common_noise:  # the same lse_samples perturbations for every sample
        noise = jnp.broadcast_to(jax.random.normal(key, (1, args.lse_samples, *knots.shape[1:])),
                                 (knots.shape[0], args.lse_samples, *knots.shape[1:]))
    else:
        noise = jax.random.normal(key, (knots.shape[0], args.lse_samples, *knots.shape[1:]))
    perturbed = (knots[:, None] + scale * noise).reshape(-1, *knots.shape[1:])
    perturbed = jnp.clip(perturbed, lo, hi)  # lo / hi: the region, or +-inf for no clipping
    costs = rollout_costs(perturbed).reshape(knots.shape[0], args.lse_samples)
    m = jnp.min(costs, axis=1, keepdims=True)
    soft = -args.lse_lambda * jnp.log(jnp.mean(jnp.exp(-(costs - m) / args.lse_lambda), axis=1, keepdims=True)) + m
    return soft[:, 0]


def smoothed(knots, key, scale, lo, hi):
    """lse_costs in chunks of --chunk samples (n x lse_samples rollouts in all)."""
    out = []
    keys = jax.random.split(key, -(-len(knots) // args.chunk))
    if args.common_noise:
        keys = [key] * len(keys)  # every chunk draws the same perturbations
    for i, k in zip(range(0, len(knots), args.chunk), keys):
        out.append(np.asarray(lse_costs(knots[i: i + args.chunk], k, scale, lo, hi)))
    return np.concatenate(out).astype(np.float64)


def true_cost(x):
    """The original cost of one flat knot vector, clipped into the box."""
    knots = jnp.asarray(np.clip(x, u_min, u_max).reshape(1, ctrl.num_knots, nu), dtype=jnp.float32)
    return float(rollout_costs(knots)[0])


sampler = TwoLevelSampler(args.n, dim, cluster_size=args.cluster_size, seed=args.seed,
                          cluster_radius=args.cluster_radius)
sample = jax.jit(sampler.sample)


def ksos(samples, f, sigma):
    """RCM -> Wendland kernel -> ADMM on these samples with costs f."""
    perm = rcm_order(samples, sigma)
    problem = build_problem(samples[perm], sigma, lambda _: f[perm], lambd=args.lambd)
    x, info = ADMM.solve_ksos(
        problem.Phi, problem.f, problem.lambd, samples=problem.samples,
        time_limit=args.time_limit, record_history=False, backend=args.admm_backend,
        cuda_options={"anderson_reg": args.anderson_reg} if args.admm_backend == "cuda" else None,
    )
    return np.asarray(x), info


def chain(use_lse):
    """One restart chain; per restart: (radius, scale, f min, chosen sample's original
    cost, x's original cost, ADMM status, iterations, time)."""
    center = np.zeros(dim)
    rows, rollouts = [], 0
    for i, scale in enumerate(args.lse_sigmas):
        radius = args.decay**i * u_max
        lo, hi = np.maximum(center - radius, u_min), np.minimum(center + radius, u_max)
        points, sigma, _ = sample(jax.random.fold_in(jax.random.PRNGKey(args.seed), i),
                                  jnp.asarray(lo, jnp.float32), jnp.asarray(hi, jnp.float32))
        samples, sigma = np.asarray(points, dtype=np.float64), float(sigma)
        knots = points.reshape(args.n, ctrl.num_knots, nu)
        f_orig = np.asarray(rollout_costs(knots), dtype=np.float64)
        rollouts += args.n
        if use_lse:
            if args.clip_region:
                box = [jnp.asarray(v.reshape(ctrl.num_knots, nu), jnp.float32) for v in (lo, hi)]
            else:  # as in GlobalMPPI: perturbations are not clipped
                box = [jnp.full((ctrl.num_knots, nu), v, jnp.float32) for v in (-jnp.inf, jnp.inf)]
            f = smoothed(knots, jax.random.fold_in(jax.random.PRNGKey(1000 + args.seed), i), scale, *box)
            rollouts += args.n * args.lse_samples
            # is f_lse smoother? mean |f(x_a) - f(x_b)| over pairs in one cluster (neighbours)
            c = np.arange(args.n) // args.cluster_size
            a, b = np.triu_indices(args.n, 1)
            near = c[a] == c[b]
            rough_orig = np.mean(np.abs(f_orig[a[near]] - f_orig[b[near]]))
            rough_lse = np.mean(np.abs(f[a[near]] - f[b[near]]))
            corr = np.corrcoef(f, f_orig)[0, 1] if np.std(f_orig) > 0 else np.nan
            print(f"  lse restart {i} (s={scale}): corr(f_lse, f) = {corr:+.3f}; neighbour |df|: "
                  f"original {rough_orig:.3f}, lse {rough_lse:.3f}; f range: original "
                  f"{f_orig.min():.2f}-{f_orig.max():.2f}, lse {f.min():.2f}-{f.max():.2f}; "
                  f"on the plateau: original {np.mean(np.isclose(f_orig, 52.2881, atol=1e-3)):.0%}, "
                  f"lse {np.mean(np.isclose(f, 52.2881, atol=1e-3)):.0%}")
        else:
            f = f_orig
        t0 = time.perf_counter()
        x, info = ksos(samples, f, sigma)
        t_admm = time.perf_counter() - t0
        x = np.clip(x, lo, hi) if args.clip_region else np.clip(x, u_min, u_max)
        rows.append((float(radius[0]), scale if use_lse else 0.0, f.min(), f_orig[np.argmin(f)],
                     true_cost(x), info["status"], info["iterations"], t_admm))
        center = x  # the next trust region is centered on this restart's solution
    return rows, rollouts


results = {}
for name, use_lse in (("original", False), ("lse", True)):
    t0 = time.perf_counter()
    rows, rollouts = chain(use_lse)
    results[name] = (rows, rollouts, time.perf_counter() - t0)

print(f"n={args.n} per restart, seed={args.seed}, {len(args.lse_sigmas)} restarts, decay {args.decay}, "
      f"LSE: L={args.lse_samples}, lam={args.lse_lambda}, scales {args.lse_sigmas}, "
      f"common noise {args.common_noise}, clip to region {args.clip_region}")
for name, (rows, rollouts, wall) in results.items():
    print(f"\n{name} chain ({rollouts} rollouts, {wall:.1f}s)")
    print(f"{'restart':>7} {'radius':>7} {'lse s':>6} {'min f':>9} {'chosen sample (orig)':>21} "
          f"{'x (orig)':>9} {'status':<15} {'iters':>6} {'admm':>7}")
    for i, (r, s, fmin, fsel, fx, st, it, ta) in enumerate(rows):
        print(f"{i:>7} {r:>7.3f} {s:>6.3f} {fmin:>9.3f} {fsel:>21.3f} {fx:>9.3f} {st:<15} {it:>6} {ta:>6.2f}s")
o, s = results["original"][0][-1][4], results["lse"][0][-1][4]
print(f"\nRESULT n={args.n} L={args.lse_samples} common={args.common_noise} clip={args.clip_region} seed={args.seed}: final x original cost | "
      f"no lse {o:.4f} | lse {s:.4f}")
