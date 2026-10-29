"""Cost-aware chordal sampling vs two-level Sobol on PushT, same rollout budget.

For a budget of P rollouts:
  two-level   P two-level Sobol samples (clusters of 10), all P go to KSOS
              (RCM order; the largest clique is the cluster size).
  cost-aware  P Sobol candidates, all rolled out; then, best cost first, a
              candidate is admitted only if its two-hop neighbourhood in the
              kernel graph of the admitted samples has fewer than max_clique
              nodes and is already a clique of the chordal certificate H
              (cost_aware_chordal_sampling.py). Only the admitted samples go to
              KSOS, in reverse admission order (a perfect elimination order of
              H; our solver factors K, so the order matters).
Both use the same kernel radius sigma (the two-level rule for P samples) and
the same ADMM; every x is judged by its original cost (a fresh rollout).

Example:
    python pusht_cost_aware_test.py --budget 1000 --seed 0 --max-clique 10
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from scipy.spatial import cKDTree
from scipy.stats import qmc

from global_mppi.algs import GlobalMPPI
from global_mppi.sampling import TwoLevelSampler
from global_mppi.tasks.pusht import PushT
from ksos_tools.problems.features import build_problem
from ksos_tools.problems.sampling import rcm_order
from ksos_tools.solvers import ADMM

parser = argparse.ArgumentParser(description="cost-aware chordal sampling vs two-level Sobol on PushT")
parser.add_argument("--pusher", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--block", type=float, nargs=3, default=[0.1, 0.1, 1.3], metavar=("X", "Y", "THETA"))
parser.add_argument("--budget", type=int, default=1000, help="rollouts for each method")
parser.add_argument("--cluster-size", type=int, default=10, help="two-level: points per cluster")
parser.add_argument("--max-clique", type=int, default=10, help="cost-aware: bound on the certificate's cliques")
parser.add_argument("--sigma", type=float, default=None,
                    help="cost-aware kernel radius (default: the two-level rule for the budget); two-level "
                         "always uses its own rule")
parser.add_argument("--accept", type=int, default=None,
                    help="cost-aware: stop after this many admitted samples (default: scan the whole pool)")
parser.add_argument("--lambd", type=float, default=1e-5)
parser.add_argument("--time-limit", type=float, default=60.0, help="ADMM time limit (s)")
parser.add_argument("--anderson-reg", type=float, default=1e-10)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
# cuKernelSOS's Python binding (GPU ADMM): the first cuKernelSOS/python in a parent folder
for parent in Path(__file__).resolve().parents:
    if (parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
        os.environ.setdefault("CUKERNELSOS_PYTHON", str(parent / "cuKernelSOS" / "python"))
        break

# ---------------------------------------------------------------- PushT
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
ctrl = GlobalMPPI(task, num_samples=256, noise_level=0.1, temperature=0.1,
                  spline_type="cubic", plan_horizon=1.0, num_knots=6)
nu = task.model.nu
dim = ctrl.num_knots * nu
tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots) + state.time
u_min = np.tile(np.asarray(task.u_min), ctrl.num_knots)
u_max = np.tile(np.asarray(task.u_max), ctrl.num_knots)


@jax.jit
def _costs(knots):
    return ctrl.rollout(state, tk, knots).costs.sum(axis=-1)


def rollout(points):
    """Total rollout cost of flat knot vectors (m, dim), clipped into the box."""
    points = np.clip(np.atleast_2d(points), u_min, u_max)
    return np.asarray(_costs(jnp.asarray(points.reshape(-1, ctrl.num_knots, nu), jnp.float32)), np.float64)


def ksos(samples, f, sigma, order):
    """Wendland kernel -> ADMM (GPU) on the samples in this order; (x, info, seconds)."""
    problem = build_problem(samples[order], sigma, lambda _: f[order], lambd=args.lambd)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, info = ADMM.solve_ksos(
            problem.Phi, problem.f, problem.lambd, samples=problem.samples,
            time_limit=args.time_limit, record_history=False, backend="cuda",
            cuda_options={"anderson_reg": args.anderson_reg},
        )
    return np.asarray(x), info, time.perf_counter() - t0


def select(pool, cost, h, max_clique, accept=None):
    """cost_aware_chordal_sampling.py's admission rule, best cost first over the
    whole pool: admit x only if its two-hop neighbourhood M in the kernel graph
    (pairs closer than h) of the admitted samples has < max_clique nodes and is
    already a clique of the certificate H. Returns the admitted pool indices in
    admission order and the reject counts."""
    admitted, G, H = [], [], []
    X = np.empty((0, pool.shape[1]))
    reject = {"clique_overflow": 0, "not_clique_in_H": 0}
    for ci in np.argsort(cost, kind="stable"):
        x = pool[ci]
        nbr = np.flatnonzero(np.sum((X - x) ** 2, axis=1) < h * h).tolist() if len(X) else []
        M = set(nbr)
        for j in nbr:
            M.update(G[j])
        if len(M) >= max_clique:
            reject["clique_overflow"] += 1
            continue
        if any(not (M - {j}).issubset(H[j]) for j in M):
            reject["not_clique_in_H"] += 1
            continue
        k = len(admitted)  # x becomes a simplicial node of H with neighbourhood M
        admitted.append(ci)
        G.append(set(nbr))
        H.append(set(M))
        for j in nbr:
            G[j].add(k)
        for j in M:
            H[j].add(k)
        X = np.vstack([X, x])
        if accept is not None and len(admitted) >= accept:
            break
    return np.asarray(admitted), reject


P = args.budget
key = jax.random.PRNGKey(args.seed)
lower, upper = jnp.asarray(u_min, jnp.float32), jnp.asarray(u_max, jnp.float32)
rows = []

# ---------------------------------------------------------------- two-level Sobol
sampler = TwoLevelSampler(P, dim, cluster_size=args.cluster_size, seed=args.seed)
points, sigma, _ = jax.jit(sampler.sample)(key, lower, upper)
sigma2 = float(sigma)  # two-level's own kernel radius
sigma = sigma2 if args.sigma is None else args.sigma  # cost-aware's kernel radius
X2 = np.asarray(points, np.float64)
f2 = rollout(X2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    x, info, t = ksos(X2, f2, sigma2, rcm_order(X2, sigma2))
rows.append(("two-level", P, P, info["admm_stats"]["max_clique"], f2.min(), rollout(x)[0],
             info["status"], info["iterations"], t, 0.0))

# ---------------------------------------------------------------- cost-aware chordal
pool = qmc.Sobol(dim, scramble=True, seed=args.seed).random(P) * (u_max - u_min) + u_min
fp = rollout(pool)
t0 = time.perf_counter()
idx, reject = select(pool, fp, sigma, args.max_clique, args.accept)
t_select = time.perf_counter() - t0
Xc, fc = pool[idx], fp[idx]
x, info, t = ksos(Xc, fc, sigma, np.arange(len(Xc))[::-1])  # reverse admission: a PEO of H
rows.append((f"cost-aware (clique<={args.max_clique})", P, len(Xc), info["admm_stats"]["max_clique"],
             fc.min(), rollout(x)[0], info["status"], info["iterations"], t, t_select))

plateau = np.isclose(fp, 52.2881, atol=1e-3)
print(f"budget {P} rollouts, seed {args.seed}, sigma: two-level {sigma2:.3f}, cost-aware {sigma:.3f}; cost-aware pool: {plateau.mean():.0%} on the "
      f"plateau, best {fp.min():.3f}; admitted {len(Xc)}, rejected {reject}; admitted on the plateau "
      f"{np.isclose(fc, 52.2881, atol=1e-3).mean():.0%}")
print(f"{'method':<24} {'rollouts':>8} {'to KSOS':>8} {'max clique':>10} {'best sample':>11} {'x (orig)':>9} "
      f"{'status':<15} {'iters':>6} {'admm':>7} {'select':>7}")
for name, p, m, mc, fb, fx, st, it, t, ts in rows:
    print(f"{name:<24} {p:>8} {m:>8} {mc:>10} {fb:>11.4f} {fx:>9.4f} {st:<15} {it:>6} {t:>6.2f}s {ts:>6.2f}s")
print(f"RESULT budget={P} seed={args.seed} clique={args.max_clique}: "
      + " | ".join(f"{r[0]}: x {r[5]:.4f} best {r[4]:.4f} n {r[2]} clique {r[3]} admm {r[8]:.2f}s" for r in rows))
