"""Sweep the local refinement after KSOS on PushT (one planning step, one start state).

KSOS runs once per seed, as in pusht_method_benchmark.py: --restarts rounds of
--n two-level Sobol samples in a box of half-width --decay^i around the last
solution, ADMM-cuda (a solve that does not converge within --admm-time-limit
counts as failed: the restart keeps its best sample), keeping the cheapest of
x, the best sample and the previous solution. Then every refinement in
--configs starts from that solution:
  mppi:NOISE:TEMP          MPPI, fixed noise
  dial:NOISE:TEMP          DIAL (annealed over iterations and along the horizon, beta 1 / 1)
each with --samples samples x --iters iterations. The plan is the refiner's
final mean, judged by a fresh rollout. Records go to --json.

Example:
    python pusht_ksos_refine_sweep.py --configs mppi:0.2:0.1 mppi:0.3:0.01 dial:0.3:0.01
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from global_mppi.algs import DIAL, MPPI, GlobalMPPI
from global_mppi.sampling import TwoLevelSampler
from global_mppi.tasks.pusht import PushT
from ksos_tools.problems.features import build_problem
from ksos_tools.problems.sampling import rcm_order
from ksos_tools.solvers import ADMM

parser = argparse.ArgumentParser(description="refinement sweep after KSOS on PushT")
parser.add_argument("--pusher", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--block", type=float, nargs=3, default=[0.1, 0.1, 1.3], metavar=("X", "Y", "THETA"))
parser.add_argument("--n", type=int, default=1000, help="KSOS samples per restart")
parser.add_argument("--restarts", type=int, default=5)
parser.add_argument("--decay", type=float, default=0.8, help="box half-width at restart i: decay^i")
parser.add_argument("--cluster-size", type=int, default=10)
parser.add_argument("--lambd", type=float, default=1e-5)
parser.add_argument("--admm-time-limit", type=float, default=2.0)
parser.add_argument("--samples", type=int, default=1000, help="refinement samples per iteration")
parser.add_argument("--iters", type=int, default=5, help="refinement iterations")
parser.add_argument("--configs", nargs="+", default=["mppi:0.2:0.1"])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--lse-samples", type=int, default=0,
                    help="KSOS: LSE smoothing as GlobalMPPI.lse_smoothing_rollout, Gaussian perturbations per sample "
                         "(0 = off)")
parser.add_argument("--lse-sigmas", type=float, nargs=2, default=[0.2, 0.01], metavar=("FIRST", "LAST"),
                    help="LSE perturbation scale at the first and last restart (geometric in between)")
parser.add_argument("--lse-lambda", type=float, default=0.1, help="LSE temperature")
parser.add_argument("--lse-chunk", type=int, default=20000, help="rollouts per GPU call for the perturbations")
parser.add_argument("--cold", action="store_true",
                    help="no KSOS: every refiner starts from zero knots (the plain baselines at --samples x --iters)")
parser.add_argument("--json", default="pusht_ksos_refine_sweep.json", help="records are merged into this file")
args = parser.parse_args()
for parent in Path(__file__).resolve().parents:
    if (parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
        os.environ.setdefault("CUKERNELSOS_PYTHON", str(parent / "cuKernelSOS" / "python"))
        break

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
ksos_ctrl = GlobalMPPI(task, num_samples=256, noise_level=0.1, temperature=0.1, spline_type="cubic",
                       plan_horizon=1.0, num_knots=6)  # only for its rollout
nu = task.model.nu
num_knots = ksos_ctrl.num_knots
dim = num_knots * nu
tk = jnp.linspace(0.0, ksos_ctrl.plan_horizon, num_knots) + state.time
u_min = np.tile(np.asarray(task.u_min), num_knots)
u_max = np.tile(np.asarray(task.u_max), num_knots)


@jax.jit
def _costs(knots):
    return ksos_ctrl.rollout(state, tk, knots).costs.sum(axis=-1)


def rollout(points):
    """Total rollout cost of flat knot vectors (m, dim), clipped into the box."""
    points = np.clip(np.atleast_2d(points), u_min, u_max)
    return np.asarray(_costs(jnp.asarray(points.reshape(-1, num_knots, nu), jnp.float32)), np.float64)


lse_sigmas = np.geomspace(*args.lse_sigmas, args.restarts)


def lse_costs(X, sig, rng):
    """GlobalMPPI.lse_smoothing_rollout: --lse-samples Gaussian perturbations of scale sig per sample,
    rolled out in fixed-size chunks; -lambda log mean exp(-J / lambda), the minimum subtracted."""
    L, c = args.lse_samples, args.lse_chunk
    P = (X[:, None, :] + sig * rng.standard_normal((len(X), L, X.shape[1]))).reshape(-1, X.shape[1])
    J = np.concatenate([rollout(np.pad(P[k:k + c], ((0, max(0, k + c - len(P))), (0, 0))))[: len(P) - k]
                        for k in range(0, len(P), c)]).reshape(len(X), L)
    m = J.min(axis=1)
    return m - args.lse_lambda * np.log(np.mean(np.exp(-(J - m[:, None]) / args.lse_lambda), axis=1))


def ksos(seed):
    """The KSOS solution after --restarts restarts: (knots, cost, seconds, trace)."""
    sample = jax.jit(TwoLevelSampler(args.n, dim, cluster_size=args.cluster_size, seed=seed).sample)
    key = jax.random.PRNGKey(seed)
    jax.block_until_ready(sample(key, jnp.asarray(u_min, jnp.float32), jnp.asarray(u_max, jnp.float32)))
    start = time.perf_counter()
    lse_rng = np.random.default_rng(seed)
    center, center_cost, trace = np.zeros(dim), np.inf, []
    for i in range(args.restarts):
        key, sub = jax.random.split(key)
        r = args.decay ** i
        lo = np.maximum(center - r * u_max, u_min) if i else u_min
        hi = np.minimum(center + r * u_max, u_max) if i else u_max
        points, sigma, _ = sample(sub, jnp.asarray(lo, jnp.float32), jnp.asarray(hi, jnp.float32))
        X, sigma = np.asarray(points, np.float64), float(sigma)
        f = lse_costs(X, lse_sigmas[i], lse_rng) if args.lse_samples else rollout(X)
        order = rcm_order(X, sigma)
        prob = build_problem(X[order], sigma, lambda _: f[order], lambd=args.lambd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, info = ADMM.solve_ksos(prob.Phi, prob.f, prob.lambd, samples=prob.samples,
                                      time_limit=args.admm_time_limit, record_history=False, backend="cuda",
                                      cuda_options={"anderson_reg": 1e-10})
        x = np.clip(np.asarray(x), u_min, u_max)
        ok = info["success"] and np.all(np.isfinite(x))
        x_cost = float(rollout(x)[0]) if ok else np.inf
        best = X[np.argmin(f)]  # with LSE, f is smoothed: judge the best sample by its own rollout
        best_cost = float(rollout(best)[0]) if args.lse_samples else float(f.min())
        for cand, c in ((x, x_cost), (best, best_cost)):
            if c < center_cost:
                center, center_cost = cand, c
        trace.append(round(center_cost, 3))
    return center, center_cost, time.perf_counter() - start, trace


def make_refiner(spec):
    kind, noise, temp = spec.split(":")
    common = dict(num_samples=args.samples, temperature=float(temp), spline_type="cubic", plan_horizon=1.0,
                  num_knots=num_knots, iterations=args.iters)
    if kind == "mppi":
        return MPPI(task, noise_level=float(noise), **common)
    if kind == "dial":
        return DIAL(task, noise_level=float(noise), beta_opt_iter=1.0, beta_horizon=1.0, **common)
    raise ValueError(spec)


rollout(np.zeros((args.n, dim)))
rollout(np.zeros((1, dim)))
if args.lse_samples:
    rollout(np.zeros((args.lse_chunk, dim)))
p0 = build_problem(np.random.default_rng(0).uniform(-1, 1, (20, dim)), 1.0, lambda _: np.zeros(20), lambd=args.lambd)
ADMM.solve_dual(p0.Phi, p0.f, p0.lambd, backend="cuda", max_iter=20)  # CUDA warm-up

starts = {}
for seed in args.seeds if not args.cold else []:
    starts[seed] = ksos(seed)
    print(f"KSOS seed {seed}: {' -> '.join(map(str, starts[seed][3]))}  ({starts[seed][2]:.2f} s)")

tag = "cold" if args.cold else f"KSOS x{args.restarts} (n {args.n}, decay {args.decay:g})" + (
    f" + LSE {args.lse_sigmas[0]:g}->{args.lse_sigmas[1]:g} L {args.lse_samples} lambda {args.lse_lambda:g}"
    if args.lse_samples else "")
records = []
for spec in args.configs:
    refiner = make_refiner(spec)
    optimize = jax.jit(refiner.optimize)
    jax.block_until_ready(optimize(state, refiner.init_params(seed=0)))  # compile
    costs = []
    for seed in args.seeds:
        center, center_cost, t_ksos, _ = starts[seed] if not args.cold else (np.zeros(dim), np.nan, 0.0, [])
        params0 = refiner.init_params(initial_knots=jnp.asarray(center.reshape(num_knots, nu), jnp.float32), seed=seed)
        t0 = time.perf_counter()
        params, _, _ = jax.block_until_ready(optimize(state, params0))
        t_ref = time.perf_counter() - t0
        cost = float(rollout(np.asarray(params.mean).reshape(1, -1))[0])
        costs.append(cost)
        records.append(dict(method=f"{tag} + {spec}", seed=seed, cost=cost, ksos_cost=center_cost,
                            time=t_ksos + t_ref, time_parts={"ksos": t_ksos, "refine": t_ref},
                            rollouts=(0 if args.cold else args.restarts * (args.n * max(args.lse_samples, 1) + 1 + bool(args.lse_samples))) + args.iters * (args.samples + 1)))
    print(f"{tag} + {spec:<16} " + " / ".join(f"{c:.2f}" for c in costs)
          + f"   mean {np.mean(costs):.2f} ± {np.std(costs, ddof=1):.2f}")

out = Path(args.json)
new = {(r["method"], r["seed"]) for r in records}
old = [r for r in json.loads(out.read_text()) if (r["method"], r["seed"]) not in new] if out.exists() else []
out.write_text(json.dumps(old + records, indent=1))
print("RESULT written", args.json)
