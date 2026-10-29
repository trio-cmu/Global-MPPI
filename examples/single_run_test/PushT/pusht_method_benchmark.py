"""One planning step on PushT from one start state: KSOS (n samples) vs PS, MPPI and DIAL.

Every method starts cold (mean knots = 0) from the same state and plans once.
  PS / MPPI / DIAL  the controllers of examples/pusht.py with the same hyperparameters
                    (256 samples x 5 iterations); one jitted ctrl.optimize call; the plan
                    is the final mean, judged by a fresh rollout
  KSOS              --restarts rounds, as GlobalMPPI: restart i draws n two-level Sobol
                    samples on the GPU (n/10 clusters of 10) in the box of half-width 0.8^i
                    around the last solution (the whole box at i = 0), one rollout each,
                    RCM order, Wendland kernel -> Phi, ADMM-cuda (Anderson on); the
                    solution is the cheaper of x = sum alpha_i x_i (a fresh rollout) and
                    the best sample so far
  KSOS + MPPI       then --refine-iters MPPI iterations from that solution (GlobalMPPI's
                    MPPI settings: noise 0.1, temperature 0.1, --samples samples)
Times exclude JIT compilation (every function is called once before timing).
Records go to --json; plot_method_benchmark.py draws them.

Example:
    python pusht_method_benchmark.py --n 1000 --seeds 0 1 2 3 4
    python pusht_method_benchmark.py --samples 1000 --restarts 5 --refine-iters 5
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

from global_mppi.algs import DIAL, MPPI, GlobalMPPI, PredictiveSampling
from global_mppi.sampling import TwoLevelSampler
from global_mppi.tasks.pusht import PushT
from ksos_tools.problems.features import build_problem
from ksos_tools.problems.sampling import rcm_order
from ksos_tools.solvers import ADMM

parser = argparse.ArgumentParser(description="KSOS vs PS / MPPI / DIAL, one planning step on PushT")
parser.add_argument("--pusher", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--block", type=float, nargs=3, default=[0.1, 0.1, 1.3], metavar=("X", "Y", "THETA"))
parser.add_argument("--n", type=int, default=1000, help="KSOS samples per restart")
parser.add_argument("--samples", type=int, default=256, help="PS / MPPI / DIAL samples per iteration")
parser.add_argument("--restarts", type=int, default=1,
                    help="KSOS restarts; restart i samples a box of half-width 0.8^i around the last solution")
parser.add_argument("--refine-iters", type=int, default=0,
                    help="MPPI iterations after KSOS (GlobalMPPI's settings: noise 0.1, temperature 0.1, "
                         "--samples samples), started at the KSOS solution")
parser.add_argument("--cluster-size", type=int, default=10)
parser.add_argument("--lambd", type=float, default=1e-5)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--repeats", type=int, default=3, help="timed runs per method and seed (median)")
parser.add_argument("--refine-noise", type=float, default=0.1, help="noise level of the MPPI refinement")
parser.add_argument("--lse-samples", type=int, default=0,
                    help="KSOS: LSE smoothing as GlobalMPPI.lse_smoothing_rollout, this many Gaussian perturbations "
                         "per sample (0 = off, plain costs)")
parser.add_argument("--lse-sigmas", type=float, nargs=2, default=[0.1, 0.01], metavar=("FIRST", "LAST"),
                    help="LSE perturbation scale at the first and last restart (geometric in between)")
parser.add_argument("--lse-lambda", type=float, default=0.1, help="LSE temperature (GlobalMPPI's lse_lambda)")
parser.add_argument("--lse-chunk", type=int, default=20000, help="rollouts per GPU call for the LSE perturbations")
parser.add_argument("--methods", nargs="+", default=["PS", "MPPI", "DIAL", "KSOS"],
                    choices=["PS", "MPPI", "DIAL", "KSOS"], help="methods to run (KSOS includes its MPPI refinement)")
parser.add_argument("--json", default="pusht_method_benchmark.json",
                    help="records are merged into this file (same method and seed replaced)")
args = parser.parse_args()
# cuKernelSOS's Python binding (GPU ADMM): the first cuKernelSOS/python in a parent folder
for parent in Path(__file__).resolve().parents:
    if (parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
        os.environ.setdefault("CUKERNELSOS_PYTHON", str(parent / "cuKernelSOS" / "python"))
        break

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

CONTROLLERS = {
    "PS": PredictiveSampling(task, num_samples=args.samples, noise_level=0.4, num_randomizations=1, plan_horizon=1.0,
                             spline_type="cubic", num_knots=6, iterations=5),
    "MPPI": MPPI(task, num_samples=args.samples, noise_level=1.0, temperature=0.1, spline_type="cubic", plan_horizon=1.0,
                 num_knots=6, iterations=5),
    "DIAL": DIAL(task, num_samples=args.samples, noise_level=1.0, beta_opt_iter=1.0, beta_horizon=1.0, temperature=0.01,
                 plan_horizon=1.0, spline_type="cubic", num_knots=6, iterations=5),
}
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


def timed(fn, repeats):
    """(result of the last call, median seconds)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return result, float(np.median(times))


records = []
# ---------------------------------------------------------------- PS / MPPI / DIAL
for name, ctrl in CONTROLLERS.items():
    if name not in args.methods:
        continue
    optimize = jax.jit(ctrl.optimize)
    jax.block_until_ready(optimize(state, ctrl.init_params(seed=0)))  # compile
    for seed in args.seeds:
        params0 = ctrl.init_params(seed=seed)
        (params, rollouts, _), t = timed(lambda: optimize(state, params0), args.repeats)
        plan_cost = float(rollout(np.asarray(params.mean).reshape(1, -1))[0])
        last_best = float(jnp.min(rollouts.costs.sum(axis=-1)))
        rollouts_used = ctrl.num_samples * ctrl.iterations + ctrl.iterations  # samples + one mean rollout per iteration
        records.append(dict(method=name, seed=seed, cost=plan_cost, best_sample=last_best, time=t,
                            rollouts=rollouts_used))
        print(f"{name:<6} seed {seed}: plan cost {plan_cost:8.4f}  best sample of the last iteration "
              f"{last_best:8.4f}  time {t * 1e3:7.1f} ms  rollouts {rollouts_used}")

# ---------------------------------------------------------------- KSOS (restarts) + MPPI refinement
refine = (MPPI(task, num_samples=args.samples, noise_level=args.refine_noise, temperature=0.1, spline_type="cubic",
               plan_horizon=1.0, num_knots=6, iterations=args.refine_iters) if args.refine_iters else None)
refine_opt = jax.jit(refine.optimize) if refine else None
ksos_name = ("KSOS" if args.restarts == 1 else f"KSOS x{args.restarts} restarts") + (
    f" + LSE {args.lse_sigmas[0]:g}->{args.lse_sigmas[1]:g}" if args.lse_samples else "")
lse_sigmas = np.geomspace(*args.lse_sigmas, args.restarts) if args.restarts > 1 else np.array(args.lse_sigmas[:1])


def lse_costs(X, sig, rng):
    """GlobalMPPI.lse_smoothing_rollout: for each sample, --lse-samples Gaussian perturbations of scale sig
    (clipped into the box), rolled out in fixed-size chunks; the smoothed cost
    -lambda log mean exp(-J / lambda), with the minimum subtracted for stability."""
    L, c = args.lse_samples, args.lse_chunk
    P = (X[:, None, :] + sig * rng.standard_normal((len(X), L, X.shape[1]))).reshape(-1, X.shape[1])
    J = np.concatenate([rollout(np.pad(P[k:k + c], ((0, max(0, k + c - len(P))), (0, 0))))[: len(P) - k]
                        for k in range(0, len(P), c)]).reshape(len(X), L)
    m = J.min(axis=1)
    return m - args.lse_lambda * np.log(np.mean(np.exp(-(J - m[:, None]) / args.lse_lambda), axis=1))
warm = None
for seed in args.seeds if "KSOS" in args.methods else []:
    sampler = TwoLevelSampler(args.n, dim, cluster_size=args.cluster_size, seed=seed)
    sample = jax.jit(sampler.sample)
    jax.block_until_ready(sample(jax.random.PRNGKey(seed), jnp.asarray(u_min, jnp.float32),
                                 jnp.asarray(u_max, jnp.float32)))  # compile
    rollout(np.zeros((args.n, dim)))
    rollout(np.zeros((1, dim)))
    if args.lse_samples:
        rollout(np.zeros((args.lse_chunk, dim)))
    if refine:
        jax.block_until_ready(refine_opt(state, refine.init_params(seed=seed)))
    if warm is None:  # CUDA context and kernels, untimed
        p0 = build_problem(np.random.default_rng(0).uniform(-1, 1, (20, dim)), 1.0, lambda _: np.zeros(20),
                           lambd=args.lambd)
        ADMM.solve_dual(p0.Phi, p0.f, p0.lambd, backend="cuda", max_iter=20)
        warm = True

    def ksos_step():
        t = dict.fromkeys(["sampling", "rollout", "rcm+kernel", "admm", "x rollout", "mppi"], 0.0)
        key = jax.random.PRNGKey(seed)
        lse_rng = np.random.default_rng(seed)
        center, center_cost, trace = np.zeros(dim), np.inf, []
        for i in range(args.restarts):
            key, sub = jax.random.split(key)
            r = 0.8 ** i
            lo = np.maximum(center - r * u_max, u_min) if i else u_min
            hi = np.minimum(center + r * u_max, u_max) if i else u_max
            t0 = time.perf_counter()
            points, sigma, _ = jax.block_until_ready(
                sample(sub, jnp.asarray(lo, jnp.float32), jnp.asarray(hi, jnp.float32)))
            t["sampling"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            X, sigma = np.asarray(points, np.float64), float(sigma)
            f = lse_costs(X, lse_sigmas[i], lse_rng) if args.lse_samples else rollout(X)
            t["rollout"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            order = rcm_order(X, sigma)
            prob = build_problem(X[order], sigma, lambda _: f[order], lambd=args.lambd)
            t["rcm+kernel"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, info = ADMM.solve_ksos(prob.Phi, prob.f, prob.lambd, samples=prob.samples, time_limit=60.0,
                                          record_history=False, backend="cuda", cuda_options={"anderson_reg": 1e-10})
            t["admm"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            x = np.clip(np.asarray(x), u_min, u_max)
            # a diverged solve can return a non-finite x; then only the best sample counts
            x_cost = float(rollout(x)[0]) if np.all(np.isfinite(x)) else np.inf
            # with LSE, f is the smoothed cost: judge the best sample by its own rollout too
            best = X[np.argmin(f)]
            best_cost = float(rollout(best)[0]) if args.lse_samples else float(f.min())
            t["x rollout"] += time.perf_counter() - t0
            # keep the cheapest (original cost) of x, this restart's best sample and the previous solution
            for cand, c in ((x, x_cost), (best, best_cost)):
                if c < center_cost:
                    center, center_cost = cand, c
            trace.append(dict(best_sample=best_cost, x_cost=x_cost, solution=center_cost,
                              status=info["status"], iterations=int(info["iterations"])))
        out = dict(ksos_cost=center_cost, trace=trace)
        if refine:
            t0 = time.perf_counter()
            params, _, _ = jax.block_until_ready(
                refine_opt(state, refine.init_params(initial_knots=jnp.asarray(center.reshape(num_knots, nu),
                                                                               jnp.float32), seed=seed)))
            t["mppi"] += time.perf_counter() - t0
            out["refined_cost"] = float(rollout(np.asarray(params.mean).reshape(1, -1))[0])
        out["t"] = t
        return out

    runs = [ksos_step() for _ in range(args.repeats)]
    r = runs[-1]
    t_parts = {k: float(np.median([q["t"][k] for q in runs])) for k in r["t"]}
    t_ksos = sum(v for k, v in t_parts.items() if k != "mppi")
    ksos_rollouts = args.restarts * (args.n * max(args.lse_samples, 1) + 1 + bool(args.lse_samples))
    records.append(dict(method=ksos_name, seed=seed, cost=r["ksos_cost"], best_sample=min(q["best_sample"] for q in r["trace"]),
                        time=t_ksos, rollouts=ksos_rollouts, trace=r["trace"],
                        time_parts={k: v for k, v in t_parts.items() if k != "mppi"}))
    line = (f"{ksos_name} seed {seed}: " + " -> ".join(f"{q['solution']:.2f}" for q in r["trace"])
            + f"  time {t_ksos * 1e3:.1f} ms")
    if refine:
        records.append(dict(method=f"{ksos_name} + MPPI x{args.refine_iters} (noise {args.refine_noise:g})", seed=seed, cost=r["refined_cost"],
                            time=t_ksos + t_parts["mppi"], time_parts=t_parts,
                            rollouts=ksos_rollouts + args.refine_iters * (args.samples + 1)))
        line += f"  | + MPPI: {r['refined_cost']:.4f}  time {(t_ksos + t_parts['mppi']) * 1e3:.1f} ms"
    print(line + "  (" + ", ".join(f"{q['status']} {q['iterations']}" for q in r["trace"]) + ")")

out = Path(args.json)
new = {(r["method"], r["seed"]) for r in records}
old = [r for r in json.loads(out.read_text()) if (r["method"], r["seed"]) not in new] if out.exists() else []
out.write_text(json.dumps(old + records, indent=1))
print("RESULT written", args.json)
