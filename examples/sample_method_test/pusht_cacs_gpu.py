"""Top-10%-seeded adaptive cost-aware chordal sampling (CACS) on PushT, GPU rollouts.

PushT version of cacs_top10_adaptive_benchmark.py (12-D knots in [-1, 1]^12):
1. Evaluate M0 Sobol points (one GPU batch); the lowest 10% are the elites.
2. Grow from the best elite in rounds. Each round proposes `batch` points a
   step of 0.75-0.995 h from accepted nodes (45% low-degree tips, 40% low-cost
   nodes, 15% random), 72% of the directions pulled toward an elite; checks the
   graph rules on the CPU BEFORE any rollout:
     - at least one kernel neighbour (distance < h): the kernel graph stays connected;
     - the two-hop neighbourhood T has < max_clique nodes;
     - T is already a clique of the chordal certificate H;
   rolls out all feasible proposals in ONE padded GPU call, then accepts up to
   `accept` of them, cheapest first, re-checking the rules against the points
   accepted earlier in the same round. (The original accepts 1 point per
   batch: n sequential GPU calls; this needs about n / accept.)
3. KSOS (Wendland kernel of radius h, GPU ADMM) on the accepted samples in
   reverse acceptance order (a perfect elimination order of H).
Baseline with the SAME number of rollouts: two-level Sobol + KSOS.
Every x is judged by its original cost (a fresh rollout).

Example:
    python pusht_cacs_gpu.py --m0 1000 --n 1000 --seed 0
"""

import argparse
import os
import time
import warnings
from collections import Counter
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

parser = argparse.ArgumentParser(description="adaptive cost-aware chordal sampling on PushT (GPU rollouts)")
parser.add_argument("--pusher", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--block", type=float, nargs=3, default=[0.1, 0.1, 1.3], metavar=("X", "Y", "THETA"))
parser.add_argument("--m0", type=int, default=1000, help="initial Sobol evaluations (top 10%% = elites)")
parser.add_argument("--n", type=int, default=1000, help="samples to accept (they go to KSOS)")
parser.add_argument("--h", type=float, default=1.2,
                    help="kernel radius (the 10-D unit-cube script's 0.569, scaled to [-1, 1]^12)")
parser.add_argument("--max-clique", type=int, default=20)
parser.add_argument("--batch", type=int, default=256, help="proposals per round (one GPU rollout call)")
parser.add_argument("--accept", type=int, default=16, help="most points accepted per round")
parser.add_argument("--lse-samples", type=int, default=0,
                    help="LSE perturbations per candidate (0: plain rollout cost). The search (elites, "
                         "acceptance, low-cost anchors) then uses the LSE cost; KSOS uses the original cost")
parser.add_argument("--lse-sigma0", type=float, default=0.1, help="LSE scale at the start")
parser.add_argument("--lse-sigma-end", type=float, default=0.001,
                    help="LSE scale once n samples are accepted (geometric in the accepted fraction)")
parser.add_argument("--lse-lambda", type=float, default=0.1, help="LSE temperature")
parser.add_argument("--lse-chunk", type=int, default=64, help="candidates per LSE GPU call")
parser.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True,
                    help="also run two-level Sobol + KSOS with the same rollouts (--no-baseline: skip)")
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
D = ctrl.num_knots * nu
tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots) + state.time
LO = np.tile(np.asarray(task.u_min), ctrl.num_knots)
HI = np.tile(np.asarray(task.u_max), ctrl.num_knots)


@jax.jit
def _costs(knots):
    return ctrl.rollout(state, tk, knots).costs.sum(axis=-1)


def rollout(points, pad_to=None):
    """Total rollout cost of flat knot vectors (m, D) on the GPU. pad_to keeps the
    batch shape fixed across calls (no XLA recompile per round)."""
    points = np.clip(np.atleast_2d(points), LO, HI)
    m = len(points)
    if pad_to is not None and m < pad_to:
        points = np.vstack([points, np.zeros((pad_to - m, D))])
    c = _costs(jnp.asarray(points.reshape(-1, ctrl.num_knots, nu), jnp.float32))
    return np.asarray(c, np.float64)[:m]


@jax.jit
def _lse(knots, key, scale):
    """Soft minimum (temperature lse_lambda) of the costs of lse_samples perturbations
    of scale `scale`, shared by every candidate of the call (common random numbers),
    clipped into the box: (B, num_knots, nu) -> (B,)."""
    L = args.lse_samples
    noise = jax.random.normal(key, (1, L, *knots.shape[1:]))
    lo = jnp.asarray(LO.reshape(ctrl.num_knots, nu), jnp.float32)
    hi = jnp.asarray(HI.reshape(ctrl.num_knots, nu), jnp.float32)
    perturbed = jnp.clip(knots[:, None] + scale * noise, lo, hi).reshape(-1, *knots.shape[1:])
    c = _costs(perturbed).reshape(knots.shape[0], L)
    m = jnp.min(c, axis=1, keepdims=True)
    return (-args.lse_lambda * jnp.log(jnp.mean(jnp.exp(-(c - m) / args.lse_lambda), axis=1, keepdims=True)) + m)[:, 0]


def evaluate(points, scale, key):
    """(search cost, original cost, rollouts) of flat knot vectors. The search cost is
    the LSE cost at this scale when --lse-samples > 0, else the original cost."""
    f = rollout(points, pad_to=args.batch if len(points) <= args.batch else None)
    if not args.lse_samples:
        return f, f, len(points)
    B = args.lse_chunk
    out = []
    for i in range(0, len(points), B):
        chunk = np.clip(points[i: i + B], LO, HI)
        m = len(chunk)
        chunk = np.vstack([chunk, np.zeros((B - m, D))])  # fixed shape: no recompile
        g = _lse(jnp.asarray(chunk.reshape(-1, ctrl.num_knots, nu), jnp.float32), jax.random.fold_in(key, i), scale)
        out.append(np.asarray(g, np.float64)[:m])
    return np.concatenate(out), f, len(points) * (1 + args.lse_samples)


def lse_scale(progress):
    """Geometric from lse_sigma0 (progress 0) to lse_sigma_end (progress 1)."""
    return args.lse_sigma0 * (args.lse_sigma_end / args.lse_sigma0) ** min(1.0, progress)


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


# ---------------------------------------------------------------- adaptive CACS
def propose(X, cost, G, elites, rng):
    """`batch` proposals a step of 0.75-0.995 h from accepted nodes (tips / low cost /
    random), 72% of the directions pulled toward an elite."""
    t, h, b = len(X), args.h, args.batch
    deg = np.fromiter((len(s) for s in G), dtype=int)
    low = np.argsort(cost)[: max(1, int(0.35 * t))]
    tips = np.flatnonzero(deg <= 4)
    if not len(tips):
        tips = np.arange(t)
    r = rng.random(b)
    anchor = np.where(r < 0.45, tips[rng.integers(len(tips), size=b)],
                      np.where(r < 0.85, low[rng.integers(len(low), size=b)], rng.integers(t, size=b)))
    direction = rng.normal(size=(b, D))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    targets = elites[rng.integers(min(len(elites), max(10, len(elites) // 3)), size=b)]
    seek = targets - X[anchor]
    seek /= np.maximum(np.linalg.norm(seek, axis=1, keepdims=True), 1e-15)
    mix = 0.84 * seek + 0.54 * direction
    mix /= np.maximum(np.linalg.norm(mix, axis=1, keepdims=True), 1e-15)
    use = rng.random(b) < 0.72
    direction[use] = mix[use]
    step = h * rng.uniform(0.75, 0.995, size=b)
    return np.clip(X[anchor] + step[:, None] * direction, LO, HI)


def check(S, G, H, rejected):
    """The admission rules for a point whose kernel neighbours are S; returns T or None."""
    if not S:
        rejected["no_neighbor"] += 1
        return None
    T = set(S)
    for j in S:
        T.update(G[j])
    if len(T) >= args.max_clique:
        rejected["oversize"] += 1
        return None
    if any(not (T - {j}).issubset(H[j]) for j in T):
        rejected["not_clique"] += 1
        return None
    return T


def sample_adaptive(rng):
    """The accepted samples (n, D), their costs, the rollout count and timings."""
    t0 = time.perf_counter()
    init = qmc.Sobol(D, scramble=True, seed=args.seed).random(args.m0) * (HI - LO) + LO
    key = jax.random.PRNGKey(1000 + args.seed)
    c0, o0, evals = evaluate(init, lse_scale(0.0), jax.random.fold_in(key, 0))
    order0 = np.argsort(c0)
    elites = init[order0[: int(np.ceil(0.1 * args.m0))]]
    X, cost, orig = [elites[0].copy()], [float(c0[order0[0]])], [float(o0[order0[0]])]
    G, H = [set()], [set()]
    rejected, rounds = Counter(), 0
    h2 = args.h**2
    while len(X) < args.n:
        rounds += 1
        Xa = np.asarray(X)
        trial = propose(Xa, np.asarray(cost), G, elites, rng)
        # the graph rules first (CPU), against the samples accepted so far
        nbrs = cKDTree(Xa).query_ball_point(trial, r=args.h - 1e-11)
        feasible = [i for i, S in enumerate(nbrs) if check(set(S), G, H, rejected) is not None]
        if not feasible:
            if rounds > 50 * args.n:
                raise RuntimeError(f"no feasible proposal at n={len(X)}: {dict(rejected)}")
            continue
        # one padded GPU call (plus the LSE calls) for every feasible proposal
        cc, oc, used = evaluate(trial[feasible], lse_scale(len(X) / args.n), jax.random.fold_in(key, rounds))
        evals += used
        accepted = 0
        for k in np.argsort(cc):  # cheapest first; re-check against this round's acceptances
            p = trial[feasible[k]]
            S = set(np.flatnonzero(np.sum((np.asarray(X) - p) ** 2, axis=1) < h2).tolist())
            T = check(S, G, H, Counter())
            if T is None:
                continue
            i = len(X)
            X.append(p.copy())
            cost.append(float(cc[k]))
            orig.append(float(oc[k]))
            G.append(set(S))
            H.append(set(T))
            for j in S:
                G[j].add(i)
            for j in T:
                H[j].add(i)
            accepted += 1
            if accepted >= args.accept or len(X) >= args.n:
                break
    return {"X": np.asarray(X), "cost": np.asarray(orig), "search_cost": np.asarray(cost), "evals": evals,
            "rounds": rounds, "rejected": dict(rejected), "seconds": time.perf_counter() - t0,
            "init_best": float(np.min(o0))}


rng = np.random.default_rng(args.seed)
rollout(np.zeros((args.batch, D)))  # compile the round batch shape
out = sample_adaptive(rng)
Xc, fc = out["X"], out["cost"]
x, info, t_admm = ksos(Xc, fc, args.h, np.arange(len(Xc))[::-1])  # reverse acceptance: a PEO of H
x_cost = rollout(x)[0]
name = f"adaptive CACS{' + LSE' if args.lse_samples else ''}"
rows = [(name, out["evals"], len(Xc), info["admm_stats"]["max_clique"], fc.min(), x_cost,
         info["status"], info["iterations"], t_admm, out["seconds"])]

# ---------------------------------------------------------------- baseline: two-level Sobol, same rollouts
# (capped: with LSE the budget is ~100x larger than a two-level KSOS problem should be)
if args.baseline:
    budget = min(int(round(out["evals"] / 10)) * 10, 20000)
    sampler = TwoLevelSampler(budget, D, cluster_size=10, seed=args.seed)
    t0 = time.perf_counter()
    points, sigma, _ = jax.jit(sampler.sample)(jax.random.PRNGKey(args.seed), jnp.asarray(LO, jnp.float32),
                                               jnp.asarray(HI, jnp.float32))
    X2 = np.asarray(points, np.float64)
    f2 = rollout(X2)
    t_two = time.perf_counter() - t0
    x2, info2, t2 = ksos(X2, f2, float(sigma), rcm_order(X2, float(sigma)))
    rows.append(("two-level Sobol", budget, budget, info2["admm_stats"]["max_clique"], f2.min(), rollout(x2)[0],
                 info2["status"], info2["iterations"], t2, t_two))

print(f"seed {args.seed}: m0={args.m0} (best initial, original cost {out['init_best']:.3f}), n={args.n}, h={args.h}, "
      f"LSE {'off' if not args.lse_samples else f'L={args.lse_samples}, scale {args.lse_sigma0}->{args.lse_sigma_end}, lam={args.lse_lambda}'}, "
      f"max clique {args.max_clique}, {out['rounds']} rounds, rejected {out['rejected']}")
print(f"{'method':<16} {'rollouts':>8} {'to KSOS':>8} {'max clique':>10} {'best sample':>11} {'x (orig)':>9} "
      f"{'status':<15} {'iters':>6} {'admm':>7} {'sampling':>9}")
for name, r, m, mc, fb, fx, st, it, ta, ts in rows:
    print(f"{name:<16} {r:>8} {m:>8} {mc:>10} {fb:>11.4f} {fx:>9.4f} {st:<15} {it:>6} {ta:>6.2f}s {ts:>8.2f}s")
print(f"RESULT seed={args.seed} m0={args.m0}: " + " | ".join(
    f"{r[0]}: rollouts {r[1]} best {r[4]:.4f} x {r[5]:.4f} clique {r[3]} admm {r[8]:.2f}s sampling {r[9]:.2f}s"
    for r in rows))
