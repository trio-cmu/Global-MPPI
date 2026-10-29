"""Budgeted global cost-aware chordal sampling (CACS) on PushT, GPU rollouts.

PushT version of cacs_budgeted_global_improved.py (12-D knots in [-1, 1]^12):
1. Roll out `initial` Sobol points (one GPU batch); seed the graph with up to
   `reuse` of them (the lowest-cost 10% first, then low cost far from the
   seeds), each kept only if the graph rules allow it.
2. Grow in rounds. Each round proposes `batch` points: 25% fresh global Sobol
   points, the rest a step of 0.24-1.03 h from a low-cost anchor / low-degree
   tip / cost-weighted node (20% pulled toward an elite). Feasible proposals
   are ranked by a cheap proxy -- the cost of the nearest accepted sample plus
   0.35 x its distance -- and the best `k` are picked one by one, re-checking
   the graph rules after each pick (they do not depend on the cost):
     - local points need a kernel neighbour (distance < h);
     - the two-hop neighbourhood T has < max_clique nodes;
     - T is already a clique of the chordal certificate H.
   The k picks are rolled out in ONE padded GPU call and all added (the
   original evaluates 1 point per step: n sequential calls). Every
   `global_every` accepted samples one isolated global point is allowed in.
3. Replacement moves (sequential, `replace` of them): drop a sample from the
   costliest 30%, propose in the graph without it, roll out the best-ranked
   feasible proposal, keep it if it is cheaper than the dropped sample.
4. KSOS (Wendland kernel of radius h, GPU ADMM) on the samples in reverse
   insertion order (a perfect elimination order of H).
Baselines with the SAME number of rollouts: two-level Sobol + KSOS, and the
best sample of plain Sobol. Every x is judged by a fresh rollout.

Example:
    python pusht_cacs_budgeted_gpu.py --n 1000 --seed 0
"""

import argparse
import os
import time
import warnings
from collections import Counter
from contextlib import contextmanager
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

parser = argparse.ArgumentParser(description="budgeted global CACS on PushT (GPU rollouts)")
parser.add_argument("--pusher", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--block", type=float, nargs=3, default=[0.1, 0.1, 1.3], metavar=("X", "Y", "THETA"))
parser.add_argument("--n", type=int, default=1000, help="samples to keep (they go to KSOS)")
parser.add_argument("--initial", type=int, default=200, help="initial Sobol rollouts")
parser.add_argument("--reuse", type=int, default=40, help="initial points used as graph seeds")
parser.add_argument("--h", type=float, default=1.2, help="kernel radius (0.569 of the 10-D unit cube, scaled)")
parser.add_argument("--max-clique", type=int, default=20)
parser.add_argument("--batch", type=int, default=96, help="proposals per round")
parser.add_argument("--k", type=int, default=16, help="points picked and rolled out per round (one GPU call)")
parser.add_argument("--global-every", type=int, default=125, help="one isolated global point per this many samples")
parser.add_argument("--replace", type=int, default=40, help="replacement moves after the n samples")
parser.add_argument("--close-weight", type=float, default=0.0, help="PushT close_to_block_weight")
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
task = PushT(close_to_block_weight=args.close_weight)
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


TIME = Counter()  # seconds per step (JIT compiles are done before the timed run)


@contextmanager
def timed(name):
    t0 = time.perf_counter()
    yield
    TIME[name] += time.perf_counter() - t0


class Rollouts:
    """GPU rollout costs of flat knot vectors, counting every rollout. pad_to keeps a
    batch shape fixed across calls (no XLA recompile per round)."""

    def __init__(self):
        self.count = 0

    def __call__(self, points, pad_to=None):
        points = np.clip(np.atleast_2d(points), LO, HI)
        m = len(points)
        self.count += m
        if pad_to is not None and m < pad_to:
            points = np.vstack([points, np.zeros((pad_to - m, D))])
        c = _costs(jnp.asarray(points.reshape(-1, ctrl.num_knots, nu), jnp.float32))
        return np.asarray(c, np.float64)[:m]


def ksos(samples, f, sigma, order, tag=""):
    """Wendland kernel -> ADMM (GPU) on the samples in this order; (x, info, seconds)."""
    with timed(f"{tag}KSOS: Wendland kernel + Cholesky"):
        problem = build_problem(samples[order], sigma, lambda _: f[order], lambd=args.lambd)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, info = ADMM.solve_ksos(
            problem.Phi, problem.f, problem.lambd, samples=problem.samples,
            time_limit=args.time_limit, record_history=False, backend="cuda",
            cuda_options={"anderson_reg": args.anderson_reg},
        )
    TIME[f"{tag}KSOS: ADMM setup (cliques, operators)"] += info["setup_time"]
    TIME[f"{tag}KSOS: ADMM iterations (GPU)"] += info["solve_time"]
    TIME[f"{tag}KSOS: ADMM check + repair"] += time.perf_counter() - t0 - info["setup_time"] - info["solve_time"]
    return np.asarray(x), info, time.perf_counter() - t0


# ---------------------------------------------------------------- the certified sample graph
class Graph:
    """G: the kernel graph (pairs closer than h); H: its chordal certificate (a
    supergraph of G^2 built from simplicial insertions)."""

    def __init__(self):
        self.X, self.cost, self.G, self.H = [], [], [], []

    def neighbours(self, y):
        if not self.X:
            return set()
        d2 = np.sum((np.asarray(self.X) - y) ** 2, axis=1)
        return set(np.flatnonzero(d2 < (args.h - 1e-10) ** 2).tolist())

    def rule(self, S, allow_isolated):
        """(S, T) if a point with kernel neighbours S may join, else None."""
        if not self.X:
            return set(), set()
        if not S and not allow_isolated:
            return None
        T = set(S)
        for j in S:
            T.update(self.G[j])
        if len(T) >= args.max_clique:
            return None
        if any(not (T - {j}).issubset(self.H[j]) for j in T):
            return None
        return S, T

    def batch_rule(self, Y, allow_isolated):
        """[(index into Y, (S, T))] for the proposals the rules allow."""
        if not self.X:
            return [(i, (set(), set())) for i in range(len(Y))]
        nb = cKDTree(np.asarray(self.X)).query_ball_point(Y, args.h - 1e-10)
        out = []
        for i, S in enumerate(nb):
            st = self.rule(set(S), allow_isolated)
            if st is not None:
                out.append((i, st))
        return out

    def append(self, y, cost, st):
        S, T = st
        i = len(self.X)
        self.X.append(np.array(y, dtype=float))
        self.cost.append(float(cost))
        self.G.append(set(S))
        self.H.append(set(T))
        for j in S:
            self.G[j].add(i)
        for j in T:
            self.H[j].add(i)

    def without(self, i):
        """The graph with sample i removed (an induced subgraph: still chordal, and the
        insertion order stays a perfect elimination order in reverse)."""
        g = Graph()
        keep = [k for k in range(len(self.X)) if k != i]
        new = {old: j for j, old in enumerate(keep)}
        g.X = [self.X[k] for k in keep]
        g.cost = [self.cost[k] for k in keep]
        g.G = [{new[j] for j in self.G[k] if j != i} for k in keep]
        g.H = [{new[j] for j in self.H[k] if j != i} for k in keep]
        return g


def propose(g, X0, c0, rng, sobol):
    """`batch` proposals: 25% fresh global Sobol points, the rest local steps."""
    b = args.batch
    X, cc = np.asarray(g.X), np.asarray(g.cost)
    deg = np.asarray([len(z) for z in g.G])
    anchors = X0[np.argsort(c0)[: max(8, int(0.1 * len(X0)))]]
    n_global = int(0.25 * b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sobol balance warning: fine for a proposal pool
        Y = np.empty((b, D))
        Y[:n_global] = sobol.random(n_global) * (HI - LO) + LO
    # cost scale: PushT costs are O(10), the 10-D test function's O(1)
    w = np.exp(-3.0 * (cc - cc.min()) / max(np.std(cc), 1e-9)) / np.sqrt(1 + deg)
    w /= w.sum()
    tips = np.flatnonzero(deg <= 3)
    for k in range(n_global, b):
        mode = rng.random()
        if mode < 0.20:
            base = anchors[rng.integers(len(anchors))]
        elif mode < 0.55 and len(tips):
            base = X[rng.choice(tips)]
        else:
            base = X[rng.choice(len(X), p=w)]
        dr = rng.normal(size=D)
        dr /= np.linalg.norm(dr)
        if rng.random() < 0.20:
            to = anchors[rng.integers(len(anchors))] - base
            if np.linalg.norm(to) > 1e-9:
                dr = 0.75 * to / np.linalg.norm(to) + 0.48 * dr
                dr /= np.linalg.norm(dr)
        Y[k] = np.clip(base + args.h * rng.uniform(0.24, 1.03) * dr, LO, HI)
    return Y, n_global


def proxy(g, pts):
    """Cheap score, no rollout: the cost of the nearest accepted sample + 0.35 x distance
    (distance in units of the 10-D unit cube, so 0.35 keeps its original weight)."""
    dist, near = cKDTree(np.asarray(g.X)).query(pts, k=1)
    return np.asarray(g.cost)[near] + 0.35 * dist * np.std(g.cost) / 0.5


def sample(rng, roll):
    t0 = time.perf_counter()
    with timed("1 initial Sobol points"):
        X0 = qmc.Sobol(D, scramble=True, seed=args.seed).random(args.initial) * (HI - LO) + LO
    with timed("2 rollout: initial (GPU)"):
        c0 = roll(X0)
    t_seed = time.perf_counter()
    g = Graph()
    # seeds: the lowest-cost 10% first, then low cost away from the seeds
    order = np.argsort(c0)
    used = set()
    for i in order[: max(1, int(np.ceil(0.1 * len(X0))))]:
        st = g.rule(g.neighbours(X0[i]), allow_isolated=True)
        if st is not None:
            g.append(X0[i], c0[i], st)
            used.add(int(i))
        if len(used) >= args.reuse:
            break
    while len(used) < args.reuse:
        rest = np.asarray([j for j in order if int(j) not in used])
        if not len(rest):
            break
        dist = cKDTree(np.asarray(g.X)).query(X0[rest], k=1)[0]
        score = c0[rest] + 0.12 * np.std(c0) / 0.5 * np.exp(-0.5 * (dist / (0.65 * args.h / 0.569)) ** 2)
        added = False
        for i in rest[np.argsort(score)[:50]]:
            st = g.rule(g.neighbours(X0[i]), allow_isolated=True)
            if st is not None:
                g.append(X0[i], c0[i], st)
                used.add(int(i))
                added = True
                break
        if not added:
            break
    n_seeds = len(g.X)
    TIME["3 graph seeds"] += time.perf_counter() - t_seed
    sobol = qmc.Sobol(D, scramble=True, seed=args.seed + 50010)
    stats, rounds, stalled = Counter(), 0, 0
    while len(g.X) < args.n:
        rounds += 1
        with timed("4 rounds: propose"):
            Y, n_global = propose(g, X0, c0, rng, sobol)
        # an isolated global point is allowed once every global_every samples
        due = (len(g.X) - n_seeds) // args.global_every > stats["global"]
        with timed("5 rounds: graph rules (k-d tree, two-hop, clique)"):
            feasible = g.batch_rule(Y[n_global:], allow_isolated=False)
            feasible = [(n_global + i, st) for i, st in feasible]
            iso = [(i, st) for i, st in g.batch_rule(Y[:n_global], allow_isolated=True) if not st[0]] if due else []
        stats["proposed"] += len(Y)
        if not feasible and not iso:
            stalled += 1
            if stalled >= 100:
                break
            continue
        stalled = 0
        with timed("6 rounds: proxy ranking"):
            cand = iso[:1] + [feasible[j] for j in np.argsort(proxy(g, Y[[i for i, _ in feasible]]))] if feasible else iso[:1]
        t_pick = time.perf_counter()
        # pick up to k, re-checking the rules against the earlier picks of this round
        picks = []
        for idx, (i, st) in enumerate(cand):
            if len(picks) >= args.k or len(g.X) + len(picks) >= args.n:
                break
            if picks:  # the graph changed: recompute S, T with the picks inserted (cost unknown yet)
                S = g.neighbours(Y[i])
                st = g.rule(S, allow_isolated=bool(iso) and idx == 0)
                if st is None:
                    continue
            g.append(Y[i], np.nan, st)  # cost filled in after the batched rollout
            picks.append(len(g.X) - 1)
            if iso and idx == 0:
                stats["global"] += 1
        TIME["7 rounds: pick k + re-check + insert"] += time.perf_counter() - t_pick
        with timed("8 rollout: rounds (GPU, k per call)"):
            costs = roll(np.asarray([g.X[p] for p in picks]), pad_to=args.k)
        for p, c in zip(picks, costs):
            g.cost[p] = float(c)
        stats["picked"] += len(picks)
    # replacement moves: drop a costly sample, roll out the best-ranked feasible proposal
    replaced = 0
    for _ in range(args.replace):
        t_rep = time.perf_counter()
        cc = np.asarray(g.cost)
        high = np.argsort(cc)[int(0.70 * len(cc)):]
        old = int(high[rng.integers(len(high))])
        gm = g.without(old)
        Y, _ = propose(gm, X0, c0, rng, sobol)
        viable = gm.batch_rule(Y, allow_isolated=True)
        if not viable:
            TIME["9 replacement: graph copy + propose + rules + proxy"] += time.perf_counter() - t_rep
            continue
        i, st = viable[int(np.argmin(proxy(gm, Y[[i for i, _ in viable]])))]
        TIME["9 replacement: graph copy + propose + rules + proxy"] += time.perf_counter() - t_rep
        with timed("10 rollout: replacement (GPU, 1 per call)"):
            val = roll(Y[i:i + 1])[0]
        if val < cc[old] - 1e-8:
            gm.append(Y[i], val, st)
            g = gm
            replaced += 1
    return g, {"rounds": rounds, "seeds": n_seeds, "replaced": replaced, "global": stats["global"],
               "proposed": stats["proposed"], "seconds": time.perf_counter() - t0, "init_best": float(c0.min())}


roll = Rollouts()
for shape in (args.initial, args.k, 1):  # compile every batch shape before timing
    roll(np.zeros((shape, D)))
roll.count = 0
g, info_s = sample(np.random.default_rng(args.seed + 101), roll)
Xc, fc = np.asarray(g.X), np.asarray(g.cost)
budget = roll.count
x, info, t_admm = ksos(Xc, fc, args.h, np.arange(len(Xc))[::-1])  # reverse insertion: a PEO of H
roll(x)  # compile the 1-row shape is done; this is the fresh check of x
rows = [("budgeted CACS", budget, len(Xc), info["admm_stats"]["max_clique"], fc.min(), roll(x)[0],
         info["status"], info["iterations"], t_admm, info_s["seconds"])]

# ---------------------------------------------------------------- baselines, same rollouts
n2 = int(round(budget / 10)) * 10
sampler = TwoLevelSampler(n2, D, cluster_size=10, seed=args.seed)
sample2 = jax.jit(sampler.sample)
lo2, hi2 = jnp.asarray(LO, jnp.float32), jnp.asarray(HI, jnp.float32)
jax.block_until_ready(sample2(jax.random.PRNGKey(args.seed), lo2, hi2))  # compile
roll(np.zeros((n2, D)))  # compile
t0 = time.perf_counter()
with timed("two-level: sampling (GPU)"):
    points, sigma, _ = sample2(jax.random.PRNGKey(args.seed), lo2, hi2)
    X2 = np.asarray(points, np.float64)
with timed("two-level: rollout (GPU)"):
    f2 = roll(X2)
t_two = time.perf_counter() - t0
with timed("two-level: RCM order"):
    perm2 = rcm_order(X2, float(sigma))
x2, info2, t2 = ksos(X2, f2, float(sigma), perm2, tag="two-level: ")
rows.append(("two-level Sobol", n2, n2, info2["admm_stats"]["max_clique"], f2.min(), roll(x2)[0],
             info2["status"], info2["iterations"], t2, t_two))
fs = roll(qmc.Sobol(D, scramble=True, seed=args.seed).random(budget) * (HI - LO) + LO)

print(f"seed {args.seed}: n={args.n}, initial {args.initial} (best {info_s['init_best']:.3f}), h={args.h}, "
      f"max clique {args.max_clique}, k={args.k}; {info_s['rounds']} rounds, {info_s['seeds']} seeds, "
      f"{info_s['global']} isolated global points, {info_s['replaced']}/{args.replace} replacements")
print(f"{'method':<16} {'rollouts':>8} {'to KSOS':>8} {'max clique':>10} {'best sample':>11} {'x (cost)':>9} "
      f"{'status':<15} {'iters':>6} {'admm':>7} {'sampling':>9}")
for name, r, m, mc, fb, fx, st, it, ta, ts in rows:
    print(f"{name:<16} {r:>8} {m:>8} {mc:>10} {fb:>11.4f} {fx:>9.4f} {st:<15} {it:>6} {ta:>6.2f}s {ts:>8.2f}s")
print(f"plain Sobol      {budget:>8}  best sample {fs.min():.4f} (no KSOS)")
print("\ntime per step (s), JIT compiles excluded:")
for name in sorted(TIME, key=lambda k: (k.startswith("two-level"), k.startswith("KSOS"), [int(t) if t.isdigit() else t for t in k.split()[:1]])):
    print(f"  {name:<58} {TIME[name]:8.3f}")
print(f"RESULT seed={args.seed} n={args.n}: rollouts {budget} | budgeted CACS best {fc.min():.4f} x {rows[0][5]:.4f} "
      f"| two-level best {f2.min():.4f} x {rows[1][5]:.4f} | plain Sobol best {fs.min():.4f}")
