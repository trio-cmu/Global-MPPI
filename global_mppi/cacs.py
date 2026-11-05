"""Budgeted global cost-aware chordal sampling (CACS).

Library version of examples/sample_method_test/pusht_cacs_budgeted_gpu.py (from
cacs_budgeted_global_improved.py), for use inside a controller:
1. Roll out `initial` Sobol points; seed the graph with up to `reuse` of them
   (the lowest-cost 10% first, then low cost far from the seeds), each kept
   only if the graph rules allow it.
2. Grow in rounds. Each round proposes `batch` points (25% fresh global Sobol
   points, the rest local steps of 0.24-1.03 h from accepted samples), keeps the
   ones the graph rules allow, ranks them by a cheap proxy (the cost of the
   nearest accepted sample plus 0.35 x its distance), picks the best `k` one by
   one (re-checking the rules after each pick) and rolls them out in one call.
   The rules keep the kernel graph G (pairs closer than h) inside a chordal
   certificate H with cliques smaller than `max_clique`:
     - a local point needs a kernel neighbour;
     - its two-hop neighbourhood T has < max_clique nodes;
     - T is already a clique of H.
3. `replace` replacement moves: drop a sample from the costliest 30%, roll out
   the best-ranked feasible proposal, keep it if it is cheaper. With
   `replace_batch` > 1 they run in batches: drop that many costly samples at
   once, pick that many feasible proposals (re-checking the rules after each
   pick), roll them all out in one call, then re-insert proposals and dropped
   samples cheapest first while the rules allow, until the graph has its
   size back; a proposal therefore only displaces costlier samples.
The samples come back in insertion order: its reverse is a perfect elimination
order of H, so KSOS should factor them in that order.
"""

import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass
class CacsConfig:
    n: int = 1000  # samples to keep (they go to KSOS)
    initial: int = 200  # initial Sobol rollouts
    reuse: int = 40  # initial points used as graph seeds
    h: float = 0.7  # kernel radius (sweep: 0.6-0.7 gives the cheapest samples; 1.2 before)
    max_clique: int = 20
    batch: int = 768  # proposals per round
    k: int = 512  # points picked and rolled out per round (one rollout call; 128 before)
    global_every: int = 125  # one isolated global point per this many samples
    replace: int = 0  # replacement moves after the n samples (0 = none; 40 before)
    replace_batch: int = 40  # replacement moves per rollout call (1 = one at a time)


class Graph:
    """G: the kernel graph (pairs closer than h); H: its chordal certificate (a
    supergraph of G^2 built from simplicial insertions)."""

    def __init__(self, cfg: CacsConfig):
        self.cfg = cfg
        self.X, self.cost, self.G, self.H = [], [], [], []
        self._buf = None  # the samples as one array (rows 0..len(X)-1), grown in place

    @property
    def arr(self):
        """The samples as an (len(X), dim) array, without rebuilding it from the list."""
        return self._buf[: len(self.X)] if self._buf is not None else np.empty((0, 0))

    def neighbours(self, y):
        if not self.X:
            return set()
        d2 = np.sum((self.arr - y) ** 2, axis=1)
        return set(np.flatnonzero(d2 < (self.cfg.h - 1e-10) ** 2).tolist())

    def rule(self, S, allow_isolated):
        """(S, T) if a point with kernel neighbours S may join, else None."""
        if not self.X:
            return set(), set()
        if not S and not allow_isolated:
            return None
        T = set(S)
        for j in S:
            T.update(self.G[j])
        if len(T) >= self.cfg.max_clique:
            return None
        if any(not (T - {j}).issubset(self.H[j]) for j in T):
            return None
        return S, T

    def batch_rule(self, Y, allow_isolated):
        """[(index into Y, (S, T))] for the proposals the rules allow."""
        if not self.X:
            return [(i, (set(), set())) for i in range(len(Y))]
        within = _within(self.arr, np.atleast_2d(Y), self.cfg.h - 1e-10)  # as neighbours()
        out = []
        for i, row in enumerate(within):
            st = self.rule(set(np.flatnonzero(row).tolist()), allow_isolated)
            if st is not None:
                out.append((i, st))
        return out

    def append(self, y, cost, st):
        S, T = st
        i = len(self.X)
        y = np.array(y, dtype=float)
        if self._buf is None or i >= len(self._buf):  # grow the array (doubling)
            buf = np.empty((max(2 * i, self.cfg.n + 64), len(y)))
            if i:
                buf[:i] = self._buf[:i]
            self._buf = buf
        self._buf[i] = y
        self.X.append(y)
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
        return self.without_many({i})

    def without_many(self, drop):
        """The graph with the samples in `drop` removed (an induced subgraph, as `without`)."""
        g = Graph(self.cfg)
        keep = [k for k in range(len(self.X)) if k not in drop]
        new = {old: j for j, old in enumerate(keep)}
        g.X = [self.X[k] for k in keep]
        g.cost = [self.cost[k] for k in keep]
        if keep:
            g._buf = np.empty((max(len(self._buf), self.cfg.n + 64), self._buf.shape[1]))
            g._buf[: len(keep)] = self._buf[keep]
        g.G = [{new[j] for j in self.G[k] if j in new} for k in keep]
        g.H = [{new[j] for j in self.H[k] if j in new} for k in keep]
        return g


def _within(X, Y, r):
    """(len(Y), len(X)) bool: |y - x| < r, decided exactly as Graph.neighbours does (sum of squared differences).
    Fast path |y|^2 + |x|^2 - 2 y.x (one matrix product); the few pairs within rounding error of r are recomputed
    exactly, so the answer matches neighbours() and the chordal rules stay consistent.
    Plain numpy instead of scipy's cKDTree: in long closed-loop runs (ADMM-cuda, CACS x 5) the process segfaulted
    inside cKDTree.query_ball_point in batch_rule, every time, and never under gdb (no ASLR); n = 1000 points in
    12 dimensions is too few for a tree to pay off anyway."""
    xx, yy = np.einsum("ij,ij->i", X, X), np.einsum("ij,ij->i", Y, Y)
    r2 = r * r
    e = Y @ X.T  # in place from here: e = |y - x|^2 - r^2
    e *= -2.0
    e += yy[:, None]
    e += xx[None, :]
    e -= r2
    out = e < 0
    tol = 1e-9 * (yy.max() + xx.max() + r2)  # rounding error bound of the expansion (with a wide margin)
    for i, j in zip(*np.nonzero(np.abs(e) <= tol)):
        out[i, j] = np.sum((X[j] - Y[i]) ** 2) < r2
    return out


def _nearest(X, Y):
    """(distance, index) of the nearest row of X for every row of Y (for scores: rounding does not matter)."""
    d2 = np.einsum("ij,ij->i", Y, Y)[:, None] + np.einsum("ij,ij->i", X, X)[None, :] - 2.0 * (Y @ X.T)
    near = np.argmin(d2, axis=1)
    return np.sqrt(np.maximum(d2[np.arange(len(Y)), near], 0.0)), near


def _propose(g, X0, c0, rng, sobol, lo, hi):
    """`batch` proposals: 25% fresh global Sobol points, the rest local steps."""
    cfg, dim = g.cfg, len(lo)
    b = cfg.batch
    X, cc = g.arr, np.asarray(g.cost)
    deg = np.asarray([len(z) for z in g.G])
    anchors = X0[np.argsort(c0)[: max(8, int(0.1 * len(X0)))]]
    n_global = int(0.25 * b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sobol balance warning: fine for a proposal pool
        Y = np.empty((b, dim))
        Y[:n_global] = sobol.random(n_global) * (hi - lo) + lo
    w = np.exp(-3.0 * (cc - cc.min()) / max(np.std(cc), 1e-9)) / np.sqrt(1 + deg)
    w /= w.sum()
    tips = np.flatnonzero(deg <= 3)
    # all local proposals at once: a base point (20% an elite anchor, 35% a low-degree tip,
    # the rest cost-weighted), a random unit direction (20% pulled toward an anchor), a step
    m = b - n_global
    mode = rng.random(m)
    base = X[rng.choice(len(X), size=m, p=w)]
    use_anchor = mode < 0.20
    base[use_anchor] = anchors[rng.integers(len(anchors), size=use_anchor.sum())]
    if len(tips):
        use_tip = (mode >= 0.20) & (mode < 0.55)
        base[use_tip] = X[rng.choice(tips, size=use_tip.sum())]
    dr = rng.normal(size=(m, dim))
    dr /= np.linalg.norm(dr, axis=1, keepdims=True)
    to = anchors[rng.integers(len(anchors), size=m)] - base
    norm_to = np.linalg.norm(to, axis=1, keepdims=True)
    pull = (rng.random(m) < 0.20) & (norm_to[:, 0] > 1e-9)
    pulled = 0.75 * to / np.maximum(norm_to, 1e-12) + 0.48 * dr
    dr[pull] = pulled[pull] / np.linalg.norm(pulled[pull], axis=1, keepdims=True)
    step = cfg.h * rng.uniform(0.24, 1.03, size=(m, 1))
    Y[n_global:] = np.clip(base + step * dr, lo, hi)
    return Y, n_global


def _proxy(g, pts):
    """Cheap score, no rollout: the cost of the nearest accepted sample + 0.35 x distance."""
    dist, near = _nearest(g.arr, pts)
    return np.asarray(g.cost)[near] + 0.35 * dist * np.std(g.cost) / 0.5


def _replace_batched(g, X0, c0, rng, sobol, lo, hi, roll):
    """Replacement moves in batches of `replace_batch` (one rollout call per batch)."""
    cfg = g.cfg
    replaced, used = 0, 0
    for start in range(0, cfg.replace, cfg.replace_batch):
        m = min(cfg.replace_batch, cfg.replace - start)
        cc = np.asarray(g.cost)
        high = np.argsort(cc)[int(0.70 * len(cc)):]
        drop = set(rng.choice(high, size=min(m, len(high)), replace=False).tolist())
        gm = g.without_many(drop)
        Y, _ = _propose(gm, X0, c0, rng, sobol, lo, hi)
        viable = gm.batch_rule(Y, allow_isolated=True)
        if not viable:
            continue
        # pick up to m proposals, best proxy first, re-checking the rules on a scratch copy
        tmp = gm.without_many(set())
        picks = []
        for j in np.argsort(_proxy(gm, Y[[i for i, _ in viable]])):
            if len(picks) >= m:
                break
            i = viable[j][0]
            st = tmp.rule(tmp.neighbours(Y[i]), allow_isolated=True)
            if st is None:
                continue
            tmp.append(Y[i], np.nan, st)
            picks.append(i)
        if not picks:
            continue
        vals = roll(Y[picks], pad_to=cfg.replace_batch)
        used += len(picks)
        # re-insert cheapest first (proposals and the dropped samples) until the size is back
        pool = [(float(v), Y[i], True) for i, v in zip(picks, vals)]
        pool += [(float(cc[d]), g.X[d], False) for d in drop]
        target = len(g.X)
        for cost, y, new in sorted(pool, key=lambda t: t[0]):
            if len(gm.X) >= target:
                break
            st = gm.rule(gm.neighbours(y), allow_isolated=True)
            if st is not None:
                gm.append(y, cost, st)
                replaced += int(new)
        g = gm
    return g, replaced, used


def cacs_sample(roll, lo, hi, cfg: CacsConfig, seed: int, rng: np.random.Generator, extra=None):
    """Run CACS in the box [lo, hi]. roll(points (m, dim), pad_to=None) -> costs (m,).
    `extra` (m, dim): points added to the initial batch (e.g. a warm start); their
    costs come back as stats["extra_costs"].
    Returns (samples (n, dim) in insertion order, their costs, stats)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    dim = len(lo)
    X0 = qmc.Sobol(dim, scramble=True, seed=seed).random(cfg.initial) * (hi - lo) + lo
    n_extra = 0 if extra is None else len(np.atleast_2d(extra))
    if n_extra:
        X0 = np.vstack([np.clip(np.atleast_2d(extra), lo, hi), X0])
    c0 = roll(X0)
    g = Graph(cfg)
    order = np.argsort(c0)
    used = set()
    for i in order[: max(1, int(np.ceil(0.1 * len(X0))))]:
        st = g.rule(g.neighbours(X0[i]), allow_isolated=True)
        if st is not None:
            g.append(X0[i], c0[i], st)
            used.add(int(i))
        if len(used) >= cfg.reuse:
            break
    while len(used) < cfg.reuse:
        rest = np.asarray([j for j in order if int(j) not in used])
        if not len(rest):
            break
        dist = _nearest(g.arr, X0[rest])[0]
        score = c0[rest] + 0.12 * np.std(c0) / 0.5 * np.exp(-0.5 * (dist / (0.65 * cfg.h / 0.569)) ** 2)
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
    sobol = qmc.Sobol(dim, scramble=True, seed=seed + 50010)
    stats, rounds, stalled, rollouts = Counter(), 0, 0, cfg.initial
    while len(g.X) < cfg.n:
        rounds += 1
        Y, n_global = _propose(g, X0, c0, rng, sobol, lo, hi)
        due = (len(g.X) - n_seeds) // cfg.global_every > stats["global"]
        feasible = [(n_global + i, st) for i, st in g.batch_rule(Y[n_global:], allow_isolated=False)]
        iso = [(i, st) for i, st in g.batch_rule(Y[:n_global], allow_isolated=True) if not st[0]] if due else []
        if not feasible and not iso:
            stalled += 1
            if stalled >= 100:
                break
            continue
        stalled = 0
        cand = iso[:1] + [feasible[j] for j in np.argsort(_proxy(g, Y[[i for i, _ in feasible]]))] if feasible else iso[:1]
        picks = []
        for idx, (i, st) in enumerate(cand):
            if len(picks) >= cfg.k or len(g.X) >= cfg.n:  # the picks are already in g.X
                break
            if picks:  # the graph changed: recompute S, T with the earlier picks inserted
                st = g.rule(g.neighbours(Y[i]), allow_isolated=bool(iso) and idx == 0)
                if st is None:
                    continue
            g.append(Y[i], np.nan, st)  # cost filled in after the batched rollout
            picks.append(len(g.X) - 1)
            if iso and idx == 0:
                stats["global"] += 1
        costs = roll(np.asarray([g.X[p] for p in picks]), pad_to=cfg.k)
        rollouts += len(picks)
        for p, c in zip(picks, costs):
            g.cost[p] = float(c)
    replaced = 0
    if cfg.replace_batch > 1:
        g, replaced, used_rollouts = _replace_batched(g, X0, c0, rng, sobol, lo, hi, roll)
        rollouts += used_rollouts
    for _ in range(cfg.replace if cfg.replace_batch <= 1 else 0):
        cc = np.asarray(g.cost)
        high = np.argsort(cc)[int(0.70 * len(cc)):]
        old = int(high[rng.integers(len(high))])
        gm = g.without(old)
        Y, _ = _propose(gm, X0, c0, rng, sobol, lo, hi)
        viable = gm.batch_rule(Y, allow_isolated=True)
        if not viable:
            continue
        i, st = viable[int(np.argmin(_proxy(gm, Y[[i for i, _ in viable]])))]
        val = roll(Y[i:i + 1])[0]
        rollouts += 1
        if val < cc[old] - 1e-8:
            gm.append(Y[i], val, st)
            g = gm
            replaced += 1
    return np.asarray(g.X), np.asarray(g.cost), {
        "extra_costs": c0[:n_extra],
        "rounds": rounds, "seeds": n_seeds, "replaced": replaced, "rollouts": rollouts + n_extra,
        "initial_best": float(c0.min()),
    }
