"""Relabel the finished CACS certificate H with RCM (a symmetric permutation P H P^T, nothing rebuilt).

Same CACS samples as plot/cacs_2d_sparsity.png (plot_cacs_2d.py: seed 3, h = 0.17, max_clique 15, 100 samples).
E is the aggregate pattern KSOS factors (Phi from the reverse insertion order, as _solve_cacs), H the sampler's
chordal certificate (E inside H). Both are renumbered with the RCM order of H. Renumbering changes the picture, not
the graphs: the cliques, their sizes and chordality stay the same, which the script checks.

    python plot_cacs_h_rcm_2d.py        ->  plot/cacs_2d_sparsity_H_rcm.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: E402

import global_mppi.cacs as cm  # noqa: E402
from global_mppi.cacs import CacsConfig, cacs_sample  # noqa: E402
from ksos_tools.problems.features import build_problem  # noqa: E402
from ksos_tools.solvers.ADMM.chordal import column_supports  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--h", type=float, default=0.17)
ap.add_argument("--max-clique", type=int, default=15)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", default=str(Path(__file__).parent / "plot"))
args = ap.parse_args()

CENTERS = np.array([[0.22, 0.27, 0.00, 1.0], [0.70, 0.22, 0.10, 1.2], [0.80, 0.74, 0.02, 0.9],
                    [0.30, 0.78, 0.18, 1.1], [0.52, 0.50, 0.26, 1.3]])


def cost(X):
    X = np.atleast_2d(X)
    dx = X[:, None, 0] - CENTERS[None, :, 0]
    dy = X[:, None, 1] - CENTERS[None, :, 1]
    e = (dx * dx + 1.4 * dy * dy + 0.6 * dx * dy) * CENTERS[None, :, 3] * 9 + CENTERS[None, :, 2]
    return e.min(1) + 0.03 * np.sin(14 * X[:, 0] + 9 * X[:, 1]) ** 2


# the sampler, exactly as plot_cacs_2d.py calls it; keep its final graph object for H
box = {}
_mc = cm.Graph.max_cliques


def _keep(self):
    box["g"] = self
    return _mc(self)


cm.Graph.max_cliques = _keep
cfg = CacsConfig(n=args.n, initial=20, reuse=5, h=args.h, max_clique=args.max_clique, batch=48, k=8,
                 global_every=50, replace=0)
X, c, _ = cacs_sample(lambda P, pad_to=None: cost(P), [0, 0], [1, 1], cfg, seed=args.seed,
                      rng=np.random.default_rng(args.seed + 101))
n = len(X)
g = box["g"]

# matrix rows = reverse insertion order (row r is sample n-1-r), as _solve_cacs
o = np.arange(n)[::-1]
prob = build_problem(X[o], args.h, lambda _: c[o])
E = np.zeros((n, n), bool)
for s in column_supports(prob.Phi):
    E[np.ix_(s, s)] = True
H = np.eye(n, dtype=bool)
for v in range(n):
    for u in g.H[v]:
        H[n - 1 - v, n - 1 - u] = True
assert not (E & ~H).any(), "E must lie inside H"


def cliques(A):
    adj = {i: set(np.flatnonzero(A[i])) - {i} for i in range(len(A))}
    out = []

    def rec(R, P, Xs):
        if not P and not Xs:
            out.append(sorted(R))
            return
        u = max(P | Xs, key=lambda v: len(adj[v] & P))
        for v in list(P - adj[u]):
            rec(R | {v}, P & adj[v], Xs & adj[v])
            P, Xs = P - {v}, Xs | {v}
    rec(set(), set(adj), set())
    return out


def chordal(A):  # maximum cardinality search: earlier-numbered neighbours of each vertex must form a clique
    m = len(A)
    w, left, order = np.zeros(m, int), set(range(m)), []
    while left:
        v = max(left, key=lambda u: (w[u], -u))
        order.append(v)
        left.remove(v)
        for u in np.flatnonzero(A[v]):
            if u in left:
                w[u] += 1
    pos = {v: i for i, v in enumerate(order)}
    for v in order:
        earlier = [u for u in np.flatnonzero(A[v]) if u != v and pos[u] < pos[v]]
        if not A[np.ix_(earlier, earlier)].all():
            return False
    return True


def band(A):
    i, j = np.nonzero(np.triu(A, 1))
    return int((j - i).max())


perm = reverse_cuthill_mckee(sp.csr_matrix(H.astype(float)), symmetric_mode=True)
Ep, Hp = E[np.ix_(perm, perm)], H[np.ix_(perm, perm)]
cE, cH, cEp, cHp = cliques(E), cliques(H), cliques(Ep), cliques(Hp)
stats = {name: (len(cl), max(map(len, cl)), chordal(A), band(A)) for name, cl, A in
         [("E", cE, E), ("H", cH, H), ("E rcm", cEp, Ep), ("H rcm", cHp, Hp)]}
for k, v in stats.items():
    print(f"{k:6s}: {v[0]} maximal cliques, largest {v[1]}, chordal {v[2]}, farthest nonzero {v[3]} rows from diagonal")

# plot: E and H (H not E lighter) before and after the relabelling, the largest clique of E in red
INK, INK2, RULE = "#14213d", "#5c677d", "#d9dee5"
C_E, C_H, C_BIG = "#134074", "#f4a261", "#c9184a"


def draw(ax, Ea, Ha, cl, title):
    rgb = np.ones((n, n, 3))
    rgb[Ha] = matplotlib.colors.to_rgb(C_H)
    rgb[Ea] = matplotlib.colors.to_rgb(C_E)
    big = max(cl, key=len)
    rgb[np.ix_(big, big)] = matplotlib.colors.to_rgb(C_BIG)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([0, n // 2, n - 1]); ax.set_yticks([0, n // 2, n - 1])
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    for s_ in ax.spines.values():
        s_.set_color(RULE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 6.9), dpi=170)
nE, mE, chE, bE = stats["E"]
nEp, mEp, chEp, bEp = stats["E rcm"]
nHp, mHp, chHp, bHp = stats["H rcm"]
draw(a1, E, H, cE, f"as built: reverse insertion order\nE: {nE} cliques, largest {mE}, chordal {chE}, "
     f"farthest nonzero {bE} rows off the diagonal")
draw(a2, Ep, Hp, cEp, f"same E and H, relabelled by RCM of H (H: largest {mHp})\nE: {nEp} cliques, largest {mEp}, "
     f"chordal {chEp}, farthest nonzero {bEp} rows off the diagonal")
fig.legend(handles=[Patch(color=C_E, label="nonzero of E (what ADMM decomposes)"),
                    Patch(color=C_H, label="in H only (certificate edges outside E)"),
                    Patch(color=C_BIG, label="one largest clique of E")],
           loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.text(0.03, 0.975, "CACS: the finished H relabelled with RCM", fontsize=14, weight="bold", color=INK, va="top")
fig.text(0.03, 0.935, f"{n} CACS samples (seed {args.seed}, h = {args.h}, max_clique = {args.max_clique}). "
         "P E P^T and P H P^T with P the RCM order of H: the graphs, their cliques and chordality do not change, "
         "only the picture.", fontsize=8.8, color=INK2, va="top")
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.84, wspace=0.12)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "cacs_2d_sparsity_H_rcm.png", facecolor="white")
print("saved", out / "cacs_2d_sparsity_H_rcm.png")
