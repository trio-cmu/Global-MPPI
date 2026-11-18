"""2-D illustration of sampling for KernelSOS: plain Sobol vs cost-aware chordal sampling (CACS).

Cost: the five-basin 2-D function of the CACS explainer page (artifact "Cost-Aware Chordal Sampling").
CACS is the real sampler, global_mppi/cacs.py (cacs_sample), run in 2-D. Both plots show 100 samples:
100 plain Sobol points, and the 100 samples CACS keeps (its evaluation count is printed). Two figures:

    plot/sobol_2d.png        Sobol samples on the cost map
    plot/cacs_2d_cliques.png CACS samples, the kernel graph G and the cliques of the chordal certificate H
                             (each maximal clique shaded; the cliques KSOS decomposes the SDP over)
    plot/sobol_2d_kernel_graph.png, plot/cacs_2d_kernel_graph.png
                             the kernel graph G of each sample set: an edge where two samples are closer than h,
                             i.e. where the Wendland kernel K_ij is nonzero
    plot/sobol_2d_kernel_pattern.png, plot/cacs_2d_kernel_pattern.png
                             the same kernel graph as an n x n matrix (K_ij != 0), rows in the order KSOS uses
                             (Sobol RCM, CACS reverse insertion), so it lines up with the sparsity plots
    plot/sobol_1level_2d.png one figure: the 100 one-level (plain) Sobol samples and their aggregate pattern E
    plot/sobol_2d_sparsity.png, plot/cacs_2d_sparsity.png
    plot/cacs_2d_sparsity_cliques.png
                             the CACS pattern with its largest cliques coloured, and each of them pulled out as the
                             dense block ADMM projects (E_k M E_k^T)
    plot/sobol_vs_cacs_2d_paper.png  the same without title, panels labelled (a)-(d), for the paper
    plot/sobol_vs_cacs_2d.png  one figure, 2 x 2: top Sobol samples | Sobol aggregate pattern (RCM),
                             bottom CACS samples and cliques | CACS aggregate pattern (RCM, as cacs_2d_sparsity_rcm)
    plot/sobol_2d_aggregate.png   the Sobol aggregate pattern exactly as generated: E = union_j supp(phi_j) x supp(phi_j),
                             Phi = chol(K)^T in the sequence's own order; no RCM, no chordal extension
    plot/sobol_2d_sparsity_natural.png   the Sobol pattern with Phi built in the sequence's own order (before RCM)
    plot/cacs_2d_sparsity_rcm.png   the CACS samples' pattern with rows in RCM order and the solver's own chordal
                             extension (what the solver would get without the insertion order)
                             the aggregate sparsity pattern E of the KSOS SDP, M(alpha) = lambda I + Phi diag(alpha)
                             Phi^T, for each sample set, built as GlobalMPPI builds it: Sobol in RCM order with the
                             solver's own chordal extension (_solve_admm_cuda), CACS in reverse insertion order with
                             the elimination cliques (_solve_cacs); E only, the solver's chordal fill is not drawn

    python plot_cacs_2d.py [--n 100] [--h 0.17] [--max-clique 10] [--seed 0]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from global_mppi.algs.global_mppi import _elimination_cliques  # noqa: E402
from global_mppi.cacs import CacsConfig, cacs_sample  # noqa: E402
from ksos_tools.problems.features import build_problem  # noqa: E402
from ksos_tools.problems.sampling import rcm_order  # noqa: E402
from ksos_tools.solvers.ADMM.chordal import chordal_decomposition, column_supports  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100, help="samples: CACS keeps n, Sobol draws n")
ap.add_argument("--initial", type=int, default=20, help="CACS initial Sobol batch")
ap.add_argument("--h", type=float, default=0.17, help="kernel radius")
ap.add_argument("--max-clique", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(Path(__file__).parent / "plot"))
args = ap.parse_args()

# the explainer page's 2-D cost: five quadratic basins (min over them) plus a small ripple
CENTERS = np.array([[0.22, 0.27, 0.00, 1.0], [0.70, 0.22, 0.10, 1.2], [0.80, 0.74, 0.02, 0.9],
                    [0.30, 0.78, 0.18, 1.1], [0.52, 0.50, 0.26, 1.3]])


def cost(X):
    X = np.atleast_2d(X)
    dx = X[:, None, 0] - CENTERS[None, :, 0]
    dy = X[:, None, 1] - CENTERS[None, :, 1]
    e = (dx * dx + 1.4 * dy * dy + 0.6 * dx * dy) * CENTERS[None, :, 3] * 9 + CENTERS[None, :, 2]
    return e.min(1) + 0.03 * np.sin(14 * X[:, 0] + 9 * X[:, 1]) ** 2


n_evals = [0]


def roll(P, pad_to=None):
    n_evals[0] += len(P)
    return cost(P)


cfg = CacsConfig(n=args.n, initial=args.initial, reuse=5, h=args.h, max_clique=args.max_clique,
                 batch=48, k=8, global_every=50, replace=0)
X, c, stats = cacs_sample(roll, [0, 0], [1, 1], cfg, seed=args.seed, rng=np.random.default_rng(args.seed + 101))
evals = n_evals[0]
init = qmc.Sobol(2, scramble=True, seed=args.seed).random(args.initial)  # CACS's own initial batch
S = qmc.Sobol(2, scramble=True, seed=args.seed + 7).random(2 ** int(np.ceil(np.log2(args.n))))[:args.n]
cS = cost(S)

# graphs on the CACS samples: kernel graph G (pairs closer than h) and the maximal cliques of H
D = np.linalg.norm(X[:, None] - X[None], axis=-1)
G_edges = [(i, j) for i in range(len(X)) for j in range(i + 1, len(X)) if D[i, j] < args.h]
cliques = sorted(stats["cliques"], key=len, reverse=True)
big = len(cliques[0])

g = np.linspace(0, 1, 500)
GX, GY = np.meshgrid(g, g)
Z = cost(np.stack([GX.ravel(), GY.ravel()], 1)).reshape(GX.shape)
true_min = np.unravel_index(np.argmin(Z), Z.shape)
tm = (GX[true_min], GY[true_min], Z[true_min])
levels = np.linspace(0, np.percentile(Z, 95), 20)

# palette: cost map deep blue (low) to pale (high); samples white; cliques in warm colours that read on blue
CMAP = LinearSegmentedColormap.from_list("cost", ["#6f8fa6", "#97b1c3", "#bfd0dc", "#dde7ee", "#f3f6f9", "#fbfcfd"])
CLIQUE = ["#ef476f", "#f8961e", "#06d6a0", "#9b5de5", "#ffd166", "#f15bb5", "#43aa8b", "#fb5607"]
EDGE = "#2b2d42"  # kernel edges and sample outlines: dark ink over the light cost map
INK, INK2, RULE = "#14213d", "#5c677d", "#d9dee5"
SAMPLE_EDGE, BEST, STAR = "#2b2d42", "#d00070", "#ffbe0b"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def base(title, subtitle):
    fig, ax = plt.subplots(figsize=(6.8, 7.2), dpi=170)
    cf = ax.contourf(GX, GY, Z, levels=levels, cmap=CMAP, extend="max")
    ax.contour(GX, GY, Z, levels=levels[::2], colors="white", linewidths=0.6, alpha=0.8)
    ax.plot(*tm[:2], marker="*", ms=19, color=STAR, mec=INK, mew=1.0, zorder=7, ls="none")
    ax.set(xlim=(-0.035, 1.035), ylim=(-0.035, 1.035), aspect="equal")  # room for samples on the box edge
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=INK2, lw=0.9, zorder=2))
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.07, 0.965, title, fontsize=14, weight="bold", color=INK, va="top")
    fig.text(0.07, 0.925, subtitle, fontsize=9, color=INK2, va="top", linespacing=1.5)
    cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label("cost", color=INK2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.05, top=0.84)
    return fig, ax


def samples(ax, P, best):
    ax.scatter(P[:, 0], P[:, 1], s=24, c="white", edgecolors=SAMPLE_EDGE, linewidths=0.9, zorder=5)
    ax.scatter(*P[best], s=95, facecolors="none", edgecolors=BEST, linewidths=2.2, zorder=6)


ZERO, NZ, FILLC, DIAG, BIGC = "#ffffff", "#2f6fdb", "#ffffff", "#0b1f4b", "#e11d48"  # matrix plots (fill not drawn)
GRID = "#e3e8ef"


def tiles(ax, n):
    """Matrix look: a thin frame, ticks every 25 rows (no per-cell grid)."""
    ax.add_patch(plt.Rectangle((-0.5, -0.5), n, n, fill=False, ec="#c5ced9", lw=0.8, zorder=4))
    t = list(range(0, n, 25)) + ([n - 1] if (n - 1) % 25 else [])
    ax.set_xticks(t); ax.set_yticks(t)
    ax.tick_params(colors=INK2, labelsize=8, length=0, pad=3)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)


def below(ax, handles, ncol=3):
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015), ncol=ncol, frameon=False,
              fontsize=8.2, handlelength=1.2, columnspacing=1.4, labelcolor=INK)


def save(fig, name):
    fig.savefig(out / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("saved", out / f"{name}.png")


# 1. Sobol
fig, ax = base("Plain Sobol samples",
               f"{args.n} scrambled Sobol points spread evenly over the box.\n"
               f"Best sample {cS.min():.3f} (pink ring); true minimum {tm[2]:.3f} (star). Darker = lower cost.")
samples(ax, S, np.argmin(cS))
save(fig, "sobol_2d")

# 2. CACS with its cliques
fig, ax = base("CACS samples and their cliques",
               f"{len(X)} samples kept ({evals} cost evaluations, {args.initial} of them the initial Sobol batch).\n"
               f"Kernel radius h = {args.h}, max_clique = {args.max_clique}: {len(cliques)} maximal cliques of the chordal\n"
               f"certificate H, largest {big}. Best sample {c.min():.3f} (pink ring); true minimum {tm[2]:.3f} (star).")
for k, q in enumerate(cliques):
    P, col = X[q], CLIQUE[k % len(CLIQUE)]
    if len(q) >= 3:
        try:
            hull = P[ConvexHull(P).vertices]
        except Exception:  # collinear
            hull = P
        ax.add_patch(Polygon(hull, closed=True, facecolor=col, alpha=0.32, edgecolor=col, linewidth=1.4, zorder=2))
    elif len(q) == 2:
        ax.plot(*P.T, color=col, lw=3, alpha=0.7, zorder=2, solid_capstyle="round")
ax.add_collection(LineCollection([X[[i, j]] for i, j in G_edges], colors=EDGE, linewidths=0.6, alpha=0.45, zorder=3))
ax.scatter(init[:, 0], init[:, 1], s=16, marker="s", c="none", edgecolors="#6c757d", linewidths=0.9, zorder=4)
samples(ax, X, np.argmin(c))
ax.plot([], [], color=EDGE, lw=1.0, alpha=0.6, label="kernel graph G (distance < h)")
ax.add_patch(Polygon([[0, 0]], facecolor=CLIQUE[0], alpha=0.45, edgecolor=CLIQUE[0], label="maximal clique of H (one colour each)"))
ax.scatter([], [], s=16, marker="s", c="none", edgecolors="#6c757d", label="initial Sobol batch (all evaluated)")
leg = ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95, edgecolor=RULE, facecolor="white")
save(fig, "cacs_2d_cliques")
print(f"n {args.n}, cacs evals {evals}, sobol best {cS.min():.4f}, cacs best {c.min():.4f}, cliques {len(cliques)}, largest {big}")


# 3-4. aggregate sparsity of the KSOS SDP for each sample set, ordered and decomposed as GlobalMPPI does
def structure_of(P, f, order, own_cliques):
    prob = build_problem(P[order], args.h, lambda _: f[order])
    return chordal_decomposition(prob.Phi, cliques=_elimination_cliques(prob.Phi) if own_cliques else None)


def sparsity(st, name, title, order_note):
    n = st.n
    A = np.zeros((n, n))
    i, j = st.pattern[:, 0], st.pattern[:, 1]
    keep = ~st.is_fill  # E itself; the solver's chordal fill is not drawn
    A[i[keep], j[keep]] = A[j[keep], i[keep]] = 1
    big = np.asarray(max(st.cliques, key=len))  # one largest clique: its entries of E in rose
    blk = np.zeros((n, n), bool)
    blk[np.ix_(big, big)] = True
    A[blk & (A == 1)] = 2
    A[np.arange(n), np.arange(n)] = 3
    A[big, big] = 2
    s_ = st.stats
    dens = 100 * (2 * s_["E_edges"] + n) / n ** 2
    fig, ax = plt.subplots(figsize=(6.2, 6.9), dpi=170)
    ax.imshow(A, cmap=ListedColormap([ZERO, NZ, BIGC, DIAG]), vmin=0, vmax=3,
              interpolation="nearest")
    ax.text(1.0, 1.075, f"largest clique: {len(big)} x {len(big)}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10.5, weight="bold", color=BIGC,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=BIGC, linewidth=1.1))
    fig.text(0.08, 0.965, title, fontsize=14, weight="bold", color=INK, va="top")
    fig.text(0.08, 0.925,
             f"Pattern E of M(alpha) = lambda I + Phi diag(alpha) Phi^T: {n} x {n}, kernel radius h = {args.h}.\n"
             f"{order_note}\n"
             f"E: {s_['E_edges']} off-diagonal pairs ({dens:.1f}% of entries)\n"
             f"{s_['num_cliques']} cliques, largest {s_['max_clique']}, mean {s_['mean_clique']:.1f}",
             fontsize=8.6, color=INK2, va="top", linespacing=1.5)
    from matplotlib.patches import Patch
    tiles(ax, n)
    below(ax, [Patch(color=NZ, label="nonzero of E"), Patch(color=DIAG, label="diagonal"),
               Patch(color=BIGC, label=f"one largest clique ({len(big)} rows)")])
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.76)
    save(fig, name)
    return s_


stS = sparsity(structure_of(S, cS, rcm_order(S, args.h), False), "sobol_2d_sparsity", "Sobol: aggregate sparsity",
               "Rows in RCM order; chordal extension by the solver (MCS / min-degree).")
stC = sparsity(structure_of(X, c, np.arange(len(X))[::-1], True), "cacs_2d_sparsity", "CACS: aggregate sparsity",
               f"Rows in reverse insertion order (perfect elimination order of H); max_clique = {args.max_clique}.")
stN = sparsity(structure_of(S, cS, np.arange(len(S)), False), "sobol_2d_sparsity_natural", "Sobol before RCM: aggregate sparsity",
               "Rows in the Sobol sequence's own order (no RCM); chordal extension by the solver.")
print("sobol natural order: E", stN["E_edges"], "fill", stN["fill_edges"], "cliques", stN["num_cliques"], "max clique", stN["max_clique"])
stR = sparsity(structure_of(X, c, rcm_order(X, args.h), False), "cacs_2d_sparsity_rcm", "CACS in RCM order: aggregate sparsity",
               "Same CACS samples, rows in RCM order; chordal extension by the solver, as for Sobol.")
print("cacs RCM: E", stR["E_edges"], "fill", stR["fill_edges"], "cliques", stR["num_cliques"], "max clique", stR["max_clique"])
print("sobol E", stS["E_edges"], "fill", stS["fill_edges"], "max clique", stS["max_clique"], "| cacs E", stC["E_edges"],
      "fill", stC["fill_edges"], "max clique", stC["max_clique"])


# 5-6. kernel graph G: an edge wherever K_ij != 0 (distance < h)
def kernel_graph(P, f, name, title, colour_components):
    n = len(P)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    I, J = np.nonzero(np.triu(D < args.h, 1))
    parent = list(range(n))  # connected components (union-find)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(I, J):
        parent[find(a)] = find(b)
    roots = np.array([find(a) for a in range(n)])
    sizes = {r: int((roots == r).sum()) for r in set(roots)}
    deg = np.bincount(np.r_[I, J], minlength=n)
    isolated = int((deg == 0).sum())
    fig, ax = base(title, f"{n} samples, kernel radius h = {args.h}: an edge wherever K_ij is nonzero (distance < h).\n"
                          f"{len(I)} edges, mean degree {deg.mean():.1f}, max degree {deg.max()}; {isolated} isolated samples;\n"
                          f"{len(sizes)} connected components, largest {max(sizes.values())} samples.")
    ax.add_collection(LineCollection([P[[a, b]] for a, b in zip(I, J)], colors=EDGE, linewidths=1.0, alpha=0.6, zorder=3))
    k = int(np.argmax(deg)) if deg.max() > 0 else 0  # scale: the radius-h disc around the best-connected sample
    ax.add_patch(plt.Circle(P[k], args.h, fill=False, ls="--", lw=1.2, color="#d00070", zorder=4))
    ax.annotate("radius h", P[k] + [args.h * 0.72, args.h * 0.72], color="#d00070", fontsize=8, zorder=6,
                xytext=(3, 3), textcoords="offset points")
    lone = deg == 0
    ax.scatter(P[~lone, 0], P[~lone, 1], s=26, c="white", edgecolors=SAMPLE_EDGE, linewidths=0.9, zorder=5)
    ax.scatter(P[lone, 0], P[lone, 1], s=26, c="#8d99ae", edgecolors=SAMPLE_EDGE, linewidths=0.9, zorder=5)
    ax.plot([], [], color=EDGE, lw=1.2, alpha=0.7, label="kernel edge (K_ij != 0)")
    ax.scatter([], [], s=26, c="white", edgecolors=SAMPLE_EDGE, label="sample with a neighbour")
    ax.scatter([], [], s=26, c="#8d99ae", edgecolors=SAMPLE_EDGE, label="isolated sample (no neighbour)")
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95, edgecolor=RULE, facecolor="white")
    save(fig, name)
    return len(I), deg.mean(), deg.max(), isolated, len(sizes), max(sizes.values())


print("kernel graph sobol (edges, mean deg, max deg, isolated, components, largest):",
      kernel_graph(S, cS, "sobol_2d_kernel_graph", "Sobol: kernel graph", False))
print("kernel graph cacs  (edges, mean deg, max deg, isolated, components, largest):",
      kernel_graph(X, c, "cacs_2d_kernel_graph", "CACS: kernel graph", False))


# 7-8. the kernel graph as a matrix: K's nonzero pattern, rows in the order KSOS uses
def kernel_pattern(P, order, name, title, order_note):
    Q = P[order]
    n = len(Q)
    A = (np.linalg.norm(Q[:, None] - Q[None], axis=-1) < args.h).astype(float)
    A[np.arange(n), np.arange(n)] = 2
    edges = int((A == 1).sum() // 2)
    I, J = np.nonzero(np.triu(A == 1, 1))
    band = int((J - I).max()) if len(I) else 0
    fig, ax = plt.subplots(figsize=(6.2, 6.9), dpi=170)
    ax.imshow(A, cmap=ListedColormap([ZERO, NZ, DIAG]), vmin=0, vmax=2, interpolation="nearest")
    tiles(ax, n)
    from matplotlib.patches import Patch
    below(ax, [Patch(color=NZ, label="K_ij != 0 (distance < h)"), Patch(color=DIAG, label="diagonal")], ncol=2)
    fig.text(0.08, 0.965, title, fontsize=14, weight="bold", color=INK, va="top")
    fig.text(0.08, 0.925,
             f"Kernel graph G, {n} x {n}: cell (i, j) filled where K_ij != 0 (distance < h = {args.h}).\n"
             f"{order_note}\n"
             f"{edges} edges ({100 * (2 * edges + n) / n ** 2:.1f}% of entries incl. diagonal), "
             f"farthest pair {band} rows apart.",
             fontsize=8.6, color=INK2, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.79)
    save(fig, name)
    return edges, band


print("kernel pattern sobol (edges, bandwidth):", kernel_pattern(S, rcm_order(S, args.h), "sobol_2d_kernel_pattern",
      "Sobol: kernel graph pattern", "Rows in RCM order, as KSOS factors Sobol samples."))
print("kernel pattern cacs  (edges, bandwidth):", kernel_pattern(X, np.arange(len(X))[::-1], "cacs_2d_kernel_pattern",
      "CACS: kernel graph pattern", "Rows in reverse insertion order, as KSOS factors CACS samples."))


# 9. one-level Sobol: the samples and the aggregate pattern in one figure
def sobol_one_level():
    p = rcm_order(S, args.h)
    st = structure_of(S, cS, p, False)
    n = st.n
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 6.3), dpi=170, gridspec_kw={"width_ratios": [1.12, 1]})
    cf = a1.contourf(GX, GY, Z, levels=levels, cmap=CMAP, extend="max")
    a1.contour(GX, GY, Z, levels=levels[::2], colors="white", linewidths=0.6, alpha=0.8)
    a1.plot(*tm[:2], marker="*", ms=17, color=STAR, mec=INK, mew=1.0, zorder=7, ls="none")
    a1.scatter(S[:, 0], S[:, 1], s=24, c="white", edgecolors=SAMPLE_EDGE, linewidths=0.9, zorder=5)
    b = np.argmin(cS)
    a1.scatter(*S[b], s=95, facecolors="none", edgecolors=BEST, linewidths=2.2, zorder=6)
    a1.set(xlim=(-0.035, 1.035), ylim=(-0.035, 1.035), aspect="equal")
    a1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=INK2, lw=0.9, zorder=2))
    a1.set_xticks([0, 0.5, 1]); a1.set_yticks([0, 0.5, 1])
    a1.tick_params(colors=INK2, labelsize=9, length=0)
    for sp in a1.spines.values():
        sp.set_visible(False)
    a1.set_title(f"{n} samples   best {cS.min():.3f} (pink ring), true minimum {tm[2]:.3f} (star)",
                 fontsize=9.5, color=INK2, loc="left")
    cb = fig.colorbar(cf, ax=a1, fraction=0.04, pad=0.02)
    cb.set_label("cost", color=INK2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    cb.outline.set_visible(False)
    A = np.zeros((n, n))
    i, j = st.pattern[:, 0], st.pattern[:, 1]
    keep = ~st.is_fill
    A[i[keep], j[keep]] = A[j[keep], i[keep]] = 1
    big = max(st.cliques, key=len)
    A[np.arange(n), np.arange(n)] = 1
    blk = np.zeros((n, n), bool)
    blk[np.ix_(big, big)] = True
    A[blk & (A == 1)] = 3
    a2.imshow(A, cmap=ListedColormap([ZERO, NZ, FILLC, BIGC]), vmin=0, vmax=3, interpolation="nearest")
    tiles(a2, n)
    s_ = st.stats
    a2.set_title(f"aggregate pattern E, rows in RCM order: {s_['E_edges']} pairs "
                 f"({100 * (2 * s_['E_edges'] + n) / n ** 2:.1f}%),\n"
                 f"{s_['num_cliques']} cliques, largest {s_['max_clique']} (red)", fontsize=9.5, color=INK2, loc="left")
    from matplotlib.patches import Patch
    below(a2, [Patch(color=NZ, label="nonzero of E"), Patch(color=BIGC, label=f"largest clique ({len(big)} rows)")], ncol=2)
    a2.text(1.0, 1.075, f"largest clique: {len(big)} x {len(big)}", transform=a2.transAxes, ha="right", va="bottom",
            fontsize=10.5, weight="bold", color=BIGC,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=BIGC, linewidth=1.1))
    fig.text(0.03, 0.97, "One-level Sobol: samples and aggregate sparsity", fontsize=14, weight="bold", color=INK, va="top")
    fig.text(0.03, 0.928, f"{n} scrambled Sobol points on the 2-D five-basin cost. KSOS pattern of "
             f"M(alpha) = lambda I + Phi diag(alpha) Phi^T\nwith the Wendland kernel of radius h = {args.h}; "
             "chordal extension by the solver, as for Sobol samples in GlobalMPPI.", fontsize=9, color=INK2, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.8, wspace=0.12)
    save(fig, "sobol_1level_2d")


sobol_one_level()


# 10. what a clique is: the CACS pattern with large cliques coloured, and each pulled out as a dense block
def clique_figure(top=4):
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    o = np.arange(len(X))[::-1]
    st = structure_of(X, c, o, True)
    n = st.n
    cl = []  # the largest cliques, skipping ones that share more than half their rows with a picked one
    for C in sorted(st.cliques, key=len, reverse=True):
        if all(len(set(C) & set(D)) <= len(C) // 2 for D in cl):
            cl.append(C)
        if len(cl) == top:
            break
    cols = ["#e11d48", "#f59e0b", "#7c3aed", "#10b981"][:len(cl)]
    SHARED = "#343a40"
    member = np.zeros((n, n), int)
    for C in cl:
        member[np.ix_(C, C)] += 1
    base = np.zeros((n, n), bool)
    i, j = st.pattern[:, 0], st.pattern[:, 1]
    base[i, j] = base[j, i] = True
    base[np.arange(n), np.arange(n)] = True
    rgb = np.ones((n, n, 3))
    rgb[:] = matplotlib.colors.to_rgb(ZERO)
    rgb[base] = matplotlib.colors.to_rgb("#bcd2f7")
    for C, col in zip(cl, cols):
        rgb[np.ix_(C, C)] = matplotlib.colors.to_rgb(col)
    rgb[member >= 2] = matplotlib.colors.to_rgb(SHARED)
    fig = plt.figure(figsize=(13.6, 7.6), dpi=170)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2.2, 1, 1], wspace=0.3, hspace=0.5, left=0.04, right=0.97, top=0.78, bottom=0.11)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(rgb, interpolation="nearest")
    tiles(ax, n)
    ax.set_title(f"aggregate pattern E ({n} x {n}, reverse insertion order)", fontsize=10, color=INK2, loc="left", pad=22)
    below(ax, [Patch(color="#bcd2f7", label="nonzero of E (other cliques)")] +
              [Patch(color=col, label=f"clique {k + 1}: {len(C)} rows") for k, (C, col) in enumerate(zip(cl, cols))] +
              ([Patch(color=SHARED, label="in two shown cliques (separator)")] if (member >= 2).any() else []), ncol=3)
    for k, (C, col) in enumerate(zip(cl, cols)):
        a = fig.add_subplot(gs[k // 2, 1 + k % 2])
        others = set().union(*[set(D) for D in cl if D is not C]) if len(cl) > 1 else set()
        sh = np.array([v in others for v in C])
        sub = np.ones((len(C), len(C), 3))
        sub[:] = matplotlib.colors.to_rgb(col)
        sub[np.ix_(sh, sh)] = matplotlib.colors.to_rgb(SHARED)
        a.imshow(sub, interpolation="nearest")
        a.add_collection(LineCollection([[(e, -0.5), (e, len(C) - 0.5)] for e in np.arange(-0.5, len(C), 1.0)] +
                                        [[(-0.5, e), (len(C) - 0.5, e)] for e in np.arange(-0.5, len(C), 1.0)],
                                        colors="white", linewidths=0.6, zorder=3))
        a.set_xticks(range(len(C))); a.set_yticks(range(len(C)))
        a.set_xticklabels(C, fontsize=5.5, rotation=90, color=INK2)
        a.set_yticklabels(C, fontsize=5.5, color=INK2)
        a.tick_params(length=0, pad=1.5)
        for sp in a.spines.values():
            sp.set_color(col); sp.set_linewidth(1.6)
        ns = int(sh.sum())
        a.set_title(f"clique {k + 1}: {len(C)} x {len(C)}, all nonzero"
                    + (f"\n{ns} row{'s' if ns > 1 else ''} shared with another clique (dark)" if ns else "\nshares no rows with the others"),
                    fontsize=8.5, color=INK, loc="left")
    s_ = st.stats
    fig.text(0.04, 0.965, "CACS: what the cliques are", fontsize=14, weight="bold", color=INK, va="top")
    fig.text(0.04, 0.925,
             "A clique C_k is a set of rows whose submatrix of E is completely filled. ADMM splits M(alpha) >= 0 into one "
             "PSD block Z_k per clique,\nM(alpha) = sum_k E_k^T Z_k E_k. Left: "
             f"{len(cl)} large cliques of the {s_['num_cliques']} coloured in place (their rows need not be contiguous). "
             "Right: each of them with\nits rows and columns gathered: the dense "
             f"block ADMM projects. Largest clique {s_['max_clique']} (max_clique = {args.max_clique}), kernel radius h = {args.h}.",
             fontsize=8.6, color=INK2, va="top", linespacing=1.5)
    save(fig, "cacs_2d_sparsity_cliques")
    return [len(C) for C in cl]


print("clique figure, shown clique sizes:", clique_figure())


# 11. the aggregate pattern as generated, nothing else: sequence order, no RCM, no chordal extension
def aggregate_direct(P, f, name, title):
    prob = build_problem(P, args.h, lambda _: f)
    n = len(P)
    A = np.zeros((n, n))
    for sup in column_supports(prob.Phi):
        A[np.ix_(sup, sup)] = 1
    A[np.arange(n), np.arange(n)] = 2
    pairs = int((np.triu(A, 1) > 0).sum())
    fig, ax = plt.subplots(figsize=(6.2, 6.8), dpi=170)
    ax.imshow(A, cmap=ListedColormap([ZERO, NZ, DIAG]), vmin=0, vmax=2, interpolation="nearest")
    tiles(ax, n)
    from matplotlib.patches import Patch
    below(ax, [Patch(color=NZ, label="nonzero of E"), Patch(color=DIAG, label="diagonal")], ncol=2)
    fig.text(0.08, 0.965, title, fontsize=14, weight="bold", color=INK, va="top")
    fig.text(0.08, 0.925,
             f"E = union over samples j of supp(phi_j) x supp(phi_j): the nonzeros of\n"
             f"M(alpha) = lambda I + Phi diag(alpha) Phi^T, Phi = chol(K)^T, kernel radius h = {args.h}.\n"
             f"Rows in the order the samples were generated; no reordering, no chordal extension.\n"
             f"{pairs} off-diagonal pairs ({100 * (2 * pairs + n) / n ** 2:.1f}% of entries).",
             fontsize=8.6, color=INK2, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.76)
    save(fig, name)
    return pairs


print("sobol direct aggregate pattern, pairs:", aggregate_direct(S, cS, "sobol_2d_aggregate", "Sobol: aggregate sparsity pattern"))


# 12. one figure: Sobol (top) and CACS (bottom), samples on the left, aggregate pattern E on the right
def comparison_figure(paper=False):
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(12.6, 12.4), dpi=170)
    # paper: top row samples (a) Sobol (b) CACS, bottom row patterns (c) Sobol (d) CACS; else one row per method
    P_SP, P_CP, P_SM, P_CM = ((0, 0), (0, 1), (1, 0), (1, 1)) if not paper else ((0, 0), (1, 0), (0, 1), (1, 1))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.12, 1] if not paper else [1, 1],
                  wspace=0.16 if not paper else 0.22, hspace=0.2 if not paper else 0.3,
                  left=0.07, right=0.94 if paper else 0.98, top=0.89 if paper else 0.87, bottom=0.085)
    lab = (lambda k, t: f"({k}) {t}") if paper else (lambda k, t: t)

    def points(ax, P, f, title):
        cf = ax.contourf(GX, GY, Z, levels=levels, cmap=CMAP, extend="max")
        ax.contour(GX, GY, Z, levels=levels[::2], colors="white", linewidths=0.6, alpha=0.8)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=INK2, lw=0.9, zorder=2))
        ax.plot(*tm[:2], marker="*", ms=17, color=STAR, mec=INK, mew=1.0, zorder=7, ls="none")
        ax.scatter(P[:, 0], P[:, 1], s=22, c="white", edgecolors=SAMPLE_EDGE, linewidths=0.9, zorder=5)
        ax.scatter(*P[np.argmin(f)], s=90, facecolors="none", edgecolors=BEST, linewidths=2.2, zorder=6)
        ax.set(xlim=(-0.035, 1.035), ylim=(-0.035, 1.035), aspect="equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=15 if paper else 10.5, weight="bold" if paper else "normal", color=INK, loc="left")
        cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("cost", color=INK2, fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5, colors=INK2, length=0)
        cb.outline.set_visible(False)

    def pattern(ax, st, title):
        n = st.n
        A = np.zeros((n, n))
        i, j = st.pattern[:, 0], st.pattern[:, 1]
        keep = ~st.is_fill
        A[i[keep], j[keep]] = A[j[keep], i[keep]] = 1
        big = np.asarray(max(st.cliques, key=len))
        blk = np.zeros((n, n), bool)
        blk[np.ix_(big, big)] = True
        A[blk & (A == 1)] = 2
        A[np.arange(n), np.arange(n)] = 3
        A[big, big] = 2
        ax.imshow(A, cmap=ListedColormap([ZERO, NZ, BIGC, DIAG]), vmin=0, vmax=3, interpolation="nearest")
        tiles(ax, n)
        ax.set_title(title, fontsize=15 if paper else 10.5, weight="bold" if paper else "normal", color=INK, loc="left",
                     pad=24 if paper else 20)
        ax.text(0.03, 0.03, f"largest clique: {len(big)} x {len(big)}", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=13 if paper else 10.5, weight="bold", color=BIGC,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=BIGC, linewidth=1.1))
        return st.stats

    # top: Sobol
    a = fig.add_subplot(gs[0, 0])
    a_sobol = a
    points(a, S, cS, f"{len(S)} samples in 2D" if paper else f"Sobol: {len(S)} samples, best {cS.min():.3f}")
    stS2 = structure_of(S, cS, rcm_order(S, args.h), False)
    pattern(fig.add_subplot(gs[1, 0] if paper else gs[0, 1]), stS2,
            "aggregate sparsity pattern" if paper else
            f"Sobol: aggregate pattern E\n{stS2.stats['E_edges']} pairs, {stS2.stats['num_cliques']} cliques")
    # bottom: CACS, samples with the kernel graph and the cliques of H
    a = fig.add_subplot(gs[0, 1] if paper else gs[1, 0])
    for k, q in enumerate(cliques):
        Pq, col = X[q], CLIQUE[k % len(CLIQUE)]
        if len(q) >= 3:
            try:
                hull = Pq[ConvexHull(Pq).vertices]
            except Exception:
                hull = Pq
            a.add_patch(Polygon(hull, closed=True, facecolor=col, alpha=0.32, edgecolor=col, linewidth=1.3, zorder=2))
        elif len(q) == 2:
            a.plot(*Pq.T, color=col, lw=2.5, alpha=0.7, zorder=2, solid_capstyle="round")
    a.add_collection(LineCollection([X[[i, j]] for i, j in G_edges], colors=EDGE, linewidths=0.55, alpha=0.4, zorder=3))
    a_cacs = a
    points(a, X, c, f"{len(X)} samples in 2D" if paper else
           f"CACS (max_clique {args.max_clique}): {len(X)} samples, best {c.min():.3f}")
    stC2 = structure_of(X, c, rcm_order(X, args.h), False)  # as cacs_2d_sparsity_rcm.png
    pattern(fig.add_subplot(gs[1, 1]), stC2,
            "aggregate sparsity pattern" if paper else
            f"CACS: aggregate pattern E\n{stC2.stats['E_edges']} pairs, {stC2.stats['num_cliques']} cliques")

    if not paper:
        fig.text(0.07, 0.975, "Sobol vs CACS: samples and the aggregate sparsity of M(alpha)", fontsize=15, weight="bold",
             color=INK, va="top")
    if not paper:
        fig.text(0.07, 0.95, f"2-D five-basin cost, Wendland kernel radius h = {args.h}, seed {args.seed}. "
             "E: nonzeros of M(alpha) = lambda I + Phi diag(alpha) Phi^T, rows in RCM order for both.\n"
             f"Star: true minimum {tm[2]:.3f}; pink ring: best sample; shaded polygons: cliques of H.",
             fontsize=9, color=INK2, va="top", linespacing=1.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=6, mfc="white", mec=SAMPLE_EDGE, label="sample"),
                        Line2D([], [], ls="none", marker="o", ms=10, mfc="none", mec=BEST, mew=2, label="best sample"),
                        Line2D([], [], ls="none", marker="*", ms=13, mfc=STAR, mec=INK, label="true minimum"),
                        Line2D([], [], color=EDGE, lw=1, alpha=0.6, label="kernel edge (distance < h)"),
                        Patch(facecolor=CLIQUE[0], alpha=0.45, edgecolor=CLIQUE[0], label="clique of H (one colour each)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.475) if paper else (0.3, 0.008), ncol=5 if paper else 3, frameon=False, fontsize=10.5 if paper else 9, labelcolor=INK,
               columnspacing=1.2, handletextpad=0.5)
    fig.legend(handles=[Patch(color=NZ, label="nonzero of E"), Patch(color=DIAG, label="diagonal"),
                        Patch(color=BIGC, label="one largest clique (one ADMM block)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.008) if paper else (0.77, 0.008), ncol=3 if paper else 2, frameon=False, fontsize=10.5 if paper else 9, labelcolor=INK,
               columnspacing=1.2)
    if paper:  # column headers: (a) Sobol on the left, (b) CACS on the right
        fig.canvas.draw()
        for ax_, txt in ((a_sobol, "(a) Sobol sampling"), (a_cacs, "(b) CACS sampling")):
            pos = ax_.get_position()
            fig.text((pos.x0 + pos.x1) / 2, 0.965, txt, ha="center", va="top", fontsize=19, weight="bold", color=INK)
    save(fig, "sobol_vs_cacs_2d_paper" if paper else "sobol_vs_cacs_2d")


comparison_figure()
comparison_figure(paper=True)
