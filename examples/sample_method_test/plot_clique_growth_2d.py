"""Sobol's KSOS cliques grow with the number of samples; CACS's stay at max_clique.

2-D, the five-basin cost of plot_cacs_2d.py, Wendland kernel radius h = 0.09 for every sample set. For each
sample count n: the largest clique of the chordal decomposition the ADMM solver uses, built as GlobalMPPI builds
it (Sobol: RCM order and the solver's own chordal extension; CACS: reverse insertion order and the elimination
cliques). Writes

    plot/clique_growth_2d.png   largest clique vs number of samples (Sobol, CACS max_clique 10 and 20)
    plot/clique_growth_2d_sparsity.png
                                the aggregate pattern E at n = 400 for Sobol and CACS (max_clique 10), the largest
                                clique highlighted

    python plot_clique_growth_2d.py [--seed 3] [--h 0.09]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from global_mppi.algs.global_mppi import _elimination_cliques  # noqa: E402
from global_mppi.cacs import CacsConfig, cacs_sample  # noqa: E402
from ksos_tools.problems.features import build_problem  # noqa: E402
from ksos_tools.problems.sampling import rcm_order  # noqa: E402
from ksos_tools.solvers.ADMM.chordal import chordal_decomposition  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--h", type=float, default=0.09)
ap.add_argument("--out", default=str(Path(__file__).parent / "plot"))
args = ap.parse_args()
h = args.h

CENTERS = np.array([[0.22, 0.27, 0.00, 1.0], [0.70, 0.22, 0.10, 1.2], [0.80, 0.74, 0.02, 0.9],
                    [0.30, 0.78, 0.18, 1.1], [0.52, 0.50, 0.26, 1.3]])


def cost(X):
    X = np.atleast_2d(X)
    dx = X[:, None, 0] - CENTERS[None, :, 0]
    dy = X[:, None, 1] - CENTERS[None, :, 1]
    e = (dx * dx + 1.4 * dy * dy + 0.6 * dx * dy) * CENTERS[None, :, 3] * 9 + CENTERS[None, :, 2]
    return e.min(1) + 0.03 * np.sin(14 * X[:, 0] + 9 * X[:, 1]) ** 2


def sobol(n):
    P = qmc.Sobol(2, scramble=True, seed=args.seed + 7).random(2 ** int(np.ceil(np.log2(n))))[:n]
    p = rcm_order(P, h)
    return chordal_decomposition(build_problem(P[p], h, lambda _: cost(P[p])).Phi)


def cacs(n, max_clique):
    cfg = CacsConfig(n=n, initial=max(20, n // 5), reuse=5, h=h, max_clique=max_clique, batch=96, k=16,
                     global_every=50, replace=0)
    X, c, _ = cacs_sample(lambda P, pad_to=None: cost(P), [0, 0], [1, 1], cfg, seed=args.seed,
                          rng=np.random.default_rng(args.seed + 101))
    _, first = np.unique(X, axis=0, return_index=True)  # points clipped onto the same box corner
    keep = np.sort(first)  # an induced subgraph: the clique bound still holds
    X, c = X[keep], c[keep]
    o = np.arange(len(X))[::-1]
    prob = build_problem(X[o], h, lambda _: c[o])
    return chordal_decomposition(prob.Phi, cliques=_elimination_cliques(prob.Phi))


NS = [100, 150, 200, 300, 400, 600, 800, 1200, 1600]
rows = {"sobol": [], "cacs10": [], "cacs20": []}
for n in NS:
    s = sobol(n)
    rows["sobol"].append((s.n, s.stats["max_clique"], s.stats["fill_edges"]))
    for mc in (10, 20):
        st = cacs(n, mc)
        rows[f"cacs{mc}"].append((st.n, st.stats["max_clique"], st.stats["fill_edges"]))
    print(n, {k: v[-1] for k, v in rows.items()}, flush=True)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "clique_growth_2d.json").write_text(json.dumps({"h": h, "seed": args.seed, "requested_n": NS, **rows}, indent=1))

INK, INK2, RULE, BG = "#14213d", "#5c677d", "#d9dee5", "#ffffff"
C_SOBOL, C10, C20 = "#2e6ea6", "#e8590c", "#c9184a"

# 1. largest clique vs samples kept
fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=170)
for key, col, lab in [("sobol", C_SOBOL, "Sobol"), ("cacs20", C20, "CACS, max_clique 20"), ("cacs10", C10, "CACS, max_clique 10")]:
    pts = sorted({(n, m) for n, m, _ in rows[key]})
    x, y = zip(*pts)
    ax.plot(x, y, "-o", color=col, lw=2.2, ms=5.5, mec="white", mew=1.0, label=lab, zorder=3)
for mc, col in ((10, C10), (20, C20)):
    ax.axhline(mc, color=col, lw=1, ls=":", zorder=1)
    nmax = max(n for n, _, _ in rows[f"cacs{mc}"])
    ax.annotate(f"fills up at {nmax} samples", (nmax, mc), xytext=(8, 7 if mc == 20 else -13),
                textcoords="offset points", fontsize=8.5, color=col)
xs, ys = zip(*sorted({(n, m) for n, m, _ in rows["sobol"]}))
ax.annotate(f"{ys[-1]}", (xs[-1], ys[-1]), xytext=(-4, 8), textcoords="offset points", fontsize=9, color=C_SOBOL,
            ha="right", weight="bold")
ax.set_xlabel("number of samples", color=INK2)
ax.set_ylabel("largest clique (largest ADMM block)", color=INK2)
ax.set_xlim(0, 1700)
ax.set_ylim(0, max(ys) * 1.1)
ax.grid(True, color=RULE, lw=0.7)
ax.set_axisbelow(True)
ax.tick_params(colors=INK2, length=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(RULE)
ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)
fig.text(0.08, 0.97, "Sobol's cliques grow with the samples; CACS's are bounded", fontsize=13, weight="bold", color=INK,
         va="top")
fig.text(0.08, 0.915, f"2-D five-basin cost, kernel radius h = {h} for every sample set. Largest clique of the\n"
         "chordal decomposition the ADMM solver uses: each clique is one dense PSD block,\n"
         "one eigendecomposition per iteration.",
         fontsize=8.8, color=INK2, va="top", linespacing=1.45)
fig.subplots_adjust(left=0.1, right=0.97, bottom=0.11, top=0.78)
fig.savefig(out / "clique_growth_2d.png", facecolor=BG)
plt.close(fig)
print("saved", out / "clique_growth_2d.png")


# 2. the pattern at n = 400, largest clique highlighted
def pattern(st):
    n = st.n
    A = np.zeros((n, n))
    i, j = st.pattern[:, 0], st.pattern[:, 1]
    A[i, j] = A[j, i] = np.where(st.is_fill, 2, 1)
    big = max(st.cliques, key=len)
    A[np.ix_(big, big)] = 3
    return A, len(big)


fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.9), dpi=170)
cmap = ListedColormap(["white", "#9fb7d0", "#ff9f1c", "#c9184a"])
for ax, st, title in [(axes[0], sobol(400), "Sobol, 400 samples"), (axes[1], cacs(400, 10), "CACS (max_clique 10), 400 samples")]:
    A, big = pattern(st)
    ax.imshow(A, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(f"{title}\nlargest clique {big}  ·  {st.stats['num_cliques']} cliques  ·  "
                 f"{100 * (2 * st.stats['E_edges'] + st.n) / st.n ** 2:.1f}% nonzero",
                 fontsize=10, color=INK, loc="left")
    ax.set_xticks([0, st.n - 1]); ax.set_yticks([0, st.n - 1])
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for sp in ax.spines.values():
        sp.set_color(RULE)
fig.legend(handles=[Patch(color="#9fb7d0", label="nonzero of E"), Patch(color="#ff9f1c", label="chordal fill"),
                    Patch(color="#c9184a", label="largest clique (one ADMM block)")],
           loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.text(0.03, 0.975, "Aggregate sparsity at the same number of samples", fontsize=13, weight="bold", color=INK, va="top")
fig.text(0.03, 0.925, f"Sobol rows in RCM order with the solver's chordal extension; CACS rows in reverse insertion order. "
         f"Kernel radius h = {h}.", fontsize=8.8, color=INK2, va="top")
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.8, wspace=0.12)
fig.savefig(out / "clique_growth_2d_sparsity.png", facecolor=BG)
plt.close(fig)
print("saved", out / "clique_growth_2d_sparsity.png")
