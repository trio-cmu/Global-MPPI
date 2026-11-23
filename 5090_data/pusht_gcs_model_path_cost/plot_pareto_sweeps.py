"""Pareto plot with every baseline's sample sweep: path cost J (drop-tail, w_final = 10) vs plan time per cycle.

Each baseline is a line through 2k, 4k, 8k, 16k, 32k, 64k samples per iteration (5 iterations, 6 seeds each):
<method>_n<N>/ from run_samples.sh (DIAL, MPPI, MTP) and run_ps_samples.sh (PS). PS 2k is the main batch (ps/,
free GPU); PS 4k-32k ran while other users' jobs shared the GPU (slower times). CBO left out (fails on most seeds).
Global-MPPI: main batch. GCS: one whole-trajectory plan counted as one cycle (RTX 4090).

    python plot_pareto_sweeps.py [--w-final=W] [--keep-tail]
"""
import glob, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
from score_path_cost import score  # noqa: E402

W_FINAL, DROP, ERRBARS, NOSTD = 10.0, True, False, False
for a in sys.argv[1:]:
    if a == "--no-std": NOSTD = True  # no spread drawn
    if a == "--errorbars": ERRBARS = True  # error bars per point instead of shaded bands
    if a.startswith("--w-final"): W_FINAL = float(a.split("=", 1)[1])
    if a == "--keep-tail": DROP = False
PCT = "--percentile" in sys.argv  # median with 25th-75th percentile spread instead of mean ± std
NO_MTP = "--no-mtp" in sys.argv  # leave MTP out
GM_VAR = "--gm-variants" in sys.argv  # add Global-MPPI no-final-MPPI and MPPI-noise-0.2 (run_gm_variants.sh)
NS = [2000, 4000, 8000, 16000, 32000, 64000]
SWEEPS = {"ps": ("PS", "#52514e"), "dial": ("DIAL", "#eb6834"), "mppi": ("MPPI", "#1baf7a"), "mtp": ("MTP", "#8e6bd1")}
if NO_MTP: SWEEPS.pop("mtp")
GM_COLOR, GCS_COLOR, INK, INK_2, GRID = "#d43d51", "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e0"


def run_stats(folder, log_prefix, partial=False):
    T, J = [], []
    for s in range(6):
        f = glob.glob(f"{folder}/s{s}/*robot_pos_history*.npy")
        bad = not f or len(np.load(f[0])) < 101 or "KSOS solver failed" in open(f"console/{log_prefix}_s{s}.log").read()
        if bad and partial: continue
        if bad: return None
        t = [float(x) for c, x in re.findall(r"^Cycle ([\d.]+): plan time: ([\d.]+)s", open(f"console/{log_prefix}_s{s}.log").read(), re.M) if float(c) > 0]
        T.append(np.mean(t)); J.append(score(f[0], W_FINAL, 0.1, 0.02, DROP)["J"])
    if not J: return None
    if PCT:
        c = np.median(J); lo, hi = np.percentile(J, [25, 75])
        return np.mean(T), c, np.array([[c - lo], [hi - c]]), len(J)
    sd = np.std(J, ddof=1) if len(J) > 1 else 0.0
    return np.mean(T), np.mean(J), np.array([[sd], [sd]]), len(J)


curves = {}
for m in SWEEPS:
    pts = []
    for n in NS:
        folder = "ps" if (m == "ps" and n == 2000) else f"{m}_n{n}"
        r = run_stats(folder, folder)
        if r: pts.append((n, *r))
    curves[m] = pts
gm = run_stats("gm", "gm")
gm1 = run_stats("gm_blas1", "gm_blas1")  # same settings, OPENBLAS_NUM_THREADS=1
gcs_t = float(re.search(r'"relaxed_gcs_optimizer_seconds": ([\d.]+)',
                        (here.parent.parent / "4090_data/gcs_path_pusher_wp0.1/gcs_run.log").read_text()).group(1))
gcs_J = score(str(here.parent / "gcs_path_pusher_wp0.1/gcs_path_pusher_wp0.1_traj.npz"), W_FINAL, 0.1, 0.02, DROP)["J"]

fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=150)
ax.set_xscale("log"); ax.set_yscale("log")
ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#c9c8c2")
ax.tick_params(colors=INK_2, labelsize=10, length=0)

allpts = []
for m, pts in curves.items():
    lab, col = SWEEPS[m]
    x = [p[1] for p in pts]; y = [p[2] for p in pts]
    sd = np.hstack([p[3] for p in pts])  # rows: below, above
    if NOSTD:
        pass
    elif ERRBARS:
        off = {"ps": 0.94, "dial": 0.98, "mppi": 1.02, "mtp": 1.06}[m]  # spread same-time points apart
        ax.errorbar(np.array(x) * off, y, yerr=sd, fmt="none", ecolor=col, elinewidth=1.2, capsize=3, alpha=0.75, zorder=2)
    else:
        ax.fill_between(x, np.array(y) - sd[0], np.array(y) + sd[1], color=col, alpha=0.13, lw=0, zorder=1)
    ax.plot(x, y, color=col, lw=2, marker="o", ms=6.5, mec="white", mew=1.2, zorder=3, label=f"{lab}  (2k → 64k samples)")
    for n, xi, yi, *_ in pts:
        if n in (2000, 64000):
            ax.annotate(f"{n // 1000}k", (xi, yi), xytext=(0, -13 if m == "dial" and n == 2000 else 7), textcoords="offset points", ha="center", fontsize=8, color=col, weight="bold")
        allpts.append((xi, yi))
ax.errorbar(gm[0], gm[1], yerr=gm[2], fmt="o", ms=10, color=GM_COLOR, mec="white", mew=1.5, capsize=4, lw=1.5, zorder=4, label="Global-MPPI")
ax.annotate(f"Global-MPPI  {gm[1]:.3f}", (gm[0], gm[1]), xytext=(10, -14), textcoords="offset points", fontsize=9.5, color=INK)
ax.plot(gcs_t, gcs_J, "*", ms=17, color=GCS_COLOR, mec="white", mew=1.5, zorder=4, label="GCS (one plan = one cycle, 4090)")
ax.annotate(f"GCS  {gcs_J:.3f}", (gcs_t, gcs_J), xytext=(10, -4), textcoords="offset points", fontsize=9.5, color=INK, va="center")
if gm1:
    ax.annotate("", (gm1[0], gm1[1]), xytext=(gm[0], gm[1]), arrowprops=dict(arrowstyle="->", color=GM_COLOR, lw=1.2, ls="--"), zorder=3)
    ax.errorbar(gm1[0], gm1[1], yerr=None if NOSTD else gm1[2], fmt="o", ms=10, color=GM_COLOR, mfc="white", mec=GM_COLOR, mew=2, capsize=4, lw=1.5, zorder=4, label="Global-MPPI (1-thread BLAS)")
    if GM_VAR:  # crowded: label from the empty area below, with a leader line
        ax.annotate(f"1-thread BLAS  {gm1[1]:.3f}", (gm1[0], gm1[1]), xytext=(4.5, 0.143), ha="right", va="center", fontsize=9.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color="#b5b4ae", lw=0.8, shrinkB=7))
    else:
        ax.annotate(f"1-thread BLAS  {gm1[1]:.3f}", (gm1[0], gm1[1]), xytext=(4, 30), textcoords="offset points", ha="center", fontsize=9.5, color=INK)
    allpts.append((gm1[0], gm1[1]))
GM_VARS = [("gm_nomppi", "no final MPPI", "D", None), ("gm_noise0.2", "MPPI noise 0.2", "s", (12, -4))]
for folder, lab, mk, off in (GM_VARS if GM_VAR else []):
    r = run_stats(folder, folder, partial=True)
    if not r: continue
    nn = "" if r[3] == 6 else f", {r[3]}/6 seeds"
    ax.errorbar(r[0], r[1], yerr=None if NOSTD else r[2], fmt=mk, ms=9, color=GM_COLOR, mfc="white", mec=GM_COLOR, mew=2, capsize=4, lw=1.5, zorder=4, label=f"Global-MPPI, 1-thread BLAS, {lab}{nn}")
    if off is None:
        ax.annotate(f"{lab}  {r[1]:.3f}", (r[0], r[1]), xytext=(4.5, 0.133), ha="right", va="center", fontsize=9.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color="#b5b4ae", lw=0.8, shrinkB=7))
    else:
        ax.annotate(f"{lab}  {r[1]:.3f}", (r[0], r[1]), xytext=off, textcoords="offset points", fontsize=9.5, color=INK)
    if r[3] == 6: allpts.append((r[0], r[1]))
allpts += [(gm[0], gm[1]), (gcs_t, gcs_J)]

# Pareto front over every point
front, best = [], np.inf
for xp, yp in sorted(allpts):
    if yp < best: front.append((xp, yp)); best = yp
fx = [p[0] for p in front]; fy = [p[1] for p in front]
ax.step(fx + [400], fy + [fy[-1]], where="post", color="#9a9992", lw=1.4, ls="--", zorder=2, label="Pareto front")

ax.set_xlim(0.1, 400); ax.set_ylim(0.145 if NOSTD else (0.08 if GM_VAR else 0.12), 0.33)
yt = ([0.1] if GM_VAR else []) + [0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3]
ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt]); ax.minorticks_off()
xt = [0.1, 1, 10, 100]  # decades only
ax.set_xticks(xt); ax.set_xticklabels([f"{v:g}" for v in xt])
ax.set_xlabel("Planning time per control cycle (s, log)", color=INK, fontsize=11, weight="bold")
ax.set_ylabel(f"Path cost J = $L_T$ + 0.1 $L_P$ + {W_FINAL:g} $D_{{final}}$", color=INK, fontsize=11, weight="bold")
ax.legend(frameon=False, loc="lower right" if GM_VAR else "upper right", fontsize=8.5 if GM_VAR else 9, labelcolor=INK_2)
tail = "pusher motion after T stops not counted" if DROP else "all pusher motion counted"
ax.set_title(f"PushT path cost vs planning time, sample sweeps\n{('median' if PCT else 'mean') + ' over 6 seeds' if NOSTD else ('median, 25-75th pct.' if PCT else 'mean ± std') + ' of 6 seeds (' + ('error bars' if ERRBARS else 'bands') + ')'}; {tail}; CBO{'' if not NO_MTP else ', MTP'} not shown",
             loc="left", fontsize=10, color=INK, weight="bold")
fig.text(0.01, 0.01, "Samples per iteration, 5 iterations, RTX 5090. PS 4k-32k ran while other users' jobs shared the GPU (slower).\n" +
         ("Global-MPPI no-final-MPPI and noise-0.2: most seeds ran while other jobs shared the GPU (times inflated).\n" if GM_VAR else "") +
         ("Global-MPPI error bar" if NOSTD else "Error bars" if ERRBARS else "Shaded bands and error bar") + (": 25th-75th percentile of the 6 seeds; points are medians." if PCT else ": ± 1 std over the 6 seeds.") + " GCS: relaxed-GCS optimizer time for its one whole-trajectory plan, RTX 4090.",
         fontsize=8, color=INK_2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
out = here / f"pusht_pareto_sweeps{'_median' if PCT else ''}{'_nomtp' if NO_MTP else ''}{'_gmvariants' if GM_VAR else ''}{'' if W_FINAL == 10 else f'_wfinal{W_FINAL:g}'}{'_droptail' if DROP else ''}{'_nostd' if NOSTD else '_errorbars' if ERRBARS else ''}.png"
fig.savefig(out, facecolor="white"); print(out)
for m, pts in curves.items(): print(m, [(n, round(t, 2), round(j, 3)) for n, t, j, *_ in pts])
print("front", [(round(x, 2), round(y, 3)) for x, y in front])
