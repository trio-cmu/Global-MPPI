"""Pareto plot (CBO left out): path cost J (drop-tail scoring, w_final = 10) vs total planning time for the 2 s task.

Controllers: sum of the per-cycle plan times of cycles 1-100 (console logs; cycle 0 = jit, left out), RTX 5090.
GCS: one open-loop plan, its relaxed-GCS optimizer time (RTX 4090 machine, 4090_data/gcs_path_pusher_wp0.1/gcs_run.log).

    python plot_pareto.py [--w-final=W] [--keep-tail]
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

W_FINAL, DROP, PER_CYCLE, PS_SWEEP, KEEP_PS = 10.0, True, False, False, False
for a in sys.argv[1:]:
    if a == "--keep-ps": KEEP_PS = True  # with --ps-sweep: keep the main batch's PS point as well
    if a == "--ps-sweep": PS_SWEEP = True  # draw PS's sample sweep (ps_n<N>, run_ps_samples.sh) in place of the PS point
    if a == "--per-cycle": PER_CYCLE = True  # x = time per control cycle; GCS's one solve counts as one cycle
    if a.startswith("--w-final"): W_FINAL = float(a.split("=", 1)[1])
    if a == "--keep-tail": DROP = False
METHODS = {"ps": ("PS", "#52514e"), "mppi": ("MPPI", "#1baf7a"), "dial": ("DIAL", "#eb6834"),
           "mtp": ("MTP", "#8e6bd1"), "gm": ("Global-MPPI", "#d43d51")}
GCS_COLOR, INK, INK_2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e0"
GCS_LOG = here.parent.parent / "4090_data/gcs_path_pusher_wp0.1/gcs_run.log"
gcs_t = float(re.search(r'"relaxed_gcs_optimizer_seconds": ([\d.]+)', GCS_LOG.read_text()).group(1))
gcs_J = score(str(here.parent / "gcs_path_pusher_wp0.1/gcs_path_pusher_wp0.1_traj.npz"), W_FINAL, 0.1, 0.02, DROP)["J"]

SWEEP_N = [n for n in (1000, 2000, 4000, 8000, 16000, 32000, 64000) if len(glob.glob(f"ps_n{n}/s*/*robot_pos_history*.npy")) == 6]
if PS_SWEEP and not KEEP_PS:
    del METHODS["ps"]
pts = []  # (label, color, total time mean, std, J mean, std, per-cycle ms)
for m, (lab, col) in METHODS.items():
    T, J = [], []
    for s in range(6):
        t = [float(x) for c, x in re.findall(r"^Cycle ([\d.]+): plan time: ([\d.]+)s", open(f"console/{m}_s{s}.log").read(), re.M) if float(c) > 0]
        T.append(sum(t))
        J.append(score(glob.glob(f"{m}/s{s}/*robot_pos_history*.npy")[0], W_FINAL, 0.1, 0.02, DROP)["J"])
    k = 100 if PER_CYCLE else 1
    pts.append((lab, col, np.mean(T) / k, np.std(T, ddof=1) / k, np.mean(J), np.std(J, ddof=1), np.mean(T) / 100 * 1e3))
sweep = []
if PS_SWEEP:
    for n in SWEEP_N:
        T, J = [], []
        for s_ in range(6):
            t = [float(x) for c, x in re.findall(r"^Cycle ([\d.]+): plan time: ([\d.]+)s", open(f"console/ps_n{n}_s{s_}.log").read(), re.M) if float(c) > 0]
            T.append(np.mean(t) if PER_CYCLE else sum(t))
            J.append(score(glob.glob(f"ps_n{n}/s{s_}/*robot_pos_history*.npy")[0], W_FINAL, 0.1, 0.02, DROP)["J"])
        sweep.append((n, np.mean(T), np.mean(J), np.std(J, ddof=1)))
pts.append(("GCS", GCS_COLOR, gcs_t, 0, gcs_J, 0, None))

fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
ax.set_xscale("log"); ax.set_yscale("log")
ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#c9c8c2")
ax.tick_params(colors=INK_2, labelsize=10, length=0)

XMIN, XMAX = (0.05, 300) if PER_CYCLE else (5, 3e3)
if sweep:
    sx = [p[1] for p in sweep]; sy = [p[2] for p in sweep]
    ax.plot(sx, sy, color="#52514e", lw=2, zorder=2, marker="s", ms=7, mec="white", label=f"PS, {SWEEP_N[0] // 1000}k → {SWEEP_N[-1] // 1000}k samples / iteration")
    ax.errorbar(sx, sy, yerr=[p[3] for p in sweep], fmt="s", ms=7, color="#52514e", mec="white", mew=1.2, capsize=3, lw=1.2, zorder=3)
    for n, x, y, _ in sweep:
        up = n in (2000, 8000, 16000)
        ax.annotate(f"{n // 1000}k\n{y:.3f}", (x, y), xytext=(-9 if n == 64000 else 9, -4 if n == 64000 else 4 if up else -4),
                    textcoords="offset points", ha="right" if n == 64000 else "left", va="bottom" if up else "top", fontsize=8.5, color=INK_2, linespacing=1.0)
    if KEEP_PS:  # the main batch's PS point (2000 samples, free GPU) to the sweep's 2k point (same setting, shared GPU)
        ps_main = next(p for p in pts if p[0] == "PS")
        ax.plot([ps_main[2] * 0.91, sx[0]], [ps_main[4], sy[0]], color="#52514e", lw=2, zorder=2)
    pts_front = pts + [("PS", None, x, 0, y, 0, None) for _, x, y, _ in sweep]
else:
    pts_front = pts
# Pareto front: sorted by time, keep points that lower the best J so far
front, best = [], np.inf
for p in sorted(pts_front, key=lambda p: (round(p[2], 2), p[4])):  # times within 1 s count as a tie: the lowest J of them
    if p[4] < best: front.append(p); best = p[4]
fx = [p[2] for p in front]; fy = [p[4] for p in front]
ax.step(fx + [XMAX], fy + [fy[-1]], where="post", color="#9a9992", lw=1.5, ls="--", zorder=1, label="Pareto front")

# baselines share the same time (~17 s): spread their labels vertically
JIT = {"PS": 0.91, "MPPI": 0.97, "DIAL": 1.03, "MTP": 1.09}  # same time: spread the error bars apart
LABEL_Y = {"MTP": 0.194, "MPPI": 0.289, "DIAL": 0.215, "PS": 0.24} if PS_SWEEP else {"MPPI": 0.289, "PS": 0.235, "DIAL": 0.215, "MTP": 0.194}
for lab, col, t, ts, J, Js, ms in pts:
    t = t * JIT.get(lab, 1.0)
    mk = "*" if lab == "GCS" else "o"
    ax.errorbar(t, J, yerr=Js if Js else None, xerr=ts if ts else None, fmt=mk, ms=16 if mk == "*" else 9,
                color=col, mec="white", mew=1.5, capsize=4, lw=1.5, zorder=3)
    note = ("one plan = one cycle (4090)" if PER_CYCLE else "one offline plan (4090)") if lab == "GCS" else f"{ms:.0f} ms / cycle"
    txt = f"{lab}  {J:.3f}\n{note}"
    if lab in ("GCS", "Global-MPPI"):
        ax.annotate(txt, (t, J), xytext=(12, -4 if lab == "GCS" else (-22 if PS_SWEEP else 22 if PER_CYCLE else 6)), textcoords="offset points", fontsize=9.5, color=INK, va="center")
    else:
        ty = LABEL_Y[lab] or J
        ax.annotate(f"{lab}  {J:.3f}", (t, J), xytext=(pts[0][2] * 0.72, ty), textcoords="data", fontsize=9.5, color=INK,
                    va="center", ha="right", arrowprops=dict(arrowstyle="-", color="#c9c8c2", lw=0.8, shrinkA=2, shrinkB=5))
ax.text(pts[0][2] * (1.6 if PER_CYCLE else 1), 0.145, ("MPPI · DIAL · MTP\n" if PS_SWEEP and not KEEP_PS else "PS · MPPI · DIAL · MTP\n") + f"≈{pts[0][6]:.0f} ms / cycle", ha="center", va="top", fontsize=9, color=INK_2, bbox=dict(fc="white", ec="none", pad=1))

ax.set_xlim(XMIN, XMAX); ax.set_ylim(0.13, 0.36)
yt = [0.15, 0.2, 0.25, 0.3, 0.35]
ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt]); ax.minorticks_off()
xt = [0.1, 0.3, 1, 3, 10, 30, 100] if PER_CYCLE else [10, 30, 100, 300, 1000]
ax.set_xticks(xt); ax.set_xticklabels([f"{v:g}" for v in xt])
ax.set_xlabel("Planning time per control cycle (s, log)" if PER_CYCLE else "Total planning time for the 2 s task (s, log)", color=INK, fontsize=11, weight="bold")
ax.set_ylabel(f"Path cost J = $L_T$ + 0.1 $L_P$ + {W_FINAL:g} $D_{{final}}$", color=INK, fontsize=11, weight="bold")
ax.legend(frameon=False, loc="upper right", fontsize=9.5, labelcolor=INK_2)
tail = "pusher motion after the T stops not counted" if DROP else "all pusher motion counted"
ax.set_title(f"PushT path cost vs planning time\nmean ± std over 6 seeds; {tail}; CBO not shown", loc="left", fontsize=11, color=INK, weight="bold")
fig.text(0.01, 0.01, ("Controllers: mean plan time of cycles 1-100, RTX 5090.  GCS: relaxed-GCS optimizer time for its one whole-trajectory plan, RTX 4090."
         + ("\nPS 2k-32k sweep: run while other users' jobs shared the GPU (2k: 377 ms/cycle there vs 171 ms in the main batch); PS 64k: free GPU."
       if PS_SWEEP else "")
    if PER_CYCLE else "Controllers: sum of 100 cycle plan times, RTX 5090.  GCS: relaxed-GCS optimizer time for the whole trajectory, RTX 4090."),
         fontsize=8, color=INK_2)
fig.tight_layout(rect=(0, 0.06 if PS_SWEEP else 0.03, 1, 1))
out = here / f"pusht_pareto{'_percycle' if PER_CYCLE else ''}{('_pssweep_keepps' if KEEP_PS else '_pssweep') if PS_SWEEP else ''}{'' if W_FINAL == 10 else f'_wfinal{W_FINAL:g}'}{'_droptail' if DROP else ''}.png"
fig.savefig(out, facecolor="white"); print(out)
for p in pts: print(f"{p[0]:12s} time {p[2]:7.1f} s  J {p[4]:.3f}")
