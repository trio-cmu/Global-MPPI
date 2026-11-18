#!/usr/bin/env python3
"""Side-by-side slide figures of several real closed-loop runs (numpy + matplotlib only; use /usr/bin/python3).

  compare_runs.py OUT_DIR RUN_DIR [RUN_DIR ...]      # e.g. runs/closed_loop_test/compare_b1 runs/closed_loop_test/real_dial_gpu_b1 ...

Writes into OUT_DIR: compare_trajectory.png, compare_error_vs_time.png, compare_cycle_timeline.png, compare_plan_time.png.
Same conventions as run_figures.py (goal frame: base = R(theta) * sim + goal_p; time 0 = start of chunk 0).
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_figures import C_FB, C_GOAL, C_LOST, C_PLAN, C_READ, C_WALK, FONT, load, poly, wrap   # noqa: E402

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
LABELS = {"ps": "PS", "dial": "DIAL", "mppi": "MPPI", "globalmppi": "GlobalMPPI"}


def read_run(run):
    A = json.load(open(os.path.join(run, "args.json")))
    C, T, S = (load(os.path.join(run, f)) for f in ("cycles.csv", "t_pose_raw.csv", "xy_state_raw.csv"))
    gx, gy, gth = A["goal_base"]
    R = np.array([[math.cos(gth), math.sin(gth)], [-math.sin(gth), math.cos(gth)]])
    tr, tt = S["tracking"], S["recv_epoch_s"]
    starts = tt[1:][(tr[:-1] == 0) & (tr[1:] == 1)]
    stops = tt[1:][(tr[:-1] == 1) & (tr[1:] == 0)]
    t0 = starts[0] - float(C["t_s"][0]) - float(C["t_start_lat_s"][0])
    ok = np.isfinite(T["x_base"])
    cam_t = T["stamp_epoch_s"][ok] - 0.06 - t0
    cam_xy = (R @ (np.vstack([T["x_base"][ok], T["y_base"][ok]]) - np.array([[gx], [gy]]))).T
    cam_th = wrap(T["theta_base_rad"][ok] - gth)
    sel = (cam_t >= -0.5) & (cam_t <= float(C["t_s"][-1]) + 3.0)
    pen_xy = (R @ (np.vstack([S["x"], S["y"]]) - np.array([[gx], [gy]]))).T
    n = len(C)
    nst = min(len(stops), n)
    return dict(name=os.path.basename(os.path.normpath(run)), algo=LABELS.get(A["algo"], A["algo"]), A=A, C=C, n=n, t0=t0,
                cam_t=cam_t[sel], cam_pos=np.hypot(cam_xy[sel, 0], cam_xy[sel, 1]) * 1000, cam_ang=np.degrees(np.abs(cam_th[sel])),
                cam_xy=cam_xy[sel], pen_t=tt - t0, pen_xy=pen_xy, stops=stops, nst=nst,
                pen_at_stop=np.array([pen_xy[np.searchsorted(tt, stops[k]) - 1] for k in range(nst)]),
                Tm=np.array([[C["T_x"][k + 1], C["T_y"][k + 1]] if k + 1 < n else [np.nan] * 2 for k in range(nst)]),
                fb=set(int(k) for k in A.get("fallback_chunks", []) or []),
                reason=" ".join(str(A.get("stop_reason", "")).split()))


def short_reason(r):
    s = r["reason"].replace("STOP: ", "")
    if s.startswith("plateau"):
        return "plateau stop"
    if s.startswith("converged"):
        return "converged"
    if "silent" in s:
        return "camera stream silent (T lost)"
    return s[:30]


def main(out, runs):
    os.makedirs(out, exist_ok=True)
    D = [read_run(r) for r in runs]
    N = len(D)
    tmax = max(float(d["C"]["t_s"][-1]) for d in D) + 1.0

    # 1. trajectories, one panel per algorithm, shared axes
    fig, axs = plt.subplots(1, N, figsize=(7.4 * N, 6.2), sharex=True, sharey=True, squeeze=False)
    for ax, d in zip(axs[0], D):
        C = d["C"]
        ax.fill(*poly(0, 0, 0).T, facecolor="none", edgecolor=C_GOAL, lw=3, ls="--", label="goal")
        ax.fill(*poly(C["T_x"][0], C["T_y"][0], math.radians(C["T_th_deg"][0])).T, facecolor="#BBBBBB", edgecolor="k", alpha=0.6, label="T start")
        ax.fill(*poly(C["T_x"][-1], C["T_y"][-1], math.radians(C["T_th_deg"][-1])).T, facecolor="#6BAED6", edgecolor="#08519C", alpha=0.55, label="T final")
        ax.plot(d["cam_xy"][:, 0] * 1000, d["cam_xy"][:, 1] * 1000, color="#08519C", lw=1.2, alpha=0.7, label="T path")
        m = (d["pen_t"] >= -0.2) & (d["pen_t"] <= float(C["t_s"][-1]) + 1.0)
        ax.scatter(d["pen_xy"][m, 0] * 1000, d["pen_xy"][m, 1] * 1000, c=d["pen_t"][m], s=2, cmap="viridis", label="pusher (color = time)")
        good = np.array([k not in d["fb"] for k in range(d["nst"])])
        ax.scatter(d["Tm"][good, 0] * 1000, d["Tm"][good, 1] * 1000, s=35, c="#08519C", edgecolor="w", zorder=5, label="T at standstills")
        if (~good).any():
            ax.scatter(d["Tm"][~good, 0] * 1000, d["Tm"][~good, 1] * 1000, s=90, marker="D", c=C_FB, edgecolor="k", zorder=6, label="fallback standstill")
        ax.set_aspect("equal")
        ax.set_title(f"{d['algo']}: {A_txt(d)}", fontsize=FONT)
        ax.set_xlabel("x in goal frame (mm)")
    axs[0][0].set_ylabel("y in goal frame (mm)")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 7), framealpha=0.9)
    fig.suptitle("T block and pusher trajectory (goal frame)", fontsize=FONT + 3)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(os.path.join(out, "compare_trajectory.png"), dpi=200)
    plt.close(fig)

    # 2. errors vs time, algorithms overlaid
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for i, d in enumerate(D):
        col = COLORS[i % len(COLORS)]
        ax[0].plot(d["cam_t"], d["cam_pos"], color=col, lw=2.2, label=f"{d['algo']} (final {float(d['C']['pos_err_mm'][-1]):.0f} mm)")
        ax[1].plot(d["cam_t"], d["cam_ang"], color=col, lw=2.2, label=d["algo"])
        for k in d["fb"]:
            if k + 1 < d["n"]:
                for a_ in ax:
                    a_.axvline(float(d["C"]["t_s"][k + 1]), color=col, lw=2, ls=":")
    ax[0].axhline(10, color="k", ls="--", lw=2.5, label="threshold 10 mm")
    ax[1].axhline(3, color="k", ls="--", lw=2.5, label="threshold 3 deg")
    ax[0].set_ylabel("position error (mm)")
    ax[1].set_ylabel("angle error (deg)")
    ax[1].set_xlabel("time since the first chunk started (s)")
    ax[0].set_title("Error to the goal vs time (camera, every frame)", fontsize=FONT + 2)
    for a_ in ax:
        a_.set_ylim(bottom=0)
        a_.set_xlim(left=-0.5, right=tmax)
        a_.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_error_vs_time.png"), dpi=200)
    plt.close(fig)

    # 3. per cycle stacked bars (walk / read pose / plan), shared scales
    fig, axs = plt.subplots(N, 1, figsize=(13, 3.6 * N), sharex=True, sharey=True, squeeze=False)
    ymax = 0
    for ax, d in zip(axs[:, 0], D):
        C = d["C"]
        walk, read, plan = (np.nan_to_num(C[k]) for k in ("t_track_s", "meas_stop_to_pose_s", "plan_s"))
        x = np.arange(d["n"])
        ax.bar(x, walk, color=C_WALK, label="walk")
        ax.bar(x, read, bottom=walk, color=C_READ, label="read pose (stop to pose)")
        ax.bar(x, plan, bottom=walk + read, color=C_PLAN, label="plan")
        tot = walk + read + plan
        ymax = max(ymax, tot.max())
        ax.set_ylabel("time (s)")
        ax.set_title(f"{d['algo']}: mean cycle {tot[tot > 0].mean():.2f} s = walk {walk[walk > 0].mean():.2f} + read {read[read > 0].mean():.2f} + "
                     f"plan {plan[plan > 0].mean():.2f}", fontsize=FONT)
    axs[0][0].set_ylim(0, ymax * 1.15)
    axs[-1][0].set_xlabel("cycle (chunk index)")
    axs[0][0].legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_cycle_timeline.png"), dpi=200)
    plt.close(fig)

    # 4. planning time per step (log scale: the algorithms differ by an order of magnitude)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    data = []
    for i, d in enumerate(D):
        pt = d["C"]["plan_s"] * 1000
        data.append(pt)
        ax[0].plot(np.arange(d["n"]), pt, "o-", color=COLORS[i % len(COLORS)], lw=2, ms=5,
                   label=f"{d['algo']}: mean {pt.mean():.0f} ms, p95 {np.percentile(pt, 95):.0f} ms")
    ax[0].axhline(500, color="k", ls="--", lw=2, label="chunk length 500 ms")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("step (chunk index)")
    ax[0].set_ylabel("planning time (ms, log scale)")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    bp = ax[1].boxplot(data, labels=[d["algo"] for d in D], patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax[1].axhline(500, color="k", ls="--", lw=2)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("planning time (ms, log scale)")
    fig.suptitle("Planning time per step (GPU; JIT compile not included)", fontsize=FONT + 2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_plan_time.png"), dpi=200)
    plt.close(fig)
    print("wrote 4 figures in", out)


def A_txt(d):
    C = d["C"]
    return f"final {float(C['pos_err_mm'][-1]):.0f} mm / {float(C['ang_err_deg'][-1]):.1f} deg, {short_reason(d)}"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
