#!/usr/bin/env python3
"""Slide-ready figures of ONE real closed-loop run (numpy + matplotlib only; run it with a python that has both, e.g. /usr/bin/python3).

  run_figures.py runs/closed_loop_test/<run> [--summary runs/closed_loop_test/summary.csv] [--no-figs]

Reads args.json, cycles.csv, t_pose_raw.csv, xy_state_raw.csv of the run directory and writes into the same directory:
  fig_trajectory.png      T block (goal frame): start, goal, last measured pose, path, pen path, every standstill (measurement) point
  fig_error_vs_time.png   position (mm) and angle (deg) error vs time with the 10 mm / 3 deg threshold lines
  fig_cycle_timeline.png  per cycle stacked bars: walk / read pose (stop -> pose) / plan
  fig_plan_time.png       planning time per step (line + histogram, mean / p95)
Measurement stops that fell back to the prediction (--max-fallbacks; args.json "fallback_chunks", fallback=1 rows of t_pose_raw.csv) are drawn in
orange; a run that ended with "T pose lost" gets a red cross at the pen position of that last stop. Goal frame: base = R(theta) * sim + goal_p.
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTLINE = np.array([[-.06, .02], [-.02, .02], [-.02, -.06], [.02, -.06], [.02, .02], [.06, .02], [.06, .06], [-.06, .06]])   # T, block frame (m)
C_WALK, C_READ, C_PLAN, C_FB, C_LOST, C_GOAL = "#4C78A8", "#F58518", "#54A24B", "#E8A100", "#D62728", "#2CA02C"
FONT = 15
plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT + 2, "axes.labelsize": FONT, "legend.fontsize": FONT - 2, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3})


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def load(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def poly(x, y, th):
    c, s = math.cos(th), math.sin(th)
    return (OUTLINE @ np.array([[c, -s], [s, c]]).T + [x, y]) * 1000.0


def main(run):
    A = json.load(open(os.path.join(run, "args.json")))
    C, T, S = (load(os.path.join(run, f)) for f in ("cycles.csv", "t_pose_raw.csv", "xy_state_raw.csv"))
    gx, gy, gth = A["goal_base"]
    R = np.array([[math.cos(gth), math.sin(gth)], [-math.sin(gth), math.cos(gth)]])         # base -> sim (goal frame)
    name = os.path.basename(os.path.normpath(run))
    algo = str(A.get("algo", "?")).upper()
    n = len(C)
    reason = str(A.get("stop_reason", ""))
    lost = "T pose lost" in reason
    fb_chunks = sorted(set(int(k) for k in A.get("fallback_chunks", []) or []))
    # --- time axis: t = 0 at the start of chunk 0 (first tracking=1 of XY_STATE)
    tr, tt = S["tracking"], S["recv_epoch_s"]
    starts = tt[1:][(tr[:-1] == 0) & (tr[1:] == 1)]
    stops = tt[1:][(tr[:-1] == 1) & (tr[1:] == 0)]
    t0 = starts[0] - float(C["t_s"][0]) - float(C["t_start_lat_s"][0]) if len(starts) else tt[0]
    # --- camera track in the goal frame (kept frames; fallback marker rows have no pose)
    ok = np.isfinite(T["x_base"])
    if "fallback" in T.dtype.names:
        fb_t = T["recv_epoch_s"][~ok & (T["fallback"] == 1)]
    else:
        fb_t = np.array([])
    cam_t = T["stamp_epoch_s"][ok] - 0.06 - t0
    cam_xy = (R @ (np.vstack([T["x_base"][ok], T["y_base"][ok]]) - np.array([[gx], [gy]]))).T
    cam_th = wrap(T["theta_base_rad"][ok] - gth)
    sel = (cam_t >= -0.5) & (cam_t <= float(C["t_s"][-1]) + 3.0)
    cam_t, cam_xy, cam_th = cam_t[sel], cam_xy[sel], cam_th[sel]
    cam_pos = np.hypot(cam_xy[:, 0], cam_xy[:, 1]) * 1000
    cam_ang = np.degrees(np.abs(cam_th))
    pen_t = tt - t0
    pen_xy = (R @ (np.vstack([S["x"], S["y"]]) - np.array([[gx], [gy]]))).T
    # --- standstill points: stop i = end of chunk i; the pose measured there is the state T of chunk i+1
    stop_t = stops - t0
    nst = min(len(stop_t), n)
    Tm = np.array([[C["T_x"][k + 1], C["T_y"][k + 1], math.radians(C["T_th_deg"][k + 1])] if k + 1 < n else [np.nan] * 3 for k in range(nst)])
    pen_at_stop = np.array([pen_xy[np.searchsorted(tt, stops[k]) - 1] for k in range(nst)])
    is_fb = np.array([k in fb_chunks for k in range(nst)])
    is_lost = np.array([lost and k == nst - 1 and k + 1 >= n for k in range(nst)])
    short = reason.replace("STOP: ", "")
    short = "T pose lost (no usable frames after the stop)" if lost else short[:70]
    sub = f"{name}  |  {algo}  |  {short}"
    fb_note = f"  |  {len(fb_chunks)} fallback stop(s)" if fb_chunks else ""

    # ===== 1. trajectory
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.fill(*poly(0, 0, 0).T, facecolor="none", edgecolor=C_GOAL, lw=3, ls="--", label="goal")
    ax.fill(*poly(C["T_x"][0], C["T_y"][0], math.radians(C["T_th_deg"][0])).T, facecolor="#BBBBBB", edgecolor="k", alpha=0.6, label="T start")
    last = n - 1
    ax.fill(*poly(C["T_x"][last], C["T_y"][last], math.radians(C["T_th_deg"][last])).T, facecolor="#6BAED6", edgecolor="#08519C", alpha=0.55,
            label="T last measured")
    ax.plot(cam_xy[:, 0] * 1000, cam_xy[:, 1] * 1000, color="#08519C", lw=1.2, alpha=0.7, label="T path (camera)")
    pm = (pen_t >= -0.2) & (pen_t <= float(C["t_s"][-1]) + 1.0)
    sc = ax.scatter(pen_xy[pm, 0] * 1000, pen_xy[pm, 1] * 1000, c=pen_t[pm], s=3, cmap="viridis", label="pusher path (color = time)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("time (s)")
    good = ~is_fb & ~is_lost & np.isfinite(Tm[:, 0])
    ax.scatter(Tm[good, 0] * 1000, Tm[good, 1] * 1000, s=70, c="#08519C", edgecolor="w", zorder=5, label="T at each standstill (measured)")
    if is_fb.any():
        ax.scatter(Tm[is_fb, 0] * 1000, Tm[is_fb, 1] * 1000, s=130, marker="D", c=C_FB, edgecolor="k", zorder=6, label="standstill: fallback (predicted)")
    ax.scatter(pen_at_stop[:, 0] * 1000, pen_at_stop[:, 1] * 1000, s=45, marker="s", facecolor="none", edgecolor="k", zorder=5, label="pusher at each standstill")
    if is_lost.any():
        ax.scatter(pen_at_stop[is_lost, 0] * 1000, pen_at_stop[is_lost, 1] * 1000, s=220, marker="X", c=C_LOST, edgecolor="k", zorder=7,
                   label="T pose lost: run stopped")
    ax.set_aspect("equal")
    ax.set_xlabel("x in goal frame (mm)")
    ax.set_ylabel("y in goal frame (mm)")
    ax.set_title("T block and pusher trajectory (goal frame)\n" + sub + fb_note, fontsize=FONT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(run, "fig_trajectory.png"), dpi=200)
    plt.close(fig)

    # ===== 2. error vs time
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    t_pt = C["t_s"]
    for a_, cam, pts, thr, lab, unit in [(ax[0], cam_pos, C["pos_err_mm"], 10.0, "position error", "mm"), (ax[1], cam_ang, C["ang_err_deg"], 3.0, "angle error", "deg")]:
        a_.plot(cam_t, cam, color="#9ecae1", lw=1.5, label="camera (every frame)")
        a_.plot(t_pt, pts, "o-", color="#08519C", ms=6, lw=1.5, label="state at each chunk start")
        a_.axhline(thr, color=C_LOST, ls="--", lw=2.5, label=f"threshold {thr:g} {unit}")
        for k in fb_chunks:
            if k + 1 < n:
                a_.axvline(t_pt[k + 1], color=C_FB, lw=3, alpha=0.8)
        if lost and nst:
            a_.axvline(stop_t[nst - 1], color=C_LOST, lw=3, alpha=0.6)
        a_.set_ylabel(f"{lab} ({unit})")
        a_.set_ylim(bottom=0)
    ax[0].plot([], [], color=C_FB, lw=3, label="fallback standstill") if fb_chunks else None
    ax[0].plot([], [], color=C_LOST, lw=3, alpha=0.6, label="T pose lost: stop") if lost else None
    ax[0].legend(loc="upper right")
    ax[1].set_xlabel("time since the first chunk started (s)")
    ax[0].set_title("Error to the goal vs time\n" + sub + fb_note, fontsize=FONT)
    fig.tight_layout()
    fig.savefig(os.path.join(run, "fig_error_vs_time.png"), dpi=200)
    plt.close(fig)

    # ===== 3. cycle timeline: walk / read pose / plan (same split as the latency breakdown)
    walk = np.nan_to_num(C["t_track_s"])
    read = np.nan_to_num(C["meas_stop_to_pose_s"])
    plan = np.nan_to_num(C["plan_s"])
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(n)
    ax.bar(x, walk, color=C_WALK, label=f"walk (mean {walk[walk > 0].mean():.2f} s)")
    ax.bar(x, read, bottom=walk, color=C_READ, label=f"read pose: stop to pose (mean {read[read > 0].mean():.2f} s)")
    ax.bar(x, plan, bottom=walk + read, color=C_PLAN, label=f"plan (mean {plan[plan > 0].mean():.2f} s)")
    for k in fb_chunks:
        if k + 1 < n:
            ax.bar(k + 1, walk[k + 1] + read[k + 1] + plan[k + 1], fill=False, edgecolor=C_FB, lw=3.5)
    if fb_chunks:
        ax.bar([], [], fill=False, edgecolor=C_FB, lw=3.5, label="cycle after a fallback standstill")
    tot = walk + read + plan
    ax.set_xlabel("cycle (chunk index)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Time per cycle (mean {tot[tot > 0].mean():.2f} s)\n" + sub + fb_note, fontsize=FONT)
    ax.set_xticks(x)
    ax.set_ylim(0, tot.max() * 1.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(run, "fig_cycle_timeline.png"), dpi=200)
    plt.close(fig)

    # ===== 4. planning time per step (the one-off JIT compile happens before the run and is not in these numbers)
    pt = C["plan_s"] * 1000
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].plot(np.arange(n), pt, "o-", color=C_PLAN, lw=2)
    ax[0].axhline(pt.mean(), color="k", ls="--", label=f"mean {pt.mean():.1f} ms")
    ax[0].axhline(np.percentile(pt, 95), color=C_LOST, ls=":", lw=2.5, label=f"p95 {np.percentile(pt, 95):.1f} ms")
    ax[0].set_xlabel("step (chunk index)")
    ax[0].set_xticks(np.arange(0, n, max(1, n // 10)))
    ax[0].set_ylabel("planning time (ms)")
    ax[0].set_ylim(0, max(pt.max() * 1.25, 1))
    ax[0].legend()
    ax[1].hist(pt, bins=min(15, max(5, n // 2)), color=C_PLAN, edgecolor="k")
    ax[1].set_xlabel("planning time (ms)")
    ax[1].set_ylabel("steps")
    fig.suptitle(f"Planning time per step, {algo} (n = {n}, max {pt.max():.1f} ms)\n{sub}", fontsize=FONT)
    fig.tight_layout()
    fig.savefig(os.path.join(run, "fig_plan_time.png"), dpi=200)
    plt.close(fig)
    print("wrote fig_trajectory.png fig_error_vs_time.png fig_cycle_timeline.png fig_plan_time.png in", run)


SUMMARY_COLS = ["run", "algo", "final_pos_mm", "final_ang_deg", "final_source", "reached_10mm_3deg", "duration_s", "plan_mean_ms", "plan_p95_ms",
                "fallbacks", "truncated_by_max_time", "cycles", "stop_reason"]


def summarize(run):
    """One summary row (dict) of a run directory; works for failed / stopped runs too (missing pieces become empty)."""
    A = json.load(open(os.path.join(run, "args.json")))
    row = dict.fromkeys(SUMMARY_COLS, "")
    row["run"], row["algo"] = os.path.basename(os.path.normpath(run)), A.get("algo", "")
    reason = " ".join(str(A.get("stop_reason", "")).split())
    row["stop_reason"] = reason.replace(",", ";")[:90]
    row["truncated_by_max_time"] = "yes" if reason.startswith("time limit") or "time limit exceeded" in reason else "no"
    row["fallbacks"] = len(A.get("fallback_chunks", []) or [])
    pos, ang, src = A.get("final_pos_mm"), A.get("final_ang_deg"), "final reading"
    C = None
    cp = os.path.join(run, "cycles.csv")
    if os.path.exists(cp) and os.path.getsize(cp) > 0:
        C = np.atleast_1d(load(cp))
        if C.size == 0 or "pos_err_mm" not in (C.dtype.names or ()):
            C = None
    if C is not None:
        row["cycles"] = len(C)
        pt = C["plan_s"][np.isfinite(C["plan_s"])] * 1000
        if len(pt):
            row["plan_mean_ms"], row["plan_p95_ms"] = f"{pt.mean():.1f}", f"{np.percentile(pt, 95):.1f}"
    if pos is None or not np.isfinite(pos):
        if C is not None and len(C):
            pos, ang, src = float(C["pos_err_mm"][-1]), float(C["ang_err_deg"][-1]), "last measured"
        else:
            pos, ang, src = None, None, "none"
    row["final_source"] = src
    if pos is not None:
        row["final_pos_mm"], row["final_ang_deg"] = f"{pos:.1f}", f"{ang:.2f}"
        row["reached_10mm_3deg"] = "yes" if (pos < A.get("pos_tol_mm", 10.0) and ang < A.get("ang_tol_deg", 3.0)) else "no"
    else:
        row["reached_10mm_3deg"] = "no"
    sp, xp = os.path.join(run, "xy_state_raw.csv"), None
    if C is not None and os.path.exists(sp) and os.path.getsize(sp) > 0:
        S = np.atleast_1d(load(sp))
        tr, tt = S["tracking"], S["recv_epoch_s"]
        st = tt[1:][(tr[:-1] == 0) & (tr[1:] == 1)]
        sp_ = tt[1:][(tr[:-1] == 1) & (tr[1:] == 0)]
        if len(st) and len(sp_):
            row["duration_s"] = f"{sp_[-1] - (st[0] - float(C['t_s'][0]) - float(C['t_start_lat_s'][0])):.1f}"
    return row


def append_summary(run, csv_path):
    import csv
    row = summarize(run)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    print("summary row appended to", csv_path, ":", ", ".join(f"{k}={row[k]}" for k in SUMMARY_COLS[:11]))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        sys.exit(__doc__)
    run, rc = args[0], 0
    if not os.path.exists(os.path.join(run, "args.json")):
        print(f"run_figures: no args.json in {run} (the run did not get far enough): nothing to do")
        sys.exit(0)
    if "--no-figs" not in args:
        try:
            main(run)
        except Exception as e:                      # a failed figure must never matter
            print(f"run_figures: figures failed ({type(e).__name__}: {e})")
            rc = 1
    if "--summary" in args:
        try:
            append_summary(run, args[args.index("--summary") + 1])
        except Exception as e:
            print(f"run_figures: summary failed ({type(e).__name__}: {e})")
            rc = 1
    sys.exit(rc)
