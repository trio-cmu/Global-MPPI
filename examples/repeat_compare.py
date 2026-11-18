#!/usr/bin/env python3
"""Compare two REAL open-loop runs of the same plan with each other (and with the sim open-loop replay).

  python repeat_compare.py [run_dir] [--a real_run1] [--b real_run2]

Reads <run_dir>/<a>/real_T.txt, real_xy.txt (t = 0 at the first tracking=1 sample) and the sim files of <run_dir>
(sim_T_traj_openloop.csv, sim_T_traj_openloop_band.csv). Writes <run_dir>/real_run1_vs_run2.png and repeat.csv
(goal frame; t, run A x y theta, run B x y theta, position difference, heading difference, and the sim
sensitivity band per time).
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/trio_sim/src/pusht_align"))
import frames as F

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", nargs="?", default=os.path.expanduser("~/trio_sim/runs/closedloop_ps1_slow"))
ap.add_argument("--a", default="real_run1")
ap.add_argument("--b", default="real_run2")
ap.add_argument("--dt", type=float, default=0.05)
a = ap.parse_args()
goal = F.load_goal()


def load(tag):
    d = os.path.join(a.run_dir, tag)
    T = np.loadtxt(os.path.join(d, "real_T.txt"), ndmin=2)
    X = np.loadtxt(os.path.join(d, "real_xy.txt"), ndmin=2)
    t0 = X[np.argmax(X[:, 7] > 0), 0]
    conv = [F.base_to_sim(r[1:3], r[4], goal) for r in T]
    P = np.array([[c[0][0], c[0][1], c[1]] for c in conv])
    P[:, 2] = np.unwrap(P[:, 2])
    return T[:, 0] - t0, P


tA, PA = load(a.a)
tB, PB = load(a.b)
t = np.arange(max(tA.min(), tB.min(), 0.0), min(tA.max(), tB.max()), a.dt)


def at(tt, P):
    return np.column_stack([np.interp(t, tt, P[:, k]) for k in range(3)])


A, B = at(tA, PA), at(tB, PB)
# headings: both start near 45 deg (unwrapped from their own first sample); compare modulo 2*pi
dth = np.degrees(np.array([F.wrap(x) for x in (A[:, 2] - B[:, 2])]))
dpos = np.linalg.norm(A[:, :2] - B[:, :2], axis=1) * 1000

Sm = np.loadtxt(os.path.join(a.run_dir, "sim_T_traj_openloop.csv"), delimiter=",", skiprows=1)
Bd = np.loadtxt(os.path.join(a.run_dir, "sim_T_traj_openloop_band.csv"), delimiter=",", skiprows=1)
sim = np.column_stack([np.interp(t, Sm[:, 0], Sm[:, k]) for k in (1, 2, 3)])
lo = np.column_stack([np.interp(t, Bd[:, 0], Bd[:, k]) for k in (1, 2, 3)])
hi = np.column_stack([np.interp(t, Bd[:, 0], Bd[:, k]) for k in (4, 5, 6)])
band_pos = np.maximum(np.linalg.norm(hi[:, :2] - sim[:, :2], axis=1), np.linalg.norm(lo[:, :2] - sim[:, :2], axis=1)) * 1000
band_ang = np.degrees(np.maximum(np.abs(hi[:, 2] - sim[:, 2]), np.abs(lo[:, 2] - sim[:, 2])))

np.savetxt(os.path.join(a.run_dir, "repeat.csv"),
           np.column_stack([t, A, B, dpos, dth, sim, band_pos, band_ang]), delimiter=",", fmt="%.6f", comments="",
           header=f"t_s,{a.a}_x,{a.a}_y,{a.a}_theta,{a.b}_x,{a.b}_y,{a.b}_theta,pos_diff_mm,heading_diff_deg,"
                  "sim_ol_x,sim_ol_y,sim_ol_theta,sim_band_pos_dev_mm,sim_band_ang_dev_deg (goal frame)")

# endpoints: mean of the last 1 s of each run
mA = PA[tA > tA.max() - 1.0].mean(0)
mB = PB[tB > tB.max() - 1.0].mean(0)
fin_pos = np.linalg.norm(mA[:2] - mB[:2]) * 1000
fin_ang = abs(math.degrees(F.wrap(mA[2] - mB[2])))
k = int(np.argmax(dpos))
ka = int(np.argmax(np.abs(dth)))
print(f"{a.a}: final T (goal frame) x={mA[0]:.4f} y={mA[1]:.4f} theta={math.degrees(F.wrap(mA[2])):.2f} deg "
      f"-> {np.linalg.norm(mA[:2]) * 1000:.1f} mm / {abs(math.degrees(F.wrap(mA[2]))):.2f} deg from goal")
print(f"{a.b}: final T (goal frame) x={mB[0]:.4f} y={mB[1]:.4f} theta={math.degrees(F.wrap(mB[2])):.2f} deg "
      f"-> {np.linalg.norm(mB[:2]) * 1000:.1f} mm / {abs(math.degrees(F.wrap(mB[2]))):.2f} deg from goal")
print(f"final pose difference between the two real runs: {fin_pos:.1f} mm, {fin_ang:.2f} deg")
print(f"whole trajectory: max position difference {dpos.max():.1f} mm at t={t[k]:.1f}s, max heading difference "
      f"{np.abs(dth).max():.1f} deg at t={t[ka]:.1f}s; mean {dpos.mean():.1f} mm / {np.abs(dth).mean():.1f} deg")
print(f"sim perturbation band for comparison: up to {band_pos.max():.1f} mm / {band_ang.max():.1f} deg")
for tt in (5, 10, 15, 20, 25, 30, 40, 47):
    i = int(np.argmin(abs(t - tt)))
    print(f"  t={tt:2d}s  pos diff {dpos[i]:6.1f} mm   heading diff {dth[i]:7.1f} deg   (sim band {band_pos[i]:6.1f} mm / {band_ang[i]:5.1f} deg)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
outline = np.array([[-.06, .02], [-.02, .02], [-.02, -.06], [.02, -.06], [.02, .02], [.06, .02], [.06, .06], [-.06, .06], [-.06, .02]])


def draw(th, xy, **kw):
    c, s = math.cos(th), math.sin(th)
    o = outline @ np.array([[c, -s], [s, c]]).T + xy
    ax[0].plot(o[:, 0], o[:, 1], **kw)


for r in Bd[::2]:
    ax[0].add_patch(Rectangle((r[1], r[2]), r[4] - r[1], r[5] - r[2], color="tab:blue", alpha=0.04, lw=0))
draw(0.0, np.zeros(2), color="green", lw=2, label="goal")
ax[0].plot(Sm[:, 1], Sm[:, 2], "b-", lw=1.5, label="sim open-loop (nominal); shaded: 5 perturbed replays")
ax[0].plot(PA[:, 0], PA[:, 1], color="tab:red", lw=1.2, label=f"real {a.a}")
ax[0].plot(PB[:, 0], PB[:, 1], color="tab:orange", lw=1.2, label=f"real {a.b}")
draw(PA[0, 2], PA[0, :2], color="tab:red", lw=1, ls="--")
draw(mA[2], mA[:2], color="tab:red", lw=1.2)
draw(PB[0, 2], PB[0, :2], color="tab:orange", lw=1, ls="--")
draw(mB[2], mB[:2], color="tab:orange", lw=1.2)
draw(Sm[-1, 3], Sm[-1, 1:3], color="b", lw=1)
ax[0].set_aspect("equal")
ax[0].grid(alpha=0.3)
ax[0].set_xlabel("x (goal frame) [m]")
ax[0].set_ylabel("y (goal frame) [m]")
ax[0].set_title("T path, same plan twice (dashed = start, solid = end)")
ax[0].legend(loc="upper left", fontsize=7)
a1 = ax[1]
a1.fill_between(t, 0, band_pos, color="gray", alpha=0.25, label="sim perturbation band (position)")
a1.plot(t, dpos, "k-", label=f"|{a.a} - {a.b}| position [mm]")
a1.set_xlabel("time since plan start [s]")
a1.set_ylabel("position difference [mm]")
a2 = a1.twinx()
a2.fill_between(t, 0, band_ang, color="plum", alpha=0.2, label="sim perturbation band (heading)")
a2.plot(t, np.abs(dth), "m-", label="heading difference [deg]")
a2.set_ylabel("heading difference [deg]")
a1.grid(alpha=0.3)
a1.set_title(f"run-to-run difference: final {fin_pos:.0f} mm / {fin_ang:.0f} deg, max {dpos.max():.0f} mm")
h1, l1 = a1.get_legend_handles_labels()
h2, l2 = a2.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
fig.tight_layout()
out = os.path.join(a.run_dir, "real_run1_vs_run2.png")
fig.savefig(out, dpi=130)
print("wrote", out, "and", os.path.join(a.run_dir, "repeat.csv"))
