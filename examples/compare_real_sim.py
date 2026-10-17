#!/usr/bin/env python3
"""One figure: real closed-loop run vs the --sim runs of the same setting (error curves, T path, prediction error).

  /usr/bin/python3 examples/compare_real_sim.py REAL_DIR SIM_DIR [SIM_DIR ...] [--out FILE]

Each dir is an output dir of closed_loop_test.py (cycles.csv, trace.csv, args.json).
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("real")
ap.add_argument("sims", nargs="+")
ap.add_argument("--out", default=None)
ap.add_argument("--others-label", default="sim")
ap.add_argument("--first-label", default="real")
a = ap.parse_args()
OUTLINE = np.array([[-.06, .02], [-.02, .02], [-.02, -.06], [.02, -.06], [.02, .02], [.06, .02], [.06, .06], [-.06, .06], [-.06, .02]])


def load(d):
    C = np.loadtxt(os.path.join(d, "cycles.csv"), delimiter=",", skiprows=1, ndmin=2)
    T = np.loadtxt(os.path.join(d, "trace.csv"), delimiter=",", skiprows=1, ndmin=2)
    info = json.load(open(os.path.join(d, "args.json")))
    return C, T, info


def errs(T):
    return np.linalg.norm(T[:, 1:3], axis=1) * 1000, np.degrees(np.abs((T[:, 3] + np.pi) % (2 * np.pi) - np.pi))


def draw(ax, xy, th, **kw):
    c, s = math.cos(th), math.sin(th)
    o = OUTLINE @ np.array([[c, -s], [s, c]]).T + xy
    ax.plot(o[:, 0], o[:, 1], **kw)


runs = [(a.first_label, *load(a.real))] + [(f"{a.others_label} {os.path.basename(d.rstrip('/'))}", *load(d)) for d in a.sims]
fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
draw(ax[0, 0], np.zeros(2), 0.0, color="green", lw=2, label="goal")
for k, (name, C, T, info) in enumerate(runs):
    real = k == 0
    col = "tab:red" if real else "tab:blue"
    kw = dict(color=col, lw=1.8 if real else 0.9, alpha=1.0 if real else 0.6)
    ax[0, 0].plot(T[:, 1], T[:, 2], label=f"{name}: {info['stop_reason'][:28]}", **kw)
    draw(ax[0, 0], T[-1, 1:3], T[-1, 3], color=col, lw=1.2, alpha=kw["alpha"])
    p, ang = errs(T)
    ax[0, 1].plot(T[:, 0], p, **kw)
    ax[1, 0].plot(T[:, 0], ang, **kw)
    ok = ~np.isnan(C[:, 21])
    ax[1, 1].plot(C[ok, 0], C[ok, 21], "o-" if real else ".-", ms=4 if real else 3, **kw)
draw(ax[0, 0], runs[0][2][0, 1:3], runs[0][2][0, 3], color="gray", lw=1, ls="--")
ax[0, 0].set_aspect("equal")
ax[0, 0].set_xlabel("x (goal frame) [m]")
ax[0, 0].set_ylabel("y (goal frame) [m]")
ax[0, 0].set_title("T path (dashed = start, solid = end)")
ax[0, 0].legend(fontsize=7)
ax[0, 1].axhline(10, color="g", ls="--")
ax[0, 1].set_ylabel("T position error [mm]")
ax[1, 0].axhline(3, color="g", ls="--")
ax[1, 0].set_ylabel("T heading error [deg]")
ax[1, 1].set_ylabel("prediction error, position [mm]")
ax[1, 1].set_xlabel("round")
ax[0, 1].set_xlabel("run time [s]")
ax[1, 0].set_xlabel("run time [s]")
ax[1, 1].set_title("planner-model prediction of the sent chunk vs next measured T")
for x in ax.ravel():
    x.grid(alpha=0.3)
fig.suptitle(f"closed loop, chunk 0.5 s: {a.first_label} (red) vs {a.others_label} (blue)")
fig.tight_layout()
out = a.out or os.path.join(a.real, "real_vs_sim.png")
fig.savefig(out, dpi=130)
print("wrote", out)
