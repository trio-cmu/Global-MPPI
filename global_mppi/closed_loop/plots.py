#!/usr/bin/env python3
"""Process figure of one episode (numpy + matplotlib only, so it also runs under a plain system python).

  plots.py --hist qpos_history.npy --out fig.png [--label TEXT] [--anchor 0 0.1] [--times 0 10 20 30 40 50 60]

Top row: snapshots (T, pusher, green goal outline, pusher trail so far). Bottom row: position error, heading error
and pusher-to-T distance versus time. qpos columns of the history: t, T_x, T_y, T_theta, root_x, root_y (goal frame).
"""
import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# T outline in the block frame (12 x 12 cm: crossbar 12 x 4 at y 0.02..0.06, stem 4 x 8 at y -0.06..0.02)
OUTLINE = np.array([[-.06, .02], [-.02, .02], [-.02, -.06], [.02, -.06], [.02, .02], [.06, .02], [.06, .06], [-.06, .06]])


def t_polygon(q):
    c, s = math.cos(q[2]), math.sin(q[2])
    return OUTLINE @ np.array([[c, -s], [s, c]]).T + q[:2]


def dist_to_poly(p, poly):
    """0 inside the polygon, else the distance to its boundary."""
    inside = False
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    if inside:
        return 0.0
    best = 1e9
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        u = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + u * ab))))
    return best


def make_figure(hist, out, label="", anchor=(0.0, 0.1), times=None):
    t = hist[:, 0]
    if times is None:                      # 7 snapshots: 0, 10, ..., 60 s, or evenly spread over a shorter run
        times = (0, 10, 20, 30, 40, 50, 60) if t[-1] >= 59 else tuple(np.linspace(0, t[-1], 7))
    q = hist[:, 1:]
    pusher = q[:, 3:5] + np.asarray(anchor)
    pos_err = np.linalg.norm(q[:, :2], axis=1) * 1000
    ang_err = np.abs(np.degrees((q[:, 2] + np.pi) % (2 * np.pi) - np.pi))
    dist = np.array([dist_to_poly(pusher[i], t_polygon(q[i])) for i in range(len(t))]) * 1000

    allxy = np.vstack([q[:, :2], pusher, [[0.1, 0.08], [-0.1, -0.08]]])
    x0, x1 = allxy[:, 0].min() - 0.09, allxy[:, 0].max() + 0.09
    y0, y1 = allxy[:, 1].min() - 0.09, allxy[:, 1].max() + 0.09
    n = len(times)
    fig = plt.figure(figsize=(2.6 * n, 8.4))
    gs = fig.add_gridspec(2, n, height_ratios=[1.25, 1])
    goal = t_polygon(np.zeros(3))
    for j, ts in enumerate(times):
        ax = fig.add_subplot(gs[0, j])
        i = int(np.argmin(np.abs(t - ts)))
        if abs(t[i] - ts) > 1.0 and ts > t[-1]:
            ax.set_axis_off()
            ax.set_title(f"t = {ts} s (run ended at {t[-1]:.0f} s)", fontsize=8)
            continue
        ax.fill(goal[:, 0], goal[:, 1], color="green", alpha=0.18)
        ax.plot(*np.vstack([goal, goal[:1]]).T, color="green", lw=1.2)
        ax.plot(pusher[: i + 1, 0], pusher[: i + 1, 1], color="gold", lw=1.3, label="pusher path")
        ax.plot(q[: i + 1, 0], q[: i + 1, 1], color="steelblue", lw=0.8, ls=":")
        poly = t_polygon(q[i])
        ax.fill(poly[:, 0], poly[:, 1], color="tab:blue", alpha=0.5)
        ax.plot(*np.vstack([poly, poly[:1]]).T, color="navy", lw=1.2)
        ax.add_patch(plt.Circle(pusher[i], 0.008, color="red"))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t[i]:.1f} s\npos {pos_err[i]:.0f} mm  ang {ang_err[i]:.0f} deg", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
    for k, (y, name, col) in enumerate(((pos_err, "T position error [mm]", "k"), (ang_err, "T heading error [deg]", "m"),
                                         (dist, "pusher to T distance [mm]", "tab:red"))):
        a0, a1 = [int(round(k * n / 3)), int(round((k + 1) * n / 3))]
        ax = fig.add_subplot(gs[1, a0:a1])
        ax.plot(t, y, color=col)
        ax.set_xlabel("sim time [s]", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    if label:
        fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--anchor", type=float, nargs=2, default=[0.0, 0.1])
    ap.add_argument("--times", type=float, nargs="+", default=None)
    a = ap.parse_args()
    make_figure(np.load(a.hist), a.out, a.label, a.anchor, a.times)
