"""Whole GCS trajectory and its total cost, under the cost shared with Global-MPPI's PushT.

Left: the plan seen from above (T outlines over time, COM and pusher paths, goal). Middle: the cost
accumulated over time, by term, computed with global_mppi.tasks.pusht.PushT's running cost on the
trajectory sampled every 0.01 s (the same cost the sampling planners use), with GCS's own value of its
formula on the knots (accumulated per mode) for reference. Right: the totals by term.

    python plot_gcs_trajectory_cost.py [--traj gcs_shared_cost_hardgoal_traj.npz]
        [--knots gcs_shared_cost_hardgoal_knot_cost.json] [--out gcs_shared_cost_hardgoal_cost.png]
"""
import argparse
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx

from global_mppi.tasks.pusht import (CONTACT_TOL, KEYPOINT_VEL_RATE, PUSHER_VEL_RATE, PushT,
                                     T_VERTICES, corner_velocities, pusher_gap)

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="gcs_shared_cost_hardgoal_traj.npz")
ap.add_argument("--knots", default="gcs_shared_cost_hardgoal_knot_cost.json")
ap.add_argument("--out", default="gcs_shared_cost_hardgoal_cost.png")
ap.add_argument("--title", default="GCS, shared cost, hard final block pose")
args = ap.parse_args()

task = PushT()
d = np.load(args.traj)
q, ts = d["qpos"].astype(np.float64), d["t"]
h = np.diff(ts)
qv = np.diff(q, axis=0) / h[:, None]  # velocity over each step, assigned to the state after it (as step())
Q, QV = jnp.asarray(q[1:]), jnp.asarray(qv)

# ---- the PushT running cost on the trajectory, term by term (sum of terms = PushT.running_cost)
speed = np.linalg.norm(np.asarray(jax.vmap(corner_velocities)(Q, QV)), axis=2)
contact = np.asarray(jax.vmap(pusher_gap)(Q)) <= CONTACT_TOL
v_p = qv[:, 3:5]
rates = {
    "contact time": task.contact_time_weight * contact,
    "T path": task.keypoint_arc_weight * speed.mean(1),
    "T speed²": task.keypoint_vel_weight * KEYPOINT_VEL_RATE * (speed ** 2).mean(1),
    "pusher speed²": task.pusher_vel_weight * PUSHER_VEL_RATE * (~contact) * (v_p ** 2).sum(1),
    "pusher path": task.pusher_arc_weight * (~contact) * np.linalg.norm(v_p, axis=1),
}
ell = np.asarray(jax.vmap(lambda a, b: task.running_cost(SimpleNamespace(qpos=a, qvel=b), jnp.zeros(2)))(Q, QV))
assert np.allclose(sum(rates.values()), ell, rtol=1e-4, atol=1e-5), "term split does not add up to running_cost"
cum = {k: np.concatenate([[0.0], np.cumsum(h * r)]) for k, r in rates.items()}
running = float(np.sum(h * ell))
# terminal cost W_T * pose cost at the last state (MuJoCo sensors for the pose)
mj = mujoco.MjData(task.mj_model)
mj.qpos[:] = q[-1]
mujoco.mj_forward(task.mj_model, mj)
terminal = float(task.terminal_cost(mjx.put_data(task.mj_model, mj)))
total = running + terminal

knots = json.load(open(args.knots))
KEYS = {"contact time": "time", "T path": "karc", "T speed²": "kvel", "pusher speed²": "pvel", "pusher path": "parc"}

# ---- figure
COLORS = {"contact time": "#0072B2", "T path": "#E69F00", "T speed²": "#009E73",
          "pusher speed²": "#CC79A7", "pusher path": "#56B4E9"}
INK, INK2, GRID = "#1f2328", "#57606a", "#d0d7de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, (ax_w, ax_c, ax_b) = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1.05, 1.5, 1.0]))
fig.patch.set_facecolor("white")


def outline(x, y, th):
    c, s = np.cos(th), np.sin(th)
    pts = np.asarray(T_VERTICES) @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.vstack([pts, pts[:1]])


# top view
g = outline(0, 0, 0)
ax_w.fill(g[:, 0], g[:, 1], color="#2da44e", alpha=0.18, lw=0)
ax_w.plot(g[:, 0], g[:, 1], color="#2da44e", lw=1.5, ls="--", label="goal")
cmap = plt.get_cmap("Greys")
snap = np.linspace(0, len(ts) - 1, 16).astype(int)
for j, i in enumerate(snap):
    o = outline(*q[i, :3])
    ax_w.plot(o[:, 0], o[:, 1], color=cmap(0.3 + 0.65 * j / (len(snap) - 1)), lw=1.0)
com = d["block_com"][:, :2]
pusher = d["pusher"]
ax_w.plot(com[:, 0], com[:, 1], color="#cf222e", lw=2, label="T COM path")
ax_w.plot(pusher[:, 0], pusher[:, 1], color="#0969da", lw=1.6, label="pusher path")
ax_w.scatter(*pusher[0], s=40, color="#0969da", zorder=5)
ax_w.scatter(*pusher[-1], s=40, color="#0969da", marker="s", zorder=5)
ax_w.set_aspect("equal")
ax_w.set_xlabel("x (m)")
ax_w.set_ylabel("y (m)")
ax_w.grid(color=GRID, lw=0.6)
ax_w.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)
ax_w.set_title("Plan seen from above (T every %.2f s, light → dark)" % (ts[-1] / 15), loc="left", fontsize=11,
               color=INK, weight="bold")

# cumulative cost over time
for m in knots["modes"]:
    if m["type"] == "contact":
        ax_c.axvspan(m["t0"], min(m["t1"], ts[-1]), color="#0072B2", alpha=0.07, lw=0)
names = list(rates)
ax_c.stackplot(ts, *[cum[k] for k in names], colors=[COLORS[k] for k in names], labels=names, alpha=0.85,
               edgecolor="white", linewidth=0.6)
gcs_t, gcs_c = [0.0], [0.0]
for m in [m for m in knots["modes"] if m["type"] != "target"]:
    gcs_t += [m["t1"]]
    gcs_c += [gcs_c[-1] + sum(m["terms"].values())]
ax_c.step(gcs_t, gcs_c, where="post", color=INK, lw=1.6, ls="--",
          label="GCS's formula on its knots (per mode)")
ax_c.annotate(f"ours {running:.3f}", (ts[-1], running), xytext=(-4, -16), textcoords="offset points", ha="right",
              color=INK, fontsize=10, weight="bold")
ax_c.set_xlabel("time (s)   (shaded: GCS contact modes)")
ax_c.set_ylabel("accumulated cost")
ax_c.set_xlim(0, ts[-1] + 0.35)
ax_c.annotate(f"GCS {knots['total']:.3f}", (gcs_t[-1], gcs_c[-1]), xytext=(-4, 6), textcoords="offset points",
              ha="right", color=INK, fontsize=10)
ax_c.grid(color=GRID, lw=0.6)
ax_c.legend(loc="upper left", frameon=False, fontsize=9)
ax_c.set_title("Cost accumulated along the plan (PushT running cost, dt 0.01 s)", loc="left", fontsize=11,
               color=INK, weight="bold")

# totals by term
y = np.arange(len(names))
ours = [cum[k][-1] for k in names]
theirs = [knots["totals"][KEYS[k]] for k in names]
ax_b.barh(y - 0.2, ours, height=0.36, color=[COLORS[k] for k in names], label="PushT cost (ours)")
ax_b.barh(y + 0.2, theirs, height=0.36, color="white", edgecolor=[COLORS[k] for k in names], hatch="///",
          label="GCS formula (knots)")
for yi, a, b in zip(y, ours, theirs):
    ax_b.text(max(a, b) + 0.05, yi, f"{a:.3f} / {b:.3f}", va="center", fontsize=9, color=INK2)
ax_b.set_yticks(y, names)
ax_b.invert_yaxis()
ax_b.set_xlim(0, max(ours + theirs) * 1.45)
ax_b.set_xlabel("total over the plan")
ax_b.grid(axis="x", color=GRID, lw=0.6)
ax_b.legend(loc="lower right", frameon=False, fontsize=9)
ax_b.set_title(f"Totals: ours {running:.3f} / GCS {knots['total']:.3f}", loc="left", fontsize=11, color=INK,
               weight="bold")

err = np.linalg.norm(com[-1] - np.array([0.0, 0.06 * 0.02 * 0.03 / (0.02 * 0.08 + 0.06 * 0.02)]))
fig.suptitle(f"{args.title}: {ts[-1]:.2f} s, final block error {1000 * err:.2f} mm / {np.degrees(q[-1, 2]):.2f}°.  "
             f"Total PushT cost = running {running:.3f} + terminal {terminal:.3f} (W_T {task.terminal_weight:g}) "
             f"= {total:.3f}", x=0.01, ha="left", fontsize=12, color=INK, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(args.out, dpi=150, facecolor="white")
fig.savefig(args.out.replace(".png", ".pdf"), facecolor="white")
print(f"RESULT running {running:.4f} terminal {terminal:.4f} total {total:.4f} | GCS knots {knots['total']:.4f}")
for k in names:
    print(f"  {k:14s} ours {cum[k][-1]:.4f}  GCS {knots['totals'][KEYS[k]]:.4f}")
print("saved", args.out)
