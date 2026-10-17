#!/usr/bin/env python3
"""Open-loop plan for the real robot from one closed-loop run (tune_mppi.py): same pusher path, xy_driver friendly.

  python make_openloop.py runs/tune_mppi/dial_d02_H2/seed0 --out runs/dial_d02_H2_openloop [--t-cut 35] [--seg-t 0.5]
        [--pusher-start X Y] [--clearance 0.04]

Plan (robot base frame, as send_xy_cmd / open_loop_test.py expect): HOME(0.40, 0) -> detour around the T (>= clearance from
its outline, inside the box) -> pusher start (the run's start, goal frame) -> 1 s hold -> the run's pusher path up to
--t-cut, sampled every --seg-t. xy_driver gives every segment the same time with a quintic (zero velocity at the points):
peak speed = 1.875 L / T, peak acceleration = 5.7735 L / T^2, so every chord is limited to
L <= min(peak_speed_max * T / 1.875, peak_acc_max * T^2 / 5.7735); where the path is faster the time is stretched
(time warp), the path itself is not changed. All points are clipped to / checked against the box.
Then the plan is replayed open loop in the sim (plant xml of the run, the pusher driven kinematically with its velocity) and
sim_T_traj_openloop.csv (t from the moment the plan is sent) is written.
"""
import argparse
import math
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
import numpy as np

from global_mppi.closed_loop.config import load_config, resolve_path
from global_mppi.closed_loop.geometry import dist_to_poly, t_polygon
from global_mppi.closed_loop.scene import Scene, rot2, wrap

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir")
ap.add_argument("--out", required=True)
ap.add_argument("--config", default="configs/pusht_real.yaml")
ap.add_argument("--t-cut", type=float, default=35.0)
ap.add_argument("--seg-t", type=float, default=0.5)
ap.add_argument("--peak-speed", type=float, default=0.2)
ap.add_argument("--peak-acc", type=float, default=2.0)
ap.add_argument("--hold-s", type=float, default=1.0)
ap.add_argument("--clearance", type=float, default=0.04)
ap.add_argument("--direct", action="store_true", help="no HOME->start detour and no hold: the first point IS the pusher start (the robot flies there first)")
ap.add_argument("--smooth-s", type=float, default=0.25)
a = ap.parse_args()
out = os.path.abspath(a.out)
os.makedirs(out, exist_ok=True)

cfg = load_config(resolve_path(a.config))
scene = Scene(cfg)
HOME = np.array([0.40, 0.0])
x0, x1, y0, y1 = scene.box
dt = a.seg_t
L_MAX = min(a.peak_speed * dt / 1.875, a.peak_acc * dt**2 / 5.7735) * 0.98
print(f"segment time {dt} s: chord limit {L_MAX * 1000:.1f} mm (peak speed <= {a.peak_speed}, acc <= {a.peak_acc})")

H = np.load(os.path.join(a.run_dir, "qpos_history.npy"))
H = H[H[:, 0] <= a.t_cut]
SR = float(np.median(np.diff(H[:, 0])))
P = np.array([scene.pusher_base(q[1:]) for q in H])                     # pusher path, base frame
q_start = H[0, 1:]
p_start_goal = scene.pusher_world(q_start)
print(f"source: {len(H)} samples to t={H[-1, 0]:.1f}s; pusher start (goal frame) {p_start_goal.round(4).tolist()}, "
      f"base {P[0].round(4).tolist()}; T start {q_start[:3].round(4).tolist()}")

# --- 1) light smoothing of the closed-loop pusher path, first / last fixed
w = max(int(round(a.smooth_s / SR)) // 2, 0)
if w:
    pad = np.vstack([np.repeat(P[:1], w, 0), P, np.repeat(P[-1:], w, 0)])
    Ps = np.stack([np.convolve(pad[:, k], np.ones(2 * w + 1) / (2 * w + 1), mode="valid") for k in (0, 1)], 1)
    Ps[0], Ps[-1] = P[0], P[-1]
else:
    Ps = P.copy()
print(f"smoothing: max deviation {np.linalg.norm(Ps - P, axis=1).max() * 1000:.1f} mm")

# --- 2) time warp where a 0.5 s chord would be too long, then sample every seg_t
step = np.linalg.norm(np.diff(Ps, axis=0), axis=1)
v_cap = L_MAX / dt
tau = np.concatenate([[0.0], np.cumsum(np.maximum(SR, step / v_cap))])
grid = np.arange(0.0, tau[-1], dt)
grid = np.append(grid, tau[-1]) if tau[-1] - grid[-1] > 1e-6 else grid
push = np.stack([np.interp(grid, tau, Ps[:, 0]), np.interp(grid, tau, Ps[:, 1])], 1)
print(f"push part: {H[-1, 0]:.1f}s of closed loop -> {tau[-1]:.1f}s ({int((step / SR > v_cap).sum())} samples slowed), {len(push)} points")

# --- 3) HOME -> pusher start, around the T (T at its start pose; goal frame -> base for the box)
T0 = q_start[:3]
poly = t_polygon(T0)
if not a.direct:
    T0 = q_start[:3]
    poly = t_polygon(T0)
    C = T0[:2]
    R = float(np.linalg.norm(poly - C, axis=1).max()) + a.clearance
    home_g = rot2(-scene.goal_th) @ (HOME - scene.goal_p)
    a_home, a_goal = math.atan2(*(home_g - C)[::-1]), math.atan2(*(p_start_goal - C)[::-1])


    def arc_points(sign):
        da = (a_goal - a_home) % (2 * math.pi) if sign > 0 else -((a_home - a_goal) % (2 * math.pi))
        n = max(int(abs(da) * R / 0.01), 2)
        return [C + R * np.array([math.cos(a_home + da * f), math.sin(a_home + da * f)]) for f in np.linspace(0, 1, n)], abs(da)


    def in_box(pg):
        b = scene.sim_to_base(pg)
        return x0 + 0.005 <= b[0] <= x1 - 0.005 and y0 + 0.005 <= b[1] <= y1 - 0.005


    best = None
    for sign in (+1, -1):
        arc, da = arc_points(sign)
        if all(in_box(q) for q in arc) and (best is None or da < best[0]):
            best = (da, arc)
    assert best is not None, "no in-box detour around the T"
    path_g = [home_g]
    if np.linalg.norm(home_g - C) < R:
        path_g.append(C + R * (home_g - C) / np.linalg.norm(home_g - C))
    path_g += best[1]
    path_g.append(p_start_goal)
    detour_b = np.array([scene.sim_to_base(q) for q in path_g])
    detour_b[0] = HOME
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(detour_b, axis=0), axis=1))])
    n_seg = int(math.ceil(s[-1] / L_MAX))
    si = np.linspace(0.0, s[-1], n_seg + 1)
    detour = np.stack([np.interp(si, s, detour_b[:, 0]), np.interp(si, s, detour_b[:, 1])], 1)
    min_clear = min(dist_to_poly(rot2(-scene.goal_th) @ (p - scene.goal_p), poly) for p in detour)
    print(f"detour HOME -> pusher start: {s[-1]:.3f} m in {n_seg} segments ({n_seg * dt:.1f}s), min distance to the T outline {min_clear * 1000:.0f} mm")
    assert min_clear >= 0.03, "detour passes closer than 3 cm to the T"


else:
    detour, n_seg, min_clear = np.zeros((0, 2)), 0, float("nan")
    print("direct plan: no detour / hold, the first point is the pusher start")

n_hold = int(round(a.hold_s / dt))
W = push if a.direct else np.vstack([detour, np.repeat(detour[-1:], n_hold, 0), push[1:]])
n_clip = int(((W[:, 0] < x0) | (W[:, 0] > x1) | (W[:, 1] < y0) | (W[:, 1] > y1)).sum())
W = np.stack([np.clip(W[:, 0], x0, x1), np.clip(W[:, 1], y0, y1)], 1)
chain = np.vstack([P[0] if a.direct else HOME, W])   # send_xy_cmd: the robot goes from its current xy (home) through the points
seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
N, dur = len(W), len(W) * dt
peak_v, peak_a = 1.875 * seg.max() / dt, 5.7735 * seg.max() / dt**2
np.savetxt(os.path.join(out, "waypoints_base.txt"), W, fmt="%.6f")
open(os.path.join(out, "duration.txt"), "w").write(f"{dur:.1f}\n")
print(f"PLAN: {N} points, {dt} s per segment, duration {dur:.1f} s, peak speed {peak_v:.3f} m/s, peak acceleration {peak_a:.2f} m/s^2, "
      f"path {seg.sum():.2f} m, x {W[:, 0].min():.3f}..{W[:, 0].max():.3f} y {W[:, 1].min():.3f}..{W[:, 1].max():.3f}, {n_clip} clipped")
assert peak_v <= a.peak_speed and peak_a <= a.peak_acc

# --- 4) open-loop replay in the sim (plant of the scene; T at its start, pusher driven along the plan)
model = scene.plant
data = mujoco.MjData(model)
home_sim = rot2(-scene.goal_th) @ ((P[0] if a.direct else HOME) - scene.goal_p)      # where the plan starts
Wsim = np.array([rot2(-scene.goal_th) @ (w_ - scene.goal_p) for w_ in W])
chain_sim = np.vstack([home_sim, Wsim])
T_end = (len(chain_sim) - 1) * dt


def ref(tt):
    i = min(int(tt / dt), len(chain_sim) - 2)
    u = min((tt - i * dt) / dt, 1.0)
    sp = 10 * u**3 - 15 * u**4 + 6 * u**5
    dsp = (30 * u**2 - 60 * u**3 + 30 * u**4) / dt if tt < T_end else 0.0
    return chain_sim[i] + sp * (chain_sim[i + 1] - chain_sim[i]), dsp * (chain_sim[i + 1] - chain_sim[i])


data.qpos[:] = q_start
data.qpos[3:5] = home_sim - scene.anchor
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)
rows, touch = [], None
gp, gb = model.geom("pusher").id, model.body("block").id
for k in range(int((T_end + 2.0) / model.opt.timestep)):
    tt = k * model.opt.timestep
    p, v = ref(min(tt, T_end))
    data.qpos[3:5] = p - scene.anchor
    data.qvel[3:5] = v
    if k % 50 == 0:
        rows.append(np.concatenate([[tt], data.qpos]))
    mujoco.mj_step(model, data)
    if touch is None:
        for c in data.contact[: data.ncon]:
            if gp in (c.geom1, c.geom2) and gb in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]):
                touch = tt
                break
rows = np.array(rows)
np.savetxt(os.path.join(out, "sim_T_traj_openloop.csv"), rows[:, :4], delimiter=",", fmt="%.6f",
           header="t_s,T_x_m,T_y_m,T_theta_rad (goal frame) -- open-loop replay of waypoints_base.txt", comments="")
np.save(os.path.join(out, "qpos_history_openloop.npy"), rows)
pe, ae = scene.errors(data.qpos)
# the original closed-loop result at the cut time, for comparison
Hc = np.load(os.path.join(a.run_dir, "qpos_history.npy"))
qc = Hc[np.argmin(abs(Hc[:, 0] - a.t_cut)), 1:]
pc, ac = scene.errors(qc)
d_T = float(np.linalg.norm(data.qpos[:2] - qc[:2])) * 1000
print(f"open-loop replay: pusher first touches the T at t={touch}s; final T {data.qpos[:3].round(4).tolist()} -> {pe * 1000:.1f} mm / {ae:.2f} deg from the goal")
print(f"original closed loop at t={a.t_cut}s: {pc * 1000:.1f} mm / {ac:.2f} deg; the two final T poses differ by {d_T:.1f} mm")
with open(os.path.join(out, "result.txt"), "w") as f:
    f.write(f"open-loop plan from {a.run_dir} (t <= {a.t_cut}s), pusher start (goal frame) {p_start_goal.round(4).tolist()} = base {P[0].round(4).tolist()}\n")
    f.write(f"waypoints {N}, {dt} s/segment, duration {dur:.1f} s, peak speed {peak_v:.3f} m/s, peak acc {peak_a:.2f} m/s^2, path {seg.sum():.2f} m, "
            f"x {W[:, 0].min():.3f}..{W[:, 0].max():.3f} y {W[:, 1].min():.3f}..{W[:, 1].max():.3f}, {n_clip} clipped\n")
    f.write(f"detour around the T: {n_seg} segments, min distance to the T outline {min_clear * 1000:.0f} mm{' (direct plan: none)' if a.direct else ''}\n")
    f.write(f"sim open-loop replay: final error {pe * 1000:.1f} mm / {ae:.2f} deg; original closed loop at {a.t_cut}s {pc * 1000:.1f} mm / {ac:.2f} deg; "
            f"final T poses differ by {d_T:.1f} mm\n")
print("outputs in", out)
