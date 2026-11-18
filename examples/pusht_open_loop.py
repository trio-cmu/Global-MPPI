#!/usr/bin/env python3
"""Send a recorded pusher trajectory (txt) to the real robot through the XY_CMD LCM interface.

  python pusht_open_loop.py <trajectory.txt> [--frame goal|base] [--slowdown 2] [--seg-t 0.5] [--dry-run] [--yes]

Order of events
  1. read XY_STATE (xy_driver must be running, idle, at the work height; this script never moves z);
  2. MOVE TO THE START: one plan from the current xy straight to the first point of the txt (time chosen so the
     speed / acceleration limits hold), wait until xy_driver is idle again and check the arrival;
  3. SEND THE TRAJECTORY: the points of the txt as ONE plan (xy_path_cmd_t on XY_CMD, same message as
     trio_infra/src/franka_driver/send_xy_cmd.py), wait until tracking = 0.

The txt
  * "t pusher_x pusher_y ..." (the robot_trajectory_*.txt written from pusht.py runs: columns t_s, pusher_x_m,
    pusher_y_m, T_x_m, T_y_m, T_theta_rad; lines starting with # are ignored), or a plain list "x y" per line;
  * --frame goal (default): the points are in the goal frame (sim frame), converted to the robot BASE frame with the
    goal pose of configs/pusht_real.yaml; --frame base: already base frame.

Why the points are re-sampled: xy_driver gives every segment of a plan the same duration and a quintic with zero speed at
each point (peak speed 1.875 L/T, peak acceleration 5.7735 L/T^2). 50 Hz points would mean 20 ms segments and a reflex stop
(the robot aborted at ~5.8 m/s^2). So the path is re-timed (time stretched where it is too fast, path unchanged) and sampled
every --seg-t seconds (>= 0.4 s, the xy_driver limit; 0.5 s is the tested value). --slowdown K (default 2) stretches the whole motion K times: K times more points, speed / K.
Note: xy_driver brings the arm to zero speed at EVERY point of a plan, so making the segments shorter does not smooth the
motion; slowing down (more time per unit of path) is what lowers speed and acceleration. Limits: --peak-speed (xy_driver rejects > 0.25 m/s), --peak-acc, workspace box.

--dry-run prints the plan and all checks and sends nothing. Otherwise you are asked before each of the two moves (--yes
skips the questions). xy_driver cannot cancel a plan that is running: stop the arm with the enable button.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import yaml

TRIO_INFRA = os.path.expanduser("~/projects/trio_infra")
sys.path.insert(0, os.path.join(TRIO_INFRA, "src", "mppi_bridge", "lcmtypes_py"))   # franka_driver_lcm (generated LCM types)
os.environ.setdefault("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=1")
import lcm  # noqa: E402
from franka_driver_lcm import xy_path_cmd_t, xy_state_t  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "..", "configs", "pusht_real.yaml")
CMD_CHANNEL, STATE_CHANNEL = "XY_CMD", "XY_STATE"
DRIVER_MAX_SPEED = 0.25          # xy_driver kMaxSpeed (peak, quintic)
XY_GAP_S, TRACK_ERR_M = 0.1, 0.02


# ------------------------------------------------------------------ txt -> base-frame path
def load_txt(path):
    """Returns (t or None, xy (N, 2))."""
    a = np.loadtxt(path, comments="#", ndmin=2)
    if a.shape[1] == 2:
        return None, a
    if a.shape[1] < 3:
        raise SystemExit(f"{path}: need 'x y' or 't x y ...' columns")
    return a[:, 0], a[:, 1:3]


def goal_to_base(xy, goal):
    c, s = math.cos(goal["theta_rad"]), math.sin(goal["theta_rad"])
    return xy @ np.array([[c, -s], [s, c]]).T + np.array([goal["x_m"], goal["y_m"]])


def plan_limits(seg_t, peak_speed, peak_acc):
    """Longest chord allowed in one segment of seg_t seconds."""
    return min(peak_speed * seg_t / 1.875, peak_acc * seg_t ** 2 / 5.7735)


def resample(t, xy, seg_t, l_max, slowdown=1.0):
    """Time-stretch where the path is too fast for l_max per seg_t, then sample every seg_t (the path is unchanged).
    Without a time column the path is simply cut into equal arc-length chords of 0.98 l_max.
    slowdown > 1 makes the whole motion that many times slower (recorded time x slowdown): the chords get shorter, so the
    plan has proportionally MORE points, and peak speed / acceleration drop by 1/slowdown and 1/slowdown^2."""
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if t is None:
        s = np.concatenate([[0.0], np.cumsum(step)])
        n = max(int(math.ceil(s[-1] * slowdown / (0.98 * l_max))), 1)
        si = np.linspace(0.0, s[-1], n + 1)
        return np.stack([np.interp(si, s, xy[:, 0]), np.interp(si, s, xy[:, 1])], 1)[1:]
    v_cap = 0.98 * l_max / seg_t
    dtau = np.maximum(np.diff(t) * slowdown, step / v_cap)       # never faster than v_cap, never faster than recorded / slowdown
    tau = np.concatenate([[0.0], np.cumsum(dtau)])
    grid = np.arange(0.0, tau[-1], seg_t)
    grid = np.append(grid, tau[-1]) if tau[-1] - grid[-1] > 1e-6 else grid
    return np.stack([np.interp(grid, tau, xy[:, 0]), np.interp(grid, tau, xy[:, 1])], 1)[1:]


def check_plan(name, cur, pts, dur, box, peak_speed, peak_acc):
    """Print the plan and return True when every limit holds. cur = where the robot is when the plan starts."""
    x0, x1, y0, y1 = box
    full = np.vstack([cur, pts])
    seg = np.linalg.norm(np.diff(full, axis=0), axis=1)
    seg_t = dur / len(pts)
    pv, pa = 1.875 * seg.max() / seg_t, 5.7735 * seg.max() / seg_t ** 2
    inbox = bool(((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).all())
    ok = inbox and pv <= peak_speed and pa <= peak_acc and pv <= DRIVER_MAX_SPEED
    print(f"  [{name}] {len(pts)} points, {seg_t:.2f} s/segment, {dur:.1f} s, path {seg.sum():.3f} m, "
          f"peak speed {pv:.3f} m/s (limit {peak_speed}), peak acc {pa:.2f} m/s^2 (limit {peak_acc}), "
          f"x {pts[:, 0].min():.3f}..{pts[:, 0].max():.3f} y {pts[:, 1].min():.3f}..{pts[:, 1].max():.3f}, "
          f"inside box x[{x0},{x1}] y[{y0},{y1}]: {inbox}  -> {'OK' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------ LCM
class Link:
    def __init__(self):
        self.lc = lcm.LCM()
        self.last, self.t_last = None, 0.0
        self.lc.subscribe(STATE_CHANNEL, self._on_state)

    def _on_state(self, _ch, data):
        self.last, self.t_last = xy_state_t.decode(data), time.time()

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.lc.handle_timeout(20)

    def state(self, timeout=2.0):
        t0 = time.time()
        while self.last is None and time.time() - t0 < timeout:
            self.lc.handle_timeout(50)
        return self.last

    def send_and_wait(self, pts, dur, label, log):
        """Publish one XY_CMD plan and wait until xy_driver has run it. Returns True when it finished normally."""
        msg = xy_path_cmd_t()
        msg.timestamp_us = int(time.time() * 1e6)
        msg.plan_id = int(time.time() * 1000) % (2 ** 31 - 1)       # must differ from the previous plan's id
        msg.num_points = len(pts)
        msg.points_xy = [[float(p[0]), float(p[1])] for p in pts]
        msg.duration_sec = float(dur)
        self.lc.publish(CMD_CHANNEL, msg.encode())
        print(f"  sent '{label}' plan_id={msg.plan_id}: {len(pts)} points, {dur:.1f} s", flush=True)
        t_send, started, t_print = time.time(), False, 0.0
        while time.time() - t_send < dur + 20:
            self.lc.handle_timeout(20)
            s = self.last
            if s is None:
                continue
            now = time.time()
            if log is not None:
                log.append([now - t_send, s.x, s.y, s.cmd_x, s.cmd_y, s.tracking, s.plan_id])
            if now - self.t_last > XY_GAP_S:
                print(f"  XY_STATE silent for {now - self.t_last:.2f} s: xy_driver dead / link down. Stopping to wait; "
                      f"release the enable button if the arm moves.")
                return False
            if math.hypot(s.cmd_x - s.x, s.cmd_y - s.y) > TRACK_ERR_M:
                print(f"  commanded - measured = {math.hypot(s.cmd_x - s.x, s.cmd_y - s.y) * 100:.1f} cm > "
                      f"{TRACK_ERR_M * 100:.0f} cm: tracking is failing. Release the enable button.")
                return False
            if s.plan_id == msg.plan_id and s.tracking:
                started = True
            if started and not s.tracking:
                print(f"  done: x={s.x:.4f} y={s.y:.4f}")
                return True
            if now - t_print > 1.0:
                t_print = now
                print(f"    t={now - t_send:5.1f}s xy=({s.x:.3f},{s.y:.3f}) tracking={s.tracking}", end="\r", flush=True)
        print("  timed out waiting for the plan to run (rejected? read the xy_driver output)")
        return False


def ask(prompt, yes):
    return yes or input(prompt + " [y/N] ").strip().lower() == "y"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("txt")
    ap.add_argument("--frame", choices=["goal", "base"], default="goal", help="frame of the points in the txt (default goal)")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="yaml with scene.goal_base and scene.box_base")
    ap.add_argument("--seg-t", type=float, default=0.5,
                    help="time per segment [s], >= 0.4: xy_driver (trio_infra_upright, plan_limits.hpp) REJECTS plans with segments "
                         "shorter than 0.4 s or a peak acceleration above 2 m/s^2. 0.5 is the tested value. Max points = duration / 0.4")
    ap.add_argument("--peak-speed", type=float, default=0.2, help="quintic peak speed limit [m/s] (xy_driver rejects > 0.25)")
    ap.add_argument("--peak-acc", type=float, default=2.0, help="quintic peak acceleration limit [m/s^2]")
    ap.add_argument("--slowdown", type=float, default=2.0,
                    help="make the trajectory this many times slower than recorded (default 2): shorter chords, so proportionally "
                         "more points; peak speed / 1x, peak acceleration / x^2")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and the checks, send nothing")
    ap.add_argument("--yes", action="store_true", help="do not ask before the moves")
    ap.add_argument("--log", default=None, help="save the XY_STATE trace of the trajectory plan to this txt")
    a = ap.parse_args()
    if a.seg_t < 0.4:
        ap.error("--seg-t must be >= 0.4 (xy_driver rejects shorter segments: plan_limits.hpp kMinSegmentSec)")

    cfg = yaml.safe_load(open(a.config))
    sc = cfg["scene"]
    b = sc["box_base"]
    box = (b["x_min"], b["x_max"], b["y_min"], b["y_max"])
    t, xy = load_txt(a.txt)
    if a.frame == "goal":
        xy = goal_to_base(xy, sc["goal_base"])
    print(f"{a.txt}: {len(xy)} points{'' if t is None else f', {t[-1] - t[0]:.1f} s'} ({a.frame} frame -> base frame); "
          f"start ({xy[0, 0]:.4f}, {xy[0, 1]:.4f}), end ({xy[-1, 0]:.4f}, {xy[-1, 1]:.4f})")

    link = Link()
    s = link.state()
    if s is None:
        print(f"No {STATE_CHANNEL}: is xy_driver running?")
        if not a.dry_run:
            return 1
        print("  (dry run: assuming the robot is at home (0.40, 0.00))")
        cur = np.array([0.40, 0.0])
    else:
        cur = np.array([s.x, s.y])
        print(f"robot now: x={s.x:.4f} y={s.y:.4f} z={s.z:.4f} tracking={s.tracking}")
        if s.tracking and not a.dry_run:
            print("xy_driver is still executing a plan; wait until it is idle.")
            return 1

    # plan 1: go to the start of the trajectory
    d = float(np.linalg.norm(xy[0] - cur))
    t_go = max(3.0, 1.875 * d / (0.6 * a.peak_speed), math.sqrt(5.7735 * d / (0.5 * a.peak_acc)))
    t_go = math.ceil(t_go * 2) / 2
    go_pts = xy[:1]
    # plan 2: the trajectory (the first segment is zero length: the robot is already at its first point)
    l_max = plan_limits(a.seg_t, a.peak_speed, a.peak_acc)
    traj = np.vstack([xy[:1], resample(t, xy, a.seg_t, l_max, a.slowdown)])
    dur = len(traj) * a.seg_t
    ok1 = check_plan("1 move to start", cur, go_pts, t_go, box, a.peak_speed, a.peak_acc)
    ok2 = check_plan("2 trajectory", xy[0], traj, dur, box, a.peak_speed, a.peak_acc)
    if not (ok1 and ok2):
        print("a check failed: nothing sent.")
        return 1
    if a.dry_run:
        print("dry run: nothing sent.")
        return 0

    if not ask(f"Robot WILL move {d:.3f} m to the start point. Send?", a.yes):
        return 1
    if not link.send_and_wait(go_pts, t_go, "move to start", None):
        return 2
    link.spin(0.5)
    err = math.hypot(link.last.x - xy[0, 0], link.last.y - xy[0, 1])
    print(f"  at the start: {err * 1000:.1f} mm from the first point")
    if err > 0.003:
        print("not at the start point: not sending the trajectory.")
        return 2
    if not ask(f"Robot WILL now run the trajectory ({dur:.0f} s). Hand on the enable button. Send?", a.yes):
        return 1
    log = [] if a.log else None
    ok = link.send_and_wait(traj, dur, "trajectory", log)
    if a.log and log:
        np.savetxt(a.log, np.array(log), fmt="%.6f", header="t_s x y cmd_x cmd_y tracking plan_id (t=0: trajectory sent)")
        print(f"  saved {a.log}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
