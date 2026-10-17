#!/usr/bin/env python3
"""REAL-ROBOT open-loop test: send a precomputed XY plan to the Franka via xy_driver on the NUC.

*** NOT the same file as ~/trio_sim/src/pusht_align/open_loop_test.py, which is a pure MuJoCo
*** simulation script. This one drives the real arm.

  python open_loop_test.py <run_dir> [--dry-run]

<run_dir> holds waypoints_base.txt (robot base frame, first point = home), duration.txt, sim_T_traj.csv
(goal frame, plan timeline) -- e.g. ~/trio_sim/runs/closedloop_ps1. Everything runs on the 5090; the NUC is
reached with `ssh junlong@192.168.50.2` (key based). NUC commands follow trio_infra
script/openloop_cmds.md (branch positions-1015, commit 47fcb1a): no tmux, xy_driver is build_noee/bin/xy_driver
started with nohup and a fifo (~/xy_in) for "down"/"up"; NOT script/run_xy_driver.sh.

--dry-run does the pre-flight only (nothing moves, nothing is started or stopped on the NUC):
  ssh works; ROBOT_STATE (~50 Hz) and T_SHAPE_POSE (~30 Hz) arrive over LCM; no leftover XY_CMD sender
  (5090, NUC, bus); NUC enp89s0 carrier_changes vs the latest carrier_changes= line of ~/linkwatch_baseline.txt
  (increase => red FAIL, stop); waypoints vs box / speed; current T pose (base -> goal frame) vs start_2
  (> 1 cm or > 3 deg => put it back).
Real run (the pre-flight must be OK):
  1) NUC: stop state_pub + aprilcube_pose; guard (exit 0 and last line PASS); start xy_driver (homes to z=0.4434,
     log "Homed; ready for plans at z=0.4434."); start aprilcube_pose; check T_SHAPE_POSE / XY_STATE arrive here;
     `down` (z=0.34874) and check the measured flange z.
  2) record the T start pose, compare with start_2.   3) "hand on the enable button" + Enter.
  4) send the whole plan as ONE XY_CMD (same message as send_xy_cmd.py), record T_SHAPE_POSE / XY_STATE to
     real_T.txt / real_xy.txt until tracking=0 (timeout duration+10 s). Ctrl-C or T leaving the box stops the
     RECORDING only: xy_driver cannot cancel a running plan, so release the enable button to stop the arm.
  Options: --xy-driver-bin PATH (default build_noee/bin/xy_driver), --max-lead-mm N (default 15, passed to xy_driver; 0 = off),
  --upright [--yaw-deg X] (passed to xy_driver; off by default), --send-hold (off). run_dir defaults to the slow plan.
  5) cleanup (always): `up`, stop xy_driver and the fifo holder, restart state_pub and aprilcube_pose.
  6) real_vs_sim.png (goal-frame T path + position/heading error vs time), compare.csv, final pose error.
"""
import argparse
import math
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import yaml

TRIO_SIM = os.path.expanduser("~/trio_sim")
TRIO_INFRA = os.path.expanduser("~/projects/trio_infra")      # on the 5090: generated lcmtypes_py live here
sys.path.insert(0, os.path.join(TRIO_SIM, "src", "pusht_align"))                     # frames.py
sys.path.insert(0, os.path.join(TRIO_INFRA, "src", "mppi_bridge", "lcmtypes_py"))    # franka_driver_lcm, camera_driver_lcm
os.environ.setdefault("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=1")

NUC = "junlong@192.168.50.2"
NUC_REPO = "~/projects/trio_infra"
NUC_PY = "~/miniforge3/envs/cam_calib/bin/python"
FIFO = "~/xy_in"
ROBOT_IP = "172.16.0.2"
NUC_IFACE = "enp89s0"
XY_DRIVER_BIN = "build_noee/bin/xy_driver"   # --xy-driver-bin; relative to NUC_REPO
XY_DRIVER_EXTRA = ""                        # " --max-lead-mm=15" [+ " --upright [--yaw-deg=X]"], set in main()
MAX_LEAD_MM = 15.0                          # --max-lead-mm (default 15; 0 = do not pass the flag)
DEFAULT_RUN_DIR = os.path.join(TRIO_SIM, "runs", "closedloop_ps1_slow")   # the slow plan
NUC_DISPLAY = ":2"      # junlong's Xwayland on the NUC (needs its XAUTHORITY, found below)
NUC_ENV = 'export LCM_DEFAULT_URL="udpm://239.255.76.67:7667?ttl=1"; '   # ttl=1, else nothing leaves the NUC
WORK_Z = 0.34874
HOME_Z = 0.4434
HOME = np.array([0.40, 0.0])                # xy_driver kInitX/kInitY
BOX = (0.26, 0.59, -0.29, 0.29)             # x0 x1 y0 y1: xy_driver flange box shrunk by 1 cm
SOFT_XY_GAP_S = 0.1                         # soft stop: XY_STATE silent for longer than this
SOFT_T_GAP_S = 1.0                          # soft stop: T_SHAPE_POSE silent for longer than this
SOFT_TRACK_ERR_M = 0.02                     # soft stop: |commanded - measured| xy larger than this
RESULT_TAG = ""                             # --tag NAME: real-run files go to <run_dir>/real_<NAME>/ instead of <run_dir>/
FLY_TO_START = False                        # --fly-to-start: plan starts at W[0]; fly there at the safe height before "down"
STRICT_START = None                         # (mm, deg): abort (no prompt) when the T is further from start_2 than this
SEND_HOLD = False                           # --send-hold: also publish a hold plan on a soft stop (does NOT stop the arm)
PEAK_MAX = 0.2                              # quintic peak speed we allow (xy_driver kMaxSpeed 0.25)
START_NAME = "start_2"
LCM_RELAY_YAML = os.path.join(TRIO_SIM, "config", "lcm_relay.yaml")
STARTS_YAML = os.path.join(TRIO_SIM, "src", "pusht_align", "pusht_starts.yaml")
RED, YEL, GRN, END = "\033[1;31m", "\033[33m", "\033[32m", "\033[0m"

ok_all = True


def report(ok, msg, warn=False):
    global ok_all
    if ok:
        print(f"  [{GRN}PASS{END}] {msg}", flush=True)
    elif warn:
        print(f"  [{YEL}WARN{END}] {msg}", flush=True)
    else:
        ok_all = False
        print(f"  [{RED}FAIL{END}] {RED}{msg}{END}", flush=True)
    return ok


# ---------------------------------------------------------------- NUC / ssh
SSH_HELP = """\
ssh to the NUC without a password does not work. One-time setup on the 5090:
    ssh-keyscan -H 192.168.50.2 >> ~/.ssh/known_hosts          # trust the host key (or ssh once and answer yes)
    ssh-copy-id -i ~/.ssh/id_ed25519.pub junlong@192.168.50.2   # asks for the NUC password once
  then check:  ssh -o BatchMode=yes junlong@192.168.50.2 true"""


def ssh(cmd, timeout=30):
    """Run cmd on the NUC in a login shell. Returns (returncode, stdout+stderr). Background jobs in cmd must
    redirect all their fds, otherwise ssh waits for them."""
    try:
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", NUC, "bash -lc " + shlex.quote(cmd)],
                           capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "ssh timed out"


# Never `pkill -f` / pgrep -f here (the pattern would match the shell's own command line): pgrep -x, or ps with comm.
PID_STATE_PUB = "ps -eo pid,comm,args | awk '$2 ~ /^python/ && /robot_state_readonly_pub.py/ {print $1}'"


def nuc_stop_state_and_pose():
    ssh(f"kill -TERM $({PID_STATE_PUB}) 2>/dev/null; kill -TERM $(pgrep -x aprilcube_pose) 2>/dev/null; sleep 3; true")
    rc, out = ssh(f"pgrep -x -a aprilcube_pose; {PID_STATE_PUB}; true")
    return not out.strip()


def nuc_bg(cmd, log):
    """Start cmd detached on the NUC: nohup setsid ... > log 2>&1 < /dev/null &  (ssh returns at once)."""
    rc, out = ssh(f"nohup setsid bash -c {shlex.quote(NUC_ENV + cmd)} > {log} 2>&1 < /dev/null & sleep 1; echo started", 20)
    return rc == 0


def nuc_start_state_pub():
    return nuc_bg(f"cd {NUC_REPO} && exec {NUC_PY} src/extrinsic_check/robot_state_readonly_pub.py",
                  f"{NUC_REPO}/log/state_pub_$(date +%Y%m%d_%H%M).log")


# junlong's graphic session on the NUC: DISPLAY=:2, XAUTHORITY = the -auth file of that Xwayland
XAUTH = ("$(ps -eo args | awk '/^\\/usr\\/bin\\/Xwayland " + NUC_DISPLAY + " /"
         "{for(i=1;i<=NF;i++) if($i==\"-auth\"){print $(i+1);exit}}')")


def nuc_start_pose(interp=False, cam_delay_ms=None, binary=None):
    """aprilcube_pose WITHOUT --show (script/t_shape_pose.sh hard-codes --show: ~30 Hz; this way ~39-41 Hz, no display needed).
    The running aprilcube_pose must be stopped first (nuc_stop_state_and_pose), otherwise the camera is busy.
    binary: path relative to the NUC repo (e.g. build_interp/bin/aprilcube_pose); default: run_pose_estimation.sh (build/bin)."""
    args = ("thrid-party/aprilcube/models/t_shape_12x4x12cm/config.json --width 848 --height 530 --fps 60 --pose-only --quiet --lcm "
            "--lcm-channel T_SHAPE_POSE --base-frame --calibration config/calibration.yaml" + (" --interp-flange" if interp else "")
            + (f" --cam-delay-ms {cam_delay_ms:g}" if interp and cam_delay_ms is not None else ""))
    if binary:
        run = (f"export LD_LIBRARY_PATH={NUC_REPO}/thrid-party/OrbbecSDK_v2/build/linux_x86_64/lib:$LD_LIBRARY_PATH && exec {binary} {args}")
    else:
        run = f"exec ./script/run_pose_estimation.sh {args}"
    return nuc_bg(f"cd {NUC_REPO} && {run}", f"{NUC_REPO}/log/t_shape_pose_$(date +%Y%m%d_%H%M).log")


def carrier_status():
    """(current carrier_changes, baseline, baseline timestamp, later events in ~/linkwatch.log).
    The baseline is the LATEST line with carrier_changes= in the NUC's ~/linkwatch_baseline.txt (the file also gets a
    sample line every 5 min, so this is the newest sample)."""
    rc, cur = ssh(f"cat /sys/class/net/{NUC_IFACE}/carrier_changes")
    if rc != 0:
        return None, None, None, [f"cannot read carrier_changes: {cur}"]
    rc, bl = ssh("grep 'carrier_changes=' ~/linkwatch_baseline.txt 2>&1 | tail -n 1")
    base, ts = None, ""
    if rc == 0 and "carrier_changes=" in bl:
        base = int(bl.split("carrier_changes=")[1].split()[0])
        ts = bl.split()[0]
    rc, log = ssh("tail -n 300 ~/linkwatch.log 2>&1")
    events = [ln for ln in log.splitlines() if ln.strip() and ln[:4] == ts[:4] and ln.split()[0] > ts]
    return int(cur), base, ts, events


# ---------------------------------------------------------------- LCM
def lcm_types():
    import lcm
    from camera_driver_lcm import object_pose_t
    from franka_driver_lcm import robot_state_t, xy_path_cmd_t, xy_state_t
    return lcm, object_pose_t, robot_state_t, xy_path_cmd_t, xy_state_t


def listen(seconds, channels=("ROBOT_STATE", "T_SHAPE_POSE", "XY_STATE", "XY_CMD")):
    lcm, *_ = lcm_types()
    lc = lcm.LCM()
    n = {c: 0 for c in channels}
    for c in channels:
        lc.subscribe(c, lambda ch, d: n.__setitem__(ch, n[ch] + 1))
    t0 = time.time()
    while time.time() - t0 < seconds:
        lc.handle_timeout(50)
    return n, time.time() - t0


def read_flange_z(seconds=1.0):
    lcm, _, robot_state_t, *_ = lcm_types()
    lc = lcm.LCM()
    z = []
    lc.subscribe("ROBOT_STATE", lambda ch, d: z.append(robot_state_t.decode(d).O_T_EE[14]))
    t0 = time.time()
    while time.time() - t0 < seconds:
        lc.handle_timeout(50)
    return float(np.mean(z)) if z else None


def read_T_pose(seconds=2.0):
    """Mean T pose in the base frame from T_SHAPE_POSE: (xy, z, theta, n, std_mm) or None."""
    import frames as F
    lcm, object_pose_t, *_ = lcm_types()
    lc = lcm.LCM()
    pos, th = [], []

    def on(_ch, d):
        m = object_pose_t.decode(d)
        if m.num_tags < 2:
            return
        F.check_base_frame(m)
        pos.append(np.array(m.position_m, float))
        th.append(F.theta_from_R(F.quat_xyzw_to_R(m.quaternion_xyzw)))

    lc.subscribe("T_SHAPE_POSE", on)
    t0 = time.time()
    while time.time() - t0 < seconds:
        lc.handle_timeout(50)
    if len(pos) < 10:
        return None
    pos = np.array(pos)
    mean_th = math.atan2(np.sin(th).mean(), np.cos(th).mean())
    return pos[:, :2].mean(0), float(pos[:, 2].mean()), mean_th, len(pos), float(np.sqrt(pos[:, 0].var() + pos[:, 1].var()) * 1000)


def start_pose_deviation(tp):
    """(dist_mm, dang_deg, (T xy, theta) in the goal frame, recorded start) of a base-frame T pose vs START_NAME."""
    import frames as F
    goal = F.load_goal()
    p_sim, th_sim = F.base_to_sim(tp[0], tp[2], goal)
    s = yaml.safe_load(open(STARTS_YAML))["starts"][START_NAME]
    d = float(np.linalg.norm(p_sim - [s["x"], s["y"]])) * 1000
    da = abs(math.degrees(F.wrap(th_sim - math.radians(s["theta_deg"]))))
    return d, da, (p_sim, th_sim), s


# ---------------------------------------------------------------- plan checks
def result_dir(run_dir):
    """Where the real-run files (real_T.txt, real_xy.txt, compare.csv, real_vs_sim.png) go."""
    d = os.path.join(run_dir, f"real_{RESULT_TAG}") if RESULT_TAG else run_dir
    os.makedirs(d, exist_ok=True)
    return d


def load_plan(run_dir):
    W = np.loadtxt(os.path.join(run_dir, "waypoints_base.txt"))
    dur = float(open(os.path.join(run_dir, "duration.txt")).read().split()[0])
    return W, dur


def check_plan(W, dur):
    x0, x1, y0, y1 = BOX
    G = yaml.safe_load(open(LCM_RELAY_YAML))
    N = len(W)
    # send_xy_cmd: the robot goes from its current XY through the points; that is home, or the pusher start after the fly-over
    full = np.vstack([W[0] if FLY_TO_START else HOME, W])
    seg = np.linalg.norm(np.diff(full, axis=0), axis=1)
    seg_t = dur / N
    peak = 1.875 * seg.max() / seg_t
    inbox = ((W[:, 0] >= x0) & (W[:, 0] <= x1) & (W[:, 1] >= y0) & (W[:, 1] <= y1)).all()
    report(inbox, f"{N} points inside box x[{x0},{x1}] y[{y0},{y1}] "
           f"(x {W[:, 0].min():.3f}..{W[:, 0].max():.3f}, y {W[:, 1].min():.3f}..{W[:, 1].max():.3f})")
    report(peak <= PEAK_MAX, f"quintic peak speed {peak:.3f} m/s <= {PEAK_MAX} (xy_driver kMaxSpeed 0.25), "
           f"segment time {seg_t:.2f} s, total {dur:.1f} s")
    acc = 5.7735 * seg.max() / seg_t**2
    report(acc <= 2.0, f"quintic peak acceleration {acc:.2f} m/s^2 <= 2.0 (reflex 'joint_velocity_discontinuity' happened at ~5.8 m/s^2)")
    avg = seg.sum() / dur
    report(avg <= G["max_speed_mps"], f"average speed {avg:.3f} m/s <= lcm_relay.yaml max_speed_mps {G['max_speed_mps']}", warn=True)
    if FLY_TO_START:
        report(True, f"first point {W[0].round(4).tolist()} is the pusher start; the robot flies there from home "
                     f"({np.linalg.norm(W[0] - HOME):.3f} m) at z={HOME_Z} before 'down'")
    else:
        report(np.linalg.norm(W[0] - HOME) < 0.005, f"first point {W[0].round(4).tolist()} is the home point {HOME.tolist()}")
    if not (N <= G["max_points"] and seg.sum() <= G["max_path_length_m"]):
        print(f"  [INFO] {N} points / {seg.sum():.2f} m path exceed the TRAJECTORY_CMD gate caps ({G['max_points']} pts / "
              f"{G['max_path_length_m']} m); those caps do not apply to XY_CMD -> xy_driver")
    return peak


def check_carrier(label=""):
    cur, base, ts, events = carrier_status()
    if cur is None:
        report(False, f"{label}NUC {NUC_IFACE} carrier_changes unreadable: {events[0]}")
        return False
    if base is None:
        report(False, f"{label}no carrier_changes= line in ~/linkwatch_baseline.txt (now {cur})", warn=True)
        ok = True
    else:
        ok = report(cur <= base, f"{label}NUC {NUC_IFACE} carrier_changes {cur} vs baseline {base} ({ts})"
                    + ("" if cur <= base else f"  -> +{cur - base} link flap(s) since the baseline. DO NOT PROCEED"))
    if events:
        print(f"  [INFO] {label}~/linkwatch.log lines after the baseline: " + " | ".join(e[:100] for e in events[-3:]))
    return ok


# ---------------------------------------------------------------- pre-flight
def preflight(run_dir):
    print("== pre-flight ==")
    rc, out = ssh("hostname", 15)
    ssh_ok = report(rc == 0, f"ssh {NUC}: {out.splitlines()[-1] if out else rc}")
    if not ssh_ok:
        print(SSH_HELP)

    n, dt = listen(3.0)
    hz = {c: n[c] / dt for c in n}
    report(35 <= hz["ROBOT_STATE"] <= 120, f"ROBOT_STATE {hz['ROBOT_STATE']:.1f} Hz (state_pub ~50, xy_driver ~100)")
    report(hz["T_SHAPE_POSE"] > 5, f"T_SHAPE_POSE {hz['T_SHAPE_POSE']:.1f} Hz (~30 expected)")
    report(n["XY_CMD"] == 0, f"no XY_CMD traffic on the bus ({n['XY_CMD']} msgs in {dt:.0f} s)")
    loc = subprocess.run("pgrep -x -a send_xy_cmd.py; ps -eo pid,comm,args | awk '$2 ~ /^python/ && /send_xy_cmd/ {print}'",
                         shell=True, capture_output=True, text=True).stdout.strip()
    report(not loc, "no XY_CMD sender process on this machine" + (f": {loc}" if loc else ""))
    route = subprocess.run("ip route get 239.255.76.67", shell=True, capture_output=True, text=True).stdout
    report("enp129s0" in route, f"5090 multicast route: {route.split(' uid')[0].strip()}", warn=True)
    if ssh_ok:
        rc, out = ssh("ps -eo pid,comm,args | awk '$2 ~ /^python/ && /send_xy_cmd/ {print}'; pgrep -x -a xy_driver; true")
        report(not out.strip(), "no XY_CMD sender / xy_driver left over on the NUC" + (f": {out}" if out.strip() else ""),
               warn=True)
        rc, out = ssh("ip route get 239.255.76.67")
        report("enp88s0" in out, f"NUC multicast route: {out.splitlines()[0] if out else '?'} (expect enp88s0)", warn=True)
        check_carrier()
        rc, out = ssh(f"cd {NUC_REPO} && git log -1 --format=%h && ls {XY_DRIVER_BIN} script/check_f_t_ee.sh "
                      f"script/t_shape_pose.sh src/extrinsic_check/robot_state_readonly_pub.py && test -x {XY_DRIVER_BIN}")
        report(rc == 0, f"NUC trio_infra checkout has {XY_DRIVER_BIN} and the scripts" + ("" if rc == 0 else f": {out}"))
        for flag in [f for f in ("--max-lead-mm", "--upright") if f in XY_DRIVER_EXTRA]:
            rc, out = ssh(f"cd {NUC_REPO} && grep -c -a -e {shlex.quote(flag)} {XY_DRIVER_BIN}")
            report(rc == 0 and out.strip().isdigit() and int(out.strip()) > 0,
                   f"xy_driver binary knows {flag}" + ("" if rc == 0 else "  (flag string not found in the binary: not built yet?)"))

    W, dur = load_plan(run_dir)
    check_plan(W, dur)

    tp = read_T_pose(2.0)
    if tp is None:
        report(False, "no usable T_SHAPE_POSE samples (>= 2 tags) to compare with the start pose")
    else:
        d, da, (p, th), s = start_pose_deviation(tp)
        print(f"  [INFO] T now (base): x={tp[0][0]:.4f} y={tp[0][1]:.4f} theta={math.degrees(tp[2]):.2f} deg "
              f"({tp[3]} samples, std {tp[4]:.2f} mm)")
        print(f"  [INFO] T now (goal frame): x={p[0]:.4f} y={p[1]:.4f} theta={math.degrees(th):.2f} deg; "
              f"{START_NAME}: x={s['x']:.4f} y={s['y']:.4f} theta={s['theta_deg']:.2f} deg")
        report(d <= 10 and da <= 3, f"T vs {START_NAME}: {d:.1f} mm, {da:.2f} deg (limit 10 mm / 3 deg)"
               + ("" if d <= 10 and da <= 3 else "  -> put the T back"))
    print("== pre-flight " + ("OK ==" if ok_all else f"{RED}NOT OK{END} ==") + "\n")
    return ok_all


# ---------------------------------------------------------------- real run
def wait_xy_state(timeout, cond, what):
    lcm, _, _, _, xy_state_t = lcm_types()
    lc = lcm.LCM()
    last = []
    lc.subscribe("XY_STATE", lambda ch, d: last.append(xy_state_t.decode(d)))
    t0 = time.time()
    while time.time() - t0 < timeout:
        lc.handle_timeout(100)
        if last and cond(last[-1]):
            return last[-1]
        del last[:-1]
    print(f"  timed out waiting for {what}")
    return None


def wait_log(logfile, needle, timeout, count=1):
    """Poll the NUC xy_driver log until `needle` appears `count` times. Returns False if the driver died / timed out."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rc, out = ssh(f"cd {NUC_REPO} && grep -c -F {shlex.quote(needle)} {logfile}; pgrep -x xy_driver >/dev/null && echo alive || echo dead")
        lines = out.split()
        if lines and lines[0].isdigit() and int(lines[0]) >= count:
            return True
        if "dead" in lines:
            return False
        time.sleep(2)
    return False


def nuc_log_tail(logfile, n=15):
    return ssh(f"cd {NUC_REPO} && tail -n {n} {logfile}")[1]


def fly_to(point, speed_peak=0.12, acc_peak=1.0):
    """Send a one-point plan (robot goes straight from its current xy to `point`) and wait until xy_driver is idle again."""
    lcm, _, _, xy_path_cmd_t, xy_state_t = lcm_types()
    lc = lcm.LCM()
    last = []
    lc.subscribe("XY_STATE", lambda ch, d: last.append(xy_state_t.decode(d)))
    t0 = time.time()
    while not last and time.time() - t0 < 3:
        lc.handle_timeout(50)
    if not last:
        raise RuntimeError("no XY_STATE before the fly-over")
    cur = np.array([last[-1].x, last[-1].y])
    d = float(np.linalg.norm(np.asarray(point) - cur))
    dur = max(3.0, 1.875 * d / speed_peak, math.sqrt(5.7735 * d / acc_peak))
    dur = math.ceil(dur * 2) / 2
    cmd = xy_path_cmd_t()
    cmd.timestamp_us = int(time.time() * 1e6)
    cmd.plan_id = int(time.time() * 1000) % (2 ** 31 - 1)
    cmd.num_points = 1
    cmd.points_xy = [[float(point[0]), float(point[1])]]
    cmd.duration_sec = dur
    print(f"  flying {d:.3f} m to ({point[0]:.4f}, {point[1]:.4f}) at z={HOME_Z} in {dur:.1f} s "
          f"(peak {1.875 * d / dur:.3f} m/s, {5.7735 * d / dur**2:.2f} m/s^2)", flush=True)
    lc.publish("XY_CMD", cmd.encode())
    started, tstart = False, time.time()
    while time.time() - tstart < dur + 15:
        lc.handle_timeout(50)
        s = last[-1]
        if s.plan_id == cmd.plan_id and s.tracking:
            started = True
        if started and not s.tracking:
            break
    else:
        raise RuntimeError("the fly-over did not finish")
    time.sleep(0.5)
    s = last[-1]
    err = math.hypot(s.x - point[0], s.y - point[1])
    print(f"  arrived at ({s.x:.4f}, {s.y:.4f}); distance to the start {err * 1000:.1f} mm", flush=True)
    return err


def record_plan(W, dur, run_dir):
    import frames as F
    lcm, object_pose_t, _, xy_path_cmd_t, xy_state_t = lcm_types()
    lc = lcm.LCM()
    Trows, Xrows, st = [], [], {"plan_t0": None, "done": None, "last_xy": None, "last_T": None}
    t_send = [None]
    cmd = xy_path_cmd_t()
    cmd.timestamp_us = int(time.time() * 1e6)
    cmd.plan_id = int(time.time() * 1000) % (2 ** 31 - 1)
    cmd.num_points = len(W)
    cmd.points_xy = [list(map(float, p)) for p in W]
    cmd.duration_sec = dur

    def on_T(_ch, d):
        if t_send[0] is None:
            return
        m = object_pose_t.decode(d)
        F.check_base_frame(m)
        st["last_T"] = time.time()
        th = F.theta_from_R(F.quat_xyzw_to_R(m.quaternion_xyzw))
        Trows.append([time.time() - t_send[0], *m.position_m, th, m.num_tags])

    def on_X(_ch, d):
        if t_send[0] is None:
            return
        s = xy_state_t.decode(d)
        now = time.time() - t_send[0]
        st["last_xy"] = time.time()
        Xrows.append([now, s.x, s.y, s.cmd_x, s.cmd_y, s.z, s.plan_id, s.tracking])
        if s.plan_id == cmd.plan_id and s.tracking and st["plan_t0"] is None:
            st["plan_t0"] = now
        if st["plan_t0"] is not None and not s.tracking and st["done"] is None:
            st["done"] = now

    lc.subscribe("T_SHAPE_POSE", on_T)
    lc.subscribe("XY_STATE", on_X)
    t_end_extra = 2.0                              # keep recording this long after tracking=0 (T settles)
    t_send[0] = time.time()
    lc.publish("XY_CMD", cmd.encode())
    print(f"  sent plan_id={cmd.plan_id}: {len(W)} points, {dur:.1f} s. Recording... (Ctrl-C stops the recording only)")
    reason = "tracking finished"
    x0, x1, y0, y1 = BOX
    last_print = 0.0
    try:
        while True:
            lc.handle_timeout(50)
            t = time.time() - t_send[0]
            if st["done"] is not None and t - st["done"] > t_end_extra:
                break
            if t > dur + 10:
                reason = "timeout"
                break
            if Trows and not (x0 <= Trows[-1][1] <= x1 and y0 <= Trows[-1][2] <= y1):
                reason = "T left the box"
                break
            now_w = time.time()
            if st["last_xy"] is not None and now_w - st["last_xy"] > SOFT_XY_GAP_S:
                reason = f"SOFT STOP: XY_STATE gap > {SOFT_XY_GAP_S} s (xy_driver dead / reflex / link?)"
                break
            if Xrows and math.hypot(Xrows[-1][3] - Xrows[-1][1], Xrows[-1][4] - Xrows[-1][2]) > SOFT_TRACK_ERR_M:
                reason = (f"SOFT STOP: commanded - measured position {math.hypot(Xrows[-1][3] - Xrows[-1][1], Xrows[-1][4] - Xrows[-1][2]) * 100:.1f} cm "
                          f"> {SOFT_TRACK_ERR_M * 100:.0f} cm")
                break
            if st["last_T"] is not None and now_w - st["last_T"] > SOFT_T_GAP_S:
                reason = f"SOFT STOP: no T_SHAPE_POSE for > {SOFT_T_GAP_S} s"
                break
            if t - last_print > 1.0 and Xrows:
                last_print = t
                print(f"    t={t:5.1f}s flange=({Xrows[-1][1]:.3f},{Xrows[-1][2]:.3f}) tracking={Xrows[-1][7]}", end="\r")
    except KeyboardInterrupt:
        reason = "Ctrl-C"
    if reason.startswith("SOFT STOP") and SEND_HOLD and Xrows:
        # xy_driver cannot preempt: this plan is only QUEUED and starts after the running one ends (see --send-hold)
        h = xy_path_cmd_t()
        h.timestamp_us = int(time.time() * 1e6)
        h.plan_id = (cmd.plan_id + 1) % (2 ** 31 - 1)
        h.num_points = 1
        h.points_xy = [[float(Xrows[-1][1]), float(Xrows[-1][2])]]
        h.duration_sec = 1.0
        lc.publish("XY_CMD", h.encode())
        print(f"  hold plan (1 point at the measured position, 1 s) published as plan_id={h.plan_id}")
    print(f"\n  recording stopped: {reason}"
          + (f" -- {RED}the arm may still be moving: release the enable button{END}" if reason != "tracking finished" else ""))
    if reason != "tracking finished":
        print(f"  {RED}recording stopped early; going to cleanup now (not waiting for the timeout){END}")
    np.savetxt(os.path.join(result_dir(run_dir), "real_T.txt"), np.array(Trows), fmt="%.6f",
               header="t_s x_base y_base z_base theta_base_rad num_tags (t=0: plan sent)")
    np.savetxt(os.path.join(result_dir(run_dir), "real_xy.txt"), np.array(Xrows), fmt="%.6f",
               header="t_s x y cmd_x cmd_y z plan_id tracking (t=0: plan sent)")
    return (st["plan_t0"] if st["plan_t0"] is not None else 0.0), reason, cmd.plan_id


def nuc_cleanup(state):
    print("== cleanup ==")
    lf = state.get("log")
    if state.get("driver"):
        if state.get("down"):
            ssh(f"echo up > {FIFO}")
            print("  NUC: 'up' sent; waiting for 'At z=0.4434' (a plan still running finishes first)")
            if not wait_log(lf, "At z=0.4434", 300):
                print(f"  {RED}WARNING: no 'At z=0.4434' in the log{END}\n" + nuc_log_tail(lf))
        ssh("kill -TERM $(pgrep -x xy_driver) 2>/dev/null; true")
        for _ in range(30):
            rc, out = ssh("pgrep -x xy_driver; true")
            if not out.strip():
                break
            time.sleep(1)
        else:
            print(f"  {RED}WARNING: xy_driver still running on the NUC{END}")
        ssh("kill $(pgrep -x -a sleep | awk '$3==100000 {print $1}') 2>/dev/null; rm -f ~/xy_in; true")
    nuc_stop_state_and_pose()
    ok1, ok2 = nuc_start_state_pub(), nuc_start_pose()
    print(f"  NUC: state_pub restart {'ok' if ok1 else 'FAILED'}; aprilcube_pose restart {'ok' if ok2 else 'FAILED'}")
    check_carrier("after run: ")


def compare(run_dir, plan_t0):
    """Real T vs the OPEN-LOOP sim replay of the same waypoints (sim_T_traj_openloop.csv); the closed-loop sim
    (sim_T_traj.csv) is kept as a light reference and sim_T_traj_openloop_band.csv (5 perturbed replays) as a band."""
    import frames as F
    goal = F.load_goal()
    rd = result_dir(run_dir)
    R = np.loadtxt(os.path.join(rd, "real_T.txt"), ndmin=2)
    Sm = np.loadtxt(os.path.join(run_dir, "sim_T_traj_openloop.csv"), delimiter=",", skiprows=1)
    clp = os.path.join(run_dir, "sim_T_traj.csv")
    Cl = np.loadtxt(clp, delimiter=",", skiprows=1) if os.path.exists(clp) else None
    bpath = os.path.join(run_dir, "sim_T_traj_openloop_band.csv")
    Bd = np.loadtxt(bpath, delimiter=",", skiprows=1) if os.path.exists(bpath) else None
    rt = R[:, 0] - plan_t0
    conv = [F.base_to_sim(r[1:3], r[4], goal) for r in R]
    rg = np.array([[c[0][0], c[0][1], c[1]] for c in conv])
    rg[:, 2] = np.unwrap(rg[:, 2])
    t = Sm[:, 0]
    sel = (t >= rt.min()) & (t <= rt.max())
    t = t[sel]
    sim = Sm[sel, 1:4]
    real = np.column_stack([np.interp(t, rt, rg[:, 0]), np.interp(t, rt, rg[:, 1]), np.interp(t, rt, rg[:, 2])])
    cl = (np.column_stack([np.interp(t, Cl[:, 0], Cl[:, k]) for k in (1, 2, 3)]) if Cl is not None
          else np.full((len(t), 3), np.nan))
    if Bd is not None:       # largest deviation of any perturbed replay from the nominal one, per time
        cols = [np.interp(t, Bd[:, 0], Bd[:, k]) for k in range(1, 7)]
        lo, hi = np.column_stack(cols[:3]), np.column_stack(cols[3:])
        dpos = np.maximum(np.linalg.norm(hi[:, :2] - sim[:, :2], axis=1), np.linalg.norm(lo[:, :2] - sim[:, :2], axis=1)) * 1000
        dang = np.degrees(np.maximum(np.abs(hi[:, 2] - sim[:, 2]), np.abs(lo[:, 2] - sim[:, 2])))
    else:
        dpos = dang = np.zeros(len(t))
    perr = np.linalg.norm(real[:, :2] - sim[:, :2], axis=1) * 1000
    aerr = np.degrees(np.array([F.wrap(a) for a in real[:, 2] - sim[:, 2]]))
    np.savetxt(os.path.join(rd, "compare.csv"), np.column_stack([t, sim, real, perr, aerr, cl, dpos, dang]),
               delimiter=",", fmt="%.6f", comments="",
               header="t_s,sim_ol_x,sim_ol_y,sim_ol_theta,real_x,real_y,real_theta,pos_err_mm,heading_err_deg,"
                      "sim_cl_x,sim_cl_y,sim_cl_theta,band_pos_dev_mm,band_ang_dev_deg (goal frame; errors are real - open-loop sim)")
    fin_real = np.array([rg[-1, 0], rg[-1, 1], F.wrap(rg[-1, 2])])
    fs = Sm[-1, 1:4]
    print("== result ==")
    print(f"  final real T (goal frame): x={fin_real[0]:.4f} y={fin_real[1]:.4f} theta={math.degrees(fin_real[2]):.2f} deg")
    print(f"  final error to GOAL         : {np.linalg.norm(fin_real[:2]) * 1000:.1f} mm, {abs(math.degrees(fin_real[2])):.2f} deg")
    print(f"  final error to OPEN-LOOP sim: {np.linalg.norm(fin_real[:2] - fs[:2]) * 1000:.1f} mm, "
          f"{abs(math.degrees(F.wrap(fin_real[2] - fs[2]))):.2f} deg  (that sim ended {np.linalg.norm(fs[:2]) * 1000:.1f} mm from goal)")
    print(f"  real vs open-loop sim over the run: mean {perr.mean():.1f} mm, max {perr.max():.1f} mm; "
          f"sim sensitivity band up to {dpos.max():.1f} mm / {dang.max():.1f} deg")
    plot(run_dir)


def plot(run_dir):
    try:
        import matplotlib
    except ImportError:    # the global-mppi env has no matplotlib: use the system python for the figure only
        subprocess.run(["/usr/bin/python3", os.path.abspath(__file__), "--plot-only", run_dir] + (["--tag", RESULT_TAG] if RESULT_TAG else []), check=False)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    C = np.loadtxt(os.path.join(result_dir(run_dir), "compare.csv"), delimiter=",", skiprows=1)
    t = C[:, 0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    # T outline (12x12 cm: crossbar 12x4 at y 0.02..0.06, stem 4x8 at y -0.06..0.02), goal frame
    outline = np.array([[-.06, .02], [-.02, .02], [-.02, -.06], [.02, -.06], [.02, .02], [.06, .02], [.06, .06], [-.06, .06], [-.06, .02]])

    def draw(th, xy, **kw):
        c, s = math.cos(th), math.sin(th)
        o = outline @ np.array([[c, -s], [s, c]]).T + xy
        ax[0].plot(o[:, 0], o[:, 1], **kw)

    bpath = os.path.join(run_dir, "sim_T_traj_openloop_band.csv")
    if os.path.exists(bpath):          # envelope of the 5 perturbed open-loop replays (min/max x, y per time)
        Bd = np.loadtxt(bpath, delimiter=",", skiprows=1)
        for r in Bd[::2]:
            ax[0].add_patch(Rectangle((r[1], r[2]), r[4] - r[1], r[5] - r[2], color="tab:blue", alpha=0.05, lw=0))
    draw(0.0, np.zeros(2), color="green", lw=2, label="goal")
    ax[0].plot(C[:, 9], C[:, 10], color="lightsteelblue", lw=1.5, label="sim closed loop (reference)")
    ax[0].plot(C[:, 1], C[:, 2], "b-", label="sim open loop (+ shaded: 5 perturbed replays)")
    ax[0].plot(C[:, 4], C[:, 5], "r-", label="real T centre")
    draw(C[0, 3], C[0, 1:3], color="b", lw=1, ls="--")
    draw(C[-1, 3], C[-1, 1:3], color="b", lw=1)
    draw(C[0, 6], C[0, 4:6], color="r", lw=1, ls="--")
    draw(C[-1, 6], C[-1, 4:6], color="r", lw=1)
    ax[0].set_aspect("equal")
    ax[0].grid(alpha=0.3)
    ax[0].set_xlabel("x (goal frame) [m]")
    ax[0].set_ylabel("y (goal frame) [m]")
    ax[0].set_title("T path: real vs open-loop sim (dashed = start, solid = end)")
    ax[0].legend(loc="best", fontsize=7)
    a1 = ax[1]
    a1.fill_between(t, 0, C[:, 12], color="gray", alpha=0.25, label="sim sensitivity band (position)")
    a1.plot(t, C[:, 7], "k-", label="position error real - sim [mm]")
    a1.set_xlabel("time since plan start [s]")
    a1.set_ylabel("position error [mm]")
    a2 = a1.twinx()
    a2.fill_between(t, 0, C[:, 13], color="plum", alpha=0.2, label="sim sensitivity band (heading)")
    a2.plot(t, C[:, 8], "m-", label="heading error real - sim [deg]")
    a2.set_ylabel("heading error [deg]")
    a1.grid(alpha=0.3)
    a1.set_title("real - open-loop sim")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir(run_dir), "real_vs_sim.png"), dpi=130)
    print(f"  wrote {os.path.join(result_dir(run_dir), 'real_vs_sim.png')}")


def real_run(run_dir, prepare_only=False):
    W, dur = load_plan(run_dir)
    state = {}
    plan_t0, reason, plan_id = 0.0, None, None
    try:
        print("== 1) NUC bring-up (script/openloop_cmds.md) ==")
        if not check_carrier("before start: "):
            sys.exit("link flapped since the baseline: not starting")
        if not nuc_stop_state_and_pose():
            sys.exit("state_pub / aprilcube_pose still running on the NUC after SIGTERM")
        rc, out = ssh(f"cd {NUC_REPO} && ./script/check_f_t_ee.sh; echo exit=$?", 120)
        last = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("exit=")]
        if "exit=0" not in out or not last or last[-1].strip() != "PASS":
            sys.exit(f"F_T_EE guard failed:\n{out[-600:]}")
        print("  F_T_EE guard: exit 0, last line PASS")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        state["log"] = lf = f"log/xy_driver_openloop_{stamp}.log"
        ssh(f"rm -f {FIFO} && mkfifo {FIFO}")
        ssh(f"(setsid sleep 100000 > {FIFO} 2>/dev/null < /dev/null &); true", 10)
        rc = 0 if nuc_bg(f"cd {NUC_REPO} && exec {XY_DRIVER_BIN} {ROBOT_IP} --work-z={WORK_Z}{XY_DRIVER_EXTRA} < {FIFO}",
                         f"{NUC_REPO}/{lf}") else 1
        out = ""
        state["driver"] = True
        if rc != 0:
            sys.exit(f"could not start xy_driver: {out}")
        print(f"  xy_driver started (NUC {NUC_REPO}/{lf}); homing to (0.40, 0, {HOME_Z}) ...")
        if not wait_log(lf, "Homed; ready for plans at z=0.4434.", 240):
            sys.exit("xy_driver did not report 'Homed; ready for plans at z=0.4434.':\n" + nuc_log_tail(lf, 25)
                     + "\n(UDP receive: Timeout -> check ~/linkwatch.log for 'Link is Down')")
        rc, tilt = ssh(f"cd {NUC_REPO} && grep -i -E 'tilt|warn' {lf}; true")
        if tilt.strip():
            print(f"  {YEL}xy_driver warnings:{END}\n{tilt}")
        print("  homed")
        if FLY_TO_START:
            # safe height, straight line (from above the T is fine); camera / T pose come afterwards
            err = fly_to(W[0])
            if err > 0.003:
                sys.exit(f"after the fly-over the flange is {err * 1000:.1f} mm from the start point")
            print("  nothing must be under the pen: sending 'down' ...")
            state["down"] = True
            ssh(f"echo down > {FIFO}")
            if not wait_log(lf, "At z=0.34874; ready for plans", 90):
                sys.exit("no 'At z=0.34874; ready for plans' in the log:\n" + nuc_log_tail(lf, 15))
            z = read_flange_z()
            print(f"  measured flange z = {z:.4f} (expect 0.3486 +- 0.0003)")
            if z is None or abs(z - 0.3486) > 0.0003:
                sys.exit("flange z not at the work height")
        if not nuc_start_pose():
            sys.exit("could not start aprilcube_pose")
        for _try in range(8):        # camera start-up takes a while: up to ~35 s for both rates to show up
            n, dt = listen(4.0, ("T_SHAPE_POSE", "XY_STATE", "ROBOT_STATE", "XY_CMD"))
            if n["XY_STATE"] / dt >= 10 and n["T_SHAPE_POSE"] / dt >= 10:
                break
        hz = {c: n[c] / dt for c in n}
        print(f"  rates: XY_STATE {hz['XY_STATE']:.0f} Hz, T_SHAPE_POSE {hz['T_SHAPE_POSE']:.1f} Hz, "
              f"ROBOT_STATE {hz['ROBOT_STATE']:.0f} Hz, XY_CMD {n['XY_CMD']} msgs (info only)")
        if hz["XY_STATE"] < 10 or hz["T_SHAPE_POSE"] < 10:
            sys.exit(f"need XY_STATE and T_SHAPE_POSE >= 10 Hz each -> abort")
        if prepare_only:
            print("== --prepare-only: T start pose check only; no down, no plan ==")
            tp = read_T_pose(2.0)
            if tp is None:
                sys.exit("no T_SHAPE_POSE samples")
            d, da, (p, th), srec = start_pose_deviation(tp)
            print(f"  T (goal frame) x={p[0]:.4f} y={p[1]:.4f} theta={math.degrees(th):.2f} deg vs {START_NAME} "
                  f"({srec['x']:.4f}, {srec['y']:.4f}, {srec['theta_deg']:.2f}): {d:.1f} mm, {da:.2f} deg "
                  f"({'OK' if d <= 10 and da <= 3 else 'OFF: put the T back'})")
            return
        if not FLY_TO_START:
            print("  nothing must be under the pen: sending 'down' ...")
            state["down"] = True
            ssh(f"echo down > {FIFO}")
            if not wait_log(lf, "At z=0.34874; ready for plans", 90):
                sys.exit("no 'At z=0.34874; ready for plans' in the log:\n" + nuc_log_tail(lf, 15))
            z = read_flange_z()
            print(f"  measured flange z = {z:.4f} (expect 0.3486 +- 0.0003)")
            if z is None or abs(z - 0.3486) > 0.001:
                sys.exit("flange z not at the work height")
        rc, plans = ssh(f"cd {NUC_REPO} && grep -c 'plan_id=' {lf}; true")
        n_ours = 1 if FLY_TO_START else 0                  # the fly-over plan is ours
        if plans.strip().isdigit() and int(plans) > n_ours:
            sys.exit(f"the xy_driver log contains {plans} plan_id line(s), expected {n_ours}: someone else publishes XY_CMD")

        print("== 2) T start pose ==")
        tp = read_T_pose(2.0)
        if tp is None:
            sys.exit("no T_SHAPE_POSE samples")
        d, da, (p, th), srec = start_pose_deviation(tp)
        print(f"  T (goal frame) x={p[0]:.4f} y={p[1]:.4f} theta={math.degrees(th):.2f} deg vs {START_NAME} "
              f"({srec['x']:.4f}, {srec['y']:.4f}, {srec['theta_deg']:.2f}): {d:.1f} mm, {da:.2f} deg")
        if STRICT_START is not None:
            if d > STRICT_START[0] or da > STRICT_START[1]:
                sys.exit(f"T is off start_2 by {d:.1f} mm / {da:.2f} deg (limit {STRICT_START[0]} mm / {STRICT_START[1]} deg): not starting")
        elif d > 10 or da > 3:
            if input(f"  {RED}T is off the recorded start by > 10 mm / 3 deg.{END} Continue anyway? [y/N] ").strip().lower() != "y":
                sys.exit("aborted: put the T back and rerun")

        print("== 3) ==")
        input("  hand on the enable button, press Enter to START ")
        print("== 4) run ==")
        plan_t0, reason, plan_id = record_plan(W, dur, run_dir)
        rc, plans = ssh(f"cd {NUC_REPO} && grep 'plan_id=' {lf}; true")
        print(f"  xy_driver log plan lines: {plans.strip() or '(none)'}   (ours: plan_id={plan_id})")
    finally:
        nuc_cleanup(state)
    if reason is not None and os.path.exists(os.path.join(run_dir, "real_T.txt")):
        compare(run_dir, plan_t0)


def main():
    global XY_DRIVER_BIN, XY_DRIVER_EXTRA, SEND_HOLD, MAX_LEAD_MM, RESULT_TAG, FLY_TO_START, STRICT_START
    SEND_HOLD = False
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=DEFAULT_RUN_DIR, help="default: %(default)s")
    ap.add_argument("--dry-run", action="store_true", help="pre-flight only; nothing moves")
    ap.add_argument("--xy-driver-bin", default=XY_DRIVER_BIN, help="xy_driver binary on the NUC, relative to the repo (default %(default)s)")
    ap.add_argument("--max-lead-mm", type=float, default=MAX_LEAD_MM, help="pass --max-lead-mm=VALUE to xy_driver (limit on how far the commanded pose may lead the measured one); 0 = do not pass it (default %(default)s)")
    ap.add_argument("--upright", action="store_true", help="start xy_driver with --upright (default off)")
    ap.add_argument("--yaw-deg", type=float, default=None, help="with --upright: also pass --yaw-deg=VALUE to xy_driver")
    ap.add_argument("--tag", default="", help="write the real-run files to <run_dir>/real_<TAG>/ (e.g. --tag run2) instead of <run_dir>/")
    ap.add_argument("--fly-to-start", action="store_true", help="the plan's first point is the pusher start: after homing fly there at "
                    "z=0.4434 (straight line), then 'down', then the camera / T check; no detour in the plan")
    ap.add_argument("--strict-start", type=float, nargs=2, metavar=("MM", "DEG"), default=None,
                    help="abort (no prompt) when the T is further from start_2 than this")
    ap.add_argument("--send-hold", action="store_true", help="on a soft stop also publish a 1-point hold plan at the measured "
                    "position. NOTE: xy_driver only QUEUES it (starts after the running plan ends, from the commanded pose), "
                    "so it does not stop the arm; off by default")
    ap.add_argument("--prepare-only", action="store_true", help="NUC start-up stage + T start pose check, then restore; no down, no plan")
    ap.add_argument("--plot-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--compare-only", type=float, metavar="PLAN_T0", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    run_dir = os.path.abspath(os.path.expanduser(a.run_dir))
    if a.yaw_deg is not None and not a.upright:
        ap.error("--yaw-deg only makes sense together with --upright")
    XY_DRIVER_BIN = a.xy_driver_bin
    SEND_HOLD = a.send_hold
    FLY_TO_START = a.fly_to_start
    STRICT_START = tuple(a.strict_start) if a.strict_start else None
    RESULT_TAG = a.tag
    XY_DRIVER_EXTRA = (f" --max-lead-mm={a.max_lead_mm:g}" if a.max_lead_mm > 0 else "") \
        + ((" --upright" + (f" --yaw-deg={a.yaw_deg}" if a.yaw_deg is not None else "")) if a.upright else "")
    MAX_LEAD_MM = a.max_lead_mm
    if a.plot_only:
        plot(run_dir)
        return 0
    if a.compare_only is not None:         # re-make the comparison from an existing real_T.txt
        compare(run_dir, a.compare_only)
        return 0
    ok = preflight(run_dir)
    if a.dry_run:
        return 0 if ok else 1
    if not ok:
        print("pre-flight failed; not starting. (fix the FAILs above)")
        return 1
    real_run(run_dir, a.prepare_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
