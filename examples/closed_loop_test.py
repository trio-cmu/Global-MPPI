#!/usr/bin/env python3
"""Closed-loop Push-T with Predictive Sampling, on the simulator (--sim) or the REAL robot (--real).

  python examples/closed_loop_test.py --sim                 # MuJoCo stands in for the camera and the arm
  python examples/closed_loop_test.py --real --dry-run      # real poses, plan and print only, nothing is sent
  python examples/closed_loop_test.py --real                # bring-up, stop at "Enter to start", then the loop

Every round: T pose (camera, base frame) + pen position (XY_STATE) -> goal frame -> state of the planner model ->
PS plan (planner exactly as in examples/pusht.py, same default parameters) -> first --chunk-s seconds of the plan
-> integrated to waypoints (base frame) -> checked against the xy_driver limits (segment >= 0.4 s, peak speed,
peak acceleration; the duration is stretched when needed) -> ONE XY_CMD -> wait until it has been executed -> next round.

Stops: T < 10 mm and < 3 deg; 90 s; T_SHAPE_POSE silent > 1 s; plan rejected (not tracked within 3 s);
commanded - measured pen position > --max-lead-mm; XY_STATE silent > 0.3 s; T left the box; carrier_changes of the NUC link
increased; T further than --max-dist-mm (350) from the goal; pen at the box edge > --max-edge-s (5 s); fewer than 5 usable T frames
in the 0.3 s read window; --max-cycles.
xy_driver cannot cancel a running plan: on a stop in --real, release the enable button.

Outputs in runs/closed_loop_test/<backend>_<stamp>/: cycles.csv (T pose, pen, errors, plan time, plan summary per round),
plans.npz (waypoints per round), trace.csv (continuous T / pen), closed_loop_traj.png, closed_loop_errors.png, args.json.
Real bring-up follows examples/open_loop_test.py (NUC over ssh, trio_infra script/openloop_cmds.md).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
TRIO_SIM = os.path.expanduser("~/trio_sim")
TRIO_INFRA = os.path.expanduser("~/projects/trio_infra")
os.environ.setdefault("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=1")

SEG_MIN_S, PEAK_V_MAX, PEAK_A_MAX = 0.4, 0.2, 1.9      # xy_driver: segment >= 0.4 s, kMaxSpeed 0.25, plan accel 2.0 (margin)
SOFT_XY_GAP_S, SOFT_T_GAP_S, REJECT_WAIT_S = 0.3, 1.0, 3.0
WAIT_AFTER_S = 0.35                                     # tracking=0 -> start of the T pose window (camera latency 0.18 s + settle 0.15 s)
WINDOW_S, MIN_FRAMES = 0.3, 5                           # T pose = median of the frames received in this window; fewer -> pose lost
READ_MAX_S, MIN_FRAMES_LOW = 1.5, 3                      # keep waiting for MIN_FRAMES up to READ_MAX_S; >= 3 frames then: used, flagged low_frames; < 3: pose lost
ADAPT_MIN_S, ADAPT_MAX_S = 0.1, 0.6                    # --adaptive-settle: earliest / latest start of the T pose window after the stop
ADAPT_DPOS_M, ADAPT_DANG_RAD = 0.0005, math.radians(0.3)   # --adaptive-settle: 3 consecutive frames within this = at rest
DRAIN_TIMEOUT_S = 8.0                                   # --pipeline: max wait for the arm to stop / the running plan to finish
CAM_LATENCY_S = 0.18
CAM_SIGMA = (0.0002, 0.001, math.radians(0.3))          # --sim measurement noise: x, y (base frame) [m], theta [rad]
PUSHER_START_YAML = os.path.join(TRIO_SIM, "src", "pusht_align", "pusher_start_pushside_near.yaml")   # default start: 2 cm from the T outline, pushing side
PUSHER_START_STD_YAML = os.path.join(TRIO_SIM, "src", "pusht_align", "pusher_start.yaml")           # the old standard start
UPRIGHT_BIN = "../trio_infra_upright/build_upright/bin/xy_driver"


def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def quintic(chain, seg_t, t):
    """Position / velocity at time t of the rest-to-rest quintic per segment through `chain` (xy_driver's tracking)."""
    i = min(int(t / seg_t), len(chain) - 2)
    u = min((t - i * seg_t) / seg_t, 1.0)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / seg_t if t < (len(chain) - 1) * seg_t else 0.0
    d = chain[i + 1] - chain[i]
    return chain[i] + s * d, ds * d


def check_limits(p0, wps, dur):
    """(ok, seg_t, peak speed, peak accel) of the plan p0 -> wps over dur seconds, equal segment times."""
    chain = np.vstack([p0, wps])
    m = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).max())
    st = dur / len(wps)
    peak, acc = 1.875 * m / st, 5.7735 * m / st**2
    return (st >= SEG_MIN_S - 1e-9 and peak <= PEAK_V_MAX and acc <= PEAK_A_MAX), st, peak, acc


class Stop(Exception):
    pass


# ------------------------------------------------------------------ planner (copied from examples/pusht.py)
class Planner:
    def __init__(self, a, xml=None):
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
        from global_mppi.algs import PredictiveSampling
        from global_mppi.tasks.pusht import PushT
        self.jax, self.jnp, self.mjx, self.mujoco = jax, jnp, mjx, mujoco
        task = PushT(xml)         # xml=None: the default scene (as pusht.py); the box walls scene otherwise
        self.n_opt = max(1, int(a.n_opt))
        if a.algo == "dial":        # examples/pusht.py DIAL defaults, none changed
            from global_mppi.algs import DIAL
            print("Running DIAL")
            self.ctrl = DIAL(task, num_samples=256, noise_level=a.dial_noise, beta_opt_iter=1.0, beta_horizon=1.0, temperature=0.01,
                             plan_horizon=1.0, spline_type="cubic", num_knots=6)
        else:
            print("Running predictive sampling")
            self.ctrl = PredictiveSampling(task, num_samples=a.num_samples, noise_level=a.noise_level, num_randomizations=1,
                                           plan_horizon=a.plan_horizon, spline_type="cubic", num_knots=a.num_knots)
        m = task.mj_model
        m.opt.timestep, m.opt.iterations, m.opt.ls_iterations = 0.001, 100, 50
        self.model, self.dt = m, m.opt.timestep
        d = mujoco.MjData(m)
        d.qpos[:] = task.start_qpos("start_2")
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self.mjx_data = mjx.put_data(m, d)
        self.mjx_data = self.mjx_data.replace(mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
        self.params = self.ctrl.init_params(initial_knots=None, seed=a.seed)
        self.jit_optimize = jax.jit(self.ctrl.optimize)
        self.jit_interp = jax.jit(self.ctrl.interp_func)
        self.vmax = float(np.abs(m.actuator_ctrlrange).max())

    def warmup(self):
        t0 = time.time()
        q0 = np.asarray(self.mjx_data.qpos)
        for _ in range(2):                      # the same call path as the loop (traced types must match)
            self.plan(q0, 0.0, 0.5)
        self.jax.block_until_ready(self.params.mean)
        print(f"Time to jit: {time.time() - t0:.2f} s")

    def plan(self, qpos, t_clock, chunk_s):
        """Plan from qpos (planner model coordinates) at planner time t_clock. Returns (pusher velocity u (n,2) in the
        goal frame over chunk_s, plan wall time)."""
        jnp = self.jnp
        s = self.mjx_data.replace(qpos=jnp.array(qpos), qvel=jnp.zeros_like(self.mjx_data.qvel), time=t_clock)
        t0 = time.time()
        for _ in range(self.n_opt):               # --n-opt: repeated optimisation of the same state, time unchanged
            self.params, _, _ = self.jit_optimize(s, self.params)
        n = int(round(chunk_s / self.dt))
        tq = jnp.arange(0, n) * self.dt + t_clock
        us = np.asarray(self.jit_interp(tq, self.params.tk, self.params.mean[None, ...]))[0]
        return np.clip(us, -self.vmax, self.vmax), time.time() - t0


# ------------------------------------------------------------------ frames (goal frame = sim frame)
class Frames:
    def __init__(self, scene):
        self.sc = scene

    def base_to_sim(self, p):
        return rot2(-self.sc.goal_th) @ (np.asarray(p, float) - self.sc.goal_p)

    def sim_to_base(self, p):
        return self.sc.sim_to_base(p)

    def clamp_base(self, p):
        b, m = self.sc.box, self.sc.box_margin
        q = np.array([np.clip(p[0], b[0] + m, b[1] - m), np.clip(p[1], b[2] + m, b[3] - m)])
        return q, bool(np.any(q != p))

    def T_in_box(self, p_sim):
        b = self.sc.box
        pb = self.sim_to_base(p_sim)
        return b[0] <= pb[0] <= b[1] and b[2] <= pb[1] <= b[3]


# ------------------------------------------------------------------ backends
class SimBackend:
    """MuJoCo stands in for camera and arm: the T lives in the plant model, the pen follows the XY_CMD with
    xy_driver's rest-to-rest quintics (kinematic, like the real tracking within < 1 mm)."""
    name = "sim"

    def __init__(self, a, scene, fr):
        import mujoco
        self.mujoco, self.sc, self.fr, self.a = mujoco, scene, fr, a
        self.data = scene.new_data()
        self.model = scene.plant
        self.rng = np.random.default_rng(a.seed + 1)
        self.t = 0.0                                          # virtual run clock [s]
        self.sim_t = 0.0
        self.rd = dict(settle=0.0, read=0.0, n=0, low=0)
        self.checks = []
        self.segs = []
        self.cmd = scene.pusher_base(self.data.qpos)          # commanded pen position (base)
        self.trace = []
        self.hist = [(0.0, np.array(self.data.qpos[:3]))]       # plant T pose history (the camera sees it 0.18 s late)
        self._log_trace()

    def now(self):
        return self.t

    def advance(self, dt):
        self.t += dt

    def _log_trace(self):
        d = self.data
        pb = self.sc.pusher_base(d.qpos)
        self.trace.append([self.t, d.qpos[0], d.qpos[1], d.qpos[2], pb[0], pb[1], self.cmd[0], self.cmd[1]])

    def _step(self, n):
        m, d = self.model, self.data
        for _ in range(n):
            mj_step = self.mujoco.mj_step
            mj_step(m, d)
            self.sim_t += m.opt.timestep
            if int(round(self.sim_t / m.opt.timestep)) % 5 == 0:
                self.hist.append((self.sim_t, np.array(d.qpos[:3])))

    def _T_at(self, t):
        ts = np.array([h[0] for h in self.hist[-400:]])
        k = max(int(np.searchsorted(ts, t, side="right")) - 1, 0)
        return self.hist[-400:][k][1]

    def state(self):
        """Like the camera: 10 frames over WINDOW_S after the wait, each the T pose WINDOW-lagged by CAM_LATENCY_S
        plus noise (x, y in the base frame), median. The plant runs on meanwhile (pen at rest)."""
        sc, m = self.sc, self.model
        fr = []
        for k in range(10):
            self._step(int(round(WINDOW_S / 10 / m.opt.timestep)))
            q = self._T_at(self.sim_t - CAM_LATENCY_S)
            pb = sc.sim_to_base(q[:2]) + self.rng.normal(0, 1, 2) * CAM_SIGMA[:2]
            fr.append([*pb, q[2] + sc.goal_th + self.rng.normal(0, CAM_SIGMA[2])])
        fr = np.array(fr)
        th = math.atan2(np.sin(fr[:, 2]).mean(), np.cos(fr[:, 2]).mean())
        th_med = th + float(np.median([wrap(x - th) for x in fr[:, 2]]))
        pm = np.median(fr[:, :2], axis=0)
        self.t += WINDOW_S
        self.rd = dict(settle=WAIT_AFTER_S, read=WINDOW_S, n=len(fr), low=0)
        return np.array([*self.fr.base_to_sim(pm), wrap(th_med - sc.goal_th)]), sc.pusher_base(self.data.qpos).copy()

    def execute(self, wps, dur):
        mj, d, m, sc = self.mujoco, self.data, self.model, self.sc
        chain = np.vstack([self.cmd, wps])
        seg_t = dur / len(wps)
        R = rot2(-sc.goal_th)
        n = int(round(dur / m.opt.timestep))
        for k in range(n):
            tt = k * m.opt.timestep
            p, v = quintic(chain, seg_t, min(tt, dur))
            ps = R @ (p - sc.goal_p)
            d.qpos[3:5] = ps - sc.anchor
            d.qvel[3:5] = R @ v if tt < dur else 0.0
            self.cmd = p
            self._step(1)
            if k % 20 == 0:
                self._log_trace_at(self.t + tt)
        d.qvel[3:5] = 0.0
        self._step(int(round(WAIT_AFTER_S / m.opt.timestep)))
        self.t += dur + WAIT_AFTER_S
        self.cmd = chain[-1]
        return "ok", 0.0, dict(publish=0.0, start_lat=0.0, track=dur, total=dur)

    def _log_trace_at(self, t):
        d = self.data
        pb = self.sc.pusher_base(d.qpos)
        self.trace.append([t, d.qpos[0], d.qpos[1], d.qpos[2], pb[0], pb[1], self.cmd[0], self.cmd[1]])

    # ---- --pipeline: xy_driver queue model. One segment runs; a plan sent meanwhile is queued and starts (from the commanded
    # end pose of the running one) the moment it ends; XY_STATE.plan_id = id of the running segment.
    def send_pipe(self, wps, dur):
        if not self.a.blend and any(sg["t_start"] is None for sg in self.segs):
            raise RuntimeError("a third plan was sent while one is already queued")
        sid = len(self.segs) + 1
        self.segs.append(dict(id=sid, wps=np.array(wps), dur=float(dur), t_send=self.sim_t, t_start=None, t_end=None, chain=None))
        return sid

    def current_id(self):
        for sg in self.segs:
            if sg["t_start"] is not None and sg["t_end"] is None:
                return sg["id"]
        return None

    def started(self, sid):
        """True once segment sid has been picked up by the driver (it is running or already finished)."""
        return self.segs[sid - 1]["t_start"] is not None

    def _advance_blend(self, dt):
        """--blend: a plan sent while another runs is picked up at once (xy_driver --blend): it interrupts the running one and starts
        from its commanded position / velocity / acceleration, ends at rest. The driver's rejection rules (speed 0.25, accel 2.0, box) apply."""
        m, d, sc = self.model, self.data, self.sc
        R = rot2(-sc.goal_th)
        if not hasattr(self, "cmd_v"):
            self.cmd_v, self.cmd_a, self.run_seg = np.zeros(2), np.zeros(2), None
        for _ in range(int(round(dt / m.opt.timestep))):
            nxt = next((sg for sg in self.segs if sg["t_start"] is None), None)
            if nxt is not None:
                tr = blend_build(self.cmd, self.cmd_v, self.cmd_a, nxt["wps"], nxt["dur"])
                ok, sp, ac, out = blend_check(tr, sc.box, 0.25, 2.0)
                if not ok:
                    raise Stop(f"sim: plan {nxt['id']} rejected like the xy_driver would (peak {sp:.3f} m/s, accel {ac:.2f} m/s^2, "
                               f"outside the box by {out * 1000:.1f} mm)")
                if self.run_seg is not None:
                    self.run_seg["t_end"] = self.sim_t
                nxt["t_start"], nxt["traj"], self.run_seg = self.sim_t, tr, nxt
            run = self.run_seg
            if run is not None:
                tt = self.sim_t - run["t_start"]
                p, v, a_ = blend_eval(run["traj"], tt)
                self.cmd, self.cmd_v, self.cmd_a = p[0], v[0], a_[0]
                d.qpos[3:5], d.qvel[3:5] = R @ (p[0] - sc.goal_p) - sc.anchor, R @ v[0]
                if tt >= run["dur"] - 1e-9:
                    run["t_end"], self.run_seg = self.sim_t, None
                    self.cmd, self.cmd_v, self.cmd_a = run["wps"][-1].copy(), np.zeros(2), np.zeros(2)
                    d.qvel[3:5] = 0.0
            else:
                d.qvel[3:5] = 0.0
            self._step(1)
            if int(round(self.sim_t / m.opt.timestep)) % 20 == 0:
                self._log_trace_at(self.sim_t)
        self.t = self.sim_t

    def advance_pipe(self, dt):
        if self.a.blend:
            return self._advance_blend(dt)
        m, d, sc = self.model, self.data, self.sc
        R = rot2(-sc.goal_th)
        for _ in range(int(round(dt / m.opt.timestep))):
            run = next((sg for sg in self.segs if sg["t_start"] is not None and sg["t_end"] is None), None)
            if run is None:
                nxt = next((sg for sg in self.segs if sg["t_start"] is None), None)
                if nxt is not None:
                    nxt["t_start"], nxt["chain"] = self.sim_t, np.vstack([self.cmd, nxt["wps"]])
                    run = nxt
            if run is not None:
                tt = self.sim_t - run["t_start"]
                p, v = quintic(run["chain"], run["dur"] / len(run["wps"]), min(tt, run["dur"]))
                self.cmd = p
                d.qpos[3:5], d.qvel[3:5] = R @ (p - sc.goal_p) - sc.anchor, R @ v
                if tt >= run["dur"] - 1e-9:
                    run["t_end"], self.cmd = self.sim_t, run["chain"][-1]
            else:
                d.qvel[3:5] = 0.0
            self._step(1)
            if int(round(self.sim_t / m.opt.timestep)) % 20 == 0:
                self._log_trace_at(self.sim_t)
        self.t = self.sim_t

    def account(self, dt):
        """Compute time on the critical path: in sim the arm keeps moving meanwhile (virtual time advances)."""
        self.advance_pipe(dt)

    def wait(self, dt):
        self.advance_pipe(dt)

    def monitor(self, chunks):
        pass

    def drain(self, settle=True):
        t_lim = self.sim_t + 20.0
        while self.current_id() is not None or any(sg["t_start"] is None for sg in self.segs):
            self.advance_pipe(0.01)
            if self.sim_t > t_lim:
                raise Stop("timeout (20 s) waiting for the queued plans to finish (sim)")
        self.t_rest = max([sg["t_end"] for sg in self.segs if sg["t_end"] is not None], default=self.sim_t)   # the arm stopped here
        self.data.qvel[3:5] = 0.0
        if settle:
            self.advance_pipe(WAIT_AFTER_S)

    def rest_read(self):
        """The T pose at rest, as the camera loop gets it: frames whose IMAGE time is in [t_rest + settle, t_rest + settle + read]
        (they arrive CAM_LATENCY_S later, so the plant is advanced until then), median. Returns (T pose, pen in the base frame)."""
        sc = self.sc
        settle = ADAPT_MIN_S + 0.05 if self.a.adaptive_settle else WAIT_AFTER_S       # --adaptive-settle: window right after the stop
        ts = getattr(self, "t_rest", self.sim_t) + settle
        te = ts + WINDOW_S
        if self.sim_t < te + CAM_LATENCY_S:
            self.advance_pipe(te + CAM_LATENCY_S - self.sim_t)
        fr = []
        for tc in np.linspace(ts, te, 7):
            q = self._T_full(tc)
            pb = sc.sim_to_base(q[:2]) + self.rng.normal(0, 1, 2) * CAM_SIGMA[:2]
            fr.append([*pb, q[2] + sc.goal_th + self.rng.normal(0, CAM_SIGMA[2])])
        fr = np.array(fr)
        th = math.atan2(np.sin(fr[:, 2]).mean(), np.cos(fr[:, 2]).mean())
        th_med = th + float(np.median([wrap(x - th) for x in fr[:, 2]]))
        pm = np.median(fr[:, :2], axis=0)
        self.rd = dict(settle=settle + CAM_LATENCY_S, read=WINDOW_S, n=len(fr), low=0)
        return np.array([*self.fr.base_to_sim(pm), wrap(th_med - sc.goal_th)]), sc.pusher_base(self.data.qpos).copy()

    def _T_full(self, t):
        if getattr(self, "_hcache", None) is None or len(self._hcache[0]) != len(self.hist):
            self._hcache = (np.array([h[0] for h in self.hist]), np.array([h[1] for h in self.hist]))
        ts, qs = self._hcache
        return qs[max(int(np.searchsorted(ts, t, side="right")) - 1, 0)]

    def camera_read(self, t_content, n=5, span=0.08):
        """Camera estimate of the T pose at time t_content: n frames whose content times span `span` around it (each frame
        shows the T CAM_LATENCY_S before it is received), noise as in state(), median. Goal frame."""
        sc = self.sc
        fr = []
        for tc in np.linspace(t_content - span / 2, t_content + span / 2, n):
            q = self._T_full(max(tc, 0.0))
            pb = sc.sim_to_base(q[:2]) + self.rng.normal(0, 1, 2) * CAM_SIGMA[:2]
            fr.append([*pb, q[2] + sc.goal_th + self.rng.normal(0, CAM_SIGMA[2])])
        fr = np.array(fr)
        th = math.atan2(np.sin(fr[:, 2]).mean(), np.cos(fr[:, 2]).mean())
        th_med = th + float(np.median([wrap(x - th) for x in fr[:, 2]]))
        pm = np.median(fr[:, :2], axis=0)
        return np.array([*self.fr.base_to_sim(pm), wrap(th_med - sc.goal_th)])

    def camera_newest(self):
        """--cam-correct: the newest camera estimate of the T pose (goal frame) and its image time (run clock)."""
        t_c = self.sim_t - CAM_LATENCY_S - 0.02
        return self.camera_read(t_c, n=3, span=0.04), t_c

    def camera_latest(self):
        """What the camera loop can know NOW: frames of the last 0.3 s, i.e. the T as it was ~0.33 s ago."""
        return self.camera_read(self.sim_t - CAM_LATENCY_S - 0.15, n=10, span=0.3)

    def check_link(self, start=True):
        pass

    def close(self):
        pass


class Feed:
    """Background LCM subscriber: latest + recorded T_SHAPE_POSE / XY_STATE / ROBOT_STATE with receive times."""

    def __init__(self, ol):
        import lcm
        _, object_pose_t, robot_state_t, _, xy_state_t = ol.lcm_types()
        import frames as F
        self.F, self.ol = F, ol
        self.lc = lcm.LCM()
        self.lock = threading.Lock()
        self.T, self.X, self.last_T, self.last_X, self.flange = [], [], None, None, None
        self.all_T = []                       # (receive time, num_tags) of EVERY T_SHAPE_POSE frame, also the unusable ones
        self.last_raw = None                  # receive time of the last T_SHAPE_POSE message of ANY kind (silence check: the camera stream, not the gates)
        self.max_reproj, self.n_drop_reproj = 0.0, 0   # --max-reproj-px (0 = off); frames dropped by it
        self.min_tags, self.n_drop_tags = 2, 0   # --min-tags (set by the backend from the arguments); frames dropped by it
        self.run = True

        def on_T(_c, d):
            m = object_pose_t.decode(d)
            with self.lock:
                self.last_raw = time.time()
                self.all_T.append((self.last_raw, m.num_tags))
            if m.num_tags < 2:
                return
            if m.num_tags < self.min_tags:
                with self.lock:
                    self.n_drop_tags += 1
                return
            if self.max_reproj > 0 and m.reprojection_error_px > self.max_reproj:
                with self.lock:
                    self.n_drop_reproj += 1
                return
            F.check_base_frame(m)
            th = F.theta_from_R(F.quat_xyzw_to_R(m.quaternion_xyzw))
            with self.lock:
                self.last_T = time.time()
                self.T.append((self.last_T, m.position_m[0], m.position_m[1], th, m.timestamp_us * 1e-6, m.num_tags, m.reprojection_error_px))

        def on_X(_c, d):
            s = xy_state_t.decode(d)
            with self.lock:
                self.last_X = time.time()
                self.X.append((self.last_X, s.x, s.y, s.cmd_x, s.cmd_y, s.plan_id, s.tracking))

        def on_R(_c, d):
            s = robot_state_t.decode(d)
            with self.lock:
                self.flange = (time.time(), s.O_T_EE[12], s.O_T_EE[13])

        self.lc.subscribe("T_SHAPE_POSE", on_T)
        self.lc.subscribe("XY_STATE", on_X)
        self.lc.subscribe("ROBOT_STATE", on_R)
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while self.run:
            self.lc.handle_timeout(50)

    def start_logging(self, out_dir):
        """Raw logs, appended every 0.5 s: xy_state_raw.csv (every XY_STATE: receive time [epoch s], x, y, cmd, plan_id, tracking) and
        t_pose_raw.csv (every usable T_SHAPE_POSE: receive time, image stamp, base x, y, theta)."""
        os.makedirs(out_dir, exist_ok=True)
        self._fx = open(os.path.join(out_dir, "xy_state_raw.csv"), "w")
        self._fx.write("recv_epoch_s,x,y,cmd_x,cmd_y,plan_id,tracking\n")
        self._ft = open(os.path.join(out_dir, "t_pose_raw.csv"), "w")
        self._ft.write("recv_epoch_s,stamp_epoch_s,x_base,y_base,theta_base_rad,num_tags,reproj_px\n")
        self._ix = self._it = 0
        self._lth = threading.Thread(target=self._log_loop, daemon=True)
        self._lth.start()

    def _flush_logs(self):
        with self.lock:
            nx, nt = self.X[self._ix:], self.T[self._it:]
            self._ix, self._it = len(self.X), len(self.T)
        self._fx.write("".join(f"{r[0]:.6f},{r[1]:.5f},{r[2]:.5f},{r[3]:.5f},{r[4]:.5f},{r[5]},{r[6]}\n" for r in nx))
        self._ft.write("".join(f"{r[0]:.6f},{r[4]:.6f},{r[1]:.5f},{r[2]:.5f},{r[3]:.5f},{r[5]},{r[6]:.3f}\n" for r in nt))
        self._fx.flush()
        self._ft.flush()

    def _log_loop(self):
        while self.run:
            time.sleep(0.5)
            self._flush_logs()

    def stop(self):
        self.run = False
        self.th.join(1.0)
        if getattr(self, "_fx", None):
            self._lth.join(1.0)
            self._flush_logs()
            self._fx.close()
            self._ft.close()

    def T_mean(self, window):
        with self.lock:
            rows = [r for r in self.T if r[0] > time.time() - window]
        if len(rows) < 2:
            return None
        r = np.array(rows)
        return r[:, 1:3].mean(0), math.atan2(np.sin(r[:, 3]).mean(), np.cos(r[:, 3]).mean()), len(rows)

    def tag_stats(self, window):
        """Frames of the last `window` s by number of detected tags, e.g. {5: 12, 1: 3} (1 tag = unusable)."""
        with self.lock:
            rows = [n for t, n in self.all_T if t > time.time() - window]
        return {k: rows.count(k) for k in sorted(set(rows))}

    def pen(self):
        with self.lock:
            if self.X and time.time() - self.last_X < 1.0:
                return np.array(self.X[-1][1:3])
            if self.flange and time.time() - self.flange[0] < 1.0:
                return np.array(self.flange[1:3])
        return None


class RealBackend:
    name = "real"

    def __init__(self, a, scene, fr):
        import open_loop_test as ol
        self.ol, self.a, self.sc, self.fr = ol, a, scene, fr
        self.state_nuc = {}
        self.feed = None
        self.t0 = None
        self.trace = []
        self.max_lead = 0.0
        self.carrier_base = None
        self.checks = []
        self.link_th, self.link_err, self.t_stop = None, None, None
        self.t_ready = 0.0                       # earliest time the next T pose window may start
        self.segs, self._xi, self.run_sid, self.pub = [], 0, None, None     # --pipeline: sent plans and the XY_STATE bookkeeping
        self.last_cam = {}
        self.out_dir = None
        self._cmdlog = None
        ol.XY_DRIVER_BIN = a.xy_driver_bin
        ol.XY_DRIVER_EXTRA = (f" --max-lead-mm={a.max_lead_mm:g}" if a.max_lead_mm > 0 else "") + \
            f" --upright --yaw-deg={a.yaw_deg}" + (" --blend" if a.blend else "")
        ol.MAX_LEAD_MM = a.max_lead_mm

    # -- run clock
    def now(self):
        return time.time() - self.t0

    def advance(self, dt):
        pass

    # -- bring-up (open_loop_test.real_run, with the safe-height fly-over to the pusher start)
    def pusher_start_base(self):
        return np.array(self.a.pusher_start_base, float)   # resolved in main(): --pusher-start-base or the pusher_start.yaml base

    def preflight(self):
        ol = self.ol
        print("== pre-flight ==")
        rc, out = ol.ssh("hostname", 15)
        ol.report(rc == 0, f"ssh {ol.NUC}: {out.splitlines()[-1] if out else rc}")
        n, dt = ol.listen(3.0)
        ol.report(n["T_SHAPE_POSE"] / dt > 5, f"T_SHAPE_POSE {n['T_SHAPE_POSE'] / dt:.1f} Hz")
        ol.report(n["ROBOT_STATE"] / dt >= 35, f"ROBOT_STATE {n['ROBOT_STATE'] / dt:.1f} Hz")
        ol.report(n["XY_CMD"] == 0, f"no XY_CMD traffic on the bus ({n['XY_CMD']})")
        if rc == 0:
            rc, out = ol.ssh("ps -eo pid,comm,args | awk '$2 ~ /^python/ && /send_xy_cmd/ {print}'; pgrep -x -a xy_driver; true")
            ol.report(not out.strip(), "no XY_CMD sender / xy_driver left over on the NUC" + (f": {out}" if out.strip() else ""), warn=True)
            ol.check_carrier()
            rc, out = ol.ssh(f"cd {ol.NUC_REPO} && test -x {ol.XY_DRIVER_BIN} && grep -c -a -e '--upright' {ol.XY_DRIVER_BIN}")
            ol.report(rc == 0, f"xy_driver binary {ol.XY_DRIVER_BIN} exists and knows --upright")
            if self.a.blend:
                rc, out = ol.ssh(f"cd {ol.NUC_REPO} && grep -c -a -e '--blend' {ol.XY_DRIVER_BIN}")
                ol.report(rc == 0, f"xy_driver binary knows --blend (velocity-continuous plan hand-over)")
            if self.a.pipeline:
                rc, out = ol.ssh("p=$(pgrep -x aprilcube_pose | head -n 1); if [ -n \"$p\" ]; then echo \"PID $p\"; ps -o args= -p $p; "
                                 "f=$(readlink -f /proc/$p/exe); echo \"BIN $f\"; grep -c -a -e '--interp''-flange' \"$f\"; else echo NONE; fi")
                lines = out.splitlines()
                import re
                if lines and lines[0] == "NONE":
                    print("  [INFO] no aprilcube_pose is running on the NUC right now")
                else:
                    args_line = lines[1] if len(lines) > 1 else ""
                    uses = "--interp-flange" in args_line
                    print(f"  [INFO] the running aprilcube_pose ({lines[2] if len(lines) > 2 else '?'}) {'USES' if uses else 'does NOT use'} --interp-flange"
                          + ("" if uses or not self.a.dry_run else " (dry-run does not restart it; a real run restarts it as below)"))
                    mm = re.search(r"--cam-delay-ms[ =]([0-9.]+)", args_line)
                    if mm and not self.a.cam_delay_explicit:
                        self.a.cam_exposure_delay_s = float(mm.group(1)) / 1000.0
                        print(f"  [INFO] camera stamp convention taken from the running process: exposure = timestamp - {mm.group(1)} ms")
                if self.a.pose_bin:                       # the binary a real run will start
                    rc, out = ol.ssh(f"cd {ol.NUC_REPO} && test -x {self.a.pose_bin} && grep -c -a -e '--interp''-flange' {self.a.pose_bin}")
                    ol.report(rc == 0 and out.strip().isdigit() and int(out.strip()) > 0,
                              f"{self.a.pose_bin} exists and knows --interp-flange (a real run starts it with --interp-flange --cam-delay-ms "
                              f"{self.a.cam_delay_ms:g}; exposure = stamp - {self.a.cam_exposure_delay_s * 1000:.0f} ms)")
        return ol.ok_all

    def log_cmd(self, plan_id, sid, dur, wps):
        """xy_cmd_log.csv: every XY_CMD this script publishes (send time as epoch s and run clock)."""
        if self.out_dir is None:
            return
        if self._cmdlog is None:
            os.makedirs(self.out_dir, exist_ok=True)
            self._cmdlog = open(os.path.join(self.out_dir, "xy_cmd_log.csv"), "w")
            self._cmdlog.write("send_epoch_s,run_s,plan_id,id,dur_s,n_points,waypoints_xy\n")
        pts = ";".join(f"{q[0]:.5f} {q[1]:.5f}" for q in wps)
        self._cmdlog.write(f"{time.time():.6f},{self.now():.4f},{plan_id},{sid},{dur:.4f},{len(wps)},{pts}\n")
        self._cmdlog.flush()

    def start_dry(self):
        self.feed = Feed(self.ol)
        self.feed.min_tags, self.feed.max_reproj = self.a.min_tags, self.a.max_reproj_px
        if self.out_dir:
            self.feed.start_logging(self.out_dir)
        time.sleep(1.5)
        self.t0 = time.time()

    def bring_up(self):
        ol = self.ol
        st, lf = self.state_nuc, None
        print("== NUC bring-up ==")
        if not ol.check_carrier("before start: "):
            raise Stop("link flapped since the baseline")
        if not ol.nuc_stop_state_and_pose():
            raise Stop("state_pub / aprilcube_pose still running on the NUC")
        rc, out = ol.ssh(f"cd {ol.NUC_REPO} && ./script/check_f_t_ee.sh; echo exit=$?", 120)
        last = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("exit=")]
        if "exit=0" not in out or not last or last[-1].strip() != "PASS":
            raise Stop(f"F_T_EE guard failed:\n{out[-600:]}")
        print("  F_T_EE guard: exit 0, last line PASS")
        st["log"] = lf = f"log/xy_driver_closedloop_{time.strftime('%Y%m%d_%H%M%S')}.log"
        ol.ssh(f"rm -f {ol.FIFO} && mkfifo {ol.FIFO}")
        ol.ssh(f"(setsid sleep 100000 > {ol.FIFO} 2>/dev/null < /dev/null &); true", 10)
        st["driver"] = True
        if not ol.nuc_bg(f"cd {ol.NUC_REPO} && exec {ol.XY_DRIVER_BIN} {ol.ROBOT_IP} --work-z={ol.WORK_Z}{ol.XY_DRIVER_EXTRA} < {ol.FIFO}",
                         f"{ol.NUC_REPO}/{lf}"):
            raise Stop("could not start xy_driver")
        print(f"  xy_driver started ({lf}); homing ...")
        if not ol.wait_log(lf, "Homed; ready for plans at z=0.4434.", 240):
            raise Stop("no 'Homed' in the xy_driver log:\n" + ol.nuc_log_tail(lf, 25))
        start = self.pusher_start_base()
        ol.fly_to(start)
        # camera + feed above the start point (xy_driver publishes ROBOT_STATE, which the camera needs for the base frame)
        if not ol.nuc_start_pose(interp=self.a.interp_flange, cam_delay_ms=self.a.cam_delay_ms, binary=self.a.pose_bin):
            raise Stop("could not start aprilcube_pose")
        for _ in range(8):
            n, dt = ol.listen(4.0, ("T_SHAPE_POSE", "XY_STATE", "ROBOT_STATE", "XY_CMD"))
            if n["XY_STATE"] / dt >= 10 and n["T_SHAPE_POSE"] / dt >= 10:
                break
        self.chk("XY_STATE / T_SHAPE_POSE alive", n["XY_STATE"] / dt >= 10 and n["T_SHAPE_POSE"] / dt >= 10,
                 f"XY_STATE {n['XY_STATE'] / dt:.0f} Hz, T_SHAPE_POSE {n['T_SHAPE_POSE'] / dt:.1f} Hz (need >= 10 each)")
        self.feed = Feed(ol)
        self.feed.min_tags, self.feed.max_reproj = self.a.min_tags, self.a.max_reproj_px
        if self.out_dir:
            self.feed.start_logging(self.out_dir)
        time.sleep(1.5)
        # --- checks above the start point (z=0.4434)
        pen = self.feed.pen()
        e_pen = float(np.linalg.norm(pen - start)) * 1000 if pen is not None else float("inf")
        self.chk("above start: pen at the start point", e_pen <= 2.0, f"{e_pen:.2f} mm (limit 2 mm)")
        Tm = self.feed.T_mean(0.5)
        self.chk("above start: T pose available", Tm is not None, "no usable T_SHAPE_POSE frames" if Tm is None else f"{Tm[2]} frames in 0.5 s")
        d, da, (p, th), srec = ol.start_pose_deviation((Tm[0], 0.0, Tm[1]))
        self.chk("above start: T at start_2", d <= 3.0 and da <= 3.0,
                 f"T (goal frame) ({p[0]:.4f}, {p[1]:.4f}, {math.degrees(th):.2f} deg): {d:.1f} mm, {da:.2f} deg off start_2 (limit 3 mm / 3 deg)")
        gap = describe_start(self.a.pusher_start_base, self.fr, np.array([p[0], p[1], th]), pen, self.a.start_label)
        from global_mppi.closed_loop.geometry import dist_to_poly, t_polygon
        dist = dist_to_poly(self.fr.base_to_sim(pen), t_polygon(np.array([p[0], p[1], th]))) * 1000
        self.chk("above start: pen to T outline", dist >= 18.0, f"{dist:.1f} mm (need >= 18 mm); {gap}")
        if self.a.confirm:
            self.wait_signal(self.a.go_file + ".down" if self.a.go_file else None,
                             "ABOVE THE START POINT: check the gap between tool and T, then create", "press Enter to go down")
        print("  all checks above the start point passed: sending 'down' ...", flush=True)
        st["down"] = True
        ol.ssh(f"echo down > {ol.FIFO}")
        if not ol.wait_log(lf, "At z=0.34874; ready for plans", 90):
            raise Stop("no 'At z=0.34874; ready for plans':\n" + ol.nuc_log_tail(lf, 15))
        # --- checks after down
        z = ol.read_flange_z()
        self.chk("after down: flange z", z is not None and abs(z - 0.3486) <= 0.0005,
                 f"z = {z if z is None else round(z, 5)} (need 0.3486 +- 0.0005)")
        t_a = time.time()
        time.sleep(1.0)
        with self.feed.lock:
            nt = [n for t, n in self.feed.all_T if t >= t_a]
        rate, min_tags = len(nt), (min(nt) if nt else 0)
        self.chk("after down: camera", rate >= self.a.min_cam_hz and min_tags >= 3,
                 f"{rate} frames/s (need >= {self.a.min_cam_hz:g}), tags per frame min {min_tags} (need >= 3), median {np.median(nt) if nt else 0:g}")
        cur, base, ts_, _ = ol.carrier_status()
        self.carrier_base = base if base is not None else cur
        self.chk("after down: carrier_changes", cur is not None and cur == self.carrier_base, f"{cur} vs baseline {self.carrier_base}")
        n, dt = ol.listen(1.0, ("XY_CMD",))
        self.chk("after down: no other XY_CMD on the bus", n["XY_CMD"] == 0, f"{n['XY_CMD']} XY_CMD messages in 1 s")
        Tm = self.feed.T_mean(0.5)
        if Tm is not None:
            d, da, (p, th), _ = ol.start_pose_deviation((Tm[0], 0.0, Tm[1]))
            self.chk("after down: T at start_2", d <= 3.0 and da <= 3.0, f"{d:.1f} mm, {da:.2f} deg (limit 3 mm / 3 deg)")
        else:
            self.chk("after down: T at start_2", False, "no T pose")
        print("  ALL BRING-UP CHECKS PASSED: starting the closed loop"
              + (" after your confirmation" if self.a.confirm else " automatically"), flush=True)
        if self.a.confirm:
            self.wait_signal(self.a.go_file, "READY. Hand on the enable button. Create", "hand on the enable button, press Enter to START")
        self.t0 = time.time()

    def chk(self, name, ok, detail):
        """Print one check result; a failed check aborts (Stop -> lift, stop xy_driver, no retry)."""
        self.checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        if not ok:
            raise Stop(f"check failed: {name}: {detail}")

    def wait_signal(self, path, msg, prompt):
        if path is None:
            input(f"  {prompt} ")
            return
        if os.path.exists(path):
            os.remove(path)
        print(f"\n  {msg} the file {path}", flush=True)
        t0 = time.time()
        while not os.path.exists(path):
            if time.time() - t0 > self.a.go_timeout:
                raise Stop("no confirmation within the timeout")
            time.sleep(0.2)
        os.remove(path)

    # ---- --pipeline (real). Dry-run: nothing is published; the plans get a virtual schedule on the wall clock.
    def _frames_pose(self, rows):
        r = np.array([x[:4] for x in rows])
        th = math.atan2(np.sin(r[:, 3]).mean(), np.cos(r[:, 3]).mean())
        th_med = th + float(np.median([wrap(x - th) for x in r[:, 3]]))
        p = self.fr.base_to_sim(np.median(r[:, 1:3], axis=0))
        return np.array([p[0], p[1], wrap(th_med - self.sc.goal_th)])

    def _content_time(self, row):
        """Image time of a frame in the run clock. T_SHAPE_POSE.timestamp_us is the IMAGE timestamp (NUC clock, NTP-synced with this
        machine; it is ~0.18 s older than the receive time, measured). --cam-exposure-delay-s: exposure-to-stamp delay on top."""
        return row[4] - self.t0 - self.a.cam_exposure_delay_s

    def account(self, dt):
        pass                                     # real time passes by itself

    def wait(self, dt):
        time.sleep(dt)

    def camera_newest(self):
        """--cam-correct: median of the 3 newest T frames (goal frame) and their mean image time (run clock); None without frames."""
        with self.feed.lock:
            rows = list(self.feed.T[-3:])
        if len(rows) < 3 or time.time() - rows[-1][0] > 0.3:
            return None
        return self._frames_pose(rows), float(np.mean([self._content_time(r) for r in rows]))

    def camera_latest(self):
        """Frames received in the last 0.3 s (stop conditions). < 3 frames: pose lost."""
        with self.feed.lock:
            rows = [r for r in self.feed.T if r[0] > time.time() - 0.3]
        if len(rows) < MIN_FRAMES_LOW:
            raise Stop(f"T pose lost: {len(rows)} frames in the last 0.3 s; by tags: {self.feed.tag_stats(0.8)}")
        return self._frames_pose(rows)

    def camera_read(self, t_content, n=5, span=0.08):
        """Post-hoc: the frames whose image time is within +-span/2 of t_content (nearest 3 if none), median."""
        with self.feed.lock:
            allr = list(self.feed.T)
        near = [r for r in allr if abs(self._content_time(r) - t_content) <= span / 2]
        if len(near) < MIN_FRAMES_LOW:
            near = sorted(allr, key=lambda r: abs(self._content_time(r) - t_content))[:MIN_FRAMES_LOW]
        return self._frames_pose(near)

    def _update(self):
        """Segment start / end times from the XY_STATE plan_id switches (real) or from the virtual schedule (dry-run)."""
        if self.a.dry_run:
            now = self.now()
            for sg in self.segs:
                if sg["t_start"] <= now < sg["t_end"]:
                    self.run_sid = sg["id"]
                    return
            self.run_sid = None
            return
        with self.feed.lock:
            new = self.feed.X[self._xi:]
            self._xi = len(self.feed.X)
        by_pid = {sg["plan_id"]: sg for sg in self.segs}
        for x in new:
            t, pid, tracking = x[0] - self.t0, x[5], x[6]
            sg = by_pid.get(pid)
            if tracking and sg is not None:
                if sg["t_start"] is None:
                    sg["t_start"] = t
                if self.run_sid is not None and self.run_sid != sg["id"]:        # plan_id switched: the previous plan ended
                    prev = self.segs[self.run_sid - 1]
                    if prev["t_end"] is None:
                        prev["t_end"] = t
                self.run_sid = sg["id"]
            elif not tracking and self.run_sid is not None:
                prev = self.segs[self.run_sid - 1]
                if prev["t_end"] is None:
                    prev["t_end"] = t
                self.run_sid = None

    def current_id(self):
        self._update()
        return self.run_sid

    def started(self, sid):
        self._update()
        sg = self.segs[sid - 1]
        return sg["t_start"] is not None and (not self.a.dry_run or sg["t_start"] <= self.now())

    def send_pipe(self, wps, dur):
        self._update()
        if not self.a.blend and any(sg["t_start"] is None or sg["t_start"] > self.now() for sg in self.segs):
            raise Stop("a third plan was about to be sent while one is already queued")
        sid, now = len(self.segs) + 1, self.now()
        sg = dict(id=sid, plan_id=int(time.time() * 1000) % (2 ** 31 - 1), wps=np.array(wps), dur=float(dur), t_send=now, t_start=None, t_end=None)
        if self.a.dry_run:                       # virtual schedule: starts when the previous one ends (or now)
            if self.a.blend:                     # starts at once, the running one is interrupted
                for x in self.segs:
                    x["t_end"] = min(x["t_end"], now)
                sg["t_start"] = now
            else:
                prev_end = max([x["t_end"] for x in self.segs], default=0.0)
                sg["t_start"] = max(prev_end, now)
            sg["t_end"] = sg["t_start"] + sg["dur"]
        else:
            _, _, _, xy_path_cmd_t, _ = self.ol.lcm_types()
            if self.pub is None:
                import lcm
                self.pub = lcm.LCM()
            cmd = xy_path_cmd_t()
            cmd.timestamp_us, cmd.plan_id, cmd.num_points = int(time.time() * 1e6), sg["plan_id"], len(wps)
            cmd.points_xy = [list(map(float, q)) for q in wps]
            cmd.duration_sec = dur
            self.pub.publish("XY_CMD", cmd.encode())
        if not self.a.dry_run:                   # dry-run publishes nothing, so it logs no command either
            self.log_cmd(sg["plan_id"], sid, dur, wps)
        self.segs.append(sg)
        return sid

    def monitor(self, chunks):
        """Soft stops while plans run: XY_STATE / T_SHAPE_POSE silent, lead too large, a sent plan that never starts (rejected)."""
        self._update()
        f, now = self.feed, time.time()
        with f.lock:
            self._log(f)
            gx, gt, X = f.last_X, f.last_raw, (f.X[-1] if f.X else None)
        if self.a.dry_run:
            return
        if gx is not None and now - gx > SOFT_XY_GAP_S:
            raise Stop(f"SOFT STOP: XY_STATE silent for {now - gx:.2f} s")
        if gt is not None and now - gt > SOFT_T_GAP_S:
            raise Stop(f"SOFT STOP: T_SHAPE_POSE silent for {now - gt:.1f} s")
        if X is not None and self.a.max_lead_mm > 0:
            lead = math.hypot(X[3] - X[1], X[4] - X[2]) * 1000
            if lead > self.a.max_lead_mm:
                raise Stop(f"SOFT STOP: commanded - measured {lead:.1f} mm > {self.a.max_lead_mm:g} mm")
        pend = [sg for sg in self.segs if sg["t_start"] is None]
        if pend and self.run_sid is None and self.now() - max([sg["t_send"] for sg in pend] + [x["t_end"] or 0.0 for x in self.segs]) > REJECT_WAIT_S:
            raise Stop("plan rejected: a sent plan did not start within %.0f s" % REJECT_WAIT_S)

    def _rest_read_adaptive(self):
        """--adaptive-settle at a measurement point: the window starts at the first of 3 consecutive frames (image time >= stop + 0.1 s)
        that differ by < 0.5 mm and < 0.3 deg (no such run by stop + 0.6 s: start there), then the frames from there on are collected
        until MIN_FRAMES and --read-s of image time. rd: settle = stop -> window found (wall), read = found -> pose available."""
        t_rest = getattr(self, "t_rest", None)
        t_rest = t_rest if t_rest is not None else self.now() - 10.0
        t_lo, t_hi = t_rest + ADAPT_MIN_S, t_rest + ADAPT_MAX_S
        f = self.feed
        t_deadline = self.now() + READ_MAX_S + ADAPT_MAX_S
        start = None                                              # image stamp (row[4]) of the first frame of the window
        while start is None:
            self.monitor([])
            with f.lock:
                rows = [r for r in f.T if self._content_time(r) >= t_lo]
            for i in range(len(rows) - 2):
                w = rows[i:i + 3]
                if all(math.hypot(w[j + 1][1] - w[j][1], w[j + 1][2] - w[j][2]) < ADAPT_DPOS_M
                       and abs(wrap(w[j + 1][3] - w[j][3])) < ADAPT_DANG_RAD for j in range(2)):
                    start = w[0][4]
                    break
            if start is None and rows and self._content_time(rows[-1]) >= t_hi:
                start = next(r[4] for r in rows if self._content_time(r) >= t_hi)
            if start is None and self.now() > t_deadline:
                raise Stop(f"T pose lost: no usable frames after the stop (by tags: {f.tag_stats(READ_MAX_S + 0.5)})")
            if start is None:
                time.sleep(0.01)
        t_found = self.now()
        while True:
            self.monitor([])
            with f.lock:
                rows = [r for r in f.T if r[4] >= start]
                gap = time.time() - f.last_raw if f.last_raw else 1e9
            span = (self._content_time(rows[-1]) - self._content_time(rows[0])) if len(rows) > 1 else 0.0
            if (len(rows) >= MIN_FRAMES and span >= WINDOW_S - 0.02) or self.now() > t_found + READ_MAX_S:
                break
            time.sleep(0.01)
        low = int(len(rows) < MIN_FRAMES)
        self.rd = dict(settle=t_found - t_rest, read=self.now() - t_found, n=len(rows), low=low)
        if len(rows) < MIN_FRAMES_LOW:
            raise Stop(f"T pose lost: {len(rows)} frames after the stop; by tags: {f.tag_stats(READ_MAX_S + 0.5)}")
        if low:
            print(f"      low_frames: only {len(rows)} frames in the rest window (used anyway)", flush=True)
        if gap > SOFT_T_GAP_S:
            raise Stop(f"T_SHAPE_POSE silent for {gap:.1f} s")
        pen = f.pen()
        if pen is None:
            raise Stop("no pen position (XY_STATE / ROBOT_STATE)")
        return self._frames_pose(rows), pen

    def _rest_read_fast(self):
        """--fast-stop at a measurement point: no settle wait. Take the first 3 consecutive frames whose IMAGE time (stamp - exposure delay)
        is >= stop - --fast-pre-s that differ by < 0.5 mm and < 0.3 deg (T has effectively stopped) and use their median. No such triple by
        stop + ADAPT_MAX_S: the newest 3 frames. rd: settle = stop -> pose available, read = 0."""
        t_rest = getattr(self, "t_rest", None)
        t_rest = t_rest if t_rest is not None else self.now() - 10.0
        t_lo, t_hi = t_rest - self.a.fast_pre_s, t_rest + ADAPT_MAX_S
        f = self.feed
        t_deadline = self.now() + READ_MAX_S + ADAPT_MAX_S
        rows, stable = None, True
        while rows is None:
            self.monitor([])
            with f.lock:
                cand = [r for r in f.T if self._content_time(r) >= t_lo]
            for i in range(len(cand) - 2):
                w = cand[i:i + 3]
                if all(math.hypot(w[j + 1][1] - w[j][1], w[j + 1][2] - w[j][2]) < ADAPT_DPOS_M
                       and abs(wrap(w[j + 1][3] - w[j][3])) < ADAPT_DANG_RAD for j in range(2)):
                    rows = w
                    break
            if rows is None and len(cand) >= 3 and self._content_time(cand[-1]) >= t_hi:
                rows, stable = cand[-3:], False
            if rows is None and self.now() > t_deadline:
                raise Stop(f"T pose lost: no usable frames after the stop (by tags: {f.tag_stats(READ_MAX_S + 0.5)})")
            if rows is None:
                time.sleep(0.002)
        with f.lock:
            gap = time.time() - f.last_raw if f.last_raw else 1e9
        t_got = self.now()
        self.rd = dict(settle=t_got - t_rest, read=0.0, n=len(rows), low=0)
        print(f"      fast-stop: stop->pose {t_got - t_rest:.3f} s, {len(rows)} frames, image time rel. stop "
              f"{[round(self._content_time(r) - t_rest, 3) for r in rows]}, stamps {[round(r[4], 3) for r in rows]}"
              f"{'' if stable else ' (NOT stable: newest 3 used)'}", flush=True)
        if gap > SOFT_T_GAP_S:
            raise Stop(f"T_SHAPE_POSE silent for {gap:.1f} s")
        pen = f.pen()
        if pen is None:
            raise Stop("no pen position (XY_STATE / ROBOT_STATE)")
        return self._frames_pose(rows), pen

    def rest_read(self):
        if self.a.fast_stop:
            return self._rest_read_fast()
        if self.a.adaptive_settle:
            return self._rest_read_adaptive()
        return self._rest_read_fixed()

    def _rest_read_fixed(self):
        """The T pose at rest: frames whose IMAGE time (stamp - exposure delay) is in [t_rest + settle-s, t_rest + settle-s + read-s];
        they arrive ~0.18 s later, so wait for a frame at or past the window end (timeout READ_MAX_S). < 3 frames: pose lost, < 5: low_frames."""
        t_rest = getattr(self, "t_rest", None)
        ts = (t_rest if t_rest is not None else self.now() - 10.0) + WAIT_AFTER_S
        te = ts + WINDOW_S
        t_enter = self.now()
        f = self.feed
        while True:
            with f.lock:
                rows = [r for r in f.T if ts <= self._content_time(r) <= te]
                newest = self._content_time(f.T[-1]) if f.T else -1e9
                gap = time.time() - f.last_raw if f.last_raw else 1e9
            now = self.now()
            if newest >= te and len(rows) >= MIN_FRAMES:
                break
            if now > max(te, t_enter) + CAM_LATENCY_S + READ_MAX_S:
                break
            self.monitor([])
            time.sleep(0.01)
        low = int(len(rows) < MIN_FRAMES)
        self.rd = dict(settle=WAIT_AFTER_S, read=self.now() - t_enter, n=len(rows), low=low)
        if len(rows) < MIN_FRAMES_LOW:
            raise Stop(f"T pose lost: {len(rows)} frames with image time in the {WINDOW_S} s rest window; by tags: {f.tag_stats(READ_MAX_S + 0.5)}")
        if low:
            print(f"      low_frames: only {len(rows)} frames in the rest window (used anyway)", flush=True)
        if gap > SOFT_T_GAP_S:
            raise Stop(f"T_SHAPE_POSE silent for {gap:.1f} s")
        pen = f.pen()
        if pen is None:
            raise Stop("no pen position (XY_STATE / ROBOT_STATE)")
        return self._frames_pose(rows), pen

    def drain(self, settle=True):
        t0 = time.time()
        while True:
            self.monitor([])
            if self.run_sid is None and not any(sg["t_start"] is None or sg["t_start"] > self.now() for sg in self.segs):
                break
            if time.time() - t0 > DRAIN_TIMEOUT_S:
                raise Stop(f"timeout ({DRAIN_TIMEOUT_S:g} s) waiting for the arm to stop (running id {self.run_sid}, "
                           f"not started: {[sg['id'] for sg in self.segs if sg['t_start'] is None]})")
            time.sleep(0.02)
        for sg in self.segs:                      # a plan that is still open when the loop stops
            if sg["t_end"] is None:
                sg["t_end"] = self.now()
        self.t_rest = max([sg["t_end"] for sg in self.segs], default=self.now())      # the arm stopped here (run clock)
        if settle:
            time.sleep(WAIT_AFTER_S)

    def check_link(self, start=True):
        """carrier_changes of the NUC robot link must not exceed the baseline (a flap during the run -> stop).
        --bg-link-check: after the baseline, the ssh read runs in a background thread (started when start=True); a flap found
        there is raised by the next call."""
        if self.carrier_base is not None and self.a.bg_link_check:
            if self.link_err:
                raise Stop(self.link_err)
            if start and (self.link_th is None or not self.link_th.is_alive()):
                self.link_th = threading.Thread(target=self._link_bg, daemon=True)
                self.link_th.start()
            return
        self._check_link_now()

    def _link_bg(self):
        try:
            self._check_link_now()
        except Stop as e:
            self.link_err = str(e)

    def _check_link_now(self):
        if self.carrier_base is None:         # first call: the full status (also reads the baseline), ~1 s
            cur, base, ts, _ = self.ol.carrier_status()
            self.carrier_base = base if base is not None else cur
        else:                                  # later calls: one ssh `cat carrier_changes` (~0.4 s) against the cached baseline
            rc, out = self.ol.ssh(f"cat /sys/class/net/{self.ol.NUC_IFACE}/carrier_changes")
            cur = int(out.strip()) if rc == 0 and out.strip().isdigit() else None
        if cur is None:
            raise Stop("NUC carrier_changes unreadable")
        if cur > self.carrier_base:
            raise Stop(f"NUC {self.ol.NUC_IFACE} carrier_changes {cur} > baseline {self.carrier_base}: link flapped")

    def state(self):
        """Wait until WAIT_AFTER_S after the last tracking=0 (first round: now), take the frames received during the
        next WINDOW_S, median. Fewer than MIN_FRAMES -> pose lost."""
        f = self.feed
        t_enter = time.time()
        if self.a.adaptive_settle and self.t_stop is not None:
            t_from, stable = self._adaptive_from(self.t_stop)
            self.adapt = stable
        else:
            t_from = max(self.t_ready, t_enter)
            time.sleep(max(0.0, t_from - time.time()))
        time.sleep(max(0.0, t_from + WINDOW_S - time.time()))
        deadline = t_from + READ_MAX_S
        while True:                                 # keep collecting until MIN_FRAMES, at most READ_MAX_S after t_from
            with f.lock:
                rows = [r for r in f.T if r[0] >= t_from]
                gap = time.time() - f.last_raw if f.last_raw else 1e9
            if len(rows) >= MIN_FRAMES or time.time() >= deadline:
                break
            time.sleep(0.02)
        low = int(len(rows) < MIN_FRAMES)
        self.rd = dict(settle=max(0.0, t_from - t_enter), read=time.time() - t_from, n=len(rows), low=low)
        if len(rows) < MIN_FRAMES_LOW:
            raise Stop(f"T pose lost: {len(rows)} usable frames (< {MIN_FRAMES_LOW}) within {READ_MAX_S} s; "
                       f"all frames of the last {READ_MAX_S + 0.5:.1f} s by number of tags: {f.tag_stats(READ_MAX_S + 0.5)}")
        if low:
            print(f"      low_frames: only {len(rows)} usable frames in {self.rd['read']:.2f} s (used anyway)", flush=True)
        if gap > SOFT_T_GAP_S:
            raise Stop(f"T_SHAPE_POSE silent for {gap:.1f} s")
        r = np.array(rows)
        th = math.atan2(np.sin(r[:, 3]).mean(), np.cos(r[:, 3]).mean())
        th_med = th + float(np.median([wrap(x - th) for x in r[:, 3]]))
        pen = f.pen()
        if pen is None:
            raise Stop("no pen position (XY_STATE / ROBOT_STATE)")
        p = self.fr.base_to_sim(np.median(r[:, 1:3], axis=0))
        return np.array([p[0], p[1], wrap(th_med - self.sc.goal_th)]), pen

    def _adaptive_from(self, t_stop):
        """--adaptive-settle: the start of the T pose window = the first of 3 consecutive frames (received >= ADAPT_MIN_S after the
        stop) that differ by < 0.5 mm and < 0.3 deg; no such run by ADAPT_MAX_S -> start there. Returns (t_from, found)."""
        f = self.feed
        t_min, t_max = t_stop + ADAPT_MIN_S, t_stop + ADAPT_MAX_S
        time.sleep(max(0.0, t_min - time.time()))
        while True:
            with f.lock:
                rows = [r for r in f.T if r[0] >= t_stop]
            for i in range(len(rows) - 2):
                w = rows[i:i + 3]
                if w[2][0] < t_min:
                    continue
                if all(math.hypot(w[j + 1][1] - w[j][1], w[j + 1][2] - w[j][2]) < ADAPT_DPOS_M
                       and abs(wrap(w[j + 1][3] - w[j][3])) < ADAPT_DANG_RAD for j in range(2)):
                    return max(w[0][0], t_stop), True
            if time.time() >= t_max:
                return t_max, False
            time.sleep(0.01)

    def execute(self, wps, dur):
        ol, f = self.ol, self.feed
        _, _, _, xy_path_cmd_t, _ = ol.lcm_types()
        import lcm
        pub = lcm.LCM()
        cmd = xy_path_cmd_t()
        cmd.timestamp_us = int(time.time() * 1e6)
        cmd.plan_id = int(time.time() * 1000) % (2 ** 31 - 1)
        cmd.num_points = len(wps)
        cmd.points_xy = [list(map(float, p)) for p in wps]
        cmd.duration_sec = dur
        t_send = time.time()
        pub.publish("XY_CMD", cmd.encode())
        t_pub = time.time() - t_send
        self.log_cmd(cmd.plan_id, 0, dur, wps)
        t_started = None
        started, lead_max = False, 0.0
        while True:
            time.sleep(0.02)
            now = time.time()
            with f.lock:
                X = f.X[-1] if f.X else None
                gx, gt = f.last_X, f.last_raw
                self._log(f)
            if X is None or gx is None:
                if now - t_send > REJECT_WAIT_S:
                    raise Stop("no XY_STATE after sending the plan")
                continue
            if now - gx > SOFT_XY_GAP_S:
                raise Stop(f"SOFT STOP: XY_STATE silent for {now - gx:.2f} s")
            if gt is not None and now - gt > SOFT_T_GAP_S:
                raise Stop(f"SOFT STOP: T_SHAPE_POSE silent for {now - gt:.1f} s")
            lead = math.hypot(X[3] - X[1], X[4] - X[2]) * 1000
            lead_max = max(lead_max, lead)
            if self.a.max_lead_mm > 0 and lead > self.a.max_lead_mm:
                raise Stop(f"SOFT STOP: commanded - measured {lead:.1f} mm > {self.a.max_lead_mm:g} mm")
            if not started:
                if X[5] == cmd.plan_id and X[6]:
                    started = True
                    t_started = now
                elif now - t_send > REJECT_WAIT_S:
                    return "rejected", lead_max, dict(publish=t_pub, start_lat=now - t_send, track=0.0, total=now - t_send)
            elif not X[6]:
                break
            if now - t_send > dur + 8:
                raise Stop("timeout waiting for the plan to finish")
        self.t_ready = time.time() + WAIT_AFTER_S
        t_done = time.time()
        self.t_stop = t_done
        return "ok", lead_max, dict(publish=t_pub, start_lat=(t_started or t_send) - t_send, track=t_done - (t_started or t_send),
                                    total=t_done - t_send)

    def _log(self, f):
        if not f.T or not f.X:
            return
        t = f.T[-1]
        if self.trace and self.trace[-1][-1] == t[0]:
            return
        p = self.fr.base_to_sim(t[1:3])
        x = f.X[-1]
        self.trace.append([t[0] - self.t0, p[0], p[1], wrap(t[3] - self.sc.goal_th), x[1], x[2], x[3], x[4], t[0]])

    def close(self):
        if self.feed:
            self.feed.stop()
        if self.state_nuc.get("driver") or self.state_nuc.get("down"):
            orig_pose = self.ol.nuc_start_pose        # the cleanup restarts the camera: with the binary of this run (e.g. the drain build), not the default
            self.ol.nuc_start_pose = lambda *a_, **k_: orig_pose(interp=self.a.interp_flange, cam_delay_ms=self.a.cam_delay_ms, binary=self.a.pose_bin)
            try:
                self.ol.nuc_cleanup(self.state_nuc)
            finally:
                self.ol.nuc_start_pose = orig_pose
        self.trace = [r[:-1] for r in self.trace]


def describe_start(start_base, fr, T_goal, pen_base=None, label="start"):
    """Start point line: label, base coordinates and the distance of the pen centre to the T outline."""
    from global_mppi.closed_loop.geometry import dist_to_poly, t_polygon
    pen = np.asarray(start_base if pen_base is None else pen_base, float)
    d = dist_to_poly(fr.base_to_sim(pen), t_polygon(T_goal)) * 1000
    return (f"start = {label} ({start_base[0]:.4f}, {start_base[1]:.4f}) base; pen now ({pen[0]:.4f}, {pen[1]:.4f}); "
            f"pen centre to the T outline: {d:.0f} mm")


def predict_T(model, qpos, pen_sim, wps_sim, dur, anchor, wait=None):
    """Planner-model prediction of the T pose (goal frame) after the sent plan: from the measured state, the pen follows the
    rest-to-rest quintics through wps_sim (kinematic, like xy_driver), then rests WAIT_AFTER_S while the T settles."""
    import mujoco
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    chain = np.vstack([pen_sim, wps_sim])
    seg_t, dt = dur / len(wps_sim), model.opt.timestep
    for k in range(int(round(dur / dt))):
        p, v = quintic(chain, seg_t, k * dt)
        d.qpos[3:5], d.qvel[3:5] = p - anchor, v
        mujoco.mj_step(model, d)
    d.qvel[3:5] = 0.0
    wait = WAIT_AFTER_S if wait is None else wait
    for _ in range(int(round(wait / dt))):
        mujoco.mj_step(model, d)
    return np.array([d.qpos[0], d.qpos[1], wrap(d.qpos[2])])


# ------------------------------------------------------------------ --blend: velocity-continuous plan hand-over (xy_driver --blend)
def blend_build(p0, v0, a0, pts, dur):
    """Port of blend::build (blend_trajectory.hpp): knot 0 = the commanded state (p0, v0, a0), knots 1..n = the received points,
    Catmull-Rom velocity / acceleration at the inner knots, rest at the last one; n quintic Hermite segments of dur/n each."""
    pts = np.asarray(pts, float)
    n, h = len(pts), dur / len(pts)
    P = np.vstack([np.asarray(p0, float), pts])
    V, A = np.zeros((n + 1, 2)), np.zeros((n + 1, 2))
    V[0], A[0] = v0, a0
    for i in range(1, n):
        V[i] = (P[i + 1] - P[i - 1]) / (2.0 * h)
        A[i] = (P[i + 1] - 2.0 * P[i] + P[i - 1]) / (h * h)
    return dict(P=P, V=V, A=A, h=h, n=n, dur=float(dur))


def blend_eval(tr, t):
    """Port of blend::eval, vectorised over t (array): returns p, v, a, each (len(t), 2). Rest at the end point after tr['dur']."""
    t = np.atleast_1d(np.asarray(t, float))
    n, h = tr["n"], tr["h"]
    tc = np.clip(t, 0.0, tr["dur"])
    s = np.minimum((tc / h).astype(int), n - 1)
    u = ((tc - s * h) / h)[:, None]
    u2, u3, u4, u5 = u ** 2, u ** 3, u ** 4, u ** 5
    B = [1 - 10 * u3 + 15 * u4 - 6 * u5, u - 6 * u3 + 8 * u4 - 3 * u5, 0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5,
         10 * u3 - 15 * u4 + 6 * u5, -4 * u3 + 7 * u4 - 3 * u5, 0.5 * u3 - u4 + 0.5 * u5]
    D1 = [-30 * u2 + 60 * u3 - 30 * u4, 1 - 18 * u2 + 32 * u3 - 15 * u4, u - 4.5 * u2 + 6 * u3 - 2.5 * u4,
          30 * u2 - 60 * u3 + 30 * u4, -12 * u2 + 28 * u3 - 15 * u4, 1.5 * u2 - 4 * u3 + 2.5 * u4]
    D2 = [-60 * u + 180 * u2 - 120 * u3, -36 * u + 96 * u2 - 60 * u3, 1 - 9 * u + 18 * u2 - 10 * u3,
          60 * u - 180 * u2 + 120 * u3, -24 * u + 84 * u2 - 60 * u3, 3 * u - 12 * u2 + 10 * u3]
    P, V, A = tr["P"], tr["V"], tr["A"]
    c = [P[s], h * V[s], h * h * A[s], P[s + 1], h * V[s + 1], h * h * A[s + 1]]
    p = sum(B[i] * c[i] for i in range(6))
    v = sum(D1[i] * c[i] for i in range(6)) / h
    a = sum(D2[i] * c[i] for i in range(6)) / (h * h)
    end = t >= tr["dur"]
    p[end], v[end], a[end] = P[n], 0.0, 0.0
    return p, v, a


def blend_check(tr, box, v_max=PEAK_V_MAX, a_max=PEAK_A_MAX, dt=0.005):
    """(ok, peak speed, peak accel, box excess [m]) of a built blend trajectory (5 ms sampling like the driver)."""
    p, v, a = blend_eval(tr, np.arange(0.0, tr["dur"] + dt, dt))
    sp, ac = float(np.hypot(v[:, 0], v[:, 1]).max()), float(np.hypot(a[:, 0], a[:, 1]).max())
    out = max(box[0] - p[:, 0].min(), p[:, 0].max() - box[1], box[2] - p[:, 1].min(), p[:, 1].max() - box[3], 0.0)
    return (sp <= v_max and ac <= a_max and out <= 0.0), sp, ac, out


class PredWorld:
    """Planner-model world (T + pen) advanced under a blend trajectory (pen kinematic, like the arm), for the predictions of --blend."""

    def __init__(self, model, scene):
        import mujoco
        self.mj, self.m, self.sc = mujoco, model, scene
        self.d = mujoco.MjData(model)
        self.R = rot2(-scene.goal_th)

    def reset(self, T, pen_sim):
        self.d.qpos[:3], self.d.qpos[3:5], self.d.qvel[:] = T, pen_sim - self.sc.anchor, 0.0
        self.mj.mj_forward(self.m, self.d)

    def copy_from(self, other):
        self.mj.mj_copyData(self.d, self.m, other.d)

    def advance(self, tr, t0, t1):
        """Step the model from plan time t0 to t1 (1 ms steps); after tr['dur'] the pen rests at the last point."""
        dt = self.m.opt.timestep
        ts = t0 + np.arange(int(round((t1 - t0) / dt))) * dt
        if len(ts) == 0:
            return
        p, v, _ = blend_eval(tr, ts)
        ps = (p - self.sc.goal_p) @ self.R.T - self.sc.anchor
        vs = v @ self.R.T
        d = self.d
        for j in range(len(ts)):
            d.qpos[3:5], d.qvel[3:5] = ps[j], vs[j]
            self.mj.mj_step(self.m, d)
        d.qvel[3:5] = 0.0

    def run_path(self, T0, P_sim, V_sim):
        """Reset to the T pose T0 and the first pen sample, then step along the given pen path (1 ms samples, goal frame)."""
        self.reset(T0, P_sim[0])
        d = self.d
        for j in range(len(P_sim)):
            d.qpos[3:5], d.qvel[3:5] = P_sim[j] - self.sc.anchor, V_sim[j]
            self.mj.mj_step(self.m, d)
        d.qvel[3:5] = 0.0

    def T(self):
        return np.array([self.d.qpos[0], self.d.qpos[1], wrap(self.d.qpos[2])])


def rest_traj(p):
    return dict(P=np.array([p, p], float), V=np.zeros((2, 2)), A=np.zeros((2, 2)), h=1.0, n=1, dur=1.0)


def cmd_path(hist, tt):
    """Commanded pen position / velocity (base frame) at the times tt from the plans sent so far: hist = [(send time, trajectory)]."""
    P, V = np.zeros((len(tt), 2)), np.zeros((len(tt), 2))
    idx = np.searchsorted(np.array([h[0] for h in hist]), tt, side="right") - 1
    for j in np.unique(np.clip(idx, 0, len(hist) - 1)):
        m = np.clip(idx, 0, len(hist) - 1) == j
        p, v, _ = blend_eval(hist[j][1], tt[m] - hist[j][0])
        P[m], V[m] = p, v
    return P, V


def blend_loop(a, be, planner, scene, fr, info, rows, plans):
    """--pipeline --blend. xy_driver --blend: a new plan interrupts the running one and starts from its commanded position / velocity /
    acceleration (and ends at rest). Chunk k+1 is planned from the PREDICTED state at the hand-over time (planner model + a port of the
    driver's blend trajectory) and sent while chunk k still runs: hand-over h after chunk k started, with the time left in chunk k about
    the planning time (h = dur - plan time, at least plan time + 30 ms so that the plan is ready). N chunks (--measure-every) run in
    a row; then the last one finishes at rest, the T settles, the camera is read and the prediction is replaced by the measurement."""
    anchor, box = scene.anchor, scene.box
    out = info["out"]
    world, scratch = PredWorld(planner.model, scene), PredWorld(planner.model, scene)

    def flush():
        pipeline_rows(a, be, scene, chunks, rows)
        write_cycles(out, rows)

    def sim_pen(p_base):
        return fr.base_to_sim(p_base)

    def finalize_last():
        """Predicted T of the last chunk at rest (+ settling): what the reading at rest is compared with."""
        c = chunks[-1]
        if c["Tp_end"] is None:
            scratch.copy_from(world)
            scratch.advance(c["traj"], 0.0, c["dur"] + settle_pred(a))
            c["Tp_end"] = scratch.T()

    pred_times, meas_log, handover = [], [], []
    info["prop_times"], info["pred_times"], info["measure_points"], info["handover"] = [], pred_times, meas_log, handover
    T0, pen0 = be.state()
    state_T, state_pen = T0.copy(), pen0.copy()
    world.reset(state_T, sim_pen(state_pen))
    k, chunks, t_edge, since, clock, plan_est = 0, [], None, 0, 0.0, 0.3
    N = max(0, int(a.measure_every))
    pend_meas = None                                    # timing of the last measurement phase, attached to the next chunk sent
    hist = [(-1e9, rest_traj(state_pen))]               # plans sent in the current group (for --cam-correct)
    Rg, cam_log = rot2(-scene.goal_th), []
    pe_hist, conv_n = [], 0
    info["cam_corr"] = cam_log
    while True:
        be.monitor(chunks)
        if not a.dry_run and k == 0:
            be.check_link()
        if N > 0 and since >= N:
            finalize_last()
            t_m0 = be.now()
            be.drain(settle=False)
            T_m, pen_m, pend_meas = meas_phase(be, a, t_m0)
            Tp = chunks[-1]["Tp_end"]
            dpos, dang = float(np.linalg.norm(Tp[:2] - T_m[:2])) * 1000, math.degrees(wrap(Tp[2] - T_m[2]))
            chunks[-1]["static"] = T_m.copy()
            meas_log.append((k, dpos, dang))
            print(f"      measure after chunk {k - 1}: predicted ({Tp[0]:+.4f},{Tp[1]:+.4f},{math.degrees(Tp[2]):+6.1f}deg) measured "
                  f"({T_m[0]:+.4f},{T_m[1]:+.4f},{math.degrees(T_m[2]):+6.1f}deg) diff {dpos:.1f} mm {dang:+.1f} deg", flush=True)
            state_T, state_pen, since = T_m, pen_m, 0
            clock += chunks[-1]["dur"]
            world.reset(state_T, sim_pen(state_pen))
            hist = [(-1e9, rest_traj(state_pen))]
            flush()
            pe, ae = float(np.linalg.norm(T_m[:2])) * 1000, abs(math.degrees(wrap(T_m[2])))
            if not fr.T_in_box(T_m[:2]):
                return "T left the box", chunks
            if pe > a.max_dist_mm:
                return f"T {pe:.0f} mm from the goal (> {a.max_dist_mm:g} mm)", chunks
            if pe < a.pos_tol_mm and ae < a.ang_tol_deg:
                info["t_converged"] = be.now()
                return f"converged (measured at rest): {pe:.1f} mm / {ae:.2f} deg", chunks
        if k >= a.max_cycles:
            if chunks:
                finalize_last()
            return f"max cycles ({a.max_cycles})", chunks
        if be.now() > a.max_time:
            if chunks:
                finalize_last()
            return f"time limit {a.max_time:g} s", chunks
        # ---- state at the hand-over time
        t_start_plan = be.now()
        t_pred0 = time.perf_counter()
        cam_info = None
        if since == 0:
            h, tr_prev, s_prev = 0.0, None, None
            T_pl, pen_pl, v_pl, a_pl = state_T, state_pen, np.zeros(2), np.zeros(2)
        else:
            prev = chunks[-1]
            tr_prev, s_prev = prev["traj"], prev["s_send"]
            h = min(max(prev["dur"] - plan_est, plan_est + 0.03), prev["dur"] - 0.05)
            scratch.copy_from(world)
            scratch.advance(tr_prev, 0.0, h)
            T_pred = scratch.T()
            if a.cam_correct:                      # replace the predicted T by the newest camera T propagated to the hand-over
                got = be.camera_newest()
                tau = s_prev + h
                if got is not None and 0.0 <= tau - got[1] <= 1.0:
                    Tc, tc = got
                    tt = tc + np.arange(int(round((tau - tc) / 0.001))) * 0.001
                    if len(tt) > 1:
                        Pc, Vc = cmd_path(hist, tt)
                        scratch.run_path(Tc, (Pc - scene.goal_p) @ Rg.T, Vc @ Rg.T)
                        dT = scratch.T() - T_pred
                        cam_info = (float(np.hypot(dT[0], dT[1])) * 1000, abs(math.degrees(wrap(dT[2]))), tau - tc)
                        cam_log.append((k,) + cam_info)
            T_pl = scratch.T()
            p_, v_, a_ = blend_eval(tr_prev, h)
            pen_pl, v_pl, a_pl = p_[0], v_[0], a_[0]
            prev["Tp_end"] = T_pl.copy()           # the prediction for the end of the interrupted chunk = its state at the hand-over
            clock += h
        if a.converge_continuous > 0:               # --converge-continuous K: camera-corrected state within the tolerance at K consecutive hand-overs
            corrected = bool(cam_log) and cam_log[-1][0] == k
            pe_c, ae_c = float(np.linalg.norm(T_pl[:2])) * 1000, abs(math.degrees(wrap(T_pl[2])))
            conv_n = conv_n + 1 if (corrected and pe_c < a.pos_tol_mm and ae_c < a.ang_tol_deg) else 0
            if conv_n >= a.converge_continuous:
                info["t_converged"] = be.now()
                return f"converged (continuous): {pe_c:.1f} mm / {ae_c:.2f} deg at {conv_n} consecutive camera-corrected hand-overs", chunks
        if a.plateau_chunks > 0:                    # --plateau-chunks: the distance of the planning state stopped improving
            pe_hist.append(float(np.linalg.norm(T_pl[:2])) * 1000)
            if len(pe_hist) >= a.plateau_chunks:
                w_ = pe_hist[-a.plateau_chunks:]
                if max(w_) - min(w_) < a.plateau_mm:
                    return (f"plateau: distance within {max(w_) - min(w_):.2f} mm (< {a.plateau_mm:g}) over the last {a.plateau_chunks} chunks, "
                            f"now {w_[-1]:.1f} mm / {abs(math.degrees(T_pl[2])):.2f} deg"), chunks
        t_pred = time.perf_counter() - t_pred0
        pen_sim = sim_pen(pen_pl)
        qpos = np.concatenate([T_pl, pen_sim - anchor])
        us, plan_s = planner.plan(qpos, clock, a.chunk_s)
        be.account(plan_s + t_pred)
        path_sim = pen_sim + np.cumsum(us, axis=0) * planner.dt
        n_wp = max(1, int(a.chunk_s // SEG_MIN_S))
        idx = [int(round((j + 1) * len(us) / n_wp)) - 1 for j in range(n_wp)]
        wps, clamped = [], False
        for i in idx:
            p2, c2 = fr.clamp_base(fr.sim_to_base(path_sim[i]))
            wps.append(p2)
            clamped |= c2
        wps = np.array(wps)
        dur = max(a.chunk_s, n_wp * SEG_MIN_S)
        for _ in range(30):
            tr = blend_build(pen_pl, v_pl, a_pl, wps, dur)
            ok, peak, acc, out_m = blend_check(tr, box)
            if ok:
                break
            dur *= 1.1
        else:
            return (f"plan would be rejected (blend: peak {peak:.3f} m/s, acc {acc:.2f} m/s^2, outside the box by {out_m * 1000:.1f} mm)"), chunks
        # ---- hand over: wait for the planned time (or send at once when late)
        late = 0.0
        if since > 0:
            t_send_plan = s_prev + h
            wait_t0 = be.now()
            while be.now() < t_send_plan:
                be.wait(0.002)
                be.monitor(chunks)
                if be.now() - wait_t0 > 5.0:
                    raise Stop("timeout waiting for the hand-over time")
            late = be.now() - t_send_plan
        sid = be.send_pipe(wps, dur)
        s_k = be.now()
        plan_est = 0.6 * plan_est + 0.4 * (plan_s + t_pred) if since > 0 else 0.6 * plan_est + 0.4 * plan_s
        handover.append((k, h, late))
        # model of what the driver does: the new plan starts from the commanded state of the running one at the actual send time
        if since > 0:
            dt_run = s_k - s_prev
            world.copy_from(scratch)                # scratch = the (possibly camera-corrected) state at the planned hand-over time
            if dt_run > h + 0.001:
                world.advance(tr_prev, h, dt_run)
            p_, v_, a_ = blend_eval(tr_prev, dt_run)
            tr = blend_build(p_[0], v_[0], a_[0], wps, dur)
        else:
            tr = blend_build(state_pen, np.zeros(2), np.zeros(2), wps, dur)
        hist.append((s_k, tr))
        meas_c = None
        if since == 0 and pend_meas is not None:            # first chunk after a measurement: it carries the timing of the standstill
            pend_meas.update(standstill=s_k - pend_meas["t_rest"], plan=plan_s)
            meas_c, pend_meas = pend_meas, None
        chunks.append(dict(sid=sid, k=k, t_plan0=t_start_plan, plan_s=plan_s, state_T=T_pl.copy(), state_pen=pen_pl.copy(),
                           wps=wps, n_wp=n_wp, dur=dur, peak=peak, acc=acc, clamped=int(clamped), Tp_end=None, static=None,
                           traj=tr, s_send=s_k, meas=meas_c))
        pred_times.append(t_pred)
        plans[f"wps_{k}"] = wps
        print(f"[{k:3d}] t={t_start_plan:5.2f}s state T=({T_pl[0]:+.3f},{T_pl[1]:+.3f},{math.degrees(T_pl[2]):+6.1f}deg) "
              f"plan {plan_s:.3f}s -> {n_wp} wp, {dur:.2f}s, v {peak:.3f} a {acc:.2f}"
              + (" [box-clamped]" if clamped else "") + f"  sent id {sid} at hand-over {h:.2f}s (late {late * 1000:.0f} ms)  predict {t_pred * 1000:.0f} ms"
              + ("" if cam_info is None else f"  cam-correct {cam_info[0]:.1f} mm {cam_info[1]:.1f} deg (image {cam_info[2] * 1000:.0f} ms old at hand-over)"),
              flush=True)
        if N > 0 and since + 1 >= N:
            finalize_last()
        state_T, state_pen = T_pl, pen_pl
        k += 1
        since += 1
        flush()


def settle_pred(a):
    """Settling time added to the prediction of the chunk before a measurement (the camera reads soon after the stop with --adaptive-settle)."""
    if a.fast_stop:
        return 0.1
    return ADAPT_MIN_S + 0.15 if a.adaptive_settle else WAIT_AFTER_S


def meas_phase(be, a, t_m0):
    """Timing of the measurement phase after be.drain(): returns (T, pen, dict). Link check (background with --bg-link-check) is timed too."""
    t_m1 = be.now()
    T_m, pen_m = be.rest_read()
    t_m2 = be.now()
    if not a.dry_run:
        be.check_link()
    t_m3 = be.now()
    t_rest = getattr(be, "t_rest", t_m1)
    return T_m, pen_m, dict(drain=t_m1 - t_m0, t_rest=t_rest, pose=t_m2 - t_rest, settle=be.rd["settle"], read=be.rd["read"], n=be.rd["n"],
                            low=be.rd["low"], link=t_m3 - t_m2)


def pipeline_loop(a, be, planner, scene, fr, info, rows, plans):
    """--pipeline. Chunks are planned while the previous one still runs, from the pen end of the previous chunk and the T pose PREDICTED
    by the planner model (predict_T); a chunk is sent only when the previous one is the running one (at most one plan queued);
    completion = XY_STATE.plan_id switching. After every --measure-every chunks the arm is allowed to finish, the T settles
    (--settle-s), the camera is read while everything is at rest (--read-s window, median) and planning continues from that measured
    pose (the only camera correction; readings taken while the arm moves are not used for planning or stopping, only logged)."""
    anchor = scene.anchor
    out = info["out"]

    def flush():
        pipeline_rows(a, be, scene, chunks, rows)
        write_cycles(out, rows)

    def wait_for(cond, timeout, what):
        """Poll cond() with a timeout (and the run time limit): Stop = lift the pen, stop xy_driver, no endless waiting."""
        t_w = be.now()
        while not cond():
            if be.now() - t_w > timeout:
                raise Stop(f"timeout ({timeout:.1f} s) waiting for {what}")
            if be.now() > a.max_time + 10.0:
                raise Stop(f"time limit exceeded while waiting for {what}")
            be.wait(0.005)
            be.monitor(chunks)

    pred_times, meas_log = [], []
    info["prop_times"], info["pred_times"], info["measure_points"] = [], pred_times, meas_log
    T0, pen0 = be.state()                                     # measured start state at rest
    state_T, state_pen = T0.copy(), pen0.copy()
    pend_meas = None
    t_plan, k, chunks, t_edge, since = 0.0, 0, [], None, 0
    pe_hist = []
    N = max(0, int(a.measure_every))
    while True:
        be.monitor(chunks)
        if not a.dry_run and k == 0:
            be.check_link()                                   # later checks only at measurement points (arm at rest): an ssh call
                                                              # (~0.4 s) inside the planning window delayed the send past the running chunk
        # ---- measurement point: N chunks sent -> let them finish, settle, read the camera at rest, continue from the measured pose
        if N > 0 and since >= N:
            t_m0 = be.now()
            be.drain(settle=False)                            # wait until the arm has stopped (plan_id / tracking)
            T_m, pen_m, pend_meas = meas_phase(be, a, t_m0)   # camera at rest (fixed window or --adaptive-settle) + link check, timed
            Tp = chunks[-1]["Tp_end"]
            dpos, dang = float(np.linalg.norm(Tp[:2] - T_m[:2])) * 1000, math.degrees(wrap(Tp[2] - T_m[2]))
            chunks[-1]["static"] = T_m.copy()
            meas_log.append((k, dpos, dang))
            print(f"      measure after chunk {k - 1}: predicted ({Tp[0]:+.4f},{Tp[1]:+.4f},{math.degrees(Tp[2]):+6.1f}deg) measured "
                  f"({T_m[0]:+.4f},{T_m[1]:+.4f},{math.degrees(T_m[2]):+6.1f}deg) diff {dpos:.1f} mm {dang:+.1f} deg", flush=True)
            state_T, state_pen, since = T_m, pen_m, 0
            flush()
            pe, ae = float(np.linalg.norm(T_m[:2])) * 1000, abs(math.degrees(wrap(T_m[2])))
            if not fr.T_in_box(T_m[:2]):
                return "T left the box", chunks
            if pe > a.max_dist_mm:
                return f"T {pe:.0f} mm from the goal (> {a.max_dist_mm:g} mm)", chunks
            if pe < a.pos_tol_mm and ae < a.ang_tol_deg:
                info["t_converged"] = be.now()
                return f"converged (measured at rest): {pe:.1f} mm / {ae:.2f} deg", chunks
        if k >= a.max_cycles:
            return f"max cycles ({a.max_cycles})", chunks
        if be.now() > a.max_time:
            return f"time limit {a.max_time:g} s", chunks
        if a.plateau_chunks > 0:                              # --plateau-chunks: the distance (state used for planning) stopped improving
            pe_hist.append(float(np.linalg.norm(state_T[:2])) * 1000)
            if len(pe_hist) >= a.plateau_chunks:
                w_ = pe_hist[-a.plateau_chunks:]
                if max(w_) - min(w_) < a.plateau_mm:
                    return (f"plateau: distance within {max(w_) - min(w_):.2f} mm (< {a.plateau_mm:g}) over the last {a.plateau_chunks} chunks, "
                            f"now {w_[-1]:.1f} mm / {abs(math.degrees(state_T[2])):.2f} deg"), chunks
        b_, m_ = scene.box, 0.006
        at_edge = bool(state_pen[0] <= b_[0] + m_ or state_pen[0] >= b_[1] - m_ or state_pen[1] <= b_[2] + m_ or state_pen[1] >= b_[3] - m_)
        t_edge = (t_edge if t_edge is not None else be.now()) if at_edge else None
        if t_edge is not None and be.now() - t_edge > a.max_edge_s:
            return f"pen held at the box edge for {be.now() - t_edge:.1f} s (> {a.max_edge_s:g} s)", chunks
        # ---- plan chunk k from the (predicted or measured) state at the start of the chunk
        t_start_plan = be.now()
        pen_sim = fr.base_to_sim(state_pen)
        qpos = np.concatenate([state_T, pen_sim - anchor])
        us, plan_s = planner.plan(qpos, t_plan, a.chunk_s)
        be.account(plan_s)                                    # the arm keeps moving while the planner computes
        path_sim = pen_sim + np.cumsum(us, axis=0) * planner.dt
        n_wp = max(1, int(a.chunk_s // SEG_MIN_S))
        idx = [int(round((j + 1) * len(us) / n_wp)) - 1 for j in range(n_wp)]
        wps, clamped = [], False
        for i in idx:
            p_, c_ = fr.clamp_base(fr.sim_to_base(path_sim[i]))
            wps.append(p_)
            clamped |= c_
        wps = np.array(wps)
        dur = max(a.chunk_s, n_wp * SEG_MIN_S)
        for _ in range(30):
            ok, seg_t, peak, acc = check_limits(state_pen, wps, dur)
            if ok:
                break
            dur *= 1.1
        else:
            return f"plan would be rejected (seg {seg_t:.2f} s, peak {peak:.3f} m/s, acc {acc:.2f} m/s^2)", chunks
        last_of_group = N > 0 and since + 1 >= N
        tq0 = time.perf_counter()
        # the chunk before a measurement is predicted including the settling time (compared with the reading at rest)
        Tp_end = predict_T(planner.model, qpos, pen_sim, np.array([fr.base_to_sim(w) for w in wps]), dur, anchor,
                           wait=(settle_pred(a) if last_of_group else 0.0))
        pred_times.append(time.perf_counter() - tq0)
        # ---- send only when the previous chunk of this group is the running one (the one before finished), at most one queued
        if since >= 2:      # the previous chunk must have been PICKED UP by the driver (started), else two plans would be queued
            prev = chunks[k - 1]
            wait_for(lambda: be.started(prev["sid"]), max(4.0, 2 * prev["dur"] + 3.0), f"chunk {k - 1} (id {prev['sid']}) to start")
        sid = be.send_pipe(wps, dur)
        meas_c = None
        if since == 0 and pend_meas is not None:
            pend_meas.update(standstill=be.now() - pend_meas["t_rest"], plan=plan_s)
            meas_c, pend_meas = pend_meas, None
        chunks.append(dict(sid=sid, k=k, t_plan0=t_start_plan, plan_s=plan_s, state_T=state_T.copy(), state_pen=state_pen.copy(),
                           wps=wps, n_wp=n_wp, dur=dur, peak=peak, acc=acc, clamped=int(clamped), Tp_end=Tp_end, static=None, meas=meas_c))
        plans[f"wps_{k}"] = wps
        print(f"[{k:3d}] t={t_start_plan:5.2f}s state T=({state_T[0]:+.3f},{state_T[1]:+.3f},{math.degrees(state_T[2]):+6.1f}deg) "
              f"plan {plan_s:.3f}s -> {n_wp} wp, {dur:.2f}s, v {peak:.3f} a {acc:.2f}"
              + (" [box-clamped]" if clamped else "") + f"  sent id {sid}  predict {pred_times[-1] * 1000:.0f} ms", flush=True)
        t_plan += dur
        state_T, state_pen = Tp_end, wps[-1].copy()
        k += 1
        since += 1
        flush()


def pipeline_timing_summary(info, chunks):
    pr, pd, pl = info.get("prop_times", []), info.get("pred_times", []), [c["plan_s"] for c in chunks]
    dur = float(np.mean([c["dur"] for c in chunks])) if chunks else float("nan")
    f = lambda v: f"mean {np.mean(v):.3f} s, max {np.max(v):.3f} s (n={len(v)})" if len(v) else "n/a"
    print(f"== compute per chunk (CPU wall): plan {f(pl)}; propagate_T (--correct) {f(pr)}; predict_T (end-of-chunk) {f(pd)}")
    if len(pl):
        crit = (np.array(pl[-len(pr):]) + np.array(pr)) if len(pr) else np.array(pl)
        print(f"== plan + propagate = mean {crit.mean():.3f} s, max {crit.max():.3f} s vs chunk duration mean {dur:.2f} s -> "
              f"{'fits' if crit.max() < min(c['dur'] for c in chunks) else 'DOES NOT always fit'} inside one chunk")


def pipeline_rows(a, be, scene, chunks, rows):
    """Rebuild `rows` (cycles.csv layout): one row per chunk sent so far. Times of a chunk that has not started / ended yet are NaN;
    the T measured at the chunk end is the reading at rest (group-last chunks) or the camera history around the stop (logged only)."""
    rows.clear()
    nan = float("nan")
    for i, c in enumerate(chunks):
        sg = be.segs[c["sid"] - 1]
        nxt = be.segs[chunks[i + 1]["sid"] - 1] if i + 1 < len(chunks) else None
        t0_, t1_ = sg["t_start"], sg["t_end"]
        if c.get("static") is not None:
            meas = c["static"]
        elif t1_ is not None and be.now() - t1_ > 0.45:            # the frames of the stop have arrived
            meas = be.camera_read(t1_, n=5, span=0.08)
        else:
            meas = np.full(3, nan)
        Tp = c["Tp_end"] if c["Tp_end"] is not None else np.full(3, nan)        # --blend: not predicted yet
        pos_err = float(np.linalg.norm(Tp[:2] - meas[:2])) * 1000
        ang_err = math.degrees(wrap(Tp[2] - meas[2]))
        sT = c["state_T"]
        t_round = (nxt["t_start"] - t0_) if (nxt and nxt["t_start"] is not None and t0_ is not None) else nan
        m = c.get("meas")                                   # measurement phase before this chunk (first chunk of a group): its timing
        row = [c["k"], c["t_plan0"], sT[0], sT[1], math.degrees(sT[2]), c["state_pen"][0], c["state_pen"][1],
               float(np.linalg.norm(sT[:2])) * 1000, abs(math.degrees(wrap(sT[2]))), c["plan_s"], c["n_wp"], c["dur"], c["peak"],
               c["acc"], c["clamped"], Tp[0], Tp[1], math.degrees(Tp[2]), meas[0], meas[1], math.degrees(meas[2]), pos_err, ang_err,
               (m["link"] if m else 0.0), (m["settle"] if m else 0.0), (m["read"] if m else 0.0), (m["n"] if m else 0.0),
               (m["low"] if m else 0.0), 0.0, (t0_ - sg["t_send"]) if t0_ is not None else nan,
               (t1_ - t0_) if (t0_ is not None and t1_ is not None) else nan,
               (t1_ - sg["t_send"]) if t1_ is not None else nan, t_round, 0.0]
        if any(x.get("meas") for x in chunks):
            row += [m["drain"], m["pose"], m["standstill"], m["standstill"] - m["pose"] - m["link"] - m["plan"]] if m else [nan] * 4
        rows.append(row)


CYCLES_HDR = ("cycle,t_s,T_x,T_y,T_th_deg,pen_bx,pen_by,pos_err_mm,ang_err_deg,plan_s,n_wp,dur_s,peak_v,peak_a,clamped,"
              "pred_x,pred_y,pred_th_deg,meas_x,meas_y,meas_th_deg,pred_err_mm,pred_err_deg,"
              "t_link_s,t_settle_s,t_read_s,n_frames,low_frames,t_publish_s,t_start_lat_s,t_track_s,t_exec_total_s,t_round_s,t_other_s")
SPEC_HDR = ",spec_hit,spec_plan_s,spec_dpos_mm,spec_dang_deg,spec_hit_rate"
MEAS_HDR = ",meas_drain_wait_s,meas_stop_to_pose_s,meas_standstill_s,meas_standstill_other_s"   # --pipeline: rows of the first chunk after a measurement


def write_cycles(out, rows):
    """(Re)write cycles.csv from all rows so far (atomic replace); called after every round, not only at the end."""
    os.makedirs(out, exist_ok=True)
    NC = max([len(r) for r in rows] + [34])                  # 34 | 38 (--pipeline measurement timing) | 39 (--spec-plan)
    R_all = np.array([list(r) + [np.nan] * (NC - len(r)) for r in rows], dtype=float).reshape(-1, NC)
    R = R_all[:, :33]
    other = R[:, 32] - (R[:, 23] + R[:, 24] + R[:, 25] + R[:, 9] + R[:, 31])     # round - (link+settle+read+plan+exec)
    tmp = os.path.join(out, "cycles.csv.tmp")
    cols = [R, other] + ([R_all[:, 34:NC]] if NC > 34 else [])
    hdr = CYCLES_HDR + (SPEC_HDR if NC == 39 else MEAS_HDR if NC == 38 else "")
    np.savetxt(tmp, np.column_stack(cols), delimiter=",", fmt="%.6f", header=hdr, comments="")
    os.replace(tmp, os.path.join(out, "cycles.csv"))
    return R, other


# ------------------------------------------------------------------ plots
def make_plots(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = np.loadtxt(os.path.join(out, "cycles.csv"), delimiter=",", skiprows=1, ndmin=2)
    Tr = np.loadtxt(os.path.join(out, "trace.csv"), delimiter=",", skiprows=1, ndmin=2)
    info = json.load(open(os.path.join(out, "args.json")))
    goal_p = np.array(info["goal_base"][:2])
    gth = info["goal_base"][2]
    R = rot2(-gth)
    outline = np.array([[-.06, .02], [-.02, .02], [-.02, -.06], [.02, -.06], [.02, .02], [.06, .02], [.06, .06], [-.06, .06], [-.06, .02]])

    def draw(ax, xy, th, **kw):
        o = outline @ rot2(th).T + xy
        ax.plot(o[:, 0], o[:, 1], **kw)

    pen = np.array([R @ (p - goal_p) for p in Tr[:, 4:6]])
    fig, ax = plt.subplots(figsize=(7.5, 7))
    draw(ax, np.zeros(2), 0.0, color="green", lw=2, label="goal")
    ax.plot(Tr[:, 1], Tr[:, 2], "r-", lw=1.2, label="T centre")
    ax.plot(pen[:, 0], pen[:, 1], color="tab:blue", lw=0.9, label="pen")
    for k in np.linspace(0, len(Tr) - 1, 8).astype(int):
        draw(ax, Tr[k, 1:3], Tr[k, 3], color="r", lw=0.6, alpha=0.5)
    draw(ax, Tr[0, 1:3], Tr[0, 3], color="r", lw=1.2, ls="--")
    draw(ax, Tr[-1, 1:3], Tr[-1, 3], color="r", lw=1.6)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (goal frame) [m]")
    ax.set_ylabel("y (goal frame) [m]")
    ax.set_title(f"{info['backend']}: {info['stop_reason']}  (final {info['final_pos_mm']:.1f} mm / {info['final_ang_deg']:.2f} deg)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "closed_loop_traj.png"), dpi=130)
    plt.close(fig)

    ang_tr = np.degrees(np.abs((Tr[:, 3] + np.pi) % (2 * np.pi) - np.pi))
    pos_tr = np.linalg.norm(Tr[:, 1:3], axis=1) * 1000
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(Tr[:, 0], pos_tr, "k-")
    ax[0].plot(C[:, 1], C[:, 7], "r.", ms=4, label="at plan time")
    ax[0].axhline(10, color="g", ls="--")
    ax[0].set_ylabel("T position error [mm]")
    ax[0].legend(fontsize=8)
    ax[1].plot(Tr[:, 0], ang_tr, "m-")
    ax[1].plot(C[:, 1], C[:, 8], "r.", ms=4)
    ax[1].axhline(3, color="g", ls="--")
    ax[1].set_ylabel("T heading error [deg]")
    ax[2].bar(C[:, 1], C[:, 9], width=0.3, color="tab:orange")
    ax[2].set_ylabel("plan time [s]")
    ax[2].set_xlabel("run time [s]")
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "closed_loop_errors.png"), dpi=130)
    plt.close(fig)
    if C.shape[1] >= 34:
        ok2 = ~np.isnan(C[:, 32])
        if ok2.any():
            parts = [("link check", C[:, 23]), ("settle wait", C[:, 24]), ("read frames", C[:, 25]), ("plan", C[:, 9]),
                     ("publish+start latency", C[:, 28] + C[:, 29]), ("tracking", C[:, 30]), ("other", C[:, 33])]
            fig, ax = plt.subplots(figsize=(9, 4.5))
            bottom = np.zeros(len(C))
            for name, v in parts:
                ax.bar(C[:, 0], np.nan_to_num(v), bottom=bottom, label=name)
                bottom += np.nan_to_num(v)
            ax.set_xlabel("round")
            ax.set_ylabel("seconds")
            ax.set_title("where each round's time goes")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "closed_loop_timing.png"), dpi=130)
            plt.close(fig)
    ok = ~np.isnan(C[:, 21])
    if ok.any():
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(C[ok, 0], C[ok, 21], "o-", color="tab:blue")
        ax[0].set_ylabel("predicted - measured T position [mm]")
        ax[1].plot(C[ok, 0], C[ok, 22], "o-", color="tab:purple")
        ax[1].set_ylabel("predicted - measured T heading [deg]")
        ax[1].set_xlabel("round")
        for a_ in ax:
            a_.grid(alpha=0.3)
        ax[0].set_title("planner-model prediction of the sent chunk vs the T pose read next round")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "closed_loop_prediction_error.png"), dpi=130)
        plt.close(fig)
    print("wrote", os.path.join(out, "closed_loop_traj.png"), "and closed_loop_errors.png")


# ------------------------------------------------------------------ main loop
def run(a, out):
    from global_mppi.closed_loop.config import load_config
    from global_mppi.closed_loop.scene import Scene
    cfg = load_config(os.path.join(REPO, a.config))
    xml = None
    if not a.no_walls:       # workspace box as 4 walls in the planner model and the plant (same as the xy_driver box)
        from global_mppi.closed_loop.walls import write_walled_scene
        sc_ = cfg["scene"]
        b, g_ = sc_["box_base"], sc_["goal_base"]
        xml = write_walled_scene((b["x_min"], b["x_max"], b["y_min"], b["y_max"]), (g_["x_m"], g_["y_m"]), g_["theta_rad"],
                                 name=f"scene_box_{os.getpid()}.xml", include=os.path.basename(sc_["xml"]))
        cfg["scene"]["xml"] = xml
    scene = Scene(cfg)
    fr = Frames(scene)
    planner = Planner(a, xml or scene.xml)           # the planner model is the scene of the config (--config), walls or not
    planner.warmup()
    scene.start_P = fr.base_to_sim(a.pusher_start_base)        # --sim starts the pen here too
    be = SimBackend(a, scene, fr) if a.sim else RealBackend(a, scene, fr)
    if not a.sim:
        be.out_dir = out
    if a.sim:
        print(describe_start(a.pusher_start_base, fr, scene.start_T, label=a.start_label))
    g = cfg["scene"]["goal_base"]
    info = dict(vars(a), out=out, backend=be.name, goal_base=[g["x_m"], g["y_m"], g["theta_rad"]], stop_reason="?",
                final_pos_mm=float("nan"), final_ang_deg=float("nan"))
    rows, plans, reason = [], {}, "?"
    spec, spec_n, spec_hits = None, 0, 0     # --spec-plan: the plan of the next chunk, made during the execution of this one
    try:
        if not a.sim:
            if not be.preflight():
                raise Stop("pre-flight failed")
            if a.dry_run:
                be.start_dry()
            else:
                be.bring_up()
        T = pen = None
        cyc = 0
        if a.pipeline:
            reason, chunks = (blend_loop if a.blend else pipeline_loop)(a, be, planner, scene, fr, info, rows, plans)
            be.drain(settle=False)                       # let the running + queued plans finish
            T, pen = be.rest_read()                      # final reading at rest (image-time window after the stop)
            pipeline_rows(a, be, scene, chunks, rows)
            pipeline_timing_summary(info, chunks)
            info["final_pos_mm"] = float(np.linalg.norm(T[:2])) * 1000
            info["final_ang_deg"] = abs(math.degrees(wrap(T[2])))
            T = None
        t_edge = None                        # run time since the pen has been at the box edge (None: not at the edge)
        pending = []                         # row index of the round whose prediction waits for the next measured T

        def fill_meas(Tm):
            if pending:
                r = rows[pending.pop()]
                dx = (r[15] - Tm[0], r[16] - Tm[1])
                r[18:23] = [Tm[0], Tm[1], math.degrees(Tm[2]), math.hypot(*dx) * 1000, math.degrees(wrap(math.radians(r[17]) - Tm[2]))]
                write_cycles(out, rows)
        t_exec_end = None
        t_plan = 0.0                         # planner time: advances by the executed plan time only (warm start)
        while not a.pipeline:
            if cyc >= a.max_cycles:
                reason = f"max cycles ({a.max_cycles})"
                break
            if be.now() > a.max_time:
                reason = f"time limit {a.max_time:g} s"
                break
            t_top = be.now()
            if not a.dry_run and (cyc % a.link_check_every == 0 or a.bg_link_check):      # ssh costs ~1 s (first) / ~0.4 s: every N rounds
                be.check_link(start=cyc % a.link_check_every == 0)
            t_link = be.now() - t_top
            T, pen = be.state()
            t_now = be.now()
            if rows:                                       # close the previous round's wall time
                rows[-1][32] = t_top - rows[-1][33]
            t_other_mark = t_top
            fill_meas(T)
            pe, ae = float(np.linalg.norm(T[:2])) * 1000, abs(math.degrees(wrap(T[2])))
            if not fr.T_in_box(T[:2]):
                reason = "T left the box"
                break
            if pe > a.max_dist_mm:
                reason = f"T {pe:.0f} mm from the goal (> {a.max_dist_mm:g} mm)"
                break
            b_, m_ = scene.box, 0.006
            at_edge = bool(pen[0] <= b_[0] + m_ or pen[0] >= b_[1] - m_ or pen[1] <= b_[2] + m_ or pen[1] >= b_[3] - m_)
            t_edge = (t_edge if t_edge is not None else t_now) if at_edge else None
            if t_edge is not None and t_now - t_edge > a.max_edge_s:
                reason = f"pen held at the box edge for {t_now - t_edge:.1f} s (> {a.max_edge_s:g} s)"
                break
            if pe < a.pos_tol_mm and ae < a.ang_tol_deg:
                reason = f"converged: {pe:.1f} mm / {ae:.2f} deg"
                info["t_converged"] = t_now
                break
            pen_sim = fr.base_to_sim(pen)
            qpos = np.concatenate([T, pen_sim - scene.anchor])
            hit = sp_plan = sp_dp = sp_da = float("nan")
            if spec is not None:                           # --spec-plan: compare the measured T with the predicted one
                spec["th"].join()
                dT = spec["Tp"] - T
                sp_dp, sp_da = float(np.hypot(dT[0], dT[1])) * 1000, abs(math.degrees(wrap(spec["Tp"][2] - T[2])))
                sp_plan = spec.get("plan_s", float("nan"))
                spec_n += 1
                if "us" in spec and sp_dp < 2.0 and sp_da < 1.0:
                    hit, us, plan_s = 1.0, spec["us"], 0.0
                    spec_hits += 1
                else:
                    hit = 0.0
                    planner.params = spec["snap"]          # the speculative plan's warm start is dropped
                if "err" in spec:
                    print(f"      spec plan failed: {spec['err']!r}", flush=True)
                spec = None
            if not (hit == 1.0):
                us, plan_s = planner.plan(qpos, t_plan, a.chunk_s)
                be.advance(plan_s)
            # integrate the planned pusher velocity (goal frame) into waypoints (base frame)
            path_sim = pen_sim + np.cumsum(us, axis=0) * planner.dt
            n_wp = max(1, int(a.chunk_s // SEG_MIN_S))
            idx = [int(round((k + 1) * len(us) / n_wp)) - 1 for k in range(n_wp)]
            wps, clamped = [], False
            for i in idx:
                p, c = fr.clamp_base(fr.sim_to_base(path_sim[i]))
                wps.append(p)
                clamped |= c
            wps = np.array(wps)
            dur = max(a.chunk_s, n_wp * SEG_MIN_S)
            for _ in range(30):
                ok, seg_t, peak, acc = check_limits(pen, wps, dur)
                if ok:
                    break
                dur *= 1.1
            else:
                reason = f"plan would be rejected (seg {seg_t:.2f} s, peak {peak:.3f} m/s, acc {acc:.2f} m/s^2)"
                break
            msg = (f"[{cyc:3d}] t={t_now:5.1f}s T=({T[0]:+.3f},{T[1]:+.3f},{math.degrees(T[2]):+6.1f}deg) err {pe:6.1f} mm {ae:5.1f} deg  "
                   f"plan {plan_s:.3f}s  -> {n_wp} wp, {dur:.2f}s, v {peak:.3f} a {acc:.2f}" + (" [box-clamped]" if clamped else "")
                   + ("" if hit != hit else f"  spec {'HIT' if hit else 'miss'} ({sp_dp:.1f} mm, {sp_da:.2f} deg)"))
            print(msg, flush=True)
            plans[f"wps_{cyc}"] = wps
            Tp = predict_T(planner.model, qpos, pen_sim, np.array([fr.base_to_sim(w) for w in wps]), dur, scene.anchor)
            rows.append([cyc, t_now, T[0], T[1], math.degrees(T[2]), pen[0], pen[1], pe, ae, plan_s, n_wp, dur, peak, acc, int(clamped),
                         Tp[0], Tp[1], math.degrees(Tp[2]), np.nan, np.nan, np.nan, np.nan, np.nan,
                         t_link, be.rd["settle"], be.rd["read"], be.rd["n"], be.rd["low"], 0.0, 0.0, 0.0, 0.0, np.nan, t_top]
                        + ([hit, sp_plan, sp_dp, sp_da, spec_hits / spec_n if spec_n else float("nan")] if a.spec_plan else []))
            pending.append(len(rows) - 1)
            if a.spec_plan:                                # plan the next chunk now, from the predicted end state, while this one runs
                q_next = np.concatenate([Tp, fr.base_to_sim(wps[-1]) - scene.anchor])
                spec = dict(Tp=np.array(Tp, float), snap=planner.params)

                def spec_work(rec=spec, q=q_next, tc=t_plan + dur):
                    try:
                        rec["us"], rec["plan_s"] = planner.plan(q, tc, a.chunk_s)
                    except Exception as e:
                        rec["err"] = e
                spec["th"] = threading.Thread(target=spec_work, daemon=True)
                spec["th"].start()
            write_cycles(out, rows)
            info["final_pos_mm"], info["final_ang_deg"] = pe, ae
            if a.dry_run:
                print(f"      waypoints (base): {np.round(wps, 4).tolist()}")
                time.sleep(a.chunk_s)
                be.advance(a.chunk_s)
                t_plan += dur
            else:
                status, lead, t_exec = be.execute(wps, dur)
                t_exec_end = be.now()
                t_plan += dur
                rows[-1][28:32] = [t_exec["publish"], t_exec["start_lat"], t_exec["track"], t_exec["total"]]
                write_cycles(out, rows)
                print(f"      sent {np.round(wps, 4).tolist()} (base), segment {dur / n_wp:.2f} s; link {t_link:.2f}s settle {be.rd['settle']:.2f}s "
                      f"read {be.rd['read']:.2f}s ({be.rd['n']} frames{', LOW' if be.rd['low'] else ''}) plan {plan_s:.2f}s | publish {t_exec['publish'] * 1000:.0f} ms, "
                      f"start latency {t_exec['start_lat']:.2f}s, tracking {t_exec['track']:.2f}s, total {t_exec['total']:.2f}s; "
                      f"max lead {lead:.1f} mm; pen was ({pen[0]:.4f},{pen[1]:.4f})", flush=True)
                if status != "ok":
                    reason = f"plan {status}"
                    break
            cyc += 1
        if reason != "?" and T is not None and not a.dry_run:
            if not a.sim:
                be.check_link(start=False)
                be._check_link_now()
            T, pen = be.state()
            fill_meas(T)
            if rows:                                   # the final reading is not part of the last round
                rows[-1][32] = (t_exec_end if t_exec_end is not None else be.now()) - rows[-1][33]
                write_cycles(out, rows)
            info["final_pos_mm"] = float(np.linalg.norm(T[:2])) * 1000
            info["final_ang_deg"] = abs(math.degrees(wrap(T[2])))
    except Stop as e:
        reason = f"STOP: {e}"
        print(f"\n{reason}")
        if a.real and not a.dry_run and rows:
            print("  (the arm may still be moving: release the enable button)")
    except KeyboardInterrupt:
        reason = "Ctrl-C"
    finally:
        if spec is not None:
            spec["th"].join(10.0)
        be.close()
    info["spec_hits"], info["spec_n"] = spec_hits, spec_n
    if a.spec_plan:
        print(f"== spec-plan: {spec_hits}/{spec_n} hits ({100.0 * spec_hits / max(spec_n, 1):.0f}%)")
    info["stop_reason"] = reason
    chk_list = getattr(be, "checks", [])
    if chk_list:
        print("\n== SUMMARY of the automatic checks ==")
        for name, ok_, detail in chk_list:
            print(f"  [{'PASS' if ok_ else 'FAIL'}] {name}: {detail}")
        print(f"  closed loop: {reason}")
        print("  cleanup: arm lifted to z=0.4434 and xy_driver stopped (no homing), state_pub / aprilcube_pose restarted")
    info["checks"] = [[n_, o_, d_] for n_, o_, d_ in chk_list]
    if xml:
        try:
            os.remove(xml)
        except OSError:
            pass
    fd = getattr(be, "feed", None)
    if fd is not None and hasattr(fd, "n_drop_tags"):
        with fd.lock:
            tg = [r[5] for r in fd.T if len(r) > 5]
            nd = fd.n_drop_tags
        nr = fd.n_drop_reproj
        info["min_tags"], info["frames_dropped_min_tags"], info["frames_dropped_reproj"] = a.min_tags, nd, nr
        print(f"== T frames: kept {len(tg)}, dropped by --min-tags {a.min_tags}: {nd}, by --max-reproj-px {a.max_reproj_px:g}: {nr}; min tags in kept frames {min(tg) if tg else '-'}; "
              f"tag histogram of kept frames {({k_: tg.count(k_) for k_ in sorted(set(tg))})}")
    print(f"\n== stop: {reason}\n== final T error: {info['final_pos_mm']:.1f} mm, {info['final_ang_deg']:.2f} deg after {be.now() if a.sim else 0:.1f} s")
    os.makedirs(out, exist_ok=True)
    R, other = write_cycles(out, rows)
    if len(R):
        ok = ~np.isnan(R[:, 32])
        m = lambda c: float(np.nanmean(c[ok])) if ok.any() else float("nan")
        print(f"== time per round (mean of {int(ok.sum())}): total {m(R[:, 32]):.2f} s = link {m(R[:, 23]):.2f} + settle-wait {m(R[:, 24]):.2f} + "
              f"read {m(R[:, 25]):.2f} ({m(R[:, 26]):.1f} frames) + plan {m(R[:, 9]):.2f} + publish {m(R[:, 28]) * 1000:.0f} ms + "
              f"start latency {m(R[:, 29]):.2f} + tracking {m(R[:, 30]):.2f} + other {m(other):.2f}; low_frames rounds: {int(R[:, 27].sum())}")
    np.savez(os.path.join(out, "plans.npz"), **plans)
    if be.trace:
        np.savetxt(os.path.join(out, "trace.csv"), np.array(be.trace), delimiter=",", fmt="%.6f", comments="",
                   header="t_s,T_x,T_y,T_th,pen_bx,pen_by,cmd_bx,cmd_by (T: goal frame, pen/cmd: base frame)")
    info["t0_epoch"] = None if a.sim else getattr(be, "t0", None)
    json.dump(info, open(os.path.join(out, "args.json"), "w"), indent=1, default=str)
    if rows and be.trace:
        plot(out)
    pt = [r[9] for r in rows]
    if pt:
        print(f"== plan time: mean {np.mean(pt):.3f} s, max {np.max(pt):.3f} s over {len(pt)} rounds; outputs in {out}")


def plot(out):
    try:
        import matplotlib  # noqa: F401
    except ImportError:    # the global-mppi env has no matplotlib
        subprocess.run(["/usr/bin/python3", os.path.abspath(__file__), "--plot-only", out], check=False)
        return
    make_plots(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true", help="MuJoCo instead of camera + arm")
    g.add_argument("--real", action="store_true", help="real robot (NUC over ssh)")
    ap.add_argument("--dry-run", action="store_true", help="with --real: real poses, plan and print only, nothing sent / started")
    ap.add_argument("--max-cycles", type=int, default=None, help="default 1000 (sim/real), 5 with --dry-run")
    ap.add_argument("--config", default=os.path.join("configs", "pusht_real.yaml"),
                    help="scene / cost / vmax config, relative to the repo root; its scene.xml (next to pusht.xml) is also the planner model")
    ap.add_argument("--chunk-s", type=float, default=0.5, help="seconds of each plan that are sent")
    ap.add_argument("--max-time", type=float, default=90.0, help="run time limit [s]")
    ap.add_argument("--pos-tol-mm", type=float, default=10.0)
    ap.add_argument("--ang-tol-deg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--correct", action="store_true", help="(kept for old command lines; no effect: the camera correction is now the static "
                    "measurement every --measure-every chunks of --pipeline)")
    ap.add_argument("--measure-every", type=int, default=3, help="--pipeline: send N chunks from predicted states, then stop, wait for rest "
                    "(--settle-s), read the camera (--read-s) and continue from the measured pose; 0 = never measure")
    ap.add_argument("--adaptive-settle", action="store_true", help="--real, sequential: start the T pose window once 3 consecutive frames differ "
                    "< 0.5 mm and < 0.3 deg (earliest 0.1 s, latest 0.6 s after the stop) instead of the fixed --settle-s; also at the measurement points of --pipeline")
    ap.add_argument("--blend", action="store_true", help="--pipeline: xy_driver --blend hand-over: the next chunk is sent before the current one "
                    "ends (remaining time about the planning time), from the predicted pen state; it interrupts the running plan and continues at its "
                    "velocity. --measure-every N chunks in a row, then stop and measure")
    ap.add_argument("--cam-correct", action="store_true", help="--pipeline --blend: at every hand-over replace the predicted T by the newest camera T "
                    "(propagated with the planner model along the commanded pen path to the hand-over time); no stop")
    ap.add_argument("--spec-plan", action="store_true", help="sequential: plan the next chunk during the execution of this one, from the predicted "
                    "end state; if the measured T at rest is within 2 mm and 1 deg of the prediction, send that plan (no re-planning), else re-plan")
    ap.add_argument("--bg-link-check", action="store_true", help="--real: carrier_changes read in a background thread (a flap is raised at the next round)")
    ap.add_argument("--settle-s", type=float, default=0.35, help="wait after the arm stops before the T pose window (default 0.35)")
    ap.add_argument("--read-s", type=float, default=0.3, help="T pose window length (default 0.3)")
    ap.add_argument("--interp-flange", action="store_true", help="--real: start aprilcube_pose with --interp-flange (flange pose at the image "
                    "timestamp); implied by --real --pipeline --correct")
    ap.add_argument("--cam-delay-ms", type=float, default=None, help="--real: passed to aprilcube_pose as --cam-delay-ms (with --interp-flange)")
    ap.add_argument("--pose-bin", default=None, help="--real: aprilcube_pose binary on the NUC, relative to its repo (e.g. build_interp/bin/aprilcube_pose); "
                    "default: script/run_pose_estimation.sh (build/bin)")
    ap.add_argument("--cam-exposure-delay-s", type=float, default=None, help="--real --pipeline --correct: exposure-to-timestamp delay subtracted "
                    "from T_SHAPE_POSE.timestamp_us to get the image time. Default: the --cam-delay-ms of the running aprilcube_pose (dry-run) / "
                    "of the one this script starts, else 0")
    ap.add_argument("--pipeline", action="store_true", help="plan the next chunk while the current one runs (sim only for now; default off)")
    ap.add_argument("--algo", choices=["ps", "dial"], default="ps", help="planner: ps (examples/pusht.py PS) or dial (examples/pusht.py DIAL defaults)")
    ap.add_argument("--n-opt", type=int, default=1, help="optimisation steps per round on the same state (time unchanged)")
    ap.add_argument("--link-check-every", type=int, default=5, help="--real: read the NUC carrier_changes every N rounds (any increase stops at once)")
    ap.add_argument("--min-cam-hz", type=float, default=32.0, help="--real: camera check at bring-up (aprilcube_pose without --show runs at ~36-60 Hz)")
    ap.add_argument("--min-tags", type=int, default=2, help="--real: drop T frames with fewer detected tags (default 2 = only the unusable ones)")
    ap.add_argument("--max-reproj-px", type=float, default=0.0, help="--real: drop T frames with a larger reprojection error (px); 0 = off")
    ap.add_argument("--converge-continuous", type=int, default=0, help="--pipeline --blend: stop (converged (continuous)) when the camera-corrected state is "
                    "within --pos-tol-mm/--ang-tol-deg at this many consecutive hand-overs (0 = off)")
    ap.add_argument("--plateau-chunks", type=int, default=0, help="--pipeline: stop when the distance of the planning state to the goal varied by less than "
                    "--plateau-mm over the last this many chunks (0 = off)")
    ap.add_argument("--plateau-mm", type=float, default=1.0, help="--plateau-chunks: max spread of the distance (mm) that counts as a plateau")
    ap.add_argument("--fast-stop", action="store_true", help="--pipeline measurement stops: no settle wait; median of the first 3 consecutive stable frames "
                    "(< 0.5 mm, < 0.3 deg) with image time >= stop - --fast-pre-s (default off: --adaptive-settle path)")
    ap.add_argument("--fast-pre-s", type=float, default=0.05, help="--fast-stop: frames taken up to this long before the arm stopped may be used")
    ap.add_argument("--max-dist-mm", type=float, default=350.0, help="stop when the T is further than this from the goal")
    ap.add_argument("--max-edge-s", type=float, default=5.0, help="stop when the pen has been at the box edge this long")
    # PS parameters: defaults = examples/pusht.py
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--noise-level", type=float, default=0.4)
    ap.add_argument("--dial-noise", type=float, default=1.0, help="--algo dial: noise_level of DIAL (absolute, like --noise-level for PS)")
    ap.add_argument("--plan-horizon", type=float, default=1.0)
    ap.add_argument("--num-knots", type=int, default=6)
    # sim camera noise
    # real
    ap.add_argument("--confirm", action="store_true", help="--real: ask for a manual confirmation before 'down' and before the loop "
                    "(default: fully automatic; the automatic checks run in both cases)")
    ap.add_argument("--go-file", default=None, help="--real: instead of Enter, wait until this file exists (then it is removed)")
    ap.add_argument("--go-timeout", type=float, default=1800.0)
    ap.add_argument("--no-walls", action="store_true", help="no box walls in the planner model / plant")
    ap.add_argument("--pusher-start-base", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="start point of the pen, BASE frame. Default: the base point of %s "
                         "(pusher_start_pushside_near); the old standard start is 0.40815 -0.0206" % PUSHER_START_YAML)
    ap.add_argument("--xy-driver-bin", default=UPRIGHT_BIN)
    ap.add_argument("--yaw-deg", type=float, default=59.7)
    ap.add_argument("--max-lead-mm", type=float, default=15.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot-only", metavar="DIR", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.plot_only:
        make_plots(a.plot_only)
        return 0
    if not (a.sim or a.real):
        ap.error("choose --sim or --real")
    if a.dry_run and not a.real:
        ap.error("--dry-run needs --real")
    if a.cam_correct and not a.blend:
        ap.error("--cam-correct needs --pipeline --blend")
    if a.blend and not a.pipeline:
        ap.error("--blend needs --pipeline")
    if a.spec_plan and a.pipeline:
        ap.error("--spec-plan is for the sequential mode (--pipeline plans ahead already)")
    if a.correct and not a.pipeline:
        ap.error("--correct needs --pipeline")
    global WAIT_AFTER_S, WINDOW_S
    WAIT_AFTER_S, WINDOW_S = a.settle_s, a.read_s
    if a.real and (a.pipeline or a.interp_flange):                   # the NUC camera of the pipeline mode: interp_flange build, exposure = stamp - 60 ms
        a.interp_flange = True
        if a.cam_delay_ms is None:
            a.cam_delay_ms = 60.0
        if a.pose_bin is None:
            a.pose_bin = "build_interp/bin/aprilcube_pose"
    a.cam_delay_explicit = a.cam_exposure_delay_s is not None
    if a.cam_exposure_delay_s is None:
        a.cam_exposure_delay_s = (a.cam_delay_ms or 0.0) / 1000.0 if a.interp_flange else 0.0
    a.start_label = "custom"
    if a.pusher_start_base is None:
        import yaml
        y = yaml.safe_load(open(PUSHER_START_YAML))["base"]
        a.pusher_start_base, a.start_label = [y["x_m"], y["y_m"]], "pusher_start_pushside_near"
    elif np.allclose(a.pusher_start_base, [0.40815, -0.0206], atol=1e-4):
        a.start_label = "standard pusher_start"
    if a.max_cycles is None:
        a.max_cycles = 5 if a.dry_run else 1000
    out = a.out or os.path.join(REPO, "runs", "closed_loop_test", f"{'sim' if a.sim else 'real'}_{time.strftime('%Y%m%d_%H%M%S')}")
    run(a, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
