"""One closed-loop episode: plan -> execute `replan_s` of the plan -> repeat, until converged / time limit / stall."""
import time
from dataclasses import dataclass, field

import jax
import mujoco
import numpy as np

from global_mppi.closed_loop.scene import Perturbation, Scene


@dataclass
class Result:
    algo: str
    seed: int
    success: bool
    t_converged: float | None
    sim_time: float
    wall_time: float
    final_pos_mm: float
    final_ang_deg: float
    idle_s: float | None            # sim time until the pusher moved more than idle_move_mm
    plan_times: list = field(default_factory=list)   # wall time per plan, after the warm-up compile
    box_hits: int = 0
    min_pos_mm: float = float('nan')   # smallest T position error seen during the run
    min_ang_deg: float = float('nan')  # smallest |heading error| seen during the run
    skipped: bool = False              # skipped with Esc in the viewer
    stopped: str = ""               # "", "stall", "viewer closed"
    qpos_hist: np.ndarray | None = None      # (t, qpos) every 50 ms
    pusher_base: np.ndarray | None = None    # pusher xy (base frame) at every plan

    def plan_stats(self) -> dict:
        pt = np.asarray(self.plan_times) if self.plan_times else np.array([np.nan])
        return {"plan_mean_s": float(np.mean(pt)), "plan_p95_s": float(np.percentile(pt, 95)), "n_plans": len(self.plan_times)}

    def summary(self) -> dict:
        d = {k: getattr(self, k) for k in ("algo", "seed", "success", "t_converged", "sim_time", "wall_time",
                                           "final_pos_mm", "final_ang_deg", "min_pos_mm", "min_ang_deg", "idle_s", "box_hits", "stopped", "skipped")}
        d.update(self.plan_stats())
        return d


def run_episode(scene: Scene, planner, seed: int, cfg: dict, data, pert: Perturbation | None = None,
                viewer=None, label: str = "", log=None) -> Result:
    """One episode. `data` is a persistent MjData (the viewer is bound to it); it is reset to the start state here.

    loop.realtime = false (paused planning): the sim waits for the planner; every plan is executed for replan_s.
    loop.realtime = true  (latency model): the sim keeps running while the planner computes. During the measured
        planning wall time the robot executes the PREVIOUS plan (zero command before the first plan); the new plan,
        computed from the state at the start of planning, only takes effect when it is ready.
    Stops at convergence (unless loop.run_to_time), loop.max_time_s of sim time, or loop.wall_limit_s of wall time.
    """
    lp, cv = cfg["loop"], cfg["convergence"]
    scene.reset_data(data, pert)
    model = scene.plant
    dt = model.opt.timestep
    n_sub = int(round(lp["replan_s"] / dt))
    params = planner.init_params(seed)
    if viewer is not None:
        viewer.begin(label)
        viewer.update(data)

    hist, plan_times, pb_hist = [], [], []
    st = {"nstep": 0, "box_hits": 0, "stopped": "", "skipped": False, "idle_s": None}
    held_since, t_conv = None, None
    q0_pusher = scene.pusher_world(data.qpos).copy()
    pe_hist = []
    run_to_time = bool(lp.get("run_to_time", False))     # fixed-duration mode: no early stop (convergence / stall)
    wall_limit = lp.get("wall_limit_s")                  # real-time deadline of the run (None: only max_time_s)
    realtime = bool(lp.get("realtime", False))
    min_pos, min_ang = float("inf"), float("inf")
    prev_params = None
    view_every = int(lp.get("viewer_every_steps", 40))   # refresh the window every N sim steps (40 ms of sim = 25 fps real time)
    N_CTRL = 1000                                        # fixed-length control query (no recompilation per latency)
    wall0 = time.time()

    def run_controls(us):
        for u in us:
            if st["nstep"] % 50 == 0:
                hist.append(np.concatenate([[data.time], data.qpos]))
            v, hit = scene.clamp_to_box(data.qpos, u)
            st["box_hits"] += int(hit)
            data.ctrl[:] = v
            mujoco.mj_step(model, data)
            st["nstep"] += 1
            if viewer is not None:
                if st["nstep"] % 50 == 0:
                    viewer.trail(data)
                if st["nstep"] % view_every == 0:
                    viewer.refresh(data)
                    if viewer.consume_skip():
                        st["skipped"], st["stopped"] = True, "skipped (Esc)"
                        return
                    if not viewer.alive():
                        st["stopped"] = "viewer closed"
                        return
            if st["idle_s"] is None and np.linalg.norm(scene.pusher_world(data.qpos) - q0_pusher) > lp["idle_move_mm"] / 1000:
                st["idle_s"] = data.time

    while data.time < lp["max_time_s"]:
        if wall_limit is not None and time.time() - wall0 >= wall_limit:
            st["stopped"] = "wall limit"
            break
        pe, ae = scene.errors(data.qpos)
        min_pos, min_ang = min(min_pos, pe), min(min_ang, ae)
        if pe * 1000 < cv["pos_tol_mm"] and ae < cv["ang_tol_deg"]:
            held_since = data.time if held_since is None else held_since
            if t_conv is None and data.time - held_since >= cv["hold_s"]:
                t_conv = held_since
            if t_conv is not None and not run_to_time:
                break
        else:
            held_since = None
        if viewer is not None and viewer.consume_skip():
            st["skipped"], st["stopped"] = True, "skipped (Esc)"
            break
        if lp.get("stall_exit_s", 0) > 0 and not run_to_time:
            pe_hist.append((data.time, pe))
            w = lp["stall_exit_s"]
            old = [e for t_, e in pe_hist if t_ <= data.time - w]
            if old and min(e for t_, e in pe_hist if t_ > data.time - w) > old[-1] - 0.001 and pe * 1000 > cv["pos_tol_mm"]:
                st["stopped"] = "stall"
                break
        pb_hist.append(scene.pusher_base(data.qpos))

        t_plan_start = data.time
        ts = time.time()
        params = planner.plan(planner.sync(data), params)
        jax.block_until_ready(params.mean)
        plan_times.append(time.time() - ts)
        if log and len(plan_times) % 25 == 1:
            log(f"  t={data.time:6.1f}s pos {pe * 1000:6.1f} mm  ang {ae:5.2f} deg  plan {plan_times[-1]:.3f}s")
        if realtime:
            n_run = int(round(min(max(plan_times[-1], dt), N_CTRL * dt) / dt))
            if prev_params is None:
                us = np.zeros((n_run, model.nu))
            else:
                us = planner.controls(prev_params, t_plan_start, N_CTRL, dt)[:n_run]
            prev_params = params
        else:
            us = planner.controls(params, data.time, n_sub, dt)
        run_controls(us)
        if st["stopped"]:
            break

    hist.append(np.concatenate([[data.time], data.qpos]))
    pe, ae = scene.errors(data.qpos)
    res = Result(algo=planner.algo, seed=seed, success=t_conv is not None, t_converged=t_conv, sim_time=data.time,
                 wall_time=time.time() - wall0, final_pos_mm=pe * 1000, final_ang_deg=ae, idle_s=st["idle_s"],
                 plan_times=plan_times, box_hits=st["box_hits"], stopped=st["stopped"], skipped=st["skipped"],
                 min_pos_mm=min(min_pos, pe) * 1000, min_ang_deg=min(min_ang, ae), qpos_hist=np.array(hist),
                 pusher_base=np.array(pb_hist))
    if viewer is not None:
        viewer.update(data, "FINAL  ")
    return res
