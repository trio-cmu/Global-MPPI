#!/usr/bin/env python3
"""Sampling-planner tuning (mppi / dial / ...) with ONE live viewer window that stays open, fed by a queue file.

  python tune_mppi.py [--queue runs/tune_mppi/queue.yaml] [--out runs/tune_mppi]

The queue is a yaml list; entries appended while the script runs are picked up (the window stays open and idles in
between). One entry = one configuration; it runs for each of its `seeds` (default [0]) for loop.max_time_s (60 s of SIM time, no early
stop; Esc in the window skips the current run). The sim runs with the plan-latency model (loop.realtime: it keeps
going while the planner computes, the new plan takes effect when it is ready), so sim time equals real robot time, the window lingers 2 s after each run.
  - id: t1
    algo: mppi                                  # mppi (default) / dial / ps
    params: {noise_level: 0.05, temperature: 0.01, plan_horizon: 1.0, iterations: 2, num_samples: 256}   # noise_level
          # is absolute [m/s]; noise_level_vmax (a multiple of vmax) is also accepted; omitted keys keep the yaml values
    w_approach: 0
    pusher_start: [-0.3346, -0.0782]    # optional, goal frame [m]: this entry only (the yaml start stays the standard)
    seeds: [0]
Each run saves a process figure <out>/<id>_seed<N>.png (7 snapshots + error / distance curves), simulation.mp4, result.json, qpos_history.npy under <out>/<id>/seed<N>/, appends a row to
<out>/results.csv and writes <out>/<id>_seed<N>.txt (a short report). Create <out>/STOP to end the script, <out>/SKIP to skip the current run (same as Esc).
"""
import argparse
import copy
import csv
import math
import os
import subprocess
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__GL_SYNC_TO_VBLANK", "0")   # no vsync (NVIDIA GLX): the window must not pace the simulation
os.environ.setdefault("vblank_mode", "0")             # same for Mesa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml

from global_mppi.closed_loop.algos import Planner
from global_mppi.closed_loop.config import load_config, resolve_path
from global_mppi.closed_loop.results import save_run
from global_mppi.closed_loop.runner import run_episode
from global_mppi.closed_loop.scene import Scene
from global_mppi.closed_loop.video import write_video
from global_mppi.closed_loop.viewer import NullViewer, Viewer

FIELDS = ["config_id", "algo", "seed", "num_samples", "noise_level", "temperature", "plan_horizon", "iterations", "num_knots",
          "beta_opt_iter", "beta_horizon",
          "w_approach", "final_pos_mm", "final_ang_deg", "min_pos_mm", "min_ang_deg", "t_first_conv_s", "t_enter_tol_s", "max_pos_after_enter_mm", "plan_mean_s",
          "plan_p95_s", "idle_s", "t_T_moved_s", "T_path_mm", "box_hits", "sim_time_s", "wall_s", "skipped", "video"]


def diagnostics(res) -> dict:
    """T first moved / travelled; first time inside the tolerance (10 mm and 3 deg) and the largest position error after it."""
    h = res.qpos_hist
    pos = np.linalg.norm(h[:, 1:3], axis=1) * 1000
    ang = np.abs(np.degrees((h[:, 3] + np.pi) % (2 * np.pi) - np.pi))
    inside = np.where((pos < 10.0) & (ang < 3.0))[0]
    t_enter = float(h[inside[0], 0]) if len(inside) else None
    overshoot = float(pos[inside[0]:].max()) if len(inside) else None
    d = np.linalg.norm(h[:, 1:3] - h[0, 1:3], axis=1)
    moved = np.where(d > 0.002)[0]
    path = float(np.sum(np.linalg.norm(np.diff(h[:, 1:3], axis=0), axis=1))) * 1000
    return {"t_T_moved_s": float(h[moved[0], 0]) if len(moved) else None, "T_path_mm": path,
            "t_enter_tol_s": t_enter, "overshoot_mm": overshoot}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="runs/tune_mppi/queue.yaml")
    ap.add_argument("--config", default="configs/pusht_real.yaml")
    ap.add_argument("--out", default="runs/tune_mppi")
    ap.add_argument("--duration", type=float, default=60.0, help="simulation time of every run [s]")
    ap.add_argument("--headless", action="store_true", help="no window; process the queue once and exit")
    ap.add_argument("--linger", type=float, default=2.0)
    args = ap.parse_args()

    base = load_config(resolve_path(args.config),
                       [f"loop.max_time_s={args.duration}", "loop.realtime=true", "loop.run_to_time=true", "loop.stall_exit_s=0"])
    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(args.queue):
        open(args.queue, "w").write("[]\n")
    csv_path = os.path.join(args.out, "results.csv")
    new_file = not os.path.exists(csv_path)
    scene = Scene(base)
    data = scene.new_data()
    viewer = NullViewer() if args.headless else Viewer(scene, data)
    viewer.skip_file = os.path.join(args.out, "SKIP")
    viewer.begin("tuner idle: waiting for the next configuration")
    done = 0
    print("tuner ready; window open; waiting for entries in", args.queue, flush=True)
    while not os.path.exists(os.path.join(args.out, "STOP")):
        if not viewer.alive():
            viewer = Viewer(scene, data)
            viewer.skip_file = os.path.join(args.out, "SKIP")
            viewer.begin("tuner idle: waiting for the next configuration")
        try:
            queue = yaml.safe_load(open(args.queue)) or []
        except Exception:
            queue = []
        if done >= len(queue):
            if args.headless:
                break
            viewer.v.sync()
            time.sleep(0.3)
            continue
        c = queue[done]
        done += 1
        cfg = copy.deepcopy(base)
        algo = c.get("algo", "mppi")
        m = dict(c.get("params") or c.get("mppi") or {})
        if "noise_level" in m:
            cfg["algos"][algo].pop("noise_level_vmax", None)
        cfg["algos"][algo].update(m)
        wa = float(c.get("w_approach", 0.0))
        scene.start_P = np.array(c["pusher_start"], float) if "pusher_start" in c else scene.start_P_default.copy()
        scene.set_cost(cfg["cost"]["w_pos"], cfg["cost"]["w_ori"], wa)
        viewer.begin(f"{c['id']}: compiling the planner ...")
        viewer.update(data)
        planner = Planner(scene, algo, cfg)
        k = planner.kwargs
        print(f"\n##### {c['id']}  {k}  w_approach={wa}", flush=True)
        print(f"compile {planner.warmup(scene.new_data()):.1f}s", flush=True)
        for seed in c.get("seeds", [0]):
            if not viewer.alive():
                viewer = Viewer(scene, data)
                viewer.skip_file = os.path.join(args.out, "SKIP")
            label = (f"{c['id']}  {algo}  seed {seed}  |  noise={k['noise_level']:g} T={k['temperature']:g} H={k['plan_horizon']:g} "
                     f"it={k.get('iterations', 1)} N={k['num_samples']} K={k['num_knots']} appr={wa:g}"
                     + (f" beta_opt={k['beta_opt_iter']:g} beta_h={k['beta_horizon']:g}" if algo == "dial" else "")
                     + (f" pstart=({scene.start_P[0]:.3f},{scene.start_P[1]:.3f})" if "pusher_start" in c else ""))
            print("===", label, flush=True)
            res = run_episode(scene, planner, seed, cfg, data, viewer=viewer, label=label, log=print)
            d = os.path.join(args.out, c["id"], f"seed{seed}")
            save_run(d, res, cfg, {"entry": c})
            vid = os.path.join(d, "simulation.mp4")
            write_video(scene, res.qpos_hist, vid, caption=label)
            s, dg = res.summary(), diagnostics(res)
            row = {"config_id": c["id"], "algo": algo, "seed": seed, "num_samples": k["num_samples"], "noise_level": round(k["noise_level"], 5),
                   "temperature": k["temperature"], "plan_horizon": k["plan_horizon"], "iterations": k.get("iterations", 1),
                   "num_knots": k["num_knots"], "beta_opt_iter": k.get("beta_opt_iter", ""), "beta_horizon": k.get("beta_horizon", ""), "w_approach": wa, "final_pos_mm": round(s["final_pos_mm"], 2),
                   "final_ang_deg": round(s["final_ang_deg"], 2), "min_pos_mm": round(s["min_pos_mm"], 2),
                   "min_ang_deg": round(s["min_ang_deg"], 2),
                   "t_first_conv_s": "" if s["t_converged"] is None else round(s["t_converged"], 1),
                   "t_enter_tol_s": "" if dg["t_enter_tol_s"] is None else round(dg["t_enter_tol_s"], 1),
                   "max_pos_after_enter_mm": "" if dg["overshoot_mm"] is None else round(dg["overshoot_mm"], 1),
                   "plan_mean_s": round(s["plan_mean_s"], 3), "plan_p95_s": round(s["plan_p95_s"], 3),
                   "idle_s": "" if s["idle_s"] is None else round(s["idle_s"], 2),
                   "t_T_moved_s": "" if dg["t_T_moved_s"] is None else round(dg["t_T_moved_s"], 1),
                   "T_path_mm": round(dg["T_path_mm"], 1), "box_hits": s["box_hits"], "sim_time_s": round(s["sim_time"], 1),
                   "wall_s": round(s["wall_time"]), "skipped": s["skipped"], "video": vid}
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                    new_file = False
                w.writerow(row)
            png = os.path.join(args.out, f"{c['id']}_seed{seed}.png")
            subprocess.run(["/usr/bin/python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_mppi", "closed_loop", "plots.py"),
                            "--hist", os.path.join(d, "qpos_history.npy"), "--out", png, "--label", label,
                            "--anchor", *[str(v) for v in scene.anchor]], check=False)
            report = (f"{c['id']} seed {seed}: sim {row['sim_time_s']} s{' (SKIPPED)' if row['skipped'] else ''}\n"
                      f"  at the end: position {row['final_pos_mm']} mm, heading {row['final_ang_deg']} deg\n"
                      f"  minimum   : position {row['min_pos_mm']} mm, heading {row['min_ang_deg']} deg\n"
                      f"  plan time : mean {row['plan_mean_s']} s, p95 {row['plan_p95_s']} s\n"
                      f"  pusher moved after {row['idle_s']} s; T first moved at {row['t_T_moved_s'] or '-'} s; T travelled "
                      f"{row['T_path_mm']} mm; box hits {row['box_hits']}; first convergence {row['t_first_conv_s'] or '-'}\n")
            open(os.path.join(args.out, f"{c['id']}_seed{seed}.txt"), "w").write(report)
            print(report, flush=True)
            viewer.linger(args.linger)
        viewer.begin("tuner idle: waiting for the next configuration")
    viewer.hold()
    return 0


if __name__ == "__main__":
    sys.exit(main())
