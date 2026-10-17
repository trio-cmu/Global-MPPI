#!/usr/bin/env python3
"""Closed-loop Push-T simulation with a sampling-based planner (ps / mppi / dial / globalmppi).

  python closed_loop.py --algo mppi --config configs/pusht_real.yaml [--seed 0 1 2] [--no-viewer]
                        [--set algos.mppi.noise_level_vmax=1 --set cost.w_approach=0]
                        [--perturb] [--out runs/closed_loop]

Scene, start/goal, pusher speed, convergence criterion, box and the hyper-parameters of every algorithm come from
the yaml (configs/pusht_real.yaml); the planner model is models/pusht/scene.xml as it is. Several seeds run in one
process (the planner is compiled once). A live MuJoCo window is opened once and reused for every episode (caption: 
algorithm / seed / config); it lingers 2 s after each episode and stays open at the end until you close it
(--no-viewer to disable). Every episode saves result.json, qpos_history.npy, waypoints_base.txt and simulation.mp4.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__GL_SYNC_TO_VBLANK", "0")   # no vsync (NVIDIA GLX): the window must not pace the simulation
os.environ.setdefault("vblank_mode", "0")             # same for Mesa      # offscreen mp4 rendering; the live viewer window is separate (glfw)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from global_mppi.closed_loop.algos import ALGOS, Planner
from global_mppi.closed_loop.config import load_config, resolve_path
from global_mppi.closed_loop.results import save_run
from global_mppi.closed_loop.runner import run_episode
from global_mppi.closed_loop.scene import Perturbation, Scene
from global_mppi.closed_loop.video import write_video
from global_mppi.closed_loop.viewer import Viewer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=list(ALGOS), required=True)
    ap.add_argument("--config", default="configs/pusht_real.yaml")
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config value, e.g. --set algos.mppi.temperature=0.01")
    ap.add_argument("--pusher-start", type=float, nargs=2, metavar=("X", "Y"), default=None,
                    help="TEMPORARY pusher start in the goal frame [m] for this run only (the yaml start is not changed)")
    ap.add_argument("--no-viewer", action="store_true", help="do not open the live window")
    ap.add_argument("--linger", type=float, default=2.0, help="seconds the window stays on the finished episode")
    ap.add_argument("--perturb", action="store_true",
                    help="start perturbation per seed: T +-2 mm / +-1 deg, pusher +-2 mm (random signs, seeded)")
    ap.add_argument("--out", default=None, help="output directory (default runs/closed_loop/<algo>_<time>)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(resolve_path(args.config), args.overrides)
    scene = Scene(cfg)
    if args.pusher_start is not None:
        scene.start_P = np.array(args.pusher_start)
        print(f"pusher start overridden for this run: {args.pusher_start} (goal frame)", flush=True)
    planner = Planner(scene, args.algo, cfg)
    print(f"algo={args.algo} kwargs={planner.kwargs} w_approach={scene.task.w_approach}", flush=True)
    t_comp = planner.warmup(scene.new_data())
    print(f"compile {t_comp:.1f}s", flush=True)

    out_root = args.out or os.path.join("runs", "closed_loop", f"{args.algo}_{time.strftime('%Y%m%d_%H%M%S')}")
    data = scene.new_data()
    viewer = None if args.no_viewer else Viewer(scene, data)
    cfg_label = " ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in planner.kwargs.items()
                         if k not in ("num_randomizations", "spline_type"))
    results = []
    for seed in args.seed:
        pert = None
        if args.perturb:
            sg = np.random.default_rng(1000 + seed).choice([-1.0, 1.0], size=5)
            pert = Perturbation(dT=(sg[0] * 0.002, sg[1] * 0.002, sg[2] * np.radians(1.0)), dP=(sg[3] * 0.002, sg[4] * 0.002))
        label = f"{args.algo}  seed {seed}  |  {cfg_label}  w_approach={scene.task.w_approach:g}"
        print(f"=== {label}", flush=True)
        res = run_episode(scene, planner, seed, cfg, data, pert, viewer=viewer, label=label, log=print)
        results.append(res)
        s = res.summary()
        print(f"seed {seed}: success={s['success']} t_conv={s['t_converged']} sim={s['sim_time']:.1f}s wall={s['wall_time']:.0f}s "
              f"final {s['final_pos_mm']:.1f} mm / {s['final_ang_deg']:.2f} deg  plan mean {s['plan_mean_s']:.3f}s "
              f"p95 {s['plan_p95_s']:.3f}s  plans {s['n_plans']} (planning total {s['plan_mean_s'] * s['n_plans']:.1f}s)  idle {s['idle_s']}s  box hits {s['box_hits']} {s['stopped']}", flush=True)
        if not args.no_save:
            d = os.path.join(out_root, f"seed{seed}")
            save_run(d, res, cfg, vars(args))
            write_video(scene, res.qpos_hist, os.path.join(d, "simulation.mp4"), caption=label)
        if viewer is not None:
            viewer.linger(args.linger)
            if not viewer.alive():
                print("viewer closed: stopping")
                break
    print(f"success {sum(r.success for r in results)}/{len(results)}")
    if viewer is not None and viewer.alive():
        print("window stays open; close it or press Ctrl-C to exit", flush=True)
        viewer.hold()
    return 0


if __name__ == "__main__":
    sys.exit(main())
