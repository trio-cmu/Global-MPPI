#!/usr/bin/env python3
"""Replay a saved run (qpos_history.npy) in the live viewer: T, translucent goal, pusher + trail, caption. No GPU needed.

  python replay_run.py runs/tune_mppi/dial_d02_H2/seed0 [--speed 1.0] [--loop]

The window stays open at the end until it is closed (Ctrl-C also ends it).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
import numpy as np

from global_mppi.closed_loop.config import load_config, resolve_path
from global_mppi.closed_loop.scene import Scene
from global_mppi.closed_loop.viewer import Viewer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", default="configs/pusht_real.yaml")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true", help="replay again and again until the window is closed")
    a = ap.parse_args()
    H = np.load(os.path.join(a.run_dir, "qpos_history.npy"))
    try:
        r = json.load(open(os.path.join(a.run_dir, "result.json")))
        e = (r.get("args") or {}).get("entry") or {}
        label = f"REPLAY {e.get('id', os.path.basename(os.path.dirname(a.run_dir)))} seed {r['seed']} (saved run, {r['sim_time']:.1f} s)"
    except Exception:
        label = f"REPLAY {a.run_dir}"
    scene = Scene(load_config(resolve_path(a.config)))
    data = mujoco.MjData(scene.plant)
    viewer = Viewer(scene, data)
    print(f"{label}: {len(H)} frames, {H[-1, 0]:.1f} s; close the window to exit", flush=True)
    try:
        while viewer.alive():
            viewer.begin(label)
            t0 = time.time()
            for i, row in enumerate(H):
                if not viewer.alive():
                    break
                data.qpos[:] = row[1:]
                mujoco.mj_forward(scene.plant, data)
                viewer.trail(data)
                viewer.update(data, "FINAL  " if i == len(H) - 1 else "REPLAY ")
                wait = (row[0] / a.speed) - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
            if not a.loop:
                break
        viewer.hold()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
