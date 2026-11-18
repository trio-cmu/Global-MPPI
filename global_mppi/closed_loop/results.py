"""Saving results: result.json, qpos_history.npy, waypoints_base.txt (pusher path, one point per plan)."""
import json
import os

import numpy as np


def save_run(out_dir: str, res, cfg: dict, args_dict: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    d = res.summary()
    d["config_algo"] = cfg["algos"][res.algo]
    d["args"] = args_dict
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(d, f, indent=1, default=float)
    np.save(os.path.join(out_dir, "qpos_history.npy"), res.qpos_hist)
    if res.pusher_base is not None and len(res.pusher_base):
        np.savetxt(os.path.join(out_dir, "waypoints_base.txt"), res.pusher_base, fmt="%.6f")
