"""One evaluator for every PushT trajectory: the PushT cost (global_mppi/tasks/pusht.py) on a dense trajectory.

    J = sum_i h_i * l(x_{i+1}, v_i) + l_T(x_T)
    l   = 50 |p_COM - goal COM|^2 + 15 * 2 (1 - cos th) + 10 * |pusher velocity| * [pusher off the T]
    l_T = 50 |p_COM - goal COM|^2 + 15 * 2 (1 - cos th)

computed from qpos samples (block x, y, th, pusher x, y relative to (0, 0.1)), with the velocity over each
sample interval (as PushTGCS.step sets qvel). The pose cost is PushTGCS's (from qpos, no MuJoCo z offset), so
GCS plans and MuJoCo runs are scored alike.

    python evaluate_pusht_cost.py TRAJ.npz [--trim-start 0.75 --trim-end 0.75] [--json out.json]

TRAJ.npz holds t and qpos (export_gcs_traj.py output), or only qpos at --dt spacing (robot_pos_history .npy
of a closed-loop run, one sample per control cycle: --dt 0.02). --trim-start / --trim-end drop GCS's source
and target vertices, which last time_non_collision in its exported trajectory but carry no cost and no motion.
"""
import argparse
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from global_mppi.tasks.pusht import CONTACT_TOL, pusher_gap
from global_mppi.tasks.pusht_gcs import PushTGCS


def load(path, dt):
    if path.endswith(".npz"):
        d = np.load(path)
        return np.asarray(d["t"], float), np.asarray(d["qpos"], float)
    q = np.load(path).astype(float)
    return np.arange(len(q)) * dt, q


def evaluate(t, q, task=None, trim_start=0.0, trim_end=0.0):
    task = task or PushTGCS()
    keep = (t >= t[0] + trim_start - 1e-9) & (t <= t[-1] - trim_end + 1e-9)
    t, q = t[keep], q[keep]
    h = np.diff(t)
    v = np.diff(q, axis=0) / h[:, None]
    Q, V = jnp.asarray(q[1:]), jnp.asarray(v)
    pose = np.asarray(jax.vmap(lambda a: task.pose_cost(SimpleNamespace(qpos=a)))(Q))
    off = np.asarray(jax.vmap(pusher_gap)(Q)) > CONTACT_TOL
    pusher = task.pusher_path_weight * off * np.linalg.norm(v[:, 3:5], axis=1)
    ell = np.asarray(jax.vmap(lambda a, b: task.running_cost(SimpleNamespace(qpos=a, qvel=b), jnp.zeros(2)))(Q, V))
    assert np.allclose(pose + pusher, ell, rtol=1e-4, atol=1e-5)
    terminal = float(task.terminal_cost(SimpleNamespace(qpos=jnp.asarray(q[-1]))))
    return dict(duration=float(t[-1] - t[0]), running_pose=float(np.sum(h * pose)),
                running_pusher_path=float(np.sum(h * pusher)), running=float(np.sum(h * ell)),
                terminal=terminal, total=float(np.sum(h * ell)) + terminal,
                pusher_path_off_contact_m=float(np.sum(h * off * np.linalg.norm(v[:, 3:5], axis=1))),
                final_qpos=[float(x) for x in q[-1]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--trim-start", type=float, default=0.0)
    ap.add_argument("--trim-end", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t, q = load(a.traj, a.dt)
    r = evaluate(t, q, trim_start=a.trim_start, trim_end=a.trim_end)
    print(json.dumps(r, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
