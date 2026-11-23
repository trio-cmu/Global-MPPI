"""Path costs of a PushT trajectory (time does not count). With v_i the T's 8 corners and v_i* the goal T's corners:

    L_T     = sum over samples (1/8) sum_i |v_i(k+1) - v_i(k)|        T path length (GCS's keypoint_arc_length)
    D_final = (1/8) sum_i |v_i(end) - v_i*|                           final distance of the corners to the goal
    L_P     = pusher path length over the intervals where the T is still (mean corner speed <= 1 mm/s),
              i.e. while not pushing (GCS's non-contact modes; GCS's pusher_arc_length)
    J       = L_T + w_P * L_P + w * D_final                           (w: --w, default 10; w_P: --wp, default 0)

and, for reference, J_path = ∫ ℓ ds with ℓ = 50 |p_COM - p_goal COM|^2 + 15 * 2 (1 - cos θ) and ds = the corner path.

    python score_path_cost.py TRAJ [TRAJ ...] [--w 10] [--wp 0] [--dt 0.02] [--json out.json]

TRAJ: export_gcs_traj.py output (.npz: t, qpos, block_com, com_in_body), or a closed-loop robot_pos_history .npy
(qpos per control cycle: block x, y, θ, pusher x, y; --dt apart). The goal is the body pose (0, 0, 0).
"""

import argparse
import json

import numpy as np

# T outline in the MuJoCo body frame (pusht.xml: "bottom" box half-size .01 x .04 at the origin,
# "top" box half-size .03 x .01 at y = .03); the same 8 vertices, in the same order, as GCS's TPusher2d
CORNERS = np.array([[-.03, .04], [.03, .04], [.03, .02], [.01, .02],
                    [.01, -.04], [-.01, -.04], [-.01, .02], [-.03, .02]])
COM_IN_BODY = np.array([0., (.06 * .02 * .03) / (.02 * .08 + .06 * .02)])  # T COM in the body frame


def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def corners(q):
    return q[:, None, :2] + np.einsum("nij,kj->nki", rot(q[:, 2]), CORNERS)


def load(path, dt):
    if path.endswith(".npz"):
        z = np.load(path)
        return z["qpos"].astype(float), z["block_com"][:, :2].astype(float), np.diff(z["t"].astype(float))
    q = np.load(path).astype(float)
    return q, q[:, :2] + np.einsum("nij,j->ni", rot(q[:, 2]), COM_IN_BODY), np.full(len(q) - 1, dt)


def score(path, w, wp, dt, drop_tail=False):
    """drop_tail: leave out of L_P the pusher path after the T's last motion (the T has finished)."""
    q, com, h = load(path, dt)
    goal_corners = CORNERS  # goal body pose (0, 0, 0)
    ell = 50 * np.sum((com - COM_IN_BODY) ** 2, 1) + 15 * 2 * (1 - np.cos(q[:, 2]))
    ds = np.linalg.norm(np.diff(corners(q), axis=0), axis=2).mean(1)
    L_T = float(ds.sum())
    still = ds / h <= 1e-3  # the T is not moving: the pusher is not pushing
    if drop_tail:
        k = len(still)
        while k > 0 and still[k - 1]:
            k -= 1
        still = still.copy()
        still[k:] = False  # after the T's last motion: not counted
    L_P = float(np.linalg.norm(np.diff(q[:, 3:5], axis=0), axis=1)[still].sum())
    D_final = float(np.linalg.norm(corners(q[-1:])[0] - goal_corners, axis=1).mean())
    return {
        "J": L_T + wp * L_P + w * D_final, "w": w, "w_P": wp, "L_T_m": L_T, "L_P_m": L_P, "D_final_m": D_final,
        "J_path_ell_ds": float(np.sum(0.5 * (ell[1:] + ell[:-1]) * ds)),
        "T_com_path_m": float(np.linalg.norm(np.diff(com, axis=0), axis=1).sum()),
        "T_rotation_rad": float(np.abs(np.diff(q[:, 2])).sum()),
        "pusher_path_m": float(np.linalg.norm(np.diff(q[:, 3:5], axis=0), axis=1).sum()),
        "final_pos_err_m": float(np.linalg.norm(com[-1] - COM_IN_BODY)),
        "final_ang_err_rad": float(abs(np.arctan2(np.sin(q[-1, 2]), np.cos(q[-1, 2])))),
        "samples": int(len(q)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", nargs="+")
    ap.add_argument("--w", type=float, default=10.0, help="weight of D_final (default 10)")
    ap.add_argument("--wp", type=float, default=0.0, help="weight of L_P (default 0)")
    ap.add_argument("--dt", type=float, default=0.02, help="sample spacing of a .npy trajectory (default 0.02 s)")
    ap.add_argument("--drop-tail", action="store_true", help="do not count the pusher path after the T's last motion")
    ap.add_argument("--json")
    args = ap.parse_args()
    out = {p: score(p, args.w, args.wp, args.dt, args.drop_tail) for p in args.traj}
    for p, r in out.items():
        print(p, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
