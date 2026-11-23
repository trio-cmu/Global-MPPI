"""Export a GCS plan saved by gcs_plan.py (gcs_<TAG>_traj.pkl) to the Global-MPPI plan format, without re-solving.

    python export_from_pkl.py TAG     -> gcs_<TAG>_traj.npz (t, qpos, block_body, block_com, pusher, com_in_body, dt)
"""
import sys
from pathlib import Path

import numpy as np
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pushing_trajectory import PlanarPushingTrajectory

here = Path(__file__).resolve().parent
tag = sys.argv[1]
COM_IN_BODY = np.array([0.0, 0.06 * 0.02 * 0.03 / (0.02 * 0.08 + 0.06 * 0.02)])  # as gcs_plan.py
PUSHER_HOME = np.array([0.0, 0.1])


class PushTGeometry(TPusher2d):  # as gcs_plan.py; needed to unpickle the plan
    @property
    def com_offset(self):
        return np.array([0.0, COM_IN_BODY[1] - 0.03]).reshape((-1, 1))


traj = PlanarPushingTrajectory.load(str(here / f"gcs_{tag}_traj.pkl"))
dt = 0.01
ts = np.arange(0.0, traj.end_time + 1e-9, dt)
ts[-1] = min(ts[-1], traj.end_time)
com = np.array([np.asarray(traj.get_value(t, "p_WB")).flatten()[:2] for t in ts])
R = [np.asarray(traj.get_value(t, "R_WB"))[:2, :2] for t in ts]
th = np.unwrap([np.arctan2(r[1, 0], r[0, 0]) for r in R])
body = com - np.array([r @ COM_IN_BODY for r in R])
pusher = np.array([np.asarray(traj.get_value(t, "p_WP")).flatten()[:2] for t in ts])
qpos = np.column_stack([body, th, pusher - PUSHER_HOME])
np.savez(here / f"gcs_{tag}_traj.npz", t=ts, qpos=qpos, block_body=np.column_stack([body, th]),
         block_com=np.column_stack([com, th]), pusher=pusher, com_in_body=COM_IN_BODY, dt=dt)
print(f"duration {traj.end_time:.2f} s; start qpos {np.round(qpos[0], 4)}; final qpos {np.round(qpos[-1], 4)}")
print("saved", here / f"gcs_{tag}_traj.npz")
