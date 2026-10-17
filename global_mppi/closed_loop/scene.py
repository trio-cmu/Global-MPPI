"""Scene: task + plant model + start state + goal-frame <-> base-frame conversions + workspace box."""
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from global_mppi.closed_loop.config import resolve_path
from global_mppi.tasks.pusht import PushT


def rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Perturbation:
    """Start perturbation: T x, y [m], T theta [rad], pusher x, y [m] (goal frame)."""

    dT: tuple = (0.0, 0.0, 0.0)
    dP: tuple = (0.0, 0.0)


class Scene:
    def __init__(self, cfg: dict):
        sc = cfg["scene"]
        self.cfg = cfg
        self.xml = resolve_path(sc["xml"])
        self.task = PushT(self.xml, w_pos=cfg["cost"]["w_pos"], w_ori=cfg["cost"]["w_ori"],
                          w_approach=cfg["cost"].get("w_approach", 0.0))
        # plant: the same xml, simulation settings from the config (a separate MjModel, not a rewritten xml)
        self.plant = mujoco.MjModel.from_xml_path(self.xml)
        pl = cfg["plant"]
        self.plant.opt.timestep = pl["timestep"]
        self.plant.opt.iterations = pl["iterations"]
        self.plant.opt.ls_iterations = pl["ls_iterations"]
        # speed limit consistency with the xml
        rng = self.plant.actuator_ctrlrange
        if not np.allclose(np.abs(rng), sc["vmax"]):
            raise ValueError(f"config vmax={sc['vmax']} differs from the xml ctrlrange {rng.tolist()}")
        self.vmax = float(sc["vmax"])
        self.anchor = np.array(self.plant.body_pos[self.plant.body("pusher").id][:2])
        g = sc["goal_base"]
        self.goal_p = np.array([g["x_m"], g["y_m"]])
        self.goal_th = float(g["theta_rad"])
        b = sc["box_base"]
        self.box = (b["x_min"], b["x_max"], b["y_min"], b["y_max"])
        self.box_margin = float(b["margin_m"])
        self.start_T = np.array([sc["start"]["x_m"], sc["start"]["y_m"], math.radians(sc["start"]["theta_deg"])])
        self.start_P_default = np.array([sc["pusher_start"]["x_m"], sc["pusher_start"]["y_m"]])
        self.start_P = self.start_P_default.copy()    # a run may override it (--pusher-start); the yaml stays the standard

    def set_cost(self, w_pos: float, w_ori: float, w_approach: float) -> None:
        """Rebuild the task with other cost weights (the plant model, and a viewer bound to it, stay the same)."""
        self.task = PushT(self.xml, w_pos=w_pos, w_ori=w_ori, w_approach=w_approach)

    # -- frames
    def sim_to_base(self, p_sim) -> np.ndarray:
        return rot2(self.goal_th) @ np.asarray(p_sim, float) + self.goal_p

    def pusher_world(self, qpos) -> np.ndarray:
        return np.asarray(qpos[3:5]) + self.anchor

    def pusher_base(self, qpos) -> np.ndarray:
        return self.sim_to_base(self.pusher_world(qpos))

    # -- state
    def start_qpos(self, pert: Perturbation | None = None) -> np.ndarray:
        pert = pert or Perturbation()
        q = np.zeros(self.plant.nq)
        q[:3] = self.start_T + np.array(pert.dT)
        q[3:5] = self.start_P + np.array(pert.dP) - self.anchor
        return q

    def new_data(self, pert: Perturbation | None = None) -> mujoco.MjData:
        d = mujoco.MjData(self.plant)
        d.qpos[:] = self.start_qpos(pert)
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.plant, d)
        return d

    def reset_data(self, d: mujoco.MjData, pert: Perturbation | None = None) -> mujoco.MjData:
        """Reset an existing MjData to the start state (the viewer stays bound to the same object)."""
        mujoco.mj_resetData(self.plant, d)
        d.qpos[:] = self.start_qpos(pert)
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.plant, d)
        return d

    def errors(self, qpos) -> tuple[float, float]:
        """(position error [m], |heading error| [deg]) of the T w.r.t. the goal (the sim origin)."""
        return float(np.linalg.norm(qpos[:2])), abs(math.degrees(wrap(float(qpos[2]))))

    def clamp_to_box(self, qpos, v_sim: np.ndarray) -> tuple[np.ndarray, bool]:
        """Zero the outward component of the commanded pusher velocity (goal frame) at the box edge."""
        pb = self.pusher_base(qpos)
        v = rot2(self.goal_th) @ np.clip(v_sim, -self.vmax, self.vmax)
        hit = False
        for i, (lo, hi) in enumerate(((self.box[0], self.box[1]), (self.box[2], self.box[3]))):
            lo, hi = lo + self.box_margin, hi - self.box_margin
            if (pb[i] <= lo and v[i] < 0) or (pb[i] >= hi and v[i] > 0):
                v[i] = 0.0
                hit = True
        return rot2(-self.goal_th) @ v, hit
