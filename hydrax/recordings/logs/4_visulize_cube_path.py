import time
import numpy as np
import mujoco
from mujoco import viewer
from hydrax.tasks.cube import CubeRotation
from hydrax.tasks.pusht import PushT

def playback_states(mj_model, qpos_traj, qvel_traj=None, hz=60):
    """
    qpos_traj: [T, nq]
    qvel_traj: [T, nv] or None
    """
    mj_data = mujoco.MjData(mj_model)

    dt = 1.0 / hz
    with viewer.launch_passive(mj_model, mj_data) as v:
        for t in range(qpos_traj.shape[0]):
            mj_data.qpos[:] = qpos_traj[t]
            if qvel_traj is not None:
                mj_data.qvel[:] = qvel_traj[t]
            mujoco.mj_forward(mj_model, mj_data)  # updates geom/site frames
            v.sync()
            time.sleep(dt)

# Example:
task = CubeRotation()
mj_model = task.mj_model
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
# set large friction for qusial static pushing
# mj_model.geom_friction[:] = np.array([5.0, 0.005, 0.0001])
mj_data = mujoco.MjData(mj_model)

qpos_traj = np.load("cube_lse2_mppiksos_robot_pos_history_seed0.npy")   # shape [T, nq]
qvel_traj = None
playback_states(mj_model, qpos_traj, qvel_traj, hz=60)