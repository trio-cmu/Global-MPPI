import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import numpy as np
import time


def make_1d_push_xml(
    margin=0.001,
    solref="0.005 1",
    solimp="0.95 0.999 0.001 0.5 2",
):
    return f"""
<mujoco model="one_d_push_box">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 0" integrator="RK4"/>

  <default>
    <geom
      friction="1 0.005 0.0001"
      margin="{margin}"
      solref="{solref}"
      solimp="{solimp}"/>
  </default>

  <worldbody>
    <geom name="ground" type="plane" size="3 1 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- Pusher: constrained to move along x -->
    <body name="pusher" pos="-0.4 0 0.05">
      <joint name="pusher_slide" type="slide" axis="1 0 0" damping="0.1"/>
      <geom name="pusher_geom"
            type="box"
            size="0.05 0.15 0.05"
            rgba="0.1 0.3 0.9 1"
            mass="0.2"/>
    </body>

    <!-- Box: also constrained to move along x -->
    <body name="box" pos="0 0 0.05">
      <joint name="box_slide" type="slide" axis="1 0 0" damping="0.05"/>
      <geom name="box_geom"
            type="box"
            size="0.05 0.15 0.05"
            rgba="0.9 0.3 0.1 1"
            mass="0.1"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Position actuator for pusher joint -->
    <position name="pusher_pos"
              joint="pusher_slide"
          kp="50"
          ctrlrange="-0.6 0.8"
          ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def rollout_push(
    model,
    data,
    u_sequence,
    steps_per_control=20,
    render=False,
):
    """
    u_sequence: array of desired pusher positions.
    """
    mujoco.mj_resetData(model, data)

    pusher_qpos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pusher_slide")
    box_qpos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_slide")

    # Joint qpos addresses
    pusher_qadr = model.jnt_qposadr[pusher_qpos_id]
    box_qadr = model.jnt_qposadr[box_qpos_id]

    pusher_x_hist = []
    box_x_hist = []
    ctrl_hist = []
    contact_hist = []

    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(model, data)

    for u in u_sequence:
        data.ctrl[0] = u

        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)

            pusher_x_hist.append(data.qpos[pusher_qadr])
            box_x_hist.append(data.qpos[box_qadr])
            ctrl_hist.append(u)
            contact_hist.append(data.ncon)

            if render:
                viewer.sync()
                # time.sleep(model.opt.timestep)
                time.sleep(0.001)

    if render and viewer is not None:
        try:
            input("Press Enter to close the viewer...")
        except EOFError:
            pass
        viewer.close()

    return {
        "pusher_x": np.array(pusher_x_hist),
        "box_x": np.array(box_x_hist),
        "ctrl": np.array(ctrl_hist),
        "num_contacts": np.array(contact_hist),
    }


def make_time_constant_position_control(
    start_pos,
    end_pos,
    N=80,
):
    """Generate a direct left-to-right position target sequence."""
    return np.linspace(start_pos, end_pos, N)


def trajectory_cost(
    theta,
    model,
    data,
    target_box_x=0.8,
    control_weight=1e-3,
):
    U = make_time_constant_position_control(theta[0], theta[1], N=80)
    out = rollout_push(model, data, U, steps_per_control=20, render=False)

    final_box_x = out["box_x"][-1]
    terminal_cost = (final_box_x - target_box_x) ** 2
    control_cost = control_weight * np.sum(U**2)

    return terminal_cost


def plot_box_position(box_histories, timestep):
    """Plot box position over simulation time."""
    plt.figure(figsize=(8, 4))

    for label, box_x in box_histories.items():
        time_steps = np.arange(len(box_x)) * timestep
        plt.plot(time_steps, box_x, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Box position x")
    plt.title("Box Position vs Time")
    plt.xlim(5, 20)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.show()


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Hard contact model
    # ------------------------------------------------------------
    hard_xml = make_1d_push_xml(
        margin=0.001,
        solref="0.05 1",
        solimp="0.95 0.999 0.01 0.5 2",
        # solref="0.05 1",
        # solimp="0.7 0.95 0.03 0.5 2",
    )

    hard_model = mujoco.MjModel.from_xml_string(hard_xml)
    hard_data = mujoco.MjData(hard_model)

    # ------------------------------------------------------------
    # Smooth contact model
    # ------------------------------------------------------------
    smooth_xml = make_1d_push_xml(
        # margin=0.2,
        # solref="0.05 1",
        # solimp="0.05 0.95 0.4 0.5 2",
        
        # best parameters
        margin=0.05,
        solref="0.1 1",
        solimp="0.05 0.95 0.35 0.5 2",
    )

    smooth_model = mujoco.MjModel.from_xml_string(smooth_xml)
    smooth_data = mujoco.MjData(smooth_model)

    # Example rollout
    theta = np.array([-0.3, 0.95])
    U = make_time_constant_position_control(
        theta[0],
        theta[1],
        N=500,
    )

    print("Running hard contact rollout...")
    hard_out = rollout_push(hard_model, hard_data, U, render=False)

    print("Running smooth contact rollout...")
    smooth_out = rollout_push(smooth_model, smooth_data, U, render=False)

    print("Hard final box x:", hard_out["box_x"][-1])
    print("Smooth final box x:", smooth_out["box_x"][-1])

    print("Hard cost:", trajectory_cost(theta, hard_model, hard_data))
    print("Smooth cost:", trajectory_cost(theta, smooth_model, smooth_data))
    plot_box_position(
        {
            "hard contact": hard_out["box_x"],
            "smooth contact": smooth_out["box_x"],
        },
        hard_model.opt.timestep,
    )

    # Render one rollout
    print("Rendering smooth contact rollout...")
    rollout_push(smooth_model, smooth_data, U, render=True)
    # rollout_push(hard_model, hard_data, U, render=True)