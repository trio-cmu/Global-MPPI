"""Sweep contact stiffness (soft -> hard) with `margin` fixed at 0.1.

Stiffness is controlled directly here via solref's "direct" mode: a NEGATIVE
first value switches solref from the implicit (timeconst, dampratio) form to
literal stiffness/damping coefficients, `solref="-stiffness -damping"` (see
the MuJoCo docs on solver parameters). This gives an explicit physical
stiffness knob instead of an indirect time constant. The paired `solimp`
impedance shape (dmin/dmax/width) is swept alongside it, since a very low
stiffness needs a correspondingly soft/wide impedance shape to actually show
an effect. `margin` (the pre-contact "force at a distance" range) is held
fixed at every level, instead of changing margin and stiffness at once.
"""

import os

# Headless offscreen rendering (for video recording) needs a GL backend
# selected before mujoco is imported. EGL uses the GPU; fall back to
# `osmesa` (CPU) by setting MUJOCO_GL yourself if EGL is unavailable.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import imageio


def make_1d_push_xml(
    margin=0.001,
    solref="0.005 1",
    solimp="0.95 0.999 0.001 0.5 2",
):
    return f"""
<mujoco model="one_d_push_box">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 0" integrator="RK4"/>

  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="64" height="64"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="100" height="100"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
  </asset>

  <default>
    <geom
      friction="1 0.005 0.0001"
      margin="{margin}"
      solref="{solref}"
      solimp="{solimp}"/>
  </default>

  <worldbody>
    <!-- size="0 0 ..." makes this an unbounded plane, so a fixed camera near
         the ground never sees "past its edge" into empty (black) void. -->
    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <!-- contype/conaffinity=0: the pusher and box only have an x-slide DOF
         (no vertical freedom), so they can never actually fall through the
         floor - floor contact is purely redundant. It's disabled here so the
         hardness sweep (which sets margin/solref/solimp via <default>) only
         affects the pusher<->box contact we actually care about; leaving it
         enabled means the floor contact gets swept too, and at the hard end
         (near-rigid, narrow solimp width) that redundant/over-constrained
         contact pair was what made the pusher itself stick and stop moving. -->
    <geom name="ground" type="plane" size="0 0 0.1" material="groundplane"
          contype="0" conaffinity="0"/>

    <!-- Fixed side-on camera framing the whole x-travel of the pusher/box -->
    <camera name="track" pos="0.3 -1.1 0.35" xyaxes="1 0 0 0 0.25 1"/>

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
    video_path=None,
    video_width=640,
    video_height=360,
    camera="track",
    push_duration_after_contact=None,
):
    """
    u_sequence: array of desired pusher positions.

    If video_path is given, records one frame per entry of u_sequence
    (offscreen, no display needed) and writes an .mp4 there; playback fps
    is set so the video runs at the same rate as the simulated time
    (1 / (steps_per_control * timestep)).

    If push_duration_after_contact (seconds) is given, the pusher keeps
    following u_sequence normally until contact is first detected (ncon>0,
    which - given `margin` - can be slightly before the geoms actually
    touch), then keeps advancing for that many more seconds before freezing
    its commanded position at whatever it reached (instead of continuing to
    ramp all the way to u_sequence's end / the actuator's ctrlrange limit).
    """
    mujoco.mj_resetData(model, data)

    pusher_qpos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pusher_slide")
    box_qpos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_slide")

    # Joint qpos addresses
    pusher_qadr = model.jnt_qposadr[pusher_qpos_id]
    box_qadr = model.jnt_qposadr[box_qpos_id]

    # Start at rest at the first commanded position: without this, the
    # pusher starts at qpos=0 while the position actuator immediately
    # targets u_sequence[0], so kp snaps it there on step one (a visible
    # jump/impulse at the start instead of starting at rest).
    data.qpos[pusher_qadr] = u_sequence[0]
    data.qvel[:] = 0.0
    data.ctrl[0] = u_sequence[0]
    mujoco.mj_forward(model, data)

    pusher_x_hist = []
    box_x_hist = []
    ctrl_hist = []
    contact_hist = []

    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(model, data)

    renderer = None
    writer = None
    if video_path is not None:
        renderer = mujoco.Renderer(model, height=video_height, width=video_width)
        fps = 1.0 / (steps_per_control * model.opt.timestep)
        writer = imageio.get_writer(
            video_path, format="FFMPEG", fps=fps, codec="libx264", quality=8
        )

    contact_step = None  # outer-loop index where contact was first detected
    frozen_u = None  # once set, overrides u_sequence for the rest of the rollout
    stop_steps = None
    if push_duration_after_contact is not None:
        stop_steps = int(round(push_duration_after_contact / (steps_per_control * model.opt.timestep)))

    for i, u in enumerate(u_sequence):
        if frozen_u is not None:
            u = frozen_u
        data.ctrl[0] = u

        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)

            pusher_x_hist.append(data.qpos[pusher_qadr])
            box_x_hist.append(data.qpos[box_qadr])
            ctrl_hist.append(u)
            contact_hist.append(data.ncon)

            if data.ncon > 0 and contact_step is None:
                contact_step = i

            if render:
                viewer.sync()

        if (
            stop_steps is not None
            and contact_step is not None
            and frozen_u is None
            and i - contact_step >= stop_steps
        ):
            frozen_u = u  # stop advancing: hold the pusher here from now on

        if writer is not None:
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())

    if render and viewer is not None:
        try:
            input("Press Enter to close the viewer...")
        except EOFError:
            pass
        viewer.close()

    if writer is not None:
        writer.close()
        renderer.close()
        print(f"Video saved to {video_path}")

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


# ----------------------------------------------------------------------
# Soft <-> hard contact parameter sweep
# ----------------------------------------------------------------------
# `margin` is held fixed at 0.1 here; only contact stiffness is swept, using
# solref's "direct" mode: a NEGATIVE first value switches solref from the
# implicit (timeconst, dampratio) formulation to literal, direct stiffness
# and damping coefficients: solref="-stiffness -damping" (MuJoCo docs). This
# gives an explicit physical stiffness knob instead of an indirect time
# constant. Damping tracks stiffness at every level (~critically damped,
# 2*sqrt(stiffness)) rather than being held fixed; solimp (dmin/dmax/width)
# is still swept alongside it since a very low stiffness needs a
# correspondingly soft/wide impedance shape to actually show an effect (a
# narrow, high-dmax solimp saturates to "rigid" almost immediately
# regardless of the underlying spring stiffness).
SOFT_PARAMS = {"margin": 0.1, "stiffness": 1.12, "dmin": 0.05, "dmax": 0.9, "width": 0.4}
HARD_PARAMS = {"margin": 0.0, "stiffness": 13.0, "dmin": 0.5, "dmax": 0.95, "width": 0.2}


def _lerp(soft, hard, t):
    return soft + (hard - soft) * t


def build_model_for_hardness(t):
    """Build a model/data pair for hardness level t in [0, 1] (0=soft, 1=hard)."""
    margin = _lerp(SOFT_PARAMS["margin"], HARD_PARAMS["margin"], t)
    stiffness = _lerp(SOFT_PARAMS["stiffness"], HARD_PARAMS["stiffness"], t)
    damping = 2.0 * stiffness**0.5  # ~critically damped for this stiffness
    dmin = _lerp(SOFT_PARAMS["dmin"], HARD_PARAMS["dmin"], t)
    dmax = _lerp(SOFT_PARAMS["dmax"], HARD_PARAMS["dmax"], t)
    width = _lerp(SOFT_PARAMS["width"], HARD_PARAMS["width"], t)

    xml = make_1d_push_xml(
        margin=margin,
        solref=f"-{stiffness:.5f} -{damping:.5f}",
        solimp=f"{dmin:.4f} {dmax:.4f} {width:.5f} 0.5 2",
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def _sequential_colors(n):
    """n colors along the sequential blue ramp, light (soft) -> dark (hard)."""
    lo = np.array([0x86, 0xB6, 0xEF], dtype=float)  # #86b6ef
    hi = np.array([0x0D, 0x36, 0x6B], dtype=float)  # #0d366b
    return [
        "#%02x%02x%02x" % tuple((lo + (hi - lo) * i / (n - 1)).round().astype(int))
        for i in range(n)
    ] if n > 1 else ["#2a78d6"]


# How many levels to sweep. Only the count changes here; the range itself
# is set by SOFT_PARAMS/HARD_PARAMS above.
NUM_LEVELS = 5
SWEEP_COLORS = _sequential_colors(NUM_LEVELS)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE = "#c3c2b7"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for name, spine in ax.spines.items():
        if name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(BASELINE)
            spine.set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9.5)


def plot_hardness_sweep(box_histories, hardness_levels, colors, timestep, save_path=None, extra_series=None):
    """Plot box position over time, one line per hardness level (soft->hard).

    extra_series: optional list of {"label", "color", "box_x"} dicts for
    reference cases plotted alongside the sweep (e.g. the real PushT task's
    contact settings), drawn dashed so they read as distinct from the sweep.
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # A legend with one entry per line only reads well for a handful of
    # levels; past that, label just the two endpoints and rely on the
    # light->dark color ramp itself to communicate soft->hard.
    show_full_legend = len(colors) <= 8
    for i, (t, color) in enumerate(zip(hardness_levels, colors)):
        box_x = box_histories[t]
        time_steps = np.arange(len(box_x)) * timestep
        stiffness = _lerp(SOFT_PARAMS["stiffness"], HARD_PARAMS["stiffness"], t)
        margin = _lerp(SOFT_PARAMS["margin"], HARD_PARAMS["margin"], t)
        if show_full_legend:
            label = f"k={stiffness:.2f}, margin={margin:.3f}"
        elif i == 0:
            label = f"k={stiffness:.2f}, margin={margin:.3f} (softest)"
        elif i == len(colors) - 1:
            label = f"k={stiffness:.2f}, margin={margin:.3f} (hardest)"
        else:
            label = None
        ax.plot(
            time_steps, box_x, color=color, linewidth=2, solid_capstyle="round",
            label=label, zorder=3,
        )

    for series in extra_series or []:
        box_x = series["box_x"]
        time_steps = np.arange(len(box_x)) * timestep
        ax.plot(
            time_steps, box_x, color=series["color"], linewidth=2.5,
            linestyle=(0, (5, 2)), solid_capstyle="round",
            label=series["label"], zorder=4,
        )

    ax.set_xlabel("Time (s)", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("Box position x", color=INK_SECONDARY, fontsize=11)
    ax.set_title(
        "Contact stiffness + margin sweep (direct mode): box position vs time",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlim(5, 30)
    _style_axes(ax)
    ax.legend(
        loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
        title="stiffness k, margin (soft → hard)", title_fontsize=9.5,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, facecolor=SURFACE)
    return fig


def plot_summary(hardness_levels, final_positions, costs, save_path=None):
    """Plot final box position and terminal cost as a function of hardness."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), dpi=200, sharex=True)
    fig.patch.set_facecolor(SURFACE)

    ax1.plot(
        hardness_levels, final_positions, color=SWEEP_COLORS[1], linewidth=2,
        marker="o", markersize=6, solid_capstyle="round", zorder=3,
    )
    ax1.set_ylabel("Final box position x", color=INK_SECONDARY, fontsize=11)
    ax1.set_title(
        "Final box position & cost vs stiffness + margin",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", pad=14,
    )
    _style_axes(ax1)

    ax2.plot(
        hardness_levels, costs, color=SWEEP_COLORS[3], linewidth=2,
        marker="o", markersize=6, solid_capstyle="round", zorder=3,
    )
    ax2.set_xlabel("direct stiffness (soft → hard); margin also drops 0.1→0 alongside it", color=INK_SECONDARY, fontsize=11)
    ax2.set_ylabel("Terminal cost", color=INK_SECONDARY, fontsize=11)
    _style_axes(ax2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, facecolor=SURFACE)
    return fig


if __name__ == "__main__":
    # Set to True to also record one .mp4 per level (slower - skip when you
    # only want the plots/data, e.g. for a quick high-resolution sweep).
    SAVE_VIDEOS = True

    # How long (seconds) to keep pushing after contact is first detected,
    # before the pusher stops advancing and holds its position. Set to
    # None to disable and push all the way to u_sequence's end, as before.
    PUSH_DURATION_AFTER_CONTACT = 5.0

    hardness_levels = np.linspace(0.0, 1.0, len(SWEEP_COLORS))
    sweep_colors = SWEEP_COLORS

    # Drop specific level(s) by index (0-based, into hardness_levels/
    # SWEEP_COLORS as generated above) without changing the spacing of the
    # ones that remain. Index 1 here is stiffness=4.09 in the 5-level sweep.
    EXCLUDE_INDICES = {1}
    if EXCLUDE_INDICES:
        keep = [i for i in range(len(hardness_levels)) if i not in EXCLUDE_INDICES]
        hardness_levels = hardness_levels[keep]
        sweep_colors = [sweep_colors[i] for i in keep]

    theta = np.array([-0.3, 0.95])
    U = make_time_constant_position_control(theta[0], theta[1], N=750)  # 750*20*0.002s = 30s

    box_histories = {}
    final_positions = []
    costs = []

    stiffnesses = [_lerp(SOFT_PARAMS["stiffness"], HARD_PARAMS["stiffness"], t) for t in hardness_levels]

    for i, (t, stiffness) in enumerate(zip(hardness_levels, stiffnesses)):
        model, data = build_model_for_hardness(t)
        print(f"Running rollout at stiffness={stiffness:.2f} ...")
        # One video per stiffness level, so each contact-parameter setting
        # has its own rollout to watch (not just the two extremes) - unless
        # SAVE_VIDEOS is off.
        video_path = f"contact_hardness_{i}_t{t:.2f}.mp4" if SAVE_VIDEOS else None
        out = rollout_push(
            model, data, U, render=False, video_path=video_path,
            push_duration_after_contact=PUSH_DURATION_AFTER_CONTACT,
        )
        box_histories[t] = out["box_x"]
        final_positions.append(out["box_x"][-1])
        costs.append(trajectory_cost(theta, model, data))

    for stiffness, fx, c in zip(stiffnesses, final_positions, costs):
        print(f"stiffness={stiffness:.2f}: final box x = {fx:.4f}, cost = {c:.4f}")

    # Reference case: the real PushT task's own contact settings
    # (global_mppi/models/pusht/pusht.xml), run with its exact XML values -
    # no direct-mode conversion needed since we just reuse its solref string.
    # Its solref="0.02 1" (timeconst, dampratio) is equivalent to a direct
    # stiffness/damping of k=1/(timeconst*dampratio)**2=2500, b=2/timeconst=100
    # - about 190x stiffer than this sweep's hardest tested case (k=13).
    print("Running rollout with pushT's actual contact settings (k≈2500) ...")
    pusht_model = mujoco.MjModel.from_xml_string(
        make_1d_push_xml(margin=0.0, solref="0.02 1", solimp="0.0 0.95 0.005 0.5 2")
    )
    pusht_data = mujoco.MjData(pusht_model)
    pusht_video_path = "contact_hardness_pusht_actual.mp4" if SAVE_VIDEOS else None
    pusht_out = rollout_push(
        pusht_model, pusht_data, U, render=False, video_path=pusht_video_path,
        push_duration_after_contact=PUSH_DURATION_AFTER_CONTACT,
    )
    print(f"pushT actual (k≈2500): final box x = {pusht_out['box_x'][-1]:.4f}")

    plot_hardness_sweep(
        box_histories, hardness_levels, sweep_colors, timestep=0.002,
        save_path="contact_hardness_sweep.png",
        extra_series=[{
            "label": "pushT actual (k≈2500, margin=0)",
            "color": "#eb6834",
            "box_x": pusht_out["box_x"],
        }],
    )
    plot_summary(
        stiffnesses, final_positions, costs,
        save_path="contact_hardness_summary.png",
    )
    plt.show()
