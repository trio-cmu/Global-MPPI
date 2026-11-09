"""Video of a GCS plan executed open loop in MuJoCo PushT (as run_gcs_open_loop.py, speed-up 1), seen from
above: the real block (MuJoCo) and, translucent orange, where the GCS plan expects the block and the pusher.

    MUJOCO_GL=egl python render_gcs_video.py                                   # matched-parameter plan
    MUJOCO_GL=egl python render_gcs_video.py --plan gcs_global_mppi_traj.npz --tag demo
    MUJOCO_GL=egl python render_gcs_video.py --plan gcs_global_mppi_traj.npz --tag gcs_model --gcs-model

Writes gcs_open_loop_<tag>.mp4 (played --playback times faster than real time).
"""

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from global_mppi import ROOT as GM_ROOT

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--plan", type=Path, default=here / "gcs_matched_traj.npz")
parser.add_argument("--tag", default="matched")
parser.add_argument("--gcs-model", action="store_true", help="GCS's own quasi-static model (tasks/pusht_gcs.py)")
parser.add_argument("--kp", type=float, default=100.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--playback", type=float, default=2.0, help="video speed relative to real time")
parser.add_argument("--size", type=int, default=720)
args = parser.parse_args()

plan = np.load(args.plan)
T, P, B = plan["t"], plan["pusher"], plan["block_body"]
V = np.gradient(P, T, axis=0)
V[-1] = 0.0
PUSHER_BODY_POS = np.array([0.0, 0.1])

model = mujoco.MjModel.from_xml_path(GM_ROOT + "/models/pusht/scene.xml")
model.opt.timestep, model.opt.iterations, model.opt.ls_iterations = 0.001, 100, 50
model.vis.global_.offwidth = model.vis.global_.offheight = args.size
data = mujoco.MjData(model)
data.qpos[:] = [0.1, 0.1, 1.3, 0.0, 0.0]
if args.gcs_model:
    from global_mppi.tasks.pusht_gcs import PushTGCS
    gcs_task = PushTGCS()
BLOCK = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, args.size, args.size)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.04, 0.04, 0.0]
cam.distance, cam.azimuth, cam.elevation = 0.42, 90.0, -90.0
GHOST = np.array([1.0, 0.55, 0.0, 0.45], np.float32)  # orange; the goal in the scene is green


def interp(t, table):
    return np.array([np.interp(t, T, table[:, k]) for k in range(table.shape[1])])


def add_geom(scene, gtype, size, pos, theta, rgba):
    c, s = np.cos(theta), np.sin(theta)
    mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64).ravel()
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype, np.asarray(size, np.float64), np.asarray(pos, np.float64), mat, rgba)
    scene.ngeom += 1


def frame(t):
    renderer.update_scene(data, cam)
    b, p = interp(t, B), interp(t, P)
    c, s = np.cos(b[2]), np.sin(b[2])
    for local, half in (((0.0, 0.0), (0.01, 0.04, 0.01)), ((0.0, 0.03), (0.03, 0.01, 0.01))):  # stem, crossbar
        xy = b[:2] + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
        add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_BOX, half, [*xy, 0.03], b[2], GHOST)
    add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.01, 0, 0], [*p, 0.04], 0.0, GHOST)
    img = Image.fromarray(renderer.render())
    (x, y), th = data.xpos[BLOCK][:2], data.qpos[2]
    err_plan = np.hypot(x - b[0], y - b[1]), np.degrees((th - b[2] + np.pi) % (2 * np.pi) - np.pi)
    err_goal = np.hypot(x, y), np.degrees((th + np.pi) % (2 * np.pi) - np.pi)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, args.size, 64], fill=(255, 255, 255))
    d.text((10, 6), f"GCS open loop in {'GCS model' if args.gcs_model else 'MuJoCo'} ({args.tag})   t = {t:5.2f} s / {T[-1]:.2f} s   ({args.playback:g}x speed)",
           fill=(0, 0, 0))
    d.text((10, 24), f"white = real block, orange = where the GCS plan expects it, green = goal;  off the plan by "
           f"{100 * err_plan[0]:4.1f} cm, {err_plan[1]:6.1f} deg", fill=(0, 0, 0))
    d.text((10, 42), f"real block off the goal by {100 * err_goal[0]:4.1f} cm, {err_goal[1]:6.1f} deg", fill=(0, 0, 0))
    return np.asarray(img)


out = here / f"gcs_open_loop_{args.tag}.mp4"
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                       f"{args.size}x{args.size}", "-r", str(args.fps), "-i", "-", "-pix_fmt", "yuv420p",
                       "-vcodec", "libx264", "-crf", "23", str(out)], stdin=subprocess.PIPE)
steps_per_frame = int(round(args.playback / args.fps / model.opt.timestep))
end = T[-1] + 1.0  # one more second after the plan ends
while data.time < end:
    ff.stdin.write(frame(data.time).tobytes())
    for _ in range(steps_per_frame):
        tp = min(data.time, T[-1])
        p_ref, v_ref = interp(tp, P), (interp(tp, V) if data.time < T[-1] else np.zeros(2))
        data.ctrl[:] = np.clip(v_ref + args.kp * (p_ref - (data.qpos[3:5] + PUSHER_BODY_POS)), -1, 1)
        gcs_task.sim_step(model, data) if args.gcs_model else mujoco.mj_step(model, data)
ff.stdin.close()
ff.wait()
print("RESULT saved", out, "final block", [*data.xpos[BLOCK][:2].round(3), round(float(data.qpos[2]), 3)])
