"""Top-down video of a closed-loop pusht.py run from its saved state history (qpos once per 20 ms cycle),
with the same camera and overlay as render_gcs_video.py.

    MUJOCO_GL=egl python render_run_video.py <robot_pos_history.npy> <out.mp4> --title "PS, GCS model, seed 0"
"""

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from global_mppi import ROOT as GM_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("history", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--title", default="")
parser.add_argument("--cycle", type=float, default=0.02, help="time between saved states (s)")
parser.add_argument("--playback", type=float, default=0.25, help="video speed relative to real time")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--size", type=int, default=720)
args = parser.parse_args()

Q = np.load(args.history)
model = mujoco.MjModel.from_xml_path(GM_ROOT + "/models/pusht/scene.xml")
model.vis.global_.offwidth = model.vis.global_.offheight = args.size
data = mujoco.MjData(model)
BLOCK = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
renderer = mujoco.Renderer(model, args.size, args.size)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.04, 0.04, 0.0]
cam.distance, cam.azimuth, cam.elevation = 0.42, 90.0, -90.0

ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                       f"{args.size}x{args.size}", "-r", str(args.fps), "-i", "-", "-pix_fmt", "yuv420p",
                       "-vcodec", "libx264", "-crf", "23", str(args.out)], stdin=subprocess.PIPE)
T = np.arange(len(Q)) * args.cycle
n_frames = int(T[-1] / args.playback * args.fps) + 1 + args.fps  # hold the last state 1 s
for f in range(n_frames):
    t = min(f * args.playback / args.fps, T[-1])
    data.qpos[:] = [np.interp(t, T, Q[:, k]) for k in range(Q.shape[1])]
    mujoco.mj_kinematics(model, data)
    renderer.update_scene(data, cam)
    img = Image.fromarray(renderer.render())
    (x, y), th = data.xpos[BLOCK][:2], data.qpos[2]
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, args.size, 46], fill=(255, 255, 255))
    d.text((10, 6), f"{args.title}   t = {t:4.2f} s / {T[-1]:.2f} s   ({args.playback:g}x speed)", fill=(0, 0, 0))
    d.text((10, 24), f"white = block, red = pusher, green = goal;  block off the goal by {100 * np.hypot(x, y):4.1f} cm, "
           f"{np.degrees((th + np.pi) % (2 * np.pi) - np.pi):6.1f} deg", fill=(0, 0, 0))
    ff.stdin.write(np.asarray(img).tobytes())
ff.stdin.close()
ff.wait()
print("RESULT saved", args.out, "final block", [*data.xpos[BLOCK][:2].round(4), round(float(data.qpos[2]), 3)])
