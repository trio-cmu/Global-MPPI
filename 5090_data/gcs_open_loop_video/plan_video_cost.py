"""The raw GCS plan (no simulation): video of the planned block and pusher, and its cost

    J = sum_i h_i * pose(x_{i+1}) + W_PATH * (pusher path length while off the T) + pose(x_T)
    pose = W_POS |p_COM - goal COM|^2 + W_ANG * 2 (1 - cos th) + W_CLOSE |p_block - p_pusher|^2
           (PushTGCS.pose_cost with W_POS 50, W_ANG 15, W_CLOSE 0 = close_to_block_weight; W_PATH 10)

As global_mppi/tasks/pusht.py (pre-release): the pusher path counts only while the pusher is off the T
(gap to the T > CONTACT_TOL = 1 mm, checked at the end of each sample interval, as evaluate_pusht_cost.py).

    python plan_video_cost.py PLAN.npz TAG [W_PATH] [W_POS] [W_ANG] [W_CLOSE]   -> plan_<TAG>.mp4, plan_<TAG>_cost.json

W_PATH: pusher path penalty (default 10); W_POS: block position weight (default 50); W_ANG: angle weight (default 15);
W_CLOSE: close-to-block weight (default 0; block body origin to pusher, as pusht.py _close_to_block_err).
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from global_mppi import ROOT as GM_ROOT
from global_mppi.tasks.pusht import CONTACT_TOL, pusher_gap
from global_mppi.tasks.pusht_gcs import COM

here = Path(__file__).resolve().parent
plan_path, tag = Path(sys.argv[1]), sys.argv[2]
PATH_WEIGHT = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
POS_WEIGHT = float(sys.argv[4]) if len(sys.argv) > 4 else 50.0
ANG_WEIGHT = float(sys.argv[5]) if len(sys.argv) > 5 else 15.0
CLOSE_WEIGHT = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
FPS, SIZE = 30, 720

plan = np.load(plan_path)
T, Q = np.asarray(plan["t"], float), np.asarray(plan["qpos"], float)
P = np.asarray(plan["pusher"], float)  # pusher position in the world

# cost of the plan
com = np.asarray(COM)
c, s_ = np.cos(Q[:, 2]), np.sin(Q[:, 2])
com_err = Q[:, :2] + np.stack([c * com[0] - s_ * com[1], s_ * com[0] + c * com[1]], -1) - com  # as PushTGCS.pose_cost
close = np.sum((Q[:, :2] - (Q[:, 3:5] + np.array([0.0, 0.1]))) ** 2, 1)  # |p_block - p_pusher|^2, pusher qpos is relative to (0, 0.1)
pose = POS_WEIGHT * np.sum(com_err ** 2, 1) + ANG_WEIGHT * 2.0 * (1.0 - np.cos(Q[:, 2])) + CLOSE_WEIGHT * close
h = np.diff(T)
off = np.asarray(jax.vmap(pusher_gap)(jnp.asarray(Q[1:]))) > CONTACT_TOL
seg = np.linalg.norm(np.diff(Q[:, 3:5], axis=0), axis=1) * off  # pusher distance per interval, off the T only
running_pose_cum = np.concatenate([[0.0], np.cumsum(h * pose[1:])])
path_cum = np.concatenate([[0.0], np.cumsum(seg)])
cost = dict(plan=plan_path.name, duration_s=float(T[-1] - T[0]),
            cost=f"sum dt * ({POS_WEIGHT:g}|p-goal|^2 + {ANG_WEIGHT:g}*2(1-cos th) + {CLOSE_WEIGHT:g}|p_block-p_pusher|^2) + {PATH_WEIGHT:g} * pusher path length off the T + terminal pose cost",
            pusher_path_weight=PATH_WEIGHT, position_weight=POS_WEIGHT, angle_weight=ANG_WEIGHT, close_to_block_weight=CLOSE_WEIGHT,
            running_close_to_block=float(CLOSE_WEIGHT * np.sum(h * close[1:])),
            running_pose=float(running_pose_cum[-1]), pusher_path_off_T_m=float(path_cum[-1]),
            pusher_path_cost=float(PATH_WEIGHT * path_cum[-1]), terminal=float(pose[-1]),
            total=float(running_pose_cum[-1] + PATH_WEIGHT * path_cum[-1] + pose[-1]),
            final_qpos=Q[-1].tolist())
(here / f"plan_{tag}_cost.json").write_text(json.dumps(cost, indent=1))

# video: the plan's own poses, set directly (kinematic replay, no dynamics)
model = mujoco.MjModel.from_xml_path(GM_ROOT + "/models/pusht/scene.xml")
model.vis.global_.offwidth = model.vis.global_.offheight = SIZE
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, SIZE, SIZE)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.04, 0.04, 0.0]
cam.distance, cam.azimuth, cam.elevation = 0.42, 90.0, -90.0
out = here / f"plan_{tag}.mp4"
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                       f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p",
                       "-vcodec", "libx264", "-crf", "23", str(out)], stdin=subprocess.PIPE)
for t in np.arange(0.0, T[-1] + 1.0, 1.0 / FPS):  # plus one second holding the final pose
    tc = min(t, T[-1])
    data.qpos[:] = [np.interp(tc, T, Q[:, k]) for k in range(Q.shape[1])]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, cam)
    img = Image.fromarray(renderer.render())
    run = np.interp(tc, T, running_pose_cum) + PATH_WEIGHT * np.interp(tc, T, path_cum)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, SIZE, 64], fill=(255, 255, 255))
    d.text((10, 6), f"GCS plan (raw, no simulation): {tag}   t = {tc:5.2f} s / {T[-1]:.2f} s   "
           f"cost: {POS_WEIGHT:g}|p|^2 + {ANG_WEIGHT:g}*2(1-cos th) + {CLOSE_WEIGHT:g}|block-pusher|^2 + {PATH_WEIGHT:g} x path off T", fill=(0, 0, 0))
    d.text((10, 24), f"pusher path off the T so far {np.interp(tc, T, path_cum):.3f} m;  cost so far {run:7.3f}"
           f"  (pose {np.interp(tc, T, running_pose_cum):.3f} + {PATH_WEIGHT:g} x path off the T)", fill=(0, 0, 0))
    d.text((10, 42), f"total cost {cost['total']:.3f} = pose {cost['running_pose']:.3f} + path "
           f"{cost['pusher_path_cost']:.3f} ({PATH_WEIGHT:g} x {path_cum[-1]:.3f} m) + terminal {cost['terminal']:.3f};  green = goal", fill=(0, 0, 0))
    ff.stdin.write(np.asarray(img).tobytes())
ff.stdin.close()
ff.wait()
print(json.dumps(cost, indent=1))
print("RESULT saved", out)
