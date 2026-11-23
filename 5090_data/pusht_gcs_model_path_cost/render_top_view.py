"""Top view video of a PushT run (closed-loop robot_pos_history or a GCS plan), with the T and pusher paths drawn and
the path cost J = L_T + 0.1 L_P + W D_final accumulated as it goes (score_path_cost.py).

    python render_top_view.py TRAJ OUT.mp4 [--title TEXT] [--slow 4] [--dt 0.02] [--w 10]

The states are set directly (kinematic replay of the recorded run, no new simulation).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
from score_path_cost import CORNERS, load, rot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("traj"); ap.add_argument("out")
ap.add_argument("--title", default="")
ap.add_argument("--slow", type=float, default=4.0, help="playback slow-down factor (default 4)")
ap.add_argument("--dt", type=float, default=0.02)
ap.add_argument("--w", type=float, default=10.0)
ap.add_argument("--wp", type=float, default=0.1)
a = ap.parse_args()

q, _, h = load(a.traj, a.dt)                       # qpos per sample: T x, y, th, pusher x, y (relative to (0, 0.1))
t = np.concatenate([[0.0], np.cumsum(h)])
P = q[:, 3:5] + np.array([0.0, 0.1])               # pusher in the world
V = np.array([q[k, :2] + CORNERS @ rot(q[k, 2]).T for k in range(len(q))])  # (N, 8, 2) T corners
dv = np.linalg.norm(np.diff(V, axis=0), axis=2).mean(1)            # mean corner step
dp = np.linalg.norm(np.diff(P, axis=0), axis=1)
still = dv <= 1e-3 * h                                             # T still: mean corner speed <= 1 mm/s
LT = np.concatenate([[0], np.cumsum(dv)]); LP = np.concatenate([[0], np.cumsum(dp * still)])
D = np.linalg.norm(V - CORNERS[None], axis=2).mean(1)              # corner distance to the goal (goal = (0, 0, 0))
J_end = LT[-1] + a.wp * LP[-1] + a.w * D[-1]

W_ = 800
M = mujoco.MjModel.from_xml_path(str(Path.home() / ".cache/global-mppi/pusht_5090/Global-MPPI-pathcost/global_mppi/models/pusht/scene.xml"))
M.vis.global_.offwidth = M.vis.global_.offheight = W_
d = mujoco.MjData(M)
r = mujoco.Renderer(M, W_, W_)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.04, 0.05, 0.0]; cam.distance, cam.azimuth, cam.elevation = 0.42, 90.0, -90.0


def _calibrate():
    """Affine map world xy -> pixel, fitted from where the (red) pusher appears when placed at known points."""
    world, pix = [], []
    for x, y in [(-0.1, 0.0), (0.15, 0.0), (0.0, 0.15), (0.12, 0.12), (-0.05, -0.05)]:
        d.qpos[:] = 0; d.qpos[:2] = [1.0, 1.0]                  # T out of view
        d.qpos[3:5] = [x, y - 0.1]
        mujoco.mj_forward(M, d); r.update_scene(d, cam)
        im = r.render().astype(int)
        red = (im[..., 0] > 150) & (im[..., 1] < 80) & (im[..., 2] < 80)
        ys, xs = np.nonzero(red)
        if len(xs) > 20:
            world.append([x, y, 1.0]); pix.append([xs.mean(), ys.mean()])
    A, *_ = np.linalg.lstsq(np.array(world), np.array(pix), rcond=None)
    return A


A_MAP = _calibrate()


def project(pts):
    return np.column_stack([pts, np.ones(len(pts))]) @ A_MAP


fps = 30
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_}x{W_}",
                       "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", a.out],
                      stdin=subprocess.PIPE)
com_world = np.array([q[k, :2] + rot(q[k, 2]) @ np.array([0, .012857]) for k in range(len(q))])
for tv in np.arange(0.0, t[-1] * a.slow + 1.5, 1.0 / fps):        # + 1.5 s holding the end
    tc = min(tv / a.slow, t[-1])
    k = int(np.searchsorted(t, tc, side="right") - 1)
    d.qpos[:5] = [np.interp(tc, t, q[:, i]) for i in range(5)]
    mujoco.mj_forward(M, d)
    r.update_scene(d, cam)
    img = Image.fromarray(r.render()); g = ImageDraw.Draw(img)
    pp = project(P[:k + 1]); cc = project(com_world[:k + 1])
    if k > 0:
        g.line([tuple(x) for x in cc], fill=(230, 120, 20), width=3)
        g.line([tuple(x) for x in pp], fill=(30, 90, 220), width=2)
    J_now = LT[k] + a.wp * LP[k] + a.w * D[k]
    g.rectangle([0, 0, W_, 88], fill=(255, 255, 255))
    g.text((10, 6), f"{a.title}   top view, {a.slow:g}x slower   t = {tc:4.2f} s / {t[-1]:.2f} s", fill=(0, 0, 0))
    g.text((10, 26), f"T path L_T {LT[k]:.3f} m   pusher path between pushes L_P {LP[k]:.3f} m   "
                     f"corner distance to goal D {1000 * D[k]:.1f} mm", fill=(0, 0, 0))
    g.text((10, 46), f"J so far = L_T + {a.wp:g} L_P + {a.w:g} D = {J_now:.3f}      final J = {J_end:.3f}", fill=(0, 0, 0))
    g.text((10, 66), "orange: T COM path   blue: pusher path   green: goal", fill=(90, 90, 90))
    ff.stdin.write(np.asarray(img).tobytes())
ff.stdin.close(); ff.wait()
print(f"final J {J_end:.4f} (L_T {LT[-1]:.4f}, L_P {LP[-1]:.4f}, D {D[-1]:.5f}); saved {a.out}")
