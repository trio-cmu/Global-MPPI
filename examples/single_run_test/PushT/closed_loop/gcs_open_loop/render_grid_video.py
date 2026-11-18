"""One video with every method side by side (3 x 2 grid), top-down, on a common clock: each panel shows
its run from start to end over the same video time, so GCS's 21.75 s plan is shown faster (in a
quasi-static model the path does not depend on the speed).

    MUJOCO_GL=egl python render_grid_video.py out.mp4 \
        "PS|ps.npy|0.02" "MPPI|mppi.npy|0.02" ... "GCS (planned once)|gcs_qpos.npy|0.01"

Each argument: label|state history (qpos per row, .npy)|time between rows (s).
"""

import argparse
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from global_mppi import ROOT as GM_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("runs", nargs="+")
parser.add_argument("--title", default="PushT in GCS's quasi-static model: block from (0.1, 0.1, 1.3 rad) to the goal")
parser.add_argument("--seconds", type=float, default=8.0, help="video time for one run")
parser.add_argument("--gcs-note", default="GCS: plans once (about 120 s), its own;plan in its own model; 21.75 s long,",
                    help="lines (';'-separated) about the GCS panel in the notes cell")
parser.add_argument("--loop-note", default="100 cycles = 2 s, closed loop (seed 0).",
                    help="lines (';'-separated) about the closed-loop panels, after the first notes line")
parser.add_argument("--freeze-pusher", action="store_true",
                    help="hold the pusher still once the T has stopped moving for good (its last push)")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--cell", type=int, default=480)
args = parser.parse_args()

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
f_title, f_label, f_text = ImageFont.truetype(BOLD, 24), ImageFont.truetype(BOLD, 22), ImageFont.truetype(FONT, 17)
runs = []
for spec in args.runs:
    label, path, dt = spec.split("|")
    Q = np.load(path)
    T = np.arange(len(Q)) * float(dt)
    t_stop = None
    if args.freeze_pusher:  # last sample where the T moves: mean corner displacement > 1e-6 m
        corners = np.array([[-.03, .02], [.03, .02], [.03, .04], [-.03, .04],
                            [-.01, -.04], [.01, -.04], [.01, .02], [-.01, .02]])
        c, s_ = np.cos(Q[:, 2]), np.sin(Q[:, 2])
        P = np.stack([c[:, None] * corners[:, 0] - s_[:, None] * corners[:, 1] + Q[:, :1],
                      s_[:, None] * corners[:, 0] + c[:, None] * corners[:, 1] + Q[:, 1:2]], -1)
        moving = np.nonzero(np.linalg.norm(np.diff(P, axis=0), axis=-1).mean(1) > 1e-6)[0]
        i_stop = moving[-1] + 1 if len(moving) else 0
        t_stop = T[i_stop]
        Q = Q.copy()
        Q[i_stop:, 3:5] = Q[i_stop, 3:5]
        print(f"{label}: T stops at {t_stop:.2f} s, pusher held from then on")
    runs.append((label, Q, T, t_stop))

model = mujoco.MjModel.from_xml_path(GM_ROOT + "/models/pusht/scene.xml")
model.vis.global_.offwidth = model.vis.global_.offheight = args.cell
data = mujoco.MjData(model)
BLOCK = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
renderer = mujoco.Renderer(model, args.cell, args.cell)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.04, 0.04, 0.0]
cam.distance, cam.azimuth, cam.elevation = 0.42, 90.0, -90.0

HEAD, LAB = 50, 84  # title bar, label bar above each panel
cols, nrows = 3, int(np.ceil(len(runs) / 3))
W, H = cols * args.cell, HEAD + nrows * (LAB + args.cell)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                       "-r", str(args.fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                       args.out], stdin=subprocess.PIPE)
n_frames = int(args.seconds * args.fps) + 1 + args.fps  # hold the end 1 s
for f in range(n_frames):
    frac = min(f / (args.seconds * args.fps), 1.0)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((14, 12), args.title, font=f_title, fill=(0, 0, 0))
    d.text((W - 380, 16), "white block, red pusher, green goal", font=f_text, fill=(80, 80, 80))
    for k, (label, Q, T, t_stop) in enumerate(runs):
        t = frac * T[-1]
        data.qpos[:] = [np.interp(t, T, Q[:, j]) for j in range(Q.shape[1])]
        mujoco.mj_kinematics(model, data)
        renderer.update_scene(data, cam)
        x0, y0 = (k % cols) * args.cell, HEAD + (k // cols) * (LAB + args.cell)
        canvas.paste(Image.fromarray(renderer.render()), (x0, y0 + LAB))
        (x, y), th = data.xpos[BLOCK][:2], data.qpos[2]
        d.text((x0 + 12, y0 + 6), label, font=f_label, fill=(0, 0, 0))
        clock = f"t = {t:5.2f} s / {T[-1]:.2f} s"
        if t_stop is not None:
            clock += f"   (T stops at {t_stop:.2f} s" + (", pusher held)" if t >= t_stop else ")")
        d.text((x0 + 12, y0 + 36), clock, font=f_text, fill=(60, 60, 60) if t_stop is None or t < t_stop else (200, 40, 40))
        d.text((x0 + 12, y0 + 58), f"off goal {100 * np.hypot(x, y):4.1f} cm, "
               f"{np.degrees((th + np.pi) % (2 * np.pi) - np.pi):6.1f}°", font=f_text, fill=(60, 60, 60))
    if len(runs) < cols * nrows:  # spare cell: what the panels are
        x0, y0 = (len(runs) % cols) * args.cell, HEAD + (len(runs) // cols) * (LAB + args.cell)
        notes = ["PS / MPPI / DIAL / CACS: replan every 20 ms,", *args.loop_note.split(";"), "",
                 *args.gcs_note.split(";"),
                 "shown faster to fit the same video time.", "", f"Every run is shown over {args.seconds:g} s of video."]
        for i, line in enumerate(notes):
            d.text((x0 + 24, y0 + LAB + 40 + 28 * i), line, font=f_text, fill=(60, 60, 60))
    ff.stdin.write(np.asarray(canvas).tobytes())
ff.stdin.close()
ff.wait()
print("RESULT saved", args.out)
