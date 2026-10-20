"""Workspace box as 4 walls in the push-T model, so the planner and the plant know the pen cannot leave the box.

The box (xy_driver flange box, BASE frame) is rotated into the goal (sim) frame. The walls collide with the pusher
only (pusher geom contype/conaffinity 3, walls 2, the T has 1) and nearly frictionless (mu 0.02, priority 1 decides the contact
friction), so the pen slides along a wall like the real clamp. The pen CENTRE can reach the box edge (the wall face
is one pusher radius outside it). The generated scene includes pusht.xml and is written next to it.
"""
import math
import os

from global_mppi import ROOT

R_PEN, THICK, LONG, H = 0.005, 0.05, 1.0, 0.03


def write_walled_scene(box, goal_p, goal_th, name="scene_box_generated.xml", include="scene.xml") -> str:
    x0, x1, y0, y1 = box
    c, s = math.cos(-goal_th), math.sin(-goal_th)
    geoms = []
    # (centre in the base frame, half extents in the base-frame axes)
    walls = {"x_min": ((x0 - R_PEN - THICK, 0.0), (THICK, LONG)), "x_max": ((x1 + R_PEN + THICK, 0.0), (THICK, LONG)),
             "y_min": ((0.0, y0 - R_PEN - THICK), (LONG, THICK)), "y_max": ((0.0, y1 + R_PEN + THICK), (LONG, THICK))}
    for k, ((cx, cy), (hx, hy)) in walls.items():
        if k.startswith("x"):
            cy = 0.5 * (y0 + y1)
        else:
            cx = 0.5 * (x0 + x1)
        dx, dy = cx - goal_p[0], cy - goal_p[1]
        px, py = c * dx - s * dy, s * dx + c * dy
        geoms.append(f'    <geom name="wall_{k}" type="box" size="{hx:.5f} {hy:.5f} {H}" pos="{px:.6f} {py:.6f} 0.02" '
                     f'euler="0 0 {math.degrees(-goal_th):.6f}" contype="2" conaffinity="2" friction="0.02 0.001 0.001" priority="1" '
                     f'rgba="0.8 0.3 0.3 0.15"/>')
    txt = (f'<mujoco model="pusht_box">\n  <include file="{include}"/>\n  <worldbody>\n' + "\n".join(geoms)
           + "\n  </worldbody>\n</mujoco>\n")
    path = os.path.join(ROOT, "models", "pusht", name)
    with open(path, "w") as f:
        f.write(txt)
    return path
