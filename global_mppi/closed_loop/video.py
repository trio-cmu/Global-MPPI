"""mp4 of an episode, rendered offscreen from the saved qpos history (T, goal, pusher + trail, caption)."""
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np


def write_video(scene, hist: np.ndarray, path: str, caption: str = "", fps: int = 20) -> None:
    """hist: rows (t, qpos) every 50 ms. Needs an offscreen GL backend (MUJOCO_GL=egl)."""
    model = scene.plant
    data = mujoco.MjData(model)
    W, H = 720, 480
    model.vis.global_.offwidth, model.vis.global_.offheight = max(W, model.vis.global_.offwidth), max(H, model.vis.global_.offheight)
    r = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.elevation, cam.azimuth, cam.distance = -90.0, 90.0, 0.9
    cam.lookat[:] = [-0.1, -0.05, 0.0]
    trail = []
    try:
        with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as w:
            for row in hist:
                data.qpos[:] = row[1:]
                mujoco.mj_forward(model, data)
                r.update_scene(data, cam)
                trail.append([*scene.pusher_world(row[1:]), 0.035])
                for p0, p1 in zip(trail[:-1], trail[1:]):
                    if r.scene.ngeom >= r.scene.maxgeom:
                        break
                    g = r.scene.geoms[r.scene.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                        np.array([1.0, 0.85, 0.0, 1.0], dtype=np.float32))
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, np.array(p0), np.array(p1))
                    r.scene.ngeom += 1
                img = np.ascontiguousarray(r.render())
                pe, ae = scene.errors(row[1:])
                cv2.putText(img, f"t = {row[0]:6.2f} s   pos err {pe * 1000:6.1f} mm   heading err {ae:5.2f} deg", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                if caption:
                    cv2.putText(img, caption, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                w.append_data(img)
    finally:
        r.close()
