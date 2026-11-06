"""Live MuJoCo viewer, opened once and reused for every episode: T, translucent goal, pusher + trail, captions."""
import os
import time

import mujoco
import mujoco.viewer
import numpy as np


class Viewer:
    def __init__(self, scene, data):
        self.scene = scene
        self._skip = False
        self.v = mujoco.viewer.launch_passive(scene.plant, data, key_callback=self._on_key)
        self.v.cam.elevation, self.v.cam.azimuth, self.v.cam.distance = -90.0, 90.0, 0.9
        self.v.cam.lookat[:] = [-0.1, -0.05, 0.0]
        self.last = None
        self.label = ""

    def _on_key(self, keycode: int) -> None:
        if keycode == 256:                       # GLFW_KEY_ESCAPE: skip the current run
            self._skip = True

    skip_file = None                             # a path: creating this file also skips the current run

    def consume_skip(self) -> bool:
        s, self._skip = self._skip, False
        if self.skip_file and os.path.exists(self.skip_file):
            os.remove(self.skip_file)
            s = True
        return s

    def alive(self) -> bool:
        return self.v.is_running()

    def begin(self, label: str) -> None:
        """New episode: clear the trail and show which configuration / seed this is."""
        with self.v.lock():
            self.v.user_scn.ngeom = 0
        self.last = None
        self.label = label

    _last_refresh = 0.0
    min_interval_s = 1.0 / 25.0                  # refresh the window at most ~25 times per wall second

    def refresh(self, data) -> None:
        """Throttled update for use inside the simulation loop (the render sync can be the slowest part of a step)."""
        now = time.time()
        if now - self._last_refresh >= self.min_interval_s:
            self._last_refresh = now
            self.update(data)

    def update(self, data, prefix: str = "") -> None:
        pe, ae = self.scene.errors(data.qpos)
        F = mujoco.mjtFontScale.mjFONTSCALE_150
        self.v.set_texts([
            (F, mujoco.mjtGridPos.mjGRID_TOPLEFT, f"{prefix}sim t = {data.time:6.2f} s",
             f"pos err {pe * 1000:6.1f} mm   heading err {ae:5.2f} deg"),
            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, self.label, ""),
        ])
        self.v.sync()

    def trail(self, data) -> None:
        p = np.array([*self.scene.pusher_world(data.qpos), 0.035])
        if self.last is not None and self.v.user_scn.ngeom < self.v.user_scn.maxgeom:
            with self.v.lock():      # without the lock the render thread can crash on the scene update
                g = self.v.user_scn.geoms[self.v.user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                    np.array([1.0, 0.85, 0.0, 1.0], dtype=np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, self.last, p)
                self.v.user_scn.ngeom += 1
        self.last = p

    def linger(self, seconds: float) -> None:
        t0 = time.time()
        while time.time() - t0 < seconds and self.v.is_running():
            self.v.sync()
            time.sleep(0.05)

    def hold(self) -> None:
        """Keep the final frame until the window is closed (or Ctrl-C)."""
        try:
            while self.v.is_running():
                self.v.sync()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        self.v.close()


class NullViewer:
    """Same interface, no window (headless runs). The skip file still works."""

    skip_file = None

    def alive(self): return True
    def begin(self, label): self.label = label
    def update(self, data, prefix=""): pass
    def refresh(self, data): pass
    def trail(self, data): pass
    def linger(self, seconds): pass
    def hold(self): pass

    def consume_skip(self) -> bool:
        if self.skip_file and os.path.exists(self.skip_file):
            os.remove(self.skip_file)
            return True
        return False
