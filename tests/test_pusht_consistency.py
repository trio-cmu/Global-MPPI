"""models/pusht/pusht.xml must agree with the real-setup definitions in ~/trio_sim.

The goal frame, the start poses and the pusher start are recorded in three yamls by the trio_sim / trio_infra
pipeline (pusht_goal.yaml, pusht_starts.yaml, pusher_start.yaml, all in the GOAL frame = the sim task frame).
This test compares them with the goal body and the `start_2` keyframe of pusht.xml and fails if they differ by
more than 1 mm / 0.1 deg. Directory: $PUSHT_ALIGN_DIR (default ~/trio_sim/src/pusht_align); skipped when the
yamls are not there.
"""
import math
import os

import mujoco
import numpy as np
import pytest
import yaml

from global_mppi import ROOT

ALIGN_DIR = os.environ.get("PUSHT_ALIGN_DIR", os.path.expanduser("~/trio_sim/src/pusht_align"))
POS_TOL_M = 1e-3
ANG_TOL_DEG = 0.1
FILES = ("pusht_goal.yaml", "pusht_starts.yaml", "pusher_start.yaml")

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(ALIGN_DIR, f)) for f in FILES),
    reason=f"{FILES} not found in {ALIGN_DIR} (set PUSHT_ALIGN_DIR)",
)


def _load(name):
    with open(os.path.join(ALIGN_DIR, name)) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(ROOT + "/models/pusht/scene.xml")


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def test_goal_is_the_sim_origin(model) -> None:
    """The goal frame is the sim frame: the goal body must sit at (0, 0) with theta = 0."""
    _load("pusht_goal.yaml")  # must exist / parse
    b = model.body("goal")
    assert np.linalg.norm(model.body_pos[b.id][:2]) <= POS_TOL_M
    w, x, y, z = model.body_quat[b.id]
    theta = math.degrees(2.0 * math.atan2(z, w))
    assert abs(_wrap_deg(theta)) <= ANG_TOL_DEG


def test_start_2_keyframe(model) -> None:
    s = _load("pusht_starts.yaml")["starts"]["start_2"]
    q = model.key("start_2").qpos
    assert math.hypot(q[0] - s["x"], q[1] - s["y"]) <= POS_TOL_M
    assert abs(_wrap_deg(math.degrees(q[2]) - s["theta_deg"])) <= ANG_TOL_DEG


def test_pusher_start_keyframe(model) -> None:
    p = _load("pusher_start.yaml")
    q = model.key("start_2").qpos
    anchor = model.body_pos[model.body("pusher").id][:2]
    world = np.asarray(q[3:5]) + anchor                      # root offsets + body position = goal-frame position
    assert np.linalg.norm(world - [p["goal_frame"]["x"], p["goal_frame"]["y"]]) <= POS_TOL_M
    # the yaml also records the root offsets directly
    assert np.linalg.norm(np.asarray(q[3:5]) - np.asarray(p["sim_pusher_qpos"])) <= POS_TOL_M
