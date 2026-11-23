"""Run the existing PushT example, directing its artifacts into a chosen folder."""
import csv
import functools
import json
from pathlib import Path
import runpy
import sys

import numpy as np

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
repo = Path.home() / ".cache/global-mppi/pusht_5090/Global-MPPI"
sys.argv = [str(repo / "examples/pusht.py"), *sys.argv[2:]]

# CUKERNELSOS_PYTHON and MUJOCO_GL are set by the launcher before imports.
from global_mppi.simulation import deterministic

original = deterministic.run_interactive


@functools.wraps(original)
def redirected(*args, **kwargs):
    kwargs["out_dir"] = str(out)
    with (out / "trajectory.csv").open("w", newline="", buffering=1) as stream:
        writer = csv.writer(stream)
        model = args[1]
        writer.writerow(["cycle", "plan_s", "best_cost", "simulation_time"]
                        + [f"qpos_{i}" for i in range(model.nq)]
                        + [f"qvel_{i}" for i in range(model.nv)])

        def record(cycle, plan_s, best_cost, model, state):
            writer.writerow([cycle, plan_s, float(np.asarray(best_cost).ravel()[0]), state.time,
                             *state.qpos.tolist(), *state.qvel.tolist()])

        kwargs["cycle_logger"] = record
        result = original(*args, **kwargs)
        state = args[2]
        (out / "final_state.json").write_text(json.dumps({
            "simulation_time": state.time,
            "qpos": state.qpos.tolist(), "qvel": state.qvel.tolist(),
        }, indent=2) + "\n")
        return result


deterministic.run_interactive = redirected
runpy.run_path(sys.argv[0], run_name="__main__")
