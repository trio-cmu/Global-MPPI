# obstacle_field: a point mass crossing a field of 9 circles

A velocity-controlled point mass (the particle of `bugtrap`: radius 0.01 m, two force motors of ±1 N, damping 1, mass 0.3 kg) goes
from the start `(-0.25, 0)` on the left to the goal `(0.25, 0)` on the right of the square arena `[-0.29, 0.29]^2`, through 9 circular obstacles
(radius 0.025 to 0.045 m). There are no inner walls. `bugtrap` is not changed.

- **Layout:** fixed, `layout_seed=19` (`make_layout(19)`). Every gap (circle-circle, circle-wall, circle-start/goal) is at least 0.06 m, 3 particle
  diameters; 3 or 4 large circles stand on or close to the start-goal line, so a greedy straight path gets stuck in front of them. The grid search
  finds the shortest path 0.530 m (1.06 x the straight line 0.500 m) and a second, different route 0.564 m. See `obstacle_field_candidates.png`
  (seeds 11, 19, 22; 19 is the middle one). Other layouts: `--layout-seed N` (searched on first use, a few seconds to a minute).
- **Cost:** the `bugtrap` cost with the wall-SDF term replaced by the distance to the nearest circle, same formula and weights:
  `5 exp(-50 d) + 5 |p - goal|^2 + 0.1 |qvel|^2 + 0.1 |u|^2`, with `d = min_i(|p - c_i| - r_i)`.
- **Episode:** 100 control cycles at 50 Hz (2 s), horizon 2.0 s, 11 knots, cubic spline (`--horizon 2.0 --num-knots 11 --spline-type cubic`).
  At full speed the particle covers 1.7 m (one axis) to 2.4 m (diagonal) in 2 s.

## Files

| file | what |
|---|---|
| `examples/obstacle_field.py` | runs one method for a range of seeds (options as `examples/bugtrap.py`, plus `--layout-seed`) |
| `global_mppi/tasks/obstacle_field.py` | the task: cost and model builder |
| `global_mppi/tasks/obstacle_field_layout.py` | `make_layout`, layout rules, grid-search shortest and second-best route (numpy only) |
| `global_mppi/models/obstacle_field/scene.xml` | MuJoCo template (outer walls, particle); the circles are added as cylinders at `<!-- OBSTACLES -->` |
| `script/obstacle_field_metrics.py` | metrics, summary table, trajectory figure, layout plots |
| `script/overnight_queue.sh` | `TASKS=obstacle_field` runs all methods and seeds, then the metrics script |

## Run

Environment (from the repo root, in the Global-MPPI conda env):

```
git submodule update --init                            # ksos-tools, pinned commit
export PYTHONPATH=$PWD:$PWD/ksos-tools                 # not the editable installs of another checkout
export CUKERNELSOS_PYTHON=/path/to/cuKernelSOS/python  # gm only: build of Zhongqi2/cuKernelSOS, branch cuKernelSOS (31b7cf7)
```

One method, one seed (`ps`, `mppi`, `dial`, `mtp`, `cbo`; options go before the method; `--max-cycles` is only checked every 10 cycles):

```
python examples/obstacle_field.py --headless --horizon 2.0 --num-knots 11 --spline-type cubic \
    --seed-start 0 --seed-end 1 --out-dir logs/of_test mppi
python examples/obstacle_field.py --headless --gpu --horizon 2.0 --num-knots 11 --spline-type cubic \
    --seed-start 0 --seed-end 1 --out-dir logs/of_test globalmppi          # gm
```

Without `--headless` the MuJoCo viewer opens. Output: `<out-dir>/<ps|mppi|dial|mtp|cbo|gm>/s<seed>/` with `cycles.csv` (per cycle: `plan_s`, `best_cost`, `x`, `y`,
`distance`, `min_clearance`, `wall_clearance`), `final.json`, `args.json`, a video.

The queue (6 methods x 6 seeds x 100 cycles, strictly sequential, detached; logs in `logs/overnight_runs/<timestamp>/`):

```
CUKS_GPU=/path/to/cuKernelSOS/python PY=/path/to/env/bin/python SYSPY=/usr/bin/python3 TASKS=obstacle_field script/overnight_queue.sh
```

Optional: `SEEDS`, `METHODS`, `MAX_CYCLES`, `RUNS_DIR`, `BASE_TIMEOUT` / `GM_TIMEOUT`, `MIN_FREE_BASE_MB` / `MIN_FREE_GM_MB` (header of the script). `SYSPY` needs matplotlib.

## Metrics

```
python script/obstacle_field_metrics.py <results dir> [--out-dir DIR]     # numpy + matplotlib only
python script/obstacle_field_metrics.py --plot-candidates out.png --seeds 11 19 22
```

Per run, from `cycles.csv` (one position per cycle): `path_length` and `path_ratio` to the grid-search shortest path, `reached` (final distance < 2 cm),
`collision` (min clearance < 0 on the path polyline; clearance = particle surface to circle surface), `min_clearance`, `final_distance`, mean plan time per
cycle without cycle 0 (JIT). Writes `metrics.csv`, `summary.txt` (one row per method), `reference_path.json`, `obstacle_field_trajectories.png`.

## Note on gm

The GPU ADMM of cuKernelSOS runs with a time limit (about 2 s per solve); on a busy GPU it stops after fewer iterations (every call hit
`status=time_limit` in our tests, with 90 to 160 iterations), and a gm cycle takes minutes. **Run gm on an idle GPU**, and keep other jobs off it.
