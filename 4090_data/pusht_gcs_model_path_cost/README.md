# PushT path-cost benchmark in GCS's quasi-static model

Which planner finds the shortest pushing path? Every method, and GCS, plans in the **same dynamics** (GCS's
quasi-static pusher-slider model) with the **same cost**, and every executed trajectory is scored by the **same
script**. Time does not count: a slow plan and a fast plan along the same path cost the same.

Run on an RTX 4090, 2026-11-20.

## Task

- PushT: push the T from `(x, y, θ) = (0.1, 0.1, 1.3)` to the goal `(0, 0, 0)`; the pusher starts and ends at
  `(0, 0.1)` (qpos is relative to it). The same start and goal as GCS (`pusht_test/run_gcs_task_cost.py`).
- Dynamics: GCS's quasi-static pusher-slider model (`global_mppi/tasks/pusht_gcs.py`, `examples/pusht.py
  --gcs-model`), used both for planning (rollouts) and for the closed-loop simulation.
- Closed loop: 100 control cycles at 50 Hz (2 s), plan horizon 1 s, 6 knots, cubic spline, 6 seeds (0-5).

## Cost

With `v_i` the T's 8 outline corners, `v_i*` the goal T's corners, and the pusher position `p`:

```
J = L_T + 0.1 · L_P + 10 · D_final

L_T     = Σ (1/8) Σ_i |Δv_i|                  T path: mean distance travelled by the 8 corners
L_P     = Σ |Δp|  while the T is still          pusher path between pushes (mean corner speed ≤ 1 mm/s)
D_final = (1/8) Σ_i |v_i(end) − v_i*|           final mean distance of the corners to the goal
```

The corners, in the T's MuJoCo body frame (the vertices GCS's `TPusher2d` uses for `keypoint_arc_length`, in
the same order): `(−.03, .04) (.03, .04) (.03, .02) (.01, .02) (.01, −.04) (−.01, −.04) (−.01, .02) (−.03, .02)`.

The same three terms appear in all three places:

| | T path | pusher path | final distance |
|---|---|---|---|
| **Sampling controllers** (`PushT(cost="path")`, `global_mppi/tasks/pusht.py`) | running `(1/8) Σ_i \|v_i'\|` × dt | running `0.1 \|v_pusher\| [T still]` × dt | terminal `10 D` at the end of each 1 s plan |
| **GCS** (`run_gcs_task_cost.py`) | `keypoint_arc_length`, weight 1 | `pusher_arc_length`, weight 0.1 | 0: the goal is a hard constraint |
| **Scorer** (`../score_path_cost.py`) | `L_T_m` | `L_P_m` | `D_final_m` |

- `dt × speed` is the distance moved in a step, so the controllers' running cost sums to `L_T + 0.1 L_P` in
  metres: dt does not weight anything. Stepping GCS's plan through `PushT(cost="path")` gives J = 0.1698, the
  scorer's value.
- The controllers see `D` at the end of each 1 s plan (receding horizon); the scorer and GCS use the end of the
  whole trajectory. That is the difference between a 1 s closed-loop planner and a whole-trajectory planner,
  and part of what the benchmark measures.
- Without the terminal term a sampling controller would not move: standing still costs 0.

## Methods

All sampling methods: 2000 samples × 5 iterations per cycle (cuKernelSOS: see below), horizon 1 s, 6 knots,
cubic spline, the per-method settings of `examples/pusht.py` (not tuned for this cost).

| Method | Settings |
|---|---|
| PS | noise 0.4 |
| MPPI | noise 1.0, temperature 0.1 |
| DIAL | noise 1.0, temperature 0.01, beta 1.0 / 1.0 |
| MTP | hydrax defaults (m_pts 3, n_per_layer 50, degree 2, 5 elites, σ 0.5 / 0.1 / 1.0, temperature 0.1, beta 0.1, alpha 0.5, akima) |
| CBO | hydrax bugtrap-example values (initial noise 1.0, temperature 0.01, consensus weight 1.0, noise weight 10, step 0.1) |
| cuKernelSOS | CACS (1000 samples, h 0.5, max clique 20) + GPU ADMM, 5 restarts, warm start, then MPPI 1000 × 5 at noise 0.05, temperature 0.001 (tuned for this cost, see below; the original settings were h 0.7, noise 0.1, temperature 0.1) |
| GCS | `--angle cos --no-pose-cost --no-nc-pose-cost --path-weight 1 --pusher-path-weight 0.1`: one open-loop plan, 3 knots per push (1.5 s), 3 per reposition (0.75 s), pusher speed ≤ 1 m/s |

## Results

J (lower is better), mean ± std over 6 seeds; GCS is one plan. Figure: `pusht_gcs_model_path_cost.png`
(`pusht_gcs_model_path_cost_wfinal100.png`: the same runs scored with 100 · D_final).

| Method | **J** | J per seed (0-5) |
|---|---|---|
| **GCS** | **0.170** | (one plan; GCS's own value 0.171) |
| MTP | 0.183 ± 0.009 | 0.177 0.200 0.180 0.185 0.177 0.177 |
| cuKernelSOS | 0.196 ± 0.029 | 0.229 0.177 0.165 0.169 0.222 0.211 |
| PS | 0.212 ± 0.034 | 0.183 0.213 0.176 0.265 0.238 0.199 |
| DIAL | 0.223 ± 0.011 | 0.241 0.210 0.215 0.227 0.216 0.228 |
| MPPI | 0.304 ± 0.023 | 0.349 0.284 0.297 0.303 0.296 0.293 |
| CBO | 1.144 ± 0.623 | 1.544 1.546 0.390 0.291 1.543 1.548 |

Notes:
- **cuKernelSOS was tuned for this cost.** Its final MPPI step weights rollouts by `softmax(−cost / temperature)`;
  the original temperature 0.1 was chosen for the pose cost (rollout costs ~10-100) and makes the weights nearly
  uniform with this cost (~0.2-0.6). A sweep of 13 settings, 2 seeds each (`../pusht_gcs_model_path_cost_sweep/`),
  chose temperature 0.001, MPPI noise 0.05 and CACS radius h 0.5: mean J 0.237 with the original settings, 0.181
  with these (seeds 0-1). The original-settings runs are in `gm_default_T0.1/` (s0 0.231; s1 stopped).
  Seeds 2 and 3 score below GCS: they end slightly short of the goal and save more path than 10 · D_final costs
  (with 20 · D_final: 0.172 and 0.175).
- The baselines use their `examples/pusht.py` settings, chosen for the pose cost, untuned for this one; CBO
  fails on 4 of 6 seeds (the T barely moves).
- Six seeds per method, no significance test.

## Reproduce

Environment: the `ksos` conda env (MuJoCo / MJX 3.3.6); cuKernelSOS found as `../cuKernelSOS/python` next to
the repo (`examples/pusht.py` sets `CUKERNELSOS_PYTHON`), built for Python 3.12 (`build-py312`).

```bash
OUT=4090_data/pusht_gcs_model_path_cost
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# baselines (ps, mppi, dial, mtp, cbo), seeds 0-5
for m in ps mppi dial mtp cbo; do for s in 0 1 2 3 4 5; do
  python examples/pusht.py --no-wandb --headless --gcs-model --cost path --num-samples 2000 \
      --seed $s --out-dir $OUT/$m/s$s $m
done; done
# cuKernelSOS, seeds 0-5
for s in 0 1 2 3 4 5; do
  python examples/pusht.py --no-wandb --headless --gcs-model --cost path --seed $s --out-dir $OUT/gm/s$s \
      --ksos-solver ADMM-cuda --ksos-sampling cacs --ksos-restarts 5 --ksos-warm-start \
      --gm-temperature 0.001 --mppi-noise 0.05 --cacs '{"h": 0.5}' globalmppi
done
# score and plot
python 4090_data/score_path_cost.py $OUT/*/s*/*robot_pos_history*.npy --w 10 --wp 0.1 --json $OUT/path_cost_all.json
python $OUT/plot_path_cost.py                    # pusht_gcs_model_path_cost.png
python $OUT/plot_path_cost.py --w-final=100      # pusht_gcs_model_path_cost_wfinal100.png
```

GCS, in the `planning-through-contact` checkout next to this repo (its `pusht_test/` is local, not in git):

```bash
cd ../planning-through-contact
.venv/bin/python pusht_test/run_gcs_task_cost.py --angle cos --no-pose-cost --no-nc-pose-cost \
    --path-weight 1 --pusher-path-weight 0.1            # prints RUN_FOLDER: pusht_test/run_<...>
.venv/bin/python pusht_test/export_gcs_traj.py --run pusht_test/run_<...> --out gcs_path_pusher_traj
cd ../Global-MPPI
python 4090_data/score_path_cost.py ../planning-through-contact/pusht_test/gcs_path_pusher_traj.npz --w 10 --wp 0.1
```

(`export_gcs_traj.py --out` treats a dot as a file extension: a name like `..._wp0.1_traj` is saved as
`..._wp0.npz`.) The GCS run of this benchmark is `../gcs_path_pusher_wp0.1/` (plan, video `gcs_path_pusher_wp0.1.mp4`,
log, `path_cost.json`); the relaxation took 36 s.

## Files

- `<method>/s<seed>/`: `*robot_pos_history_seed<s>.npy` (qpos per cycle: T x, y, θ, pusher x, y; the scored
  trajectory), `*best_cost_history_seed<s>.npy` (the planner's predicted cost per cycle), the video
- `console/<method>_s<seed>.log`: the run's output
- `path_cost_all.json`: L_T, L_P, D_final, J and more for every run (`score_path_cost.py`)
- `plot_path_cost.py`: the figures; draws only finished runs (101 states) and every method folder present
  (`gm` = cuKernelSOS)
- Methods are folders `ps mppi dial mtp cbo gm`; `gm_default_T0.1/` holds cuKernelSOS with its original settings
  (`s1` is a stopped, incomplete run).
