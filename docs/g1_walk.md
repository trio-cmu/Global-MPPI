# G1 walk (hydrax `HumanoidMocap`): motion-capture tracking with the Unitree G1

`examples/g1_walk.py` runs six planners (`ps mppi dial mtp cbo globalmppi`) on hydrax's `HumanoidMocap` task, unchanged
(`global_mppi/tasks/humanoid_mocap.py`, from hydrax `bd53d43`; only the imports and the `impl` / `make_data` parts of the
hydrax class, which this repo's `Task` base class does not have, were removed). Upstream hydrax main is at `bd53d43` too.

## The reference motion is NOT included

The task tracks a retargeted motion capture and downloads it on first use (as hydrax does) into the Hugging Face cache
(`~/.cache/huggingface`, or `$HF_HOME`); it is never stored in this repository and must not be added to it.

| | |
|---|---|
| Source | Hugging Face dataset [`robfiras/loco-mujoco-datasets`](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) (LocoMuJoCo) |
| File | `Lafan1/mocap/UnitreeG1/walk1_subject1.npz` (the default of hydrax's task and example) |
| Size | 2,476,776 bytes (2.4 MB): 10451 frames at 40 Hz (261 s), `qpos` shape (10451, 30) |
| License | **CC-BY-NC-ND 4.0** (the license tag of the dataset card): non-commercial, no derivatives. Check it before any other use |

To get it without running a simulation:

```bash
python -c "from huggingface_hub import hf_hub_download as d; print(d('robfiras/loco-mujoco-datasets', 'Lafan1/mocap/UnitreeG1/walk1_subject1.npz', repo_type='dataset'))"
```

The G1 model is not in the repo either: `script/fetch_g1_model.sh` copies hydrax's `models/g1` (from MuJoCo Menagerie, BSD-3-Clause,
Unitree Robotics) into `global_mppi/models/g1/` (checked against `script/g1_model.sha256`).

## The task

- Model `scene_23dof.xml`: 23 actuated joints (position actuators, `ctrlrange` about -3.09 to 2.97 rad), nq 30, nv 29, planning step 0.02 s.
  Control dimension of the planners: 23 actuators x 4 knots = 92.
- Running cost (hydrax): five tracking errors against the reference at the current time, each as `J = 1 - exp(-|error|^2)`, weighted and
  normalised so that the weights add up to 1: joint positions 0.1 (0.083 after normalising), joint velocities 0.01 (0.008), body positions 1.0 (0.826),
  body orientations 0.1 (0.083), body twists 0 (0). The terminal cost is the running cost (zero control) times dt. The planner cost
  (`best_cost` in `cycles.csv`) is the predicted cost over the horizon.
- The reference advances with the simulation time: frame = `int(time * 40)`, clipped to the last frame (it does not loop). Each cycle the planner starts
  at the current simulation time, so the horizon looks 0.8 s ahead in the reference.
- Horizon 0.8 s, 4 knots, zero-order hold: the values of hydrax's `examples/humanoid_mocap.py` (which uses CEM; there is no hydrax MTP example for G1).
- Simulation as that example: timestep 0.01 s, 10 solver iterations, 50 line-search iterations, soft-constraint override
  `o_solimp = [0.9, 0.95, 0.001, 0.5, 2]`; control at 50 Hz; the robot starts at reference frame 0 with the first reference joint positions as initial knots.
- Planner settings as `examples/cube.py` and `examples/g1.py`: 2000 samples x 5 iterations for PS, MPPI (noise 0.2, temperature 0.001) and DIAL (noise 1.0,
  temperature 0.01, beta 1.0 / 1.0), PS noise 0.2; MTP and CBO with the untuned hydrax defaults (`global_mppi/algs/mtp.py`, `cbo.py`); GlobalMPPI with 1000 KSOS candidates,
  CACS (h = 0.7 x 3.09), ADMM-cuda, 5 restarts, warm start, `gpu_solve`, then MPPI 1000 x 5 at noise 0.1. 100 cycles per run, seeds 0-5.

**What 100 cycles cover.** 100 cycles are 2.0 s of simulation, i.e. the first 80 frames of the reference. There the reference base is almost stationary
(it moves 4 mm in 2 s) while the joints already move (up to 1.1 rad from frame 0): the benchmark as set is "start of a walk",
not a walking gait. A longer run (`--max-cycles 500` and more) is needed to see the robot walk; the base moves 3.6 m in the first 10 s of the reference.

## cuKernelSOS requirement (GlobalMPPI only)

`ssisyg/cuKernelSOS`, branch `gpu-solve`, commit `2768612` (a WIP, not merged). `KSOS_GPU_SOLVE=1` switches `gpu_solve` on and
`CUKERNELSOS_PYTHON=<cuKernelSOS checkout>/python` selects the build. The baselines never call cuKernelSOS. Environment: MuJoCo / mujoco-mjx **3.3.6**.

## Run

```bash
script/fetch_g1_model.sh                      # once: the G1 model
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
# ps | mppi | dial | mtp | cbo   (seed S; the reference is downloaded on the first run)
python examples/g1_walk.py --headless --seed-start S --seed-end $((S+1)) --max-cycles 100 --out-dir OUT ps
# GlobalMPPI
KSOS_GPU_SOLVE=1 CUKERNELSOS_PYTHON=/path/to/cuKernelSOS_gpu/python \
python examples/g1_walk.py --headless --seed-start S --seed-end $((S+1)) --max-cycles 100 --out-dir OUT --gpu globalmppi
# the whole batch (6 methods x 6 seeds), figure and table
TASKS=g1_walk CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
```

Each run writes `OUT/<ps|mppi|dial|mtp|cbo|gm>/s<S>/{cycles.csv, final.json, args.json, *.npy, video}`. `cycles.csv`: `plan_s`, `best_cost`,
base `x y z`, `tilt_deg`, `ref_base_err` (base distance to the reference position, m), `ref_joint_err` (2-norm of the joint error, rad), `gpu_mem_mb`,
`solve_paths`. `final.json`: final base state and errors, `fell` (final base height below 0.5 m; the standing pelvis is at about 0.8 m), mean plan time, `solve_paths`.
`script/final_figs.py <queue folder> g1_walk` draws the figure and table; `script/package_task_results.py` packages the runs without the videos.
