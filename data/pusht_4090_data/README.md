# PushT benchmark rerun on the desktop RTX 4090

This is a fresh rerun of the six-method benchmark in `BENCHMARK_MAP.md`:
PS, MPPI, DIAL, MTP, CBO, and GlobalMPPI; seeds 0–5; 100 control cycles
(101 planning calls); horizon 1 second; 6 knots; cubic spline; 5 iterations.
Baselines use 2000 samples. GlobalMPPI uses 1000 MPPI samples and 1000
CACS candidates per restart, 5 restarts, warm start, noise 0.1, ADMM-cuda,
and GPU affine solves. Other controller settings come from the unchanged
`examples/pusht.py` and match `data/pusht_results/README.md`.

See `STATUS.txt` or `status.json` for live progress. `batch.log` records
run starts, completions, and failures. Runs execute sequentially; a crash
is retried up to twice. A timeout is recorded and the batch continues.
These files are the runner's last heartbeat and can remain stale after a
power loss or reboot. `python check_status.py --refresh` verifies the PID
and system boot identity and marks an absent runner as INTERRUPTED.
Sleep inhibition keeps the desktop awake; the detached job does not survive
a reboot. Resume with the command below. Completed runs are kept; the
interrupted run restarts from cycle 0. Its prior CSV, NPY and video artifacts
are preserved under `<method>/s<seed>/attempt_artifacts/attempt<N>/`.

Each `<method>/s<seed>/` contains:

- `cycles.csv`: cost at cycles 0–100 and plan times at cycles 0–99, using
  the same log extraction convention as the original benchmark.
- `trajectory.csv`: full precision per-cycle cost, plan time, and physical
  state before execution, for cycles 0–99; flushed after every cycle.
- `final.json`: completion validation, final planner cost, median plan time
  excluding cycle 0, wall time, and GlobalMPPI solve-path counts.
- `args.json`: exact command and arguments.
- `run.log`: link to the latest attempt's raw log; prior attempt logs retained.
- `gpu_before_attempt*.txt`, native cost/state NPY histories, rollout video,
  and `final_state.json`.

While a run is in progress, parsed values also appear in `cycles.partial.csv`.
Only validated completed runs get `cycles.csv` and `final.json`.
`pusht_final.png` and `pusht_summary.txt` are updated after each completed run
and therefore represent a partial batch until `STATUS.txt` says COMPLETE.
Costs are predicted planner costs, matching the original benchmark.

## Software and provenance

- Main checkout: `benchmarks`, commit `7d71c56880fe43d329374275788639abf2a730a1`.
- cuKernelSOS: `276861259a8e6028c68301cd611cd1365a7e9734` (`gpu-solve`).
- ksos-tools: `86e6163040ddea1d8df5af947bbc5bac64ce1b36` (`gpu-solve-exp`).
- Python 3.12.12, JAX/JAXLIB 0.5.3, MuJoCo/MJX 3.3.6.
- cuKernelSOS built locally with CUDA toolkit 13.1.80 for the desktop.
- `manifest.json`, `pip_freeze.txt`, and GPU snapshots record the actual setup.

The separate environment and pinned solver checkouts live in
`/home/zhongqi/.cache/global-mppi/pusht_4090/`. The environment inherits
packages from the existing `benchmark` conda environment and overrides
MuJoCo/MJX to 3.3.6. This preserves the user's existing environments and
solver checkouts. Other package versions differ from the original 5090
environment; see the freeze file. The desktop display also uses this GPU.

`run_one.py` calls the existing example with an output-directory wrapper
and an additional state logger; controller and simulation logic are unchanged.
The original example also writes its GPU snapshot under the repo's `logs/`.

## Resume

To launch detached with sleep inhibition:

```bash
/home/zhongqi/.cache/global-mppi/pusht_4090/venv/bin/python /home/zhongqi/Documents/workspace/kernelSOS_project/Global-MPPI/data/pusht_4090_data/launch.py
```

Run from any directory; completed runs in this folder are skipped:

```bash
/home/zhongqi/.cache/global-mppi/pusht_4090/venv/bin/python /home/zhongqi/Documents/workspace/kernelSOS_project/Global-MPPI/data/pusht_4090_data/run_benchmark.py
```

An advisory lock prevents two copies of this runner from executing at once.
`runner.pid` and `status.json` identify the runner and current child process.
The initial launch uses a sleep inhibitor and is detached from the terminal.
Preflight results, if present, are under `_preflight/` and excluded from the
36 benchmark runs and summary.
