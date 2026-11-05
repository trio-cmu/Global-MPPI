# Reproducing the PushT closed-loop benchmark

2026-11-05 · reproduces `examples/single_run_test/PushT/closed_loop/pusht_closed_loop_6seeds_main.png` (and `pusht_pareto.png`)

The figure compares PS, MPPI, DIAL, CACS (ours) and GCS on PushT closed loop: 100 control cycles, 6 seeds per method. The code comes from four repos. Three are ours (Global-MPPI, cuKernelSOS, ksos-tools); the fourth (planning-through-contact) only provides the GCS baseline.

## Repos and branches

| Repo | Role | Remote | Branch | Commit used | Pushed? |
| --- | --- | --- | --- | --- | --- |
| **Global-MPPI** | planners (PS, MPPI, DIAL, GlobalMPPI + CACS), PushT task, run and plot scripts | `https://github.com/trio-cmu/Global-MPPI` | `cukernelsosTest` | `ee12f03` + uncommitted changes | **No.** The branch is not on the remote, and the CACS code (`global_mppi/cacs.py`, `global_mppi/algs/global_mppi.py`, `examples/pusht.py`) is not committed yet |
| **cuKernelSOS** | GPU ADMM solver for KSOS (`--ksos-solver ADMM-cuda`), C++/CUDA with a Python binding | `https://github.com/Zhongqi2/cuKernelSOS` | `cuKernelSOS` | `a27f5c5` | **No.** 3 commits are not pushed (`45059fb`, `671ed00`, `a27f5c5`: Ruiz scaling and Anderson acceleration). There are no local edits |
| **ksos-tools** | KSOS problem building (features, sampling, RCM order) and the CPU solvers | `https://github.com/Zhongqi2/ksos-tools` (fork of `Simple-Robotics/ksos-tools`) | `sparse_admm` | `97d0748` | Yes, on the fork |
| planning-through-contact | GCS baseline only | `https://github.com/bernhardpg/planning-through-contact` (upstream, not ours) | `main` | `a3998f2` + local `pusht_test/` | **No.** `pusht_test/` is untracked and 2 files are modified; there is no fork to push to |

How they fit together:

- **ksos-tools is a git submodule of Global-MPPI** (`Global-MPPI/ksos-tools`, pinned at `97d0748`), and the `ksos` env installs it from there (`pip -e`). `.gitmodules` points to `Simple-Robotics/ksos-tools`, which does **not** have `97d0748`, so `git submodule update` on a fresh clone fails. Point the submodule URL to `Zhongqi2/ksos-tools` (or push `sparse_admm` upstream).
- **cuKernelSOS is not pip-installed.** `examples/pusht.py` looks for a folder `cuKernelSOS/python` in any parent directory and sets `CUKERNELSOS_PYTHON` to it. `global_mppi/__init__.py` then loads the extension **before** jax, because loading it after jax made it pick up jax's cuSPARSE and corrupt memory. So cuKernelSOS must sit next to Global-MPPI:

```
kernelSOS_project/
├── Global-MPPI/            (branch cukernelsosTest)
│   └── ksos-tools/         (submodule, branch sparse_admm)
├── cuKernelSOS/            (branch cuKernelSOS, built into cuKernelSOS/python)
└── planning-through-contact/   (GCS baseline, own .venv)
```

## Environment

- GPU: NVIDIA RTX 4090, driver 590.48.
- Conda env `ksos`: Python 3.12.11, jax / jaxlib 0.7.1, mujoco / mujoco-mjx 3.3.6, `global-mppi` and `ksos_tools` installed editable.
- cuKernelSOS is built with the system CUDA 13.1 (`/usr/local/cuda/bin/nvcc`) against the `ksos` Python. The build used is `cuKernelSOS/build-py312`, which matches `a27f5c5`.

## Setup from scratch

```bash
mkdir kernelSOS_project && cd kernelSOS_project

# 1. Global-MPPI and its ksos-tools submodule
git clone -b cukernelsosTest https://github.com/trio-cmu/Global-MPPI      # after the branch is pushed
cd Global-MPPI
git config submodule.ksos-tools.url https://github.com/Zhongqi2/ksos-tools.git
git submodule update --init ksos-tools                                     # -> 97d0748 (sparse_admm)
conda create -n ksos python=3.12 && conda activate ksos
pip install -e . && pip install -e ksos-tools
cd ..

# 2. cuKernelSOS, next to Global-MPPI
git clone -b cuKernelSOS https://github.com/Zhongqi2/cuKernelSOS           # after the 3 commits are pushed
cd cuKernelSOS
cmake -S . -B build-py312 -DCMAKE_BUILD_TYPE=Release -DCUKERNELSOS_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python)
cmake --build build-py312 -j4          # puts _cukernelsos.*.so into cuKernelSOS/python/cukernelsos/
cd ..
```

If nvcc picks a host compiler without C++ support, add `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CXX_COMPILER=/usr/bin/g++` (see the cuKernelSOS README).

## 1. Run the sampling planners (Global-MPPI)

Run from the `Global-MPPI` root. Each run writes one log per method and seed; the plot scripts read the `Cycle N: Best cost` and `Cycle N: plan time` lines.

```bash
D=examples/single_run_test/PushT/closed_loop/seeds
for s in 0 1 2 3 4 5; do
  # baselines: 2000 samples x 5 iterations
  for m in ps mppi dial; do
    python examples/pusht.py --no-wandb --headless --seed $s --num-samples 2000 $m > $D/${m}2000_seed$s.log 2>&1
  done
  # CACS (ours)
  python examples/pusht.py --no-wandb --headless --seed $s --ksos-solver ADMM-cuda --ksos-sampling cacs \
    --ksos-restarts 5 --ksos-warm-start --mppi-noise 0.1 globalmppi > $D/globalmppi_cacs5new_seed$s.log 2>&1
done
```

- **Task:** block starts at (0.1, 0.1, 1.3), pusher at (0, 0). 100 control cycles at 50 Hz, with a 1 s horizon and 6 knots.
- **CACS:**
  - GlobalMPPI with 5 warm-started KSOS restarts in shrinking boxes (decay 0.8).
  - Each restart samples with CACS; the main config is the current `CacsConfig` default: n 1000, initial 200, reuse 40, h 0.7, max_clique 20, batch 768, k 512, global_every 125, replace 0.
  - KSOS is then solved by the cuKernelSOS GPU ADMM (time limit 2 s), followed by MPPI with 1000 samples × 5 iterations and noise 0.1.
  - To change a CACS setting without editing code, pass `--cacs '{"batch": 512}'`.
- **Run time:** about 15 min per CACS seed (about 7.3 s per cycle) and a few minutes per baseline seed.
- **Startup crashes:** a run sometimes segfaults or aborts during JAX / ptxas compilation at cycle 0. Just rerun it. A run that gets past cycle 0 always finishes, so the crashes do not affect results.
- Videos go to `Global-MPPI/logs/pushT_<alg>/<timestamp>/simulation_*.mp4`.
- The original CACS seed-0 log is named `globalmppi_cacs5_warm_nokd_seed0.log`. It is the same command and config, run on the same night as seeds 1–5.

## 2. Time the GCS baseline (planning-through-contact)

GCS (Graph of Convex Sets) runs in its own repo and venv. Run it from the `planning-through-contact` root:

```bash
.venv/bin/python pusht_test/time_gcs_solves.py --runs 6
```

- Each run solves the same PushT problem in a fresh process (one solve needs about 15 GB of RAM) and appends a line to `pusht_test/gcs_solve_times.jsonl`.
- The GCS planning time in the figure is `relaxed_gcs_time + binary_flows_time + feasible_time` for each run: 119.84, 119.94, 119.91, 119.59, 119.57 and 120.35 s, mean 119.87 s. Almost all of it (about 119.7 s) is the relaxed GCS solve.
- The GCS final cost (1e-5) is entered by hand.
- Both values are typed into `GCS_TIMES` in `plot_closed_loop_seeds.py` and `plot_pareto.py`; the plots do not read the GCS logs.

## 3. Make the plots

```bash
cd examples/single_run_test/PushT/closed_loop
mkdir -p six_seeds_main
for m in ps2000 mppi2000 dial2000 globalmppi_cacs5new; do
  for s in 0 1 2 3 4 5; do ln -sf ../seeds/${m}_seed$s.log six_seeds_main/${m}_seed$s.log; done
done
python plot_closed_loop_seeds.py six_seeds_main pusht_closed_loop_6seeds_main.png --gcs   # main figure
python plot_pareto.py                       # pusht_pareto.png: cost vs planning time, Pareto front
python plot_paper_style.py                  # pusht_closed_loop_paper.png: median + 25-75% band
```

Each script also writes a `.pdf` next to the `.png`.

## Expected result

| Method | Final cost (cycle 100), mean ± std over 6 seeds | Planning time per cycle |
| --- | --- | --- |
| PS (2000 × 5) | 5.9e-1 ± 1.4 | 0.36 s |
| MPPI (2000 × 5) | 3.4e-1 ± 8.3e-2 | 0.36 s |
| DIAL (2000 × 5) | 2.2e-2 ± 1.3e-2 | 0.36 s |
| **CACS (ours)** | **2.1e-4 ± 1.8e-4** | 7.31 ± 0.32 s |
| GCS | 1e-5 | 119.87 s |

- The final cost varies by more than 10× between seeds, so compare the 6-seed means, not single runs.
- Pareto front (cost vs time): DIAL → CACS → GCS.
- The hyperparameter sweep behind the CACS config is in `docs/cacs_closed_loop_sweep.md`.

## To do before others can reproduce it

1. **Global-MPPI:** commit the staged changes and `git push -u origin cukernelsosTest`.
2. **cuKernelSOS:** `git push origin cuKernelSOS` (3 commits).
3. **ksos-tools submodule:** change the URL in `.gitmodules` to `https://github.com/Zhongqi2/ksos-tools.git`, or get `97d0748` into `Simple-Robotics/ksos-tools`.
4. **Logs:** `.gitignore` excludes `*.log`, so the seed logs in `closed_loop/seeds/` are not in git. To share the raw results, force-add them (`git add -f`) or upload them elsewhere. The `six_seeds_main/` symlinks currently use absolute paths on this machine; step 3 above recreates them as relative links.
5. **GCS:** the `pusht_test/` scripts live only in a local clone of someone else's repo. Fork `planning-through-contact`, or copy `pusht_test/` into Global-MPPI, so the GCS numbers can be reproduced.
