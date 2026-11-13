# Reproducing the cube (in-hand rotation) benchmark

2026-11-13 · reproduces the cost convergence plot of the LEAP-hand cube task: PS, MPPI, DIAL, GlobalMPPI, 100 control cycles, seeds 0–5.

Results below were produced on the code of `58f9add` (clean tree, `git_dirty: false` in every `args.json`).

## Repos and branches

| Repo | Remote | Branch | Commit | Role |
| --- | --- | --- | --- | --- |
| **Global-MPPI** | `https://github.com/trio-cmu/Global-MPPI` | `cube-gpu` | `58f9add` (+ the commit that adds this file) | planners, cube task, `examples/cube.py`, `script/` |
| **ksos-tools** (submodule of Global-MPPI) | `https://github.com/Zhongqi2/ksos-tools` | `gpu-solve-exp` | `86e6163` | KSOS problem building, `ADMM.solve_ksos` (records `solve_path`) |
| **cuKernelSOS** | `https://github.com/ssisyg/cuKernelSOS` | `gpu-solve` | `2768612` (on top of `4180908`) | GPU ADMM solver with the `gpu_solve` option |

`examples/cube.py` takes the cuKernelSOS binding from `CUKERNELSOS_PYTHON`; if it is unset it uses the first `cuKernelSOS/python` in a parent folder, which is **not** the `gpu-solve` build (then `gpu_solve` is not used, see "Check that the GPU path was used"). Export it explicitly:

```
kernelSOS_project/
├── Global-MPPI/            (cube-gpu; ksos-tools/ inside, gpu-solve-exp)
└── cuKernelSOS_gpu/        (gpu-solve, built in place)
export CUKERNELSOS_PYTHON=$PWD/cuKernelSOS_gpu/python
```

## Environment

- GPU: NVIDIA GeForce RTX 5090 (32 GB), driver 595.99.02. It was **shared** with other users' jobs, so planning times are noisy.
- Conda env `ksos`: Python 3.12.16, jax / jaxlib 0.5.3 (`jax-cuda12-plugin` 0.5.3, CUDA 12.9 runtime from pip), numpy 2.5.4, `global-mppi` and `ksos_tools` installed editable.
- **mujoco / mujoco-mjx 3.3.6.** `pyproject.toml` does not pin them (`mujoco>=3.2.4`), and a plain install now gets 4.0.0, whose contact dynamics differ, so results change. Install the pinned version: `pip install mujoco==3.3.6 mujoco-mjx==3.3.6`. `args.json` records both versions of every run.
- `mujoco-warp` 4.0.0 is also installed (listed in `pyproject.toml`); pip warns that it wants `mujoco>=3.13`. `cube.py` does not use it, and we left it alone.
- cuKernelSOS is built with CUDA 12.9 (`/usr/local/cuda-12.9/bin/nvcc`) for the 5090 (`sm_120`).

## Build cuKernelSOS (gpu-solve)

```bash
git clone -b gpu-solve https://github.com/ssisyg/cuKernelSOS cuKernelSOS_gpu    # 2768612
cd cuKernelSOS_gpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUKERNELSOS_BUILD_PYTHON=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DPython_EXECUTABLE=$(which python)
cmake --build build -j4        # puts _cukernelsos.cpython-312-*.so into python/cukernelsos/
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU. The `.so` is for Python 3.12 (ours was built with another 3.12 env and loads in `ksos`). Global-MPPI submodule: `git submodule update --init ksos-tools` (→ `86e6163`).

## Run

From the `Global-MPPI` root, with the `ksos` env active:

```bash
export CUKERNELSOS_PYTHON=<path>/cuKernelSOS_gpu/python
script/run_cube_benchmark.sh runs/cube_benchmark        # configs ps mppi dial gm, seeds 0-5, 100 cycles
```

One run is:

```bash
python examples/cube.py --headless --seed-start 3 --seed-end 4 --max-cycles 100 --out-dir runs/cube_benchmark/ps ps     # also mppi, dial
python examples/cube.py --headless --seed-start 3 --seed-end 4 --max-cycles 100 --out-dir runs/cube_benchmark/gm --gpu globalmppi
```

- **`--gpu` only for GlobalMPPI** (CACS, 5 restarts × 1000 samples, KSOS on ADMM-cuda with `gpu_solve`, then MPPI with noise 0.1, warm start on). PS, MPPI and DIAL (2000 samples, 5 iterations) do not take it.
- The script loops seeds outermost, skips runs whose `final.json` exists, retries a failed run up to 2 times (GPU runs sometimes crash in the JIT of cycle 0), and logs everything to `<out>/batch.log` (per-run logs in `<out>/logs/`). Env variables `SEEDS`, `CONFIGS` (add `gm_noise0.02` for `--mppi-noise 0.02`), `MAXCYC`, `PYTHON` change the defaults.
- Time on the shared 5090: PS / MPPI / DIAL 1–3.5 min per run, GlobalMPPI 12–35 min (about 13 s per cycle). The whole batch (6 seeds × 5 configs including `gm_noise0.02`) took about 5 h.
- Same-seed reruns on the GPU are not bit-identical. Compare medians over seeds, not single runs.

### Check that the GPU path was used

`final.json` of a GlobalMPPI run has `solve_paths`, the count of ADMM solves per path:

```bash
python -c "import json;print(json.load(open('runs/cube_benchmark/gm/gm/s0/final.json'))['solve_paths'])"
# {'gpu': 505}  -> all solves used gpu_solve.  A key "cpu (reason)" means the GPU solve was not used,
# typically because CUKERNELSOS_PYTHON points to a build without gpu_solve.
```

`args.json` also has `cukernelsos_python`, the `git_commit` / `git_dirty` of Global-MPPI and the mujoco versions.

## Output files

`<out>/<config>/<algo>/s<seed>/` (`<algo>` is `ps`, `mppi`, `dial` or `gm`):

- `cycles.csv`: per cycle `cycle, plan_s, best_cost, pos_err_mm, ang_err_deg, gpu_util, gpu_mem_mb`. Cycles 0–99, the error is measured before the cycle's step. `plan_s` of cycle 0 includes JIT.
- `final.json`: `pos_err_mm`, `ang_err_deg` after the 100th cycle, `cycles`, and `solve_paths` (GlobalMPPI).
- `args.json`: command line, controller settings, code provenance.
- `cube_nomppi_<alg>_best_cost_history_seed<k>.npy`: `best_cost` for cycles 0..100 (101 values; same numbers as the csv column for 0–99). It is the planner's predicted cost of its planned mean trajectory (running + terminal, summed over the horizon), not the realised cost. The cube cost has a position term (xy running, xyz terminal) and a control term besides the orientation term.
- `cube_nomppi_<alg>_robot_pos_history_seed<k>.npy`: qpos at cycles 0..100 (101 × 23).
- `*.mp4`, `<alg>_timing_seed<k>.txt` (GlobalMPPI).

## Plot the cost convergence

```bash
python script/plot_cube_cost.py runs/cube_benchmark --out cube_cost.png            # median + 25–75 % band, log y
python script/plot_cube_cost.py runs/cube_benchmark --old data/logs --out cube_cost_vs_old.png   # next to the old runs in data/logs
```

It prints the median and quartiles at iteration 100.

## Reference result

Median over 6 seeds at iteration 100 (the `best_cost` of cycle 100), 25–75 % range in brackets. Final error after 100 cycles is mean ± std (ddof 1) over seeds. Run on a **shared RTX 5090**, mujoco 3.3.6, commit `58f9add`.

| Method | Median cost, iteration 100 | Final position error (mm) | Final orientation error (deg) | Plan time per cycle (without cycle 0) |
| --- | --- | --- | --- | --- |
| PS | 0.467 (0.362–0.551) | 22.2 ± 4.9 | 63.4 ± 21.5 | 0.98 s |
| MPPI | 0.089 (0.049–0.173) | 20.1 ± 5.9 | 37.9 ± 27.9 | 1.01 s |
| DIAL | 0.031 (0.020–0.044) | 18.0 ± 7.6 | 18.1 ± 7.5 | 0.98 s |
| **GlobalMPPI** (`--gpu`) | **0.0048 (0.0047–0.0055)** | **11.4 ± 5.4** | **6.2 ± 4.6** | 12.9 s |

- `gm_noise0.02` (`--mppi-noise 0.02`): final error 12.5 ± 7.2 mm and 7.0 ± 6.2°, 11.2 s per cycle, not different from the default at 6 seeds.
- Planning times were measured while other jobs used the GPU; use them as an order of magnitude.
- The position error at cycle 100 is not converged (it moves by a few mm between cycles 75 and 100); the orientation error is.
