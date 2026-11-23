#!/bin/bash
# Global-MPPI (cuKernelSOS) rerun with OPENBLAS_NUM_THREADS=1 (gm_profile/: the banded Cholesky in build_problem
# took ~350 ms per restart with OpenBLAS's default threads, ~27 ms with one). Same settings as run_batch.sh's gm.
# Output: gm_blas1/s<seed>.
S=$HOME/.cache/global-mppi/pusht_5090
W=$S/Global-MPPI-pathcost
OUT=$(cd "$(dirname "$0")" && pwd)
PY=$S/venv/bin/python
LOG=$OUT/gm_blas1.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
say "=== gm OPENBLAS_NUM_THREADS=1 start (pid $$), code $(git -C $W rev-parse --short HEAD)"
for s in 0 1 2 3 4 5; do
  if ls $OUT/gm_blas1/s$s/*robot_pos_history*.npy >/dev/null 2>&1; then say "SKIP s$s"; continue; fi
  for att in 1 2 3; do
    say "START s$s (attempt $att)"; rm -rf $OUT/gm_blas1/s$s
    (cd $W && env OPENBLAS_NUM_THREADS=1 PYTHONPATH=$W XLA_PYTHON_CLIENT_PREALLOCATE=false CUKERNELSOS_PYTHON=$S/cuKernelSOS/python KSOS_GPU_SOLVE=1 \
      timeout 3600 $PY $OUT/run_with_own_cache.py examples/pusht.py --no-wandb --headless --gcs-model --cost path \
      --seed $s --out-dir $OUT/gm_blas1/s$s --ksos-solver ADMM-cuda --ksos-sampling cacs --ksos-restarts 5 --ksos-warm-start \
      --mppi-noise 0.1 --gm-temperature 0.1 globalmppi) > $OUT/console/gm_blas1_s$s.log 2>&1
    if ls $OUT/gm_blas1/s$s/*robot_pos_history*.npy >/dev/null 2>&1; then say "END   s$s ok"; break; fi
    say "FAILED s$s attempt $att: $(tail -1 $OUT/console/gm_blas1_s$s.log)"
  done
done
say "=== gm OPENBLAS_NUM_THREADS=1 end"
