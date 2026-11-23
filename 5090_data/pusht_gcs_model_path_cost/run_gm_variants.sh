#!/bin/bash
# Global-MPPI variants of the final MPPI step (OPENBLAS_NUM_THREADS=1, else as gm_blas1):
#   gm_nomppi  : --no-ksos-mppi (execute the KSOS plan)
#   gm_noise0.2: --mppi-noise 0.2
S=$HOME/.cache/global-mppi/pusht_5090
W=$S/Global-MPPI-pathcost
OUT=$(cd "$(dirname "$0")" && pwd)
PY=$S/venv/bin/python
LOG=$OUT/gm_variants.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
ok() { f=$(ls $OUT/$1/s$2/*robot_pos_history*.npy 2>/dev/null) && [ "$($PY -c "import numpy;print(len(numpy.load('$f')))")" = 101 ] && ! grep -q "KSOS solver failed" $OUT/console/${1}_s$2.log; }
# wait for >= 12 GB free GPU memory (other jobs filling the GPU made cuSOLVER fail with error 7)
waitgpu() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 12000 ]; do sleep 60; done; }
say "=== gm variants start (pid $$), code $(git -C $W rev-parse --short HEAD)"
for V in gm_nomppi gm_noise0.2; do
  case $V in gm_nomppi) X="--mppi-noise 0.1 --no-ksos-mppi";; gm_noise0.2) X="--mppi-noise 0.2";; esac
  for s in 0 1 2 3 4 5; do
    if ok $V $s; then say "SKIP $V s$s"; continue; fi
    for att in 1 2 3 4 5; do
      waitgpu; say "START $V s$s (attempt $att)"; rm -rf $OUT/$V/s$s
      (cd $W && env OPENBLAS_NUM_THREADS=1 PYTHONPATH=$W XLA_PYTHON_CLIENT_PREALLOCATE=false CUKERNELSOS_PYTHON=$S/cuKernelSOS/python KSOS_GPU_SOLVE=1 \
        timeout 3600 $PY $OUT/run_with_own_cache.py examples/pusht.py --no-wandb --headless --gcs-model --cost path \
        --seed $s --out-dir $OUT/$V/s$s --ksos-solver ADMM-cuda --ksos-sampling cacs --ksos-restarts 5 --ksos-warm-start \
        $X --gm-temperature 0.1 globalmppi) > $OUT/console/${V}_s$s.log 2>&1
      if ok $V $s; then say "END   $V s$s ok"; break; fi
      say "FAILED $V s$s attempt $att: $(tail -1 $OUT/console/${V}_s$s.log)"
    done
  done
done
say "=== gm variants end"
