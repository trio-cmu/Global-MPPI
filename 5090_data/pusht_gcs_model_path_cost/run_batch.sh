#!/bin/bash
# PushT path-cost benchmark in GCS's model (4090_data/pusht_gcs_model_path_cost/README.md, commit d69dfa4) on the RTX 5090.
# Same commands as the README; runs through run_with_own_cache.py (JAX cache in ~/.cache, see there).
# A run whose robot_pos_history exists is skipped; a failed run is retried twice, then logged. Then scores and plots.
S=$HOME/.cache/global-mppi/pusht_5090
W=$S/Global-MPPI-pathcost
OUT=$(cd "$(dirname "$0")" && pwd)
PY=$S/venv/bin/python
LOG=$OUT/batch.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
run() {
  local m=$1 s=$2 att=0 extra
  if ls $OUT/$m/s$s/*robot_pos_history*.npy >/dev/null 2>&1; then say "SKIP  $m s$s"; return; fi
  if [ $m = gm ]; then
    extra="--ksos-solver ADMM-cuda --ksos-sampling cacs --ksos-restarts 5 --ksos-warm-start --mppi-noise 0.1 --gm-temperature 0.1 globalmppi"
  else
    extra="--num-samples 2000 $m"
  fi
  while [ $att -lt 3 ]; do
    att=$((att + 1)); say "START $m s$s (attempt $att)"
    rm -rf $OUT/$m/s$s
    (cd $W && env PYTHONPATH=$W XLA_PYTHON_CLIENT_PREALLOCATE=false CUKERNELSOS_PYTHON=$S/cuKernelSOS/python KSOS_GPU_SOLVE=1 \
      timeout 3600 $PY $OUT/run_with_own_cache.py examples/pusht.py --no-wandb --headless --gcs-model --cost path \
      --seed $s --out-dir $OUT/$m/s$s $extra) > $OUT/console/${m}_s$s.log 2>&1
    if ls $OUT/$m/s$s/*robot_pos_history*.npy >/dev/null 2>&1; then say "END   $m s$s ok"; return; fi
    say "FAILED $m s$s attempt $att: $(tail -1 $OUT/console/${m}_s$s.log)"
  done
  say "GAVE UP $m s$s"
}
mkdir -p $OUT/console
say "=== batch start (pid $$), code $(git -C $W rev-parse --short HEAD)"
for m in ps mppi dial mtp cbo gm; do for s in 0 1 2 3 4 5; do run $m $s; done; done
$PY $W/4090_data/score_path_cost.py $OUT/*/s*/*robot_pos_history*.npy --w 10 --wp 0.1 --json $OUT/path_cost_all.json > $OUT/score.out 2>&1
cp $W/4090_data/pusht_gcs_model_path_cost/plot_path_cost.py $OUT/
(cd $OUT && $PY plot_path_cost.py && $PY plot_path_cost.py --w-final=100) > $OUT/plot.out 2>&1
say "=== batch end: $(grep -c 'GAVE UP' $LOG) gave up"
