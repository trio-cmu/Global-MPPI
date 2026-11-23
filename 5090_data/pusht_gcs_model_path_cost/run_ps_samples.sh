#!/bin/bash
# PS sample-count sweep: 2000 (timing reference, same GPU conditions), 4000, 8000, 16000, 32000 samples
# per iteration, 5 iterations (unchanged), seeds 0-5. Same command as run_batch.sh otherwise. Output: ps_n<N>/s<seed>.
S=$HOME/.cache/global-mppi/pusht_5090
W=$S/Global-MPPI-pathcost
OUT=$(cd "$(dirname "$0")" && pwd)
PY=$S/venv/bin/python
LOG=$OUT/ps_samples.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
run() {
  local n=$1 s=$2 att=0 d=ps_n$1
  if ls $OUT/$d/s$s/*robot_pos_history*.npy >/dev/null 2>&1; then say "SKIP  $d s$s"; return; fi
  while [ $att -lt 3 ]; do
    att=$((att + 1)); say "START $d s$s (attempt $att)"
    rm -rf $OUT/$d/s$s
    (cd $W && env PYTHONPATH=$W XLA_PYTHON_CLIENT_PREALLOCATE=false CUKERNELSOS_PYTHON=$S/cuKernelSOS/python KSOS_GPU_SOLVE=1 \
      timeout 3600 $PY $OUT/run_with_own_cache.py examples/pusht.py --no-wandb --headless --gcs-model --cost path \
      --seed $s --out-dir $OUT/$d/s$s --num-samples $n ps) > $OUT/console/${d}_s$s.log 2>&1
    if ls $OUT/$d/s$s/*robot_pos_history*.npy >/dev/null 2>&1; then say "END   $d s$s ok"; return; fi
    say "FAILED $d s$s attempt $att: $(tail -1 $OUT/console/${d}_s$s.log)"
  done
  say "GAVE UP $d s$s"
}
say "=== PS sample sweep start (pid $$), code $(git -C $W rev-parse --short HEAD)"
NS=${@:-2000 4000 8000 16000 32000}
for n in $NS; do for s in 0 1 2 3 4 5; do run $n $s; done; done
say "=== PS sample sweep end: $(grep -c 'GAVE UP' $LOG) gave up"
