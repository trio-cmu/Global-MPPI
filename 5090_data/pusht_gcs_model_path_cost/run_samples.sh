#!/bin/bash
# Sample-count sweep for the baselines: METHODS x NS x seeds 0-5, 5 iterations (unchanged), otherwise exactly as
# run_batch.sh. Output: <method>_n<N>/s<seed>. Usage: METHODS="dial mppi mtp cbo" NS="2000 ... 64000" ./run_samples.sh
S=$HOME/.cache/global-mppi/pusht_5090
W=$S/Global-MPPI-pathcost
OUT=$(cd "$(dirname "$0")" && pwd)
PY=$S/venv/bin/python
LOG=$OUT/samples_sweep.log
METHODS=${METHODS:-dial mppi mtp cbo}
NS=${NS:-2000 4000 8000 16000 32000 64000}
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
run() {
  local m=$1 n=$2 s=$3 att=0 d=$1_n$2
  if ls $OUT/$d/s$s/*robot_pos_history*.npy >/dev/null 2>&1; then say "SKIP  $d s$s"; return; fi
  while [ $att -lt 3 ]; do
    att=$((att + 1)); say "START $d s$s (attempt $att)"
    rm -rf $OUT/$d/s$s
    (cd $W && env PYTHONPATH=$W XLA_PYTHON_CLIENT_PREALLOCATE=false CUKERNELSOS_PYTHON=$S/cuKernelSOS/python KSOS_GPU_SOLVE=1 \
      timeout 3600 $PY $OUT/run_with_own_cache.py examples/pusht.py --no-wandb --headless --gcs-model --cost path \
      --seed $s --out-dir $OUT/$d/s$s --num-samples $n $m) > $OUT/console/${d}_s$s.log 2>&1
    if ls $OUT/$d/s$s/*robot_pos_history*.npy >/dev/null 2>&1; then say "END   $d s$s ok"; return; fi
    say "FAILED $d s$s attempt $att: $(tail -1 $OUT/console/${d}_s$s.log)"
  done
  say "GAVE UP $d s$s"
}
say "=== sample sweep start (pid $$), methods [$METHODS], samples [$NS], code $(git -C $W rev-parse --short HEAD)"
for n in $NS; do for m in $METHODS; do for s in 0 1 2 3 4 5; do run $m $n $s; done; done; done
say "=== sample sweep end: $(grep -c 'GAVE UP' $LOG) gave up"
