#!/bin/bash
# PS / MPPI / DIAL (2000 samples x 5) on the GCS-shared PushT cost, 6 seeds, 100 cycles (2 s), one run at a time
# (clean planning times). Retries on crashes; pinned off CPU core 16 (CPUs 8, 9).
# Usage (from this folder): bash run_newcost_seeds.sh   [env: SEEDS, METHODS, CYCLES]
cd "$(dirname "$0")/../../../.."   # Global-MPPI root
D=examples/single_run_test/PushT/closed_loop/seeds_newcost
PY="taskset -c 0-7,10-31 $HOME/miniconda3/envs/ksos/bin/python"
C=${CYCLES:-100}
for s in ${SEEDS:-0 1 2 3 4 5}; do
  for m in ${METHODS:-ps mppi dial}; do
    log=$D/${m}2000_seed$s.log
    for attempt in 1 2 3 4 5; do
      $PY examples/pusht.py --no-wandb --headless --seed $s --num-samples 2000 --max-cycles $C $m > $log 2>&1
      grep -q "Cycle $C.0: Best cost" $log && { echo "done $m seed $s" >> $D/progress.log; break; }
      echo "$m seed $s attempt $attempt failed" >> $D/progress.log
    done
  done
done
echo "all done" >> $D/progress.log
