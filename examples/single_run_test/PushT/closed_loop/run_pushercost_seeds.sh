#!/bin/bash
# PS / MPPI / DIAL (2000 samples x 5), PushT cost = pose (50 |dCOM|^2 + 15*2(1-cos th)) + 10 * pusher path off
# contact (global_mppi/tasks/pusht.py), 6 seeds, 100 cycles, one run at a time. Retries crashes; off CPU core 16.
cd "$(dirname "$0")/../../../.."   # Global-MPPI root
D=examples/single_run_test/PushT/closed_loop/${OUT:-seeds_pushercost}
PY="taskset -c 0-7,10-31 $HOME/miniconda3/envs/ksos/bin/python"
for s in ${SEEDS:-0 1 2 3 4 5}; do
  for m in ${METHODS:-ps mppi dial}; do
    log=$D/${m}2000_seed$s.log
    for attempt in 1 2 3 4 5; do
      $PY examples/pusht.py --no-wandb --headless --seed $s --num-samples 2000 $m > $log 2>&1
      grep -q "Cycle 100.0: Best cost" $log && { echo "done $m seed $s $(ls -td logs/pushT_globalmppi_$m/*/ | head -1)" >> $D/progress.log; break; }
      echo "$m seed $s attempt $attempt failed" >> $D/progress.log
    done
  done
done
echo "all done" >> $D/progress.log
