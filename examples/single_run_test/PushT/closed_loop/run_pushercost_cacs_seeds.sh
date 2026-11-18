#!/bin/bash
# CACS x5 (main config: ADMM-cuda, cacs sampling, 5 warm-started restarts, MPPI noise 0.1) on the current PushT
# cost (pose + 10 * pusher path off contact), 6 seeds, 100 cycles, into seeds_pushercost/globalmppi_cacs5new_seed<s>.log.
# Retries crashes (attempts 4-5 disable XLA constant folding); off CPU core 16.
cd "$(dirname "$0")/../../../.."   # Global-MPPI root
D=examples/single_run_test/PushT/closed_loop/seeds_pushercost
PY="taskset -c 0-7,10-31 $HOME/miniconda3/envs/ksos/bin/python"
for s in ${SEEDS:-0 1 2 3 4 5}; do
  log=$D/globalmppi_cacs5new_seed$s.log
  for a in 1 2 3 4 5; do
    if [ $a -ge 4 ]; then export XLA_FLAGS=--xla_disable_hlo_passes=constant_folding; else unset XLA_FLAGS; fi
    $PY examples/pusht.py --no-wandb --headless --seed $s --ksos-solver ADMM-cuda --ksos-sampling cacs \
      --ksos-restarts 5 --ksos-warm-start --mppi-noise 0.1 globalmppi > $log 2>&1
    grep -q "Cycle 100.0: Best cost" $log && { echo "done cacs seed $s $(ls -td logs/pushT_globalmppi_globalmppi/*/ | head -1)" >> $D/progress_cacs.log; break; }
    echo "cacs seed $s attempt $a failed" >> $D/progress_cacs.log
  done
done
echo "all done" >> $D/progress_cacs.log
