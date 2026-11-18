#!/bin/bash
# CACS x5 closed loop (same command as run_pushercost_cacs_seeds.sh) with tuned CacsConfig overrides (pusht.py --cacs),
# 6 seeds each, into seeds_pushercost_tuned/globalmppi_cacs5<name>_seed<s>.log. Retries crashes; pinned off the CPU
# cores with machine-check errors (core 16 = CPUs 8, 9; core 20 = CPUs 10, 11).
cd "$(dirname "$0")/../../../.."   # Global-MPPI root
D=examples/single_run_test/PushT/closed_loop/seeds_pushercost_tuned
PY="taskset -c 0-7,12-31 $HOME/miniconda3/envs/ksos/bin/python"
for cfg in ${CFGS:-'h12:{"h": 1.2}' 'mc10:{"max_clique": 10}'}; do
  name=${cfg%%:*}; json=${cfg#*:}
  for s in ${SEEDS:-0 1 2 3 4 5}; do
    log=$D/globalmppi_cacs5$name\_seed$s.log
    for a in 1 2 3 4 5; do
      if [ $a -ge 4 ]; then export XLA_FLAGS=--xla_disable_hlo_passes=constant_folding; else unset XLA_FLAGS; fi
      $PY examples/pusht.py --no-wandb --headless --seed $s --ksos-solver ADMM-cuda --ksos-sampling cacs \
        --ksos-restarts 5 --ksos-warm-start --mppi-noise 0.1 --cacs "$json" globalmppi > $log 2>&1
      grep -q "Cycle 100.0: Best cost" $log && { echo "done $name seed $s $(ls -td logs/pushT_globalmppi_globalmppi/*/ | head -1)" >> $D/progress.log; break; }
      echo "$name seed $s attempt $a failed" >> $D/progress.log
    done
  done
done
echo "all done" >> $D/progress.log
