#!/bin/bash
# PushT benchmark in GCS's quasi-static pusher-slider model (global_mppi/tasks/pusht_gcs.py): the same
# commands as docs/pusht_benchmark_reproduce.md section 1, plus --gcs-model. Run from the Global-MPPI root.
# A run that stops before cycle 100 (e.g. a ptxas crash while compiling) is retried up to 4 times.
# NO_CACS=1 skips CACS (about 12 min per seed).
# CACS seeds 2, 3 and 5 always crash XLA's GPU compiler (segfault compiling controller.optimize); they were
# run with XLA_FLAGS=--xla_disable_hlo_passes=constant_folding, which compiles them (same time per cycle).
D=examples/single_run_test/PushT/closed_loop/gcs_model_seeds
PY=${PY:-python}
mkdir -p $D
run() {  # run <log name> <args...>
  local log=$D/$1.log; shift
  for attempt in 1 2 3 4 5; do
    [ -f $log ] && grep -q "Cycle 100" $log && return
    $PY examples/pusht.py --no-wandb --headless --gcs-model "$@" > $log 2>&1
  done
}
for s in 0 1 2 3 4 5; do
  for m in ps mppi dial; do
    run ${m}2000_seed$s --seed $s --num-samples 2000 $m; echo "done ${m}2000 seed $s"
  done
  [ -n "$NO_CACS" ] && continue  # NO_CACS=1: PS / MPPI / DIAL only
  run globalmppi_cacs5new_seed$s --seed $s --ksos-solver ADMM-cuda --ksos-sampling cacs --ksos-restarts 5 \
    --ksos-warm-start --mppi-noise 0.1 globalmppi; echo "done globalmppi_cacs5new seed $s"
done
echo ALL DONE
