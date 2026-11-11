#!/bin/bash
# CACS x5 (main config) with the Cholesky-fill-aware CACS rule (global_mppi/cacs.py), MuJoCo PushT, seeds 0-5.
# Same command as docs/pusht_benchmark_reproduce.md; logs: seeds/globalmppi_cacs5fill_seed<s>.log.
# A run can segfault while JAX/XLA compiles at cycle 0: retry; attempts 4-5 add
# XLA_FLAGS=--xla_disable_hlo_passes=constant_folding (same time per cycle; see run_gcs_model_seeds.sh).
cd "$(dirname "$0")/../../../.."   # Global-MPPI root
PY=${PY:-python}
D=examples/single_run_test/PushT/closed_loop/seeds
for s in ${SEEDS:-0 1 2 3 4 5}; do
  log=$D/globalmppi_cacs5fill_seed$s.log
  for a in 1 2 3 4 5; do
    if [ $a -ge 4 ]; then export XLA_FLAGS=--xla_disable_hlo_passes=constant_folding; else unset XLA_FLAGS; fi
    $PY examples/pusht.py --no-wandb --headless --seed $s --ksos-solver ADMM-cuda --ksos-sampling cacs \
      --ksos-restarts 5 --ksos-warm-start --mppi-noise 0.1 globalmppi > $log 2>&1
    grep -q "Cycle 100" $log && break
    echo "seed $s attempt $a failed"
  done
  echo "done seed $s"
done
