#!/bin/bash
# Resume pusht_cacs_param_sweep.py one parameter per process (results: pusht_cacs_param_sweep_<param>.json),
# retrying crashes. Pinned off the CPU cores with machine-check errors (core 16 = CPUs 8, 9; core 20 = CPUs 10, 11).
cd "$(dirname "$0")"
PY="taskset -c 0-7,12-31 $HOME/miniconda3/envs/ksos/bin/python"
for p in ${PARAMS:-reuse max_clique n global_every mppi_noise restarts}; do
  for a in 1 2 3; do
    $PY pusht_cacs_param_sweep.py --only $p --seeds 0 1 2 3 4 --json pusht_cacs_param_sweep_$p.json > pusht_cacs_param_sweep_$p.log 2>&1 \
      && { echo "done $p" >> pusht_cacs_param_sweep_progress.log; break; }
    echo "$p attempt $a failed" >> pusht_cacs_param_sweep_progress.log
  done
done
echo "all done" >> pusht_cacs_param_sweep_progress.log
