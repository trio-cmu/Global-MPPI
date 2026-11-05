#!/bin/bash
# LSE-smoothed cube cost surfaces, same sigmas as examples/pusht_cost_plot
cd "$(dirname "$0")"
for s in 0.001 0.005 0.01 0.025 0.05 0.1 0.25 0.3; do
  ~/miniconda3/envs/ksos/bin/python cube_cost_surface.py --lse-sigma $s > run_surface_lse$s.log 2>&1
  echo "sigma $s: $(tail -1 run_surface_lse$s.log)"
done
