#!/bin/bash
# Overnight 5090 queue (benchmarks code @ 7d71c56, pinned env in /home/robopt/.cache/global-mppi/pusht_5090), one GPU job at a time:
#   1. G1 standup   -> g1_benchmark/            (script/run_g1_benchmark.sh, 6 methods x seeds 0-5, 100 cycles)
#   2. cube         -> cube_benchmark/          (script/run_cube_benchmark.sh, 6 methods x seeds 0-5, 100 cycles)
#   3. G1 walk      -> g1_walk_benchmark/<ts>/  (script/overnight_queue.sh TASKS=g1_walk, 6 x 6, 400 cycles)
#   4. double cart-pole with CACS h = 0.3 -> double_cart_pole_h0.3/ (6 x 6, 100 cycles; h = 0.7 stalls CACS in this 4-D control space)
# CACS h of 1-3 checked with 10-cycle GlobalMPPI runs (CACS fills 1000 samples in 3-4 rounds): unchanged defaults.
S=/home/robopt/.cache/global-mppi/pusht_5090
D=/home/robopt/Documents/zhongqi/Global-MPPI/5090_data
PY=$S/venv/bin/python
W=$S/Global-MPPI
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
say() { echo "$(date -Is) $*"; }

say "G1 start"
(cd $W && env PYTHON=$PY CUKERNELSOS_PYTHON=$S/cuKernelSOS/python ALGOS="ps mppi dial mtp cbo gm" SEEDS="0 1 2 3 4 5" MAXCYC=100 \
    bash $W/script/run_g1_benchmark.sh $D/g1_benchmark)
say "G1 exit $?"

say "cube start"
(cd $W && env PYTHON=$PY CUKERNELSOS_PYTHON=$S/cuKernelSOS/python CONFIGS="ps mppi dial mtp cbo gm" SEEDS="0 1 2 3 4 5" MAXCYC=100 \
    bash $W/script/run_cube_benchmark.sh $D/cube_benchmark)
say "cube exit $?"

say "g1_walk start"
(cd $W && env QUEUE_TS=$(date +%Y%m%d_%H%M%S) TASKS=g1_walk RUNS_DIR=$D/g1_walk_benchmark CUKS_GPU=$S/cuKernelSOS/python PY=$PY \
    bash $W/script/overnight_queue.sh --run)
say "g1_walk exit $?"

say "double_cart_pole h=0.3 start"
O=$D/double_cart_pole_h0.3; mkdir -p $O/logs
for s in 0 1 2 3 4 5; do for m in ps mppi dial mtp cbo gm; do
  [ -f $O/$m/s$s/final.json ] && { say "SKIP dcp $m s$s"; continue; }
  if [ $m = gm ]; then arg="--cacs-h 0.3 --gpu globalmppi"; to=7200; else arg=$m; to=3600; fi
  for a in 1 2 3; do
    say "START dcp $m s$s attempt $a"
    (cd $W && timeout $to env PYTHONPATH=$W KSOS_GPU_SOLVE=1 CUKERNELSOS_PYTHON=$S/cuKernelSOS/python \
        $PY examples/double_cart_pole.py --headless --seed-start $s --seed-end $((s+1)) --max-cycles 100 --out-dir $O $arg) \
        > $O/logs/${m}_s$s.attempt$a.log 2>&1
    rc=$?
    if [ -f $O/$m/s$s/final.json ]; then say "END dcp $m s$s ok"; break; fi
    say "FAILED dcp $m s$s attempt $a (exit $rc)"; [ $rc -eq 124 ] && break
  done
done; done
$PY $W/script/final_figs.py $O double_cart_pole $O > $O/plot.log 2>&1
say "double_cart_pole h=0.3 end"
say "ALL DONE"
