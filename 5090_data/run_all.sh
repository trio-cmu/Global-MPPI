#!/bin/bash
# Sequential 5090 queue: PushT (6 methods x 6 seeds), then double cart-pole (6 x 6), benchmarks code @ 7d71c56.
S=/home/robopt/.cache/global-mppi/pusht_5090
echo "$(date -Is) PushT start"
$S/venv/bin/python -u /home/robopt/Documents/zhongqi/Global-MPPI/5090_data/pusht_benchmark/run_benchmark.py
echo "$(date -Is) PushT exit $?"
echo "$(date -Is) double_cart_pole start"
env QUEUE_TS=$(date +%Y%m%d_%H%M%S) TASKS=double_cart_pole RUNS_DIR=/home/robopt/Documents/zhongqi/Global-MPPI/5090_data/double_cart_pole_benchmark \
    CUKS_GPU=$S/cuKernelSOS/python PY=$S/venv/bin/python $S/Global-MPPI/script/overnight_queue.sh --run
echo "$(date -Is) double_cart_pole exit $?"
