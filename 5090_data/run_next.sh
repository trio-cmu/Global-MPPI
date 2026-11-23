#!/bin/bash
# Continues the 5090 queue after the PushT benchmark (runner PID 676525): PushT with the GCS cost, then double cart-pole.
S=/home/robopt/.cache/global-mppi/pusht_5090
while [ -e /proc/676525 ]; do sleep 30; done
echo "$(date -Is) PushT benchmark runner exited"
echo "$(date -Is) PushT GCS-cost start"
$S/venv/bin/python -u /home/robopt/Documents/zhongqi/Global-MPPI/5090_data/pusht_gcs_cost/run_benchmark.py
echo "$(date -Is) PushT GCS-cost exit $?"
echo "$(date -Is) double_cart_pole start"
env QUEUE_TS=$(date +%Y%m%d_%H%M%S) TASKS=double_cart_pole RUNS_DIR=/home/robopt/Documents/zhongqi/Global-MPPI/5090_data/double_cart_pole_benchmark \
    CUKS_GPU=$S/cuKernelSOS/python PY=$S/venv/bin/python $S/Global-MPPI/script/overnight_queue.sh --run
echo "$(date -Is) double_cart_pole exit $?"
