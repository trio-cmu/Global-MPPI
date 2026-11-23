#!/bin/bash
# G1 standup benchmark on the 5090 (benchmarks code @ 7d71c56), after the queue in run_next.sh (PID 755857) finishes.
S=/home/robopt/.cache/global-mppi/pusht_5090
while [ -e /proc/755857 ]; do sleep 60; done
echo "$(date -Is) G1 start"
cd $S/Global-MPPI && env PYTHON=$S/venv/bin/python CUKERNELSOS_PYTHON=$S/cuKernelSOS/python MUJOCO_GL=egl \
    ALGOS="ps mppi dial mtp cbo gm" SEEDS="0 1 2 3 4 5" MAXCYC=100 \
    bash $S/Global-MPPI/script/run_g1_benchmark.sh /home/robopt/Documents/zhongqi/Global-MPPI/5090_data/g1_benchmark
echo "$(date -Is) G1 exit $?"
