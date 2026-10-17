#!/usr/bin/env bash
# Baseline REAL-robot run: pipelined closed loop with the velocity-continuous plan hand-over (xy_driver --blend).
#
#   examples/run_baseline_pipe_blend.sh [run_name]        # output: runs/closed_loop_test/<run_name>
#
# Before you start (the script does NOT ask, hand on the enable button):
#   - T block on start_2 (the script's bring-up aborts when it is off by more than 3 mm / 3 deg).
#   - Pen start: pusher_start_pushside_near (default, ~2 cm from the T outline on the pushing side;
#     ~/trio_sim/src/pusht_align/pusher_start_pushside_near.yaml).
#   - NUC driver: ~/projects/trio_infra_upright (branch noee-z-upright-blend), build_upright/bin/xy_driver started with
#     --blend --upright --yaw-deg=59.7 --max-lead-mm=15 (the script starts it; no other XY_CMD sender on the bus).
#   - Camera: aprilcube_pose from build_interp/bin with --interp-flange --cam-delay-ms 60 (set automatically for --pipeline).
#   - NUC link: carrier_changes of enp89s0 must equal the baseline in ~/linkwatch_baseline.txt; multicast route
#     239.255.76.67 on both machines; T_SHAPE_POSE >= 38 Hz (the bring-up checks all of this and aborts otherwise).
#
# What it does: DIAL plans the next 0.5 s chunk from the predicted state while the current one runs; the new plan interrupts the
# running one at the hand-over (remaining time ~ planning time) and continues at its velocity. Every 2 chunks the arm stops, the T
# settles (--adaptive-settle: 3 consecutive frames within 0.5 mm / 0.3 deg), the camera is read at rest (--read-s 0.1 window) and
# planning continues from the measurement. --bg-link-check: carrier_changes is read in a background thread.
#
# Result so far (one run, runs/closed_loop_test/real_pipe_blend_am2, 2026-10-17): converged at 5.4 mm / 0.59 deg after 12 chunks
# (5.2 s of motion, standstill 0.75 s per measurement). SINGLE RESULT - to be confirmed by repeating (the same setup with
# --measure-every 4 diverged: 71 mm / 143 deg).
#
# Notes
#   - --correct is a legacy no-op option (kept only so old command lines still parse).
#   - Plots: if the global-mppi environment has no matplotlib, closed_loop_test.py falls back to /usr/bin/python3 for plotting.
#
# External dependencies (outside this repository; NOT versioned here)
#   - ~/trio_sim (git repo, HEAD acf03b6 on 2026-10-14, with local uncommitted changes in config/lcm_relay.yaml and
#     src/pusht_align/frames.py): src/pusht_align/frames.py (imported by open_loop_test.py), pusht_starts.yaml, pusher_start.yaml,
#     pusher_start_pushside_near.yaml (a copy is in configs/pusher_start_pushside_near.yaml for the record; the code still reads
#     ~/trio_sim/src/pusht_align/), config/lcm_relay.yaml.
#   - ~/projects/trio_infra on this machine: src/mppi_bridge/lcmtypes_py (generated LCM types).
#   - NUC (junlong@192.168.50.2):
#       ~/projects/trio_infra (branch positions-1015, 47fcb1a): script/check_f_t_ee.sh, src/extrinsic_check/robot_state_readonly_pub.py,
#         ~/linkwatch_baseline.txt, build/bin/aprilcube_pose;
#       ~/projects/trio_infra_upright (branch noee-z-upright-blend, base 8175c40): build_upright/bin/xy_driver built WITH --blend.
#         The blend driver (src/franka_driver/blend_trajectory.hpp, xy_driver.cpp changes) is NOT committed yet.
#       build_interp/bin/aprilcube_pose (camera with --interp-flange) in ~/projects/trio_infra.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate global-mppi
export JAX_PLATFORMS=cpu
NAME="${1:-real_pipe_blend_$(date +%Y%m%d_%H%M%S)}"
exec python examples/closed_loop_test.py --real --algo dial --pipeline --blend --measure-every 2 --max-cycles 24 \
    --adaptive-settle --read-s 0.1 --bg-link-check --out "runs/closed_loop_test/${NAME}"
