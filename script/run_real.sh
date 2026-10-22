#!/usr/bin/env bash
# Standard real-robot Push-T closed-loop run (GPU planner, drain camera, no manual go files).
#
#   ./script/run_real.sh <ps|dial|mppi|globalmppi> <run_name> [extra closed_loop_test.py flags]     # output: runs/closed_loop_test/<run_name>
#
# Before you start: T on start_2 (the bring-up aborts above 3 mm / 3 deg), hand on the enable button (half-pressed), end effector
# within 30 deg of the wanted yaw (xy_driver refuses otherwise), state_pub / camera alive on the NUC. The script runs the pre-flight,
# prints a 5 s countdown (Ctrl-C cancels, nothing has moved yet) and then moves, goes down and runs (--auto-start, EXPERIMENTAL).
# Always use a NEW run name. After the run: fig_trajectory / fig_error_vs_time / fig_cycle_timeline / fig_plan_time .png in the run directory and
# one row appended to runs/closed_loop_test/summary.csv (global_mppi/closed_loop/run_figures.py; /usr/bin/python3, override with FIGPY=).
# Override the interpreter with PYTHON=/path/to/python.
set -euo pipefail
if [ $# -lt 2 ]; then echo "usage: $0 <ps|dial|mppi|globalmppi> <run_name> [extra flags]" >&2; exit 2; fi
ALGO=$1; NAME=$2; shift 2
case "$ALGO" in ps|dial|mppi|globalmppi) ;; *) echo "algo must be ps, dial, mppi or globalmppi, got '$ALGO'" >&2; exit 2;; esac
cd "$(dirname "$0")/.."
OUT="runs/closed_loop_test/$NAME"
if [ -e "$OUT" ]; then echo "$OUT already exists: use a new run name" >&2; exit 2; fi
PYTHON=${PYTHON:-/home/junlong/miniforge3/envs/global-mppi/bin/python}
rm -f /tmp/go /tmp/go.down
unset JAX_PLATFORMS                                   # GPU (jax default)
export LCM_DEFAULT_URL="${LCM_DEFAULT_URL:-udpm://239.255.76.67:7667?ttl=1}"
# The run (including its own cleanup: arm up, camera restored) finishes inside python; only THEN the figures / summary are made, and
# whatever they do (or fail to do) cannot change the run's exit code. Ctrl-C (also during the countdown) goes to python; the script then
# still writes the summary of what exists.
trap ':' INT
set +e
"$PYTHON" examples/closed_loop_test.py --real --algo "$ALGO" --pipeline --blend --measure-every 1 --cam-correct \
  --max-cycles 40 --max-time 60 --plateau-chunks 15 --plateau-mm 1.0 --adaptive-settle --read-s 0.1 --bg-link-check \
  --fast-stop --min-tags 3 --max-reproj-px 2.0 --max-fallbacks 2 --pose-bin build_interp_drain/bin/aprilcube_pose \
  --auto-start --out "$OUT" "$@"
RC=$?
echo "== run finished (exit code $RC): figures + summary"
timeout 180 "${FIGPY:-/usr/bin/python3}" global_mppi/closed_loop/run_figures.py "$OUT" --summary runs/closed_loop_test/summary.csv \
  || echo "(figures / summary failed: the run itself is not affected)"
exit $RC
