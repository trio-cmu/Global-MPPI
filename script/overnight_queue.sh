#!/bin/bash
# script/overnight_queue.sh -- unattended, strictly sequential queue of six-method batches (ps mppi dial mtp cbo gm x 6 seeds x 100 cycles).
#
#   CUKS_GPU=/path/to/cuKernelSOS/python  script/overnight_queue.sh      start detached (nohup + setsid) and print where to look
#   (the queue itself is `script/overnight_queue.sh --run`, started by the launcher; do not call it by hand)
#
# Environment (all optional except CUKS_GPU):
#   CUKS_GPU   cuKernelSOS `python` dir of the gpu-solve build (ssisyg/cuKernelSOS, branch gpu-solve, 2768612); REQUIRED, gm needs it
#   CUKS_OLD   cuKernelSOS python dir for the baselines (default: CUKS_GPU; ps/mppi/dial/mtp/cbo never call cuKernelSOS)
#   PY         python that runs the planners (default: python in PATH; needs the Global-MPPI env, MuJoCo 3.3.6)
#   SYSPY      python that draws the figures (default: PY; needs matplotlib)
#   RUNS_DIR   where the queue writes (default: <repo>/logs/overnight_runs); one folder <RUNS_DIR>/<timestamp> per queue
#   TASKS      tasks to run, any of "pusht walker double_cart_pole bugtrap" (default: pusht; the others come from their own branches
#              `walker` / `double-cart-pole` / `bugtrap`, which must exist locally, through git worktrees in $WT)
#   WT         worktree folder for walker / double_cart_pole / bugtrap (default: <repo>/../overnight_wt)
#   MAX_CYCLES control cycles per run of walker / double_cart_pole / bugtrap (default 100)
#   SEEDS, METHODS  (default: "0 1 2 3 4 5", "ps mppi dial mtp cbo gm")
# PushT runs the code of the checkout this script lives in (check out the branch pusht-bench first).
#
# Nothing is waited for: the queue starts at once and runs one process at a time (no two of our runs overlap).
# Code of walker / double_cart_pole / bugtrap comes from its own git branch through a worktree outside the repo ($WT/<branch>):
#   PushT -> pusht-bench, walker -> walker, double_cart_pole -> double-cart-pole, bugtrap -> bugtrap   (committed code only; PYTHONPATH selects it)
# Per task: <Q>/<task>/batch.log, <Q>/<task>/gpu_snapshots/<method>_s<seed>.txt (nvidia-smi before every run), logs/, results/;
# overall: <Q>/STATUS.txt (rewritten after every run). A failed run is retried (max 2) after ANY crash (nonzero exit, or no finished
# marker), except a timeout (exit code 124), which is not retried; after the retries it is logged "GAVE UP" and the queue goes on. A task that cannot start is "SKIPPED"; a failed or
# skipped task never blocks the next one. At the end of each task the figure and the tables are made (no GCS anywhere).
set -u
REPO=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
export QUEUE_TS=${QUEUE_TS:-$(date +%Y%m%d_%H%M%S)}
Q=${RUNS_DIR:-$REPO/logs/overnight_runs}/$QUEUE_TS
STATUS=$Q/STATUS.txt
WT=${WT:-$REPO/../overnight_wt}; mkdir -p "$WT" && WT=$(cd "$WT" && pwd -P)   # canonical path (the import check compares strings)
PY=${PY:-python}                                 # runs the planners
SYSPY=${SYSPY:-$PY}                              # draws the figures (matplotlib)
TOOLS=$REPO/script
SEEDS=${SEEDS:-"0 1 2 3 4 5"}
METHODS=${METHODS:-"ps mppi dial mtp cbo gm"}
TASKS=${TASKS:-pusht}
PUSHT_SEEDS_DIR=$REPO/examples/single_run_test/PushT/closed_loop/seeds   # (gitignored *.log, as the senior's doc)
BASE_TIMEOUT=3600      # s, ps / mppi / dial / mtp / cbo
GM_TIMEOUT=14400       # s, globalmppi
RETRIES=2              # extra attempts after a crash (any crash except a timeout)
GPU_CUKS=${CUKS_GPU:-}
OLD_CUKS=${CUKS_OLD:-$GPU_CUKS}
[ -n "$GPU_CUKS" ] || { echo "set CUKS_GPU=<cuKernelSOS gpu-solve build>/python (see the header of this script)"; exit 1; }

if [ "${1:-}" != "--run" ]; then
  mkdir -p "$Q"
  nohup setsid systemd-inhibit --what=sleep:idle:handle-lid-switch --who="overnight queue" \
    --why="Global-MPPI batch queue" --mode=block env QUEUE_TS="$QUEUE_TS" "$0" --run > "$Q/queue.out" 2>&1 < /dev/null &
  echo "queue $QUEUE_TS started (detached)."
  echo "  status : $STATUS"
  echo "  logs   : $Q/<task>/batch.log      (tail -f $Q/pusht/batch.log)"
  echo "  stop   : kill the PID of 'overnight_queue.sh --run' and of the current python child (by PID, not pkill -f)"
  exit 0
fi

mkdir -p "$Q"
declare -A TASK_STATE TASK_DONE TASK_TOTAL TASK_FAILED
for t in $TASKS; do TASK_STATE[$t]=PENDING; TASK_DONE[$t]=0; TASK_TOTAL[$t]=36; TASK_FAILED[$t]=""; done
CURRENT="-"

say() { echo "$(date '+%F %T') $*" >> "$Q/$TASK/batch.log"; }
status() {
  {
    echo "overnight queue $QUEUE_TS   updated $(date '+%F %T')   queue pid $$"
    echo "current: $CURRENT"
    for t in $TASKS; do
      echo "$t: ${TASK_STATE[$t]}   runs finished ${TASK_DONE[$t]}/${TASK_TOTAL[$t]}   gave up: ${TASK_FAILED[$t]:-none}"
    done
    echo "gpu now: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
  } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
}
gpu_snapshot() {   # file
  { echo "=== $(date '+%F %T') $CURRENT"; nvidia-smi; echo; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > "$1" 2>&1
}

# run_with_retries <kind> <label> <log> <run_dir> <timeout> <done-test-function> -- command...
run_with_retries() {
  local kind=$1 label=$2 log=$3 rdir=$4 to=$5 donefn=$6; shift 7
  local attempt=0
  while [ $attempt -le $RETRIES ]; do
    attempt=$((attempt + 1))
    CURRENT="$TASK $label (attempt $attempt)"; status
    gpu_snapshot "$Q/$TASK/gpu_snapshots/${label// /_}_a${attempt}.txt"
    say "START $label (attempt $attempt)"
    timeout "$to" "$@" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && $donefn; then say "END   $label ok"; return 0; fi
    say "FAILED $label (attempt $attempt, exit code $rc); last log line: $(tail -n 1 "$log" | cut -c1-160)"
    cp "$log" "$log.attempt$attempt" 2>/dev/null
    if [ $rc -eq 124 ]; then say "  timeout (not retried)"; break; fi
    if [ $attempt -le $RETRIES ]; then
      say "  crash (exit code $rc, not a timeout): retrying"
      continue
    fi
    break
  done
  say "GAVE UP $label (batch continues)"
  return 1
}

make_worktree() {   # branch -> $WT/<branch>, or return 1
  local b=$1
  mkdir -p "$WT"
  if [ ! -d "$WT/$b" ]; then git -C "$REPO" worktree add --detach "$WT/$b" "$b" >> "$Q/$TASK/batch.log" 2>&1 || return 1; fi
  # an existing worktree may be older than the branch: bring it to the branch head (only when it is clean)
  if [ -z "$(git -C "$WT/$b" status --porcelain 2>/dev/null)" ]; then git -C "$WT/$b" checkout -q --detach "$b" >> "$Q/$TASK/batch.log" 2>&1; fi
  # the code must come from the worktree, not from the main checkout
  local f; f=$(cd "$WT/$b" && PYTHONPATH="$WT/$b" "$PY" -c "import global_mppi; print(global_mppi.__file__)" 2>/dev/null | tail -1)
  case $f in "$WT/$b"/*) say "code: $f (branch $b, $(git -C "$WT/$b" rev-parse --short HEAD))"; return 0;; *) say "import resolves to '$f', not to the worktree"; return 1;; esac
}

# ------------------------------------------------------------------ (a) PushT
task_pusht() {
  TASK=pusht; local W=$REPO
  mkdir -p "$Q/pusht/gpu_snapshots" "$Q/pusht/logs" "$PUSHT_SEEDS_DIR"
  say "=== PushT batch: ps mppi dial mtp cbo (2000 x 5) and globalmppi (CACS, ADMM-cuda, gpu_solve), seeds $SEEDS, 100 cycles"
  [ -f "$W/examples/pusht.py" ] || { TASK_STATE[pusht]=SKIPPED; say "SKIPPED (no examples/pusht.py in $W)"; return 1; }
  say "code: $W ($(git -C "$W" rev-parse --short HEAD 2>/dev/null), branch $(git -C "$W" branch --show-current 2>/dev/null))"
  TASK_STATE[pusht]=RUNNING; status
  local s m log marker
  for s in $SEEDS; do
    say "--- seed $s"
    for m in $METHODS; do
      if [ $m = gm ]; then log=$PUSHT_SEEDS_DIR/globalmppi_cacs5new_seed$s.log; else log=$PUSHT_SEEDS_DIR/${m}2000_seed$s.log; fi
      if [ -f "$log" ] && grep -q "finish the seed" "$log"; then say "SKIP  $m s$s (finished log exists)"; TASK_DONE[pusht]=$((TASK_DONE[pusht] + 1)); continue; fi
      pusht_done() { grep -q "finish the seed" "$log"; }
      if [ $m = gm ]; then
        # gpu_solve is switched on by KSOS_GPU_SOLVE=1 and the gpu-solve build of cuKernelSOS
        ( cd "$W" && run_with_retries pusht "$m s$s" "$log" "" $GM_TIMEOUT pusht_done -- \
            env PYTHONPATH="$W" KSOS_GPU_SOLVE=1 CUKERNELSOS_PYTHON=$GPU_CUKS XLA_PYTHON_CLIENT_PREALLOCATE=false \
            "$PY" examples/pusht.py --no-wandb --headless --seed "$s" --ksos-solver ADMM-cuda --ksos-sampling cacs \
            --ksos-restarts 5 --ksos-warm-start --mppi-noise 0.1 globalmppi )
      else
        ( cd "$W" && run_with_retries pusht "$m s$s" "$log" "" $BASE_TIMEOUT pusht_done -- \
            env PYTHONPATH="$W" CUKERNELSOS_PYTHON=$OLD_CUKS XLA_PYTHON_CLIENT_PREALLOCATE=false \
            "$PY" examples/pusht.py --no-wandb --headless --seed "$s" --num-samples 2000 "$m" )
      fi
      if [ $? -eq 0 ]; then
        TASK_DONE[pusht]=$((TASK_DONE[pusht] + 1))
        if [ $m = gm ]; then
          if grep -q "solve_paths: {'gpu': " "$log" && ! grep -q "cpu (" "$log"; then say "  solve_paths: $(grep solve_paths "$log" | tail -1)"; else say "  WARNING gm s$s: solve_paths not all gpu: $(grep solve_paths "$log" | tail -1)"; fi
        fi
      else TASK_FAILED[pusht]="${TASK_FAILED[pusht]} $m/s$s"; fi
      status
    done
  done
  # figures and tables (no GCS): the doc's six_seeds_main directory, built here with the six methods
  local D=$Q/pusht/six_seeds_main; mkdir -p "$D" "$Q/pusht/figures"
  for s in $SEEDS; do
    for m in ps mppi dial mtp cbo; do [ -f "$PUSHT_SEEDS_DIR/${m}2000_seed$s.log" ] && ln -sf "$PUSHT_SEEDS_DIR/${m}2000_seed$s.log" "$D/${m}2000_seed$s.log"; done
    [ -f "$PUSHT_SEEDS_DIR/globalmppi_cacs5new_seed$s.log" ] && ln -sf "$PUSHT_SEEDS_DIR/globalmppi_cacs5new_seed$s.log" "$D/globalmppi_cacs5new_seed$s.log"
  done
  local C=$W/examples/single_run_test/PushT/closed_loop
  ( cd "$C" && $SYSPY plot_closed_loop_seeds.py "$D" "$Q/pusht/figures/pusht_closed_loop_six.png" > "$Q/pusht/figures/plot_closed_loop_seeds.txt" 2>&1
    $SYSPY plot_pareto.py "$D" "$Q/pusht/figures/pusht_pareto_six.png" > "$Q/pusht/figures/plot_pareto.txt" 2>&1
    $SYSPY plot_paper_style.py "$D" "$Q/pusht/figures/pusht_cost_median_iqr_six.png" > "$Q/pusht/figures/plot_paper_style.txt" 2>&1 )
  $SYSPY "$TOOLS/pusht_summary.py" "$D" "$Q/pusht/summary.txt" >> "$Q/pusht/batch.log" 2>&1
  TASK_STATE[pusht]=DONE; say "=== PushT batch end (gave up:${TASK_FAILED[pusht]:- none}); tables in $Q/pusht/summary.txt, figures in $Q/pusht/figures/"
}

# ------------------------------------------------------------------ (b) walker, (c) double_cart_pole, (d) bugtrap
task_example() {   # <task name = example name> <branch>
  TASK=$1; local b=$2 W=$WT/$2 RES=$Q/$1/results
  mkdir -p "$Q/$TASK/gpu_snapshots" "$Q/$TASK/logs" "$RES"
  say "=== $TASK batch: $METHODS, seeds $SEEDS, 100 cycles (settings as the cube runs; walker / double_cart_pole: horizon, knots, spline from the hydrax example; bugtrap: horizon 2.0, 11 knots, cubic)"
  make_worktree "$b" || { TASK_STATE[$TASK]=SKIPPED; say "SKIPPED (no usable worktree of $b)"; return 1; }
  [ -f "$W/examples/$TASK.py" ] || { TASK_STATE[$TASK]=SKIPPED; say "SKIPPED (no examples/$TASK.py on $b)"; return 1; }
  TASK_STATE[$TASK]=RUNNING; status
  local s m arg algo final log
  local EXTRA=""   # per-task benchmark setting
  case $TASK in bugtrap) EXTRA="--horizon 2.0 --num-knots 11 --spline-type cubic";; esac
  for s in $SEEDS; do
    say "--- seed $s"
    for m in $METHODS; do
      if [ $m = gm ]; then arg="--gpu globalmppi"; else arg=$m; fi
      final=$RES/$m/s$s/final.json; log=$Q/$TASK/logs/${m}_s$s.log
      if [ -f "$final" ]; then say "SKIP  $m s$s (final.json exists)"; TASK_DONE[$TASK]=$((TASK_DONE[$TASK] + 1)); continue; fi
      example_done() { [ -f "$final" ]; }
      if [ $m = gm ]; then
        ( cd "$W" && run_with_retries example "$m s$s" "$log" "$RES/$m/s$s" $GM_TIMEOUT example_done -- \
            env PYTHONPATH="$W" KSOS_GPU_SOLVE=1 CUKERNELSOS_PYTHON=$GPU_CUKS XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
            "$PY" examples/$TASK.py --headless --seed-start "$s" --seed-end $((s + 1)) --max-cycles ${MAX_CYCLES:-100} --out-dir "$RES" $EXTRA $arg )
      else
        ( cd "$W" && run_with_retries example "$m s$s" "$log" "$RES/$m/s$s" $BASE_TIMEOUT example_done -- \
            env PYTHONPATH="$W" CUKERNELSOS_PYTHON=$OLD_CUKS XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
            "$PY" examples/$TASK.py --headless --seed-start "$s" --seed-end $((s + 1)) --max-cycles ${MAX_CYCLES:-100} --out-dir "$RES" $EXTRA $arg )
      fi
      if [ $? -eq 0 ]; then
        TASK_DONE[$TASK]=$((TASK_DONE[$TASK] + 1))
        if [ $m = gm ]; then
          if "$PY" -c "import json,sys; sys.exit(0 if set(json.load(open('$final')).get('solve_paths',{'?':1}))=={'gpu'} else 1)"; then say "  solve_paths: all gpu"; else say "  WARNING gm s$s: solve_paths not all gpu"; fi
        fi
      else TASK_FAILED[$TASK]="${TASK_FAILED[$TASK]} $m/s$s"; fi
      status
    done
  done
  [ "$TASK" = bugtrap ] || $SYSPY "$TOOLS/plot_task.py" "$TASK" "$RES" "$Q/$TASK/figures" >> "$Q/$TASK/batch.log" 2>&1   # bugtrap: final_figs only
  TASK_STATE[$TASK]=DONE; say "=== $TASK batch end (gave up:${TASK_FAILED[$TASK]:- none}); figure and tables in $Q/$TASK/figures/"
}

TASK=queue; mkdir -p "$Q/queue"; echo "$(date '+%F %T') queue start" >> "$Q/queue/batch.log"
status
# plain calls (not subshells), so that STATUS.txt sees the state of every task; a failed or skipped task never stops the next one
for t in $TASKS; do
  case $t in
    pusht) task_pusht || true;;
    walker) task_example walker walker || true;;
    double_cart_pole) task_example double_cart_pole double-cart-pole || true;;
    bugtrap) task_example bugtrap bugtrap || true;;
    *) echo "unknown task $t" >> "$Q/queue/batch.log";;
  esac
done
$SYSPY "$TOOLS/final_figs.py" "$Q" "$(echo $TASKS | tr ' ' ',')" >> "$Q/queue/batch.log" 2>&1   # final six-method figures + tables in $Q/final_figs/
CURRENT="finished"; status
echo "$(date '+%F %T') queue end" >> "$Q/queue/batch.log"
