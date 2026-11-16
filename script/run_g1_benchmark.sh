#!/bin/bash
# G1 standup benchmark: seeds x algorithms, headless, one examples/g1.py run per (algorithm, seed).
#   algorithms: ps mppi dial mtp cbo gm (GlobalMPPI --gpu)
#   output:  $OUT/<algo>/s<seed>/   (examples/g1.py writes <out-dir>/<algo>/s<seed>; algo = ps|mppi|dial|gm)
#   a run whose final.json exists is skipped; a failed run is retried, then logged and the batch goes on.
#   log:     $OUT/batch.log, one log per run in $OUT/logs/
# usage:  script/fetch_g1_model.sh   (once)
#         CUKERNELSOS_PYTHON=<cuKernelSOS_gpu>/python script/run_g1_benchmark.sh [out dir]
# env:    PYTHON (default python), SEEDS ("0 1 2 3 4 5"), ALGOS ("ps mppi dial gm"), MAXCYC (100),
#         RUN_TIMEOUT (s per attempt, 3600), RETRIES (extra attempts, 2)
# for an unattended run:  nohup setsid systemd-inhibit --what=sleep:idle --why="g1 batch" script/run_g1_benchmark.sh > /dev/null 2>&1 &
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$(mkdir -p "${1:-$REPO/runs/g1_benchmark}" && cd "${1:-$REPO/runs/g1_benchmark}" && pwd)
PY=${PYTHON:-python}
SEEDS=${SEEDS:-"0 1 2 3 4 5"}
ALGOS=${ALGOS:-"ps mppi dial gm"}
MAXCYC=${MAXCYC:-100}
RUN_TIMEOUT=${RUN_TIMEOUT:-3600}
RETRIES=${RETRIES:-2}
LOG=$OUT/batch.log
mkdir -p "$OUT/logs"

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

cd "$REPO" || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
say "=== batch start (pid $$): seeds $SEEDS; algorithms $ALGOS; max-cycles $MAXCYC; $($PY -c 'import mujoco;print("mujoco",mujoco.__version__)')"
case " $ALGOS " in *" gm "*)
  say "CUKERNELSOS_PYTHON=${CUKERNELSOS_PYTHON:-<unset: gm will fall back to ../cuKernelSOS/python, check solve_paths>}";; esac

run_one() {   # algo seed
  local algo=$1 seed=$2 arg
  case $algo in
    ps|mppi|dial|mtp|cbo)  arg=$algo;;
    gm)            arg="--gpu globalmppi";;
    *) say "unknown algorithm $algo"; return 1;;
  esac
  local final=$OUT/$algo/s$seed/final.json run_log=$OUT/logs/${algo}_s$seed.log attempt=0
  if [ -f "$final" ]; then say "SKIP  $algo s$seed (final.json exists)"; return 0; fi
  while [ $attempt -le $RETRIES ]; do
    attempt=$((attempt + 1))
    say "START $algo s$seed (attempt $attempt)"
    timeout "$RUN_TIMEOUT" $PY examples/g1.py --headless --seed-start "$seed" --seed-end $((seed + 1)) \
      --max-cycles "$MAXCYC" --out-dir "$OUT" $arg > "$run_log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$final" ]; then
      say "END   $algo s$seed ok: $(tail -n 3 "$run_log" | grep 'final torso') $(tail -n 3 "$run_log" | grep 'solve_paths')"
      return 0
    fi
    say "FAILED $algo s$seed (attempt $attempt, exit code $rc); last log line: $(tail -n 1 "$run_log" | cut -c1-160)"
  done
  say "GAVE UP $algo s$seed after $attempt attempts (batch continues)"
  return 1
}

nfail=0
for seed in $SEEDS; do
  say "--- seed $seed"
  for algo in $ALGOS; do
    run_one "$algo" "$seed" || nfail=$((nfail + 1))
  done
done
say "=== batch end: failed runs (gave up) = $nfail"
