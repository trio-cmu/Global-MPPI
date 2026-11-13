#!/bin/bash
# Cube (LEAP hand) benchmark: seeds x configs, headless, one cube.py run per (config, seed).
#   configs: ps mppi dial gm (GlobalMPPI --gpu) gm_noise0.02 (gm with --mppi-noise 0.02)
#   output:  $OUT/<config>/<algo>/s<seed>/   (cube.py writes <out-dir>/<algo>/s<seed>; algo = ps|mppi|dial|gm)
#   a run whose final.json exists is skipped; a failed run is retried, then logged and the batch goes on.
#   log:     $OUT/batch.log, one log per run in $OUT/logs/
# usage:  CUKERNELSOS_PYTHON=<cuKernelSOS_gpu>/python script/run_cube_benchmark.sh [out dir]
# env:    PYTHON (default python), SEEDS ("0 1 2 3 4 5"), CONFIGS ("ps mppi dial gm"), MAXCYC (100),
#         RUN_TIMEOUT (s per attempt, 3600), RETRIES (extra attempts, 2)
# for an unattended run:  nohup setsid systemd-inhibit --what=sleep:idle --why="cube batch" script/run_cube_benchmark.sh > /dev/null 2>&1 &
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$(mkdir -p "${1:-$REPO/runs/cube_benchmark}" && cd "${1:-$REPO/runs/cube_benchmark}" && pwd)
PY=${PYTHON:-python}
SEEDS=${SEEDS:-"0 1 2 3 4 5"}
CONFIGS=${CONFIGS:-"ps mppi dial gm"}
MAXCYC=${MAXCYC:-100}
RUN_TIMEOUT=${RUN_TIMEOUT:-3600}
RETRIES=${RETRIES:-2}
LOG=$OUT/batch.log
mkdir -p "$OUT/logs"

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

cd "$REPO" || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
say "=== batch start (pid $$): seeds $SEEDS; configs $CONFIGS; max-cycles $MAXCYC; $($PY -c 'import mujoco;print("mujoco",mujoco.__version__)')"
case " $CONFIGS " in *" gm"*)
  say "CUKERNELSOS_PYTHON=${CUKERNELSOS_PYTHON:-<unset: gm will fall back to ../cuKernelSOS/python, check solve_paths>}";; esac

run_one() {   # config seed
  local cfg=$1 seed=$2 algo args
  case $cfg in
    ps|mppi|dial)  algo=$cfg; args=$cfg;;
    gm)            algo=gm;   args="--gpu globalmppi";;
    gm_noise0.02)  algo=gm;   args="--gpu --mppi-noise 0.02 globalmppi";;
    *) say "unknown config $cfg"; return 1;;
  esac
  local final=$OUT/$cfg/$algo/s$seed/final.json run_log=$OUT/logs/${cfg}_s$seed.log attempt=0
  if [ -f "$final" ]; then say "SKIP  $cfg s$seed (final.json exists)"; return 0; fi
  while [ $attempt -le $RETRIES ]; do
    attempt=$((attempt + 1))
    say "START $cfg s$seed (attempt $attempt)"
    timeout "$RUN_TIMEOUT" $PY examples/cube.py --headless --seed-start "$seed" --seed-end $((seed + 1)) \
      --max-cycles "$MAXCYC" --out-dir "$OUT/$cfg" $args > "$run_log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$final" ]; then
      say "END   $cfg s$seed ok: $(tail -n 3 "$run_log" | grep 'final cube error') solve_paths=$($PY -c 'import json,sys;print(json.load(open(sys.argv[1])).get("solve_paths",""))' "$final")"
      return 0
    fi
    say "FAILED $cfg s$seed (attempt $attempt, exit code $rc); last log line: $(tail -n 1 "$run_log" | cut -c1-160)"
  done
  say "GAVE UP $cfg s$seed after $attempt attempts (batch continues)"
  return 1
}

nfail=0
for seed in $SEEDS; do
  say "--- seed $seed"
  for cfg in $CONFIGS; do
    run_one "$cfg" "$seed" || nfail=$((nfail + 1))
  done
done
say "=== batch end: failed runs (gave up) = $nfail"
