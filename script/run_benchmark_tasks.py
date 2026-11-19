"""Run imported benchmark tasks with this branch's existing CACS implementation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
TASKS = ("pusht", "cube", "double_cart_pole", "g1", "bugtrap", "walker", "g1_walk")
METHODS = ("ps", "mppi", "dial", "mtp", "cbo", "gm")


def command_for(task, method, seed, output, max_cycles):
    example = "pusht_benchmark" if task == "pusht" else task
    command = [sys.executable, str(REPO / "examples" / f"{example}.py"),
               "--headless", "--out-dir", str(output / task)]
    if task == "pusht":
        command += ["--no-wandb", "--seed", str(seed)]
        if method == "gm":
            command += ["--ksos-solver", "ADMM-cuda", "--ksos-sampling", "cacs",
                        "--ksos-restarts", "5", "--ksos-warm-start", "--mppi-noise", "0.1"]
        else:
            command += ["--num-samples", "2000"]
    else:
        command += ["--seed-start", str(seed), "--seed-end", str(seed + 1)]
        if task == "bugtrap":
            command += ["--horizon", "2.0", "--num-knots", "11", "--spline-type", "cubic"]
        if method == "gm":
            command += ["--gpu"]
    if max_cycles is not None:
        command += ["--max-cycles", str(max_cycles)]
    command += ["globalmppi" if method == "gm" else method]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--max-cycles", type=int,
                        help="Override run length (default: 100; G1 walk: 400)")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "logs" / "benchmark_tasks" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running or writing files")
    args = parser.parse_args()
    if not args.seeds or min(args.seeds) < 0:
        parser.error("seeds must be nonnegative")
    if args.max_cycles is not None and args.max_cycles < 1:
        parser.error("max-cycles must be positive")
    output = args.out_dir.resolve()
    if not args.dry_run and any(t in ("g1", "g1_walk") for t in args.tasks):
        if not (REPO / "global_mppi/models/g1/scene.xml").exists():
            parser.error("G1 model missing: run script/fetch_g1_model.sh first")
    environment = dict(os.environ)
    # Use this checkout and its pinned submodule, even with a different editable install.
    environment["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "ksos-tools"),
                                               environment.get("PYTHONPATH", "")])
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("MUJOCO_GL", "egl")
    environment["PYTHONUNBUFFERED"] = "1"
    finished, failed = [], []
    for task in args.tasks:
        for seed in args.seeds:
            for method in args.methods:
                label = f"{task}/{method}/s{seed}"
                run = output / task / method / f"s{seed}"
                command = command_for(task, method, seed, output, args.max_cycles)
                if args.dry_run:
                    print(shlex.join(command))
                    continue
                final = run / "final.json"
                cycles = args.max_cycles or (400 if task == "g1_walk" else 100)
                if final.exists() and json.loads(final.read_text()).get("cycles") == cycles:
                    print(f"SKIP {label}: completed {cycles} cycles", flush=True)
                    finished.append(label)
                    continue
                run.mkdir(parents=True, exist_ok=True)
                # A fresh attempt directory keeps interrupted CSV, NPY and video artifacts intact.
                attempt = len(list(run.glob("attempts/*"))) + 1
                if any(run.iterdir()):
                    archive = run / "attempts" / str(attempt)
                    archive.mkdir(parents=True, exist_ok=True)
                    for artifact in list(run.iterdir()):
                        if artifact.is_file():
                            artifact.rename(archive / artifact.name)
                print(f"START {label}: {shlex.join(command)}", flush=True)
                with (run / "run.log").open("w") as stream:
                    result = subprocess.run(command, cwd=REPO, env=environment,
                                            stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode == 0 and final.exists():
                    finished.append(label)
                    print(f"END {label}", flush=True)
                else:
                    failed.append(label)
                    print(f"FAILED {label}: exit {result.returncode}; see {run / 'run.log'}", flush=True)
                status = dict(updated=datetime.now().astimezone().isoformat(),
                              finished=finished, failed=failed)
                temporary = output / "status.json.tmp"
                temporary.write_text(json.dumps(status, indent=2) + "\n")
                temporary.replace(output / "status.json")
    if not args.dry_run:
        print(f"Finished: {len(finished)}, failed: {len(failed)}; outputs: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
