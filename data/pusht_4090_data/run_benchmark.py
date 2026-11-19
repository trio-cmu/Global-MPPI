"""Sequential, resumable 4090 rerun of BENCHMARK_MAP.md's 36 PushT runs."""
import ast
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
SETUP = Path.home() / ".cache/global-mppi/pusht_4090"
PY = SETUP / "venv/bin/python"
METHODS = ("ps", "mppi", "dial", "mtp", "cbo", "gm")
BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
ENV = dict(os.environ, PYTHONPATH=f"{SETUP / 'ksos-tools'}:{REPO}",
           CUKERNELSOS_PYTHON=str(SETUP / "cuKernelSOS/python"), KSOS_GPU_SOLVE="1",
           XLA_PYTHON_CLIENT_PREALLOCATE="false", MUJOCO_GL="egl", PYTHONUNBUFFERED="1")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def capture(command):
    return subprocess.run(command, cwd=REPO, env=ENV, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30).stdout


def log(message):
    line = f"{now()} {message}"
    print(line, flush=True)
    with (OUT / "batch.log").open("a") as stream:
        stream.write(line + "\n")


def parsed(path):
    text = path.read_text(errors="replace") if path.exists() else ""
    costs = {int(float(k)): float(v) for k, v in re.findall(
        r"Cycle ([\d.]+): Best cost: \[?([-\d.e+]+)", text)}
    plans = {int(float(k)): float(v) for k, v in re.findall(
        r"Cycle ([\d.]+): plan time: ([\d.]+)s", text)}
    paths = re.findall(r"solve_paths: (\{[^\n]*\})", text)
    return text, costs, plans, ast.literal_eval(paths[-1]) if paths else {}


def write_csv(path, costs, plans):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cycle", "best_cost", "plan_s"])
        for cycle in sorted(costs):
            writer.writerow([cycle, costs[cycle], plans.get(cycle, "")])
    tmp.replace(path)


def refresh_figures():
    with (OUT / "plot.log").open("a") as stream:
        result = subprocess.run([str(PY), str(REPO / "script/final_figs.py"),
                                 str(OUT), "pusht", str(OUT)],
                                env=ENV, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode == 0


def main():
    lock = (OUT / ".runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / "status.json").exists():
        history = OUT / "launch_history"
        history.mkdir(exist_ok=True)
        atomic_json(history / (datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"),
                    dict(restarted=now(), new_boot_id=BOOT_ID,
                         previous_status=json.loads((OUT / "status.json").read_text())))
    (OUT / "runner.pid").write_text(str(os.getpid()) + "\n")
    completed, failed = [], []

    def status(current, state="RUNNING", cycle=None, child_pid=None):
        info = dict(state=state, updated=now(), boot_id=BOOT_ID, runner_pid=os.getpid(),
                    current=current, current_cycle=cycle, child_pid=child_pid,
                    completed=len(completed), total=36, finished_runs=completed, failed_runs=failed)
        atomic_json(OUT / "status.json", info)
        tmp = OUT / "STATUS.txt.tmp"
        tmp.write_text(f"PushT RTX 4090: {state}\nUpdated: {info['updated']}\n"
                       f"Completed: {len(completed)}/36\nCurrent: {current}\n"
                       f"Current cycle: {cycle}\nFailed: {', '.join(failed) or 'none'}\n"
                       f"Runner PID: {os.getpid()}\nChild PID: {child_pid}\n")
        tmp.replace(OUT / "STATUS.txt")

    status("initializing")
    (OUT / "gpu_start.txt").write_text(capture(["nvidia-smi"]))
    (OUT / "pip_freeze.txt").write_text(capture([str(PY), "-m", "pip", "freeze"]))
    versions = capture([str(PY), "-c", "import sys, jax, mujoco, ksos_tools; "
                        "print(sys.version); print('jax',jax.__version__); "
                        "print('mujoco',mujoco.__version__); "
                        "print('ksos_tools',ksos_tools.__file__); print(jax.devices())"])
    manifest = dict(started=now(), python=str(PY), versions=versions,
                    environment={k: ENV[k] for k in ("PYTHONPATH", "CUKERNELSOS_PYTHON",
                    "KSOS_GPU_SOLVE", "XLA_PYTHON_CLIENT_PREALLOCATE", "MUJOCO_GL")},
                    repositories={str(p): capture(["git", "-C", str(p), "rev-parse", "HEAD"]).strip()
                                  for p in (REPO, SETUP / "cuKernelSOS", SETUP / "ksos-tools")},
                    build_toolkit="CUDA 13.1.80", methods=list(METHODS), seeds=list(range(6)),
                    cycles=100, baseline_samples=2000, iterations=5,
                    globalmppi=dict(samples=1000, ksos_candidates=1000, restarts=5,
                                    sampling="cacs", solver="ADMM-cuda", warm_start=True,
                                    gpu_solve=True, mppi_noise=0.1),
                    differences_from_5090="RTX 4090; Python 3.12.12; CUDA toolkit 13.1.80; "
                    "other installed package versions are in pip_freeze.txt. "
                    "MuJoCo/MJX 3.3.6, JAX 0.5.3 and solver source commits match the benchmark map.")
    if not (OUT / "manifest.json").exists():
        atomic_json(OUT / "manifest.json", manifest)
    log("Starting fresh/resuming 36 runs, one process at a time")
    for seed in range(6):
        for method in METHODS:
            label = f"{method}/s{seed}"
            folder = OUT / method / f"s{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            final = folder / "final.json"
            if final.exists() and json.loads(final.read_text()).get("complete"):
                completed.append(label)
                status(f"resumed {label}")
                continue
            args = ["--no-wandb", "--headless", "--seed", str(seed), "--max-cycles", "100"]
            if method == "gm":
                args += ["--ksos-solver", "ADMM-cuda", "--ksos-sampling", "cacs",
                         "--ksos-restarts", "5", "--ksos-warm-start", "--mppi-noise", "0.1", "globalmppi"]
            else:
                args += ["--num-samples", "2000", method]
            command = [str(PY), str(OUT / "run_one.py"), str(folder), *args]
            atomic_json(folder / "args.json", dict(method=method, seed=seed, command=command,
                                                   example_arguments=args, manifest="../../manifest.json"))
            success = False
            for attempt in range(1, 4):
                status(f"{label} attempt {attempt}: waiting for free GPU memory")
                while True:
                    free = capture(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
                    if int(free.strip().splitlines()[0]) >= (6144 if method == "gm" else 2048):
                        break
                    time.sleep(30)
                # Append a new attempt number when resuming a previously interrupted run.
                number = len(list(folder.glob("run.attempt*.log"))) + 1
                if number > 1:
                    archive = folder / "attempt_artifacts" / f"attempt{number - 1}"
                    if not archive.exists():
                        archive.mkdir(parents=True)
                        for artifact in folder.iterdir():
                            if artifact.is_file() and not artifact.is_symlink() and (
                                artifact.suffix in (".npy", ".mp4", ".csv")
                                or artifact.name == "final_state.json"
                                or artifact.name.startswith("globalmppi_timing")):
                                shutil.copy2(artifact, archive / artifact.name)
                        log(f"Preserved {label} attempt {number - 1} artifacts in {archive}")
                (folder / f"gpu_before_attempt{number}.txt").write_text(capture(["nvidia-smi"]))
                runlog = folder / f"run.attempt{number}.log"
                link = folder / "run.log"
                link.unlink(missing_ok=True)
                link.symlink_to(runlog.name)
                start = time.monotonic()
                timed_out = False
                log(f"START {label} attempt {number}")
                with runlog.open("w") as stream:
                    child = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    status(label, child_pid=child.pid)
                    while child.poll() is None:
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            pass
                        _, costs, plans, _ = parsed(runlog)
                        if costs:
                            write_csv(folder / "cycles.partial.csv", costs, plans)
                        status(label, cycle=max(costs, default=None), child_pid=child.pid)
                        if time.monotonic() - start > (14400 if method == "gm" else 3600):
                            timed_out = True
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                            break
                text, costs, plans, solve_paths = parsed(runlog)
                success = (child.returncode == 0 and "finish the seed" in text
                           and sorted(costs) == list(range(101)) and sorted(plans) == list(range(100))
                           and all(math.isfinite(v) for v in (*costs.values(), *plans.values()))
                           and (method != "gm" or solve_paths == {"gpu": 505}))
                result = dict(complete=success, method=method, seed=seed, finished=now(),
                              returncode=child.returncode, timed_out=timed_out,
                              wall_s=time.monotonic() - start, attempt=number,
                              last_cycle=max(costs, default=None), solve_paths=solve_paths)
                atomic_json(folder / f"attempt{number}.json", result)
                if success:
                    write_csv(folder / "cycles.csv", costs, plans)
                    result.update(final_cost=costs[100],
                                  plan_s_median_excluding_jit=statistics.median(plans[i] for i in range(1, 100)))
                    atomic_json(final, result)
                    (folder / "cycles.partial.csv").unlink(missing_ok=True)
                    completed.append(label)
                    log(f"END {label}: cost@100={costs[100]}, completed {len(completed)}/36")
                    if not refresh_figures():
                        log("Figure update failed; see plot.log")
                    break
                log(f"FAILED {label}: exit {child.returncode}, last cycle {max(costs, default=None)}, "
                    f"gpu paths {solve_paths}; see {runlog}")
                if timed_out:
                    break
            if not success:
                failed.append(label)
            status(label)
    plots_ok = refresh_figures()
    state = "COMPLETE" if len(completed) == 36 and plots_ok else "INCOMPLETE"
    (OUT / "gpu_end.txt").write_text(capture(["nvidia-smi"]))
    status("finished", state=state)
    log(f"{state}: {len(completed)}/36 runs; failures {failed}")
    return 0 if state == "COMPLETE" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BlockingIOError:
        print("A benchmark runner already holds the lock; not starting another.", flush=True)
        sys.exit(2)
    except Exception as exc:
        log(f"RUNNER ERROR: {type(exc).__name__}: {exc}")
        (OUT / "ERROR.txt").write_text(f"{now()} {type(exc).__name__}: {exc}\n")
        status_path = OUT / "status.json"
        if status_path.exists():
            info = json.loads(status_path.read_text())
            if info.get("runner_pid") == os.getpid():
                info.update(state="ERROR", updated=now(), error=str(exc))
                atomic_json(status_path, info)
                (OUT / "STATUS.txt").write_text(
                    f"PushT RTX 4090: ERROR\nUpdated: {info['updated']}\n"
                    f"Completed: {info['completed']}/36\nError: {exc}\n")
        raise
