"""Detach the benchmark and keep the desktop awake for the duration."""
from pathlib import Path
import subprocess

folder = Path(__file__).resolve().parent
python = Path.home() / ".cache/global-mppi/pusht_5090/venv/bin/python"
with (folder / "queue.out").open("a") as output:
    process = subprocess.Popen([
        "systemd-inhibit", "--what=sleep:idle", "--who=PushT-5090",
        "--why=PushT benchmark rerun", "--mode=block",
        str(python), "-u", str(folder / "run_benchmark.py"),
    ], cwd=folder, stdin=subprocess.DEVNULL, stdout=output,
       stderr=subprocess.STDOUT, start_new_session=True)
(folder / "launcher.pid").write_text(str(process.pid) + "\n")
print(f"Detached launcher PID: {process.pid}")
print(f"Progress: {folder / 'STATUS.txt'}")
