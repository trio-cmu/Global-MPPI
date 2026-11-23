"""Read progress and verify that a reported running process still exists."""
import json
from pathlib import Path
import sys
from datetime import datetime

folder = Path(__file__).resolve().parent
path = folder / "status.json"
info = json.loads(path.read_text())
boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
try:
    command = Path(f"/proc/{int(info['runner_pid'])}/cmdline").read_bytes().split(b"\0")
    alive = str(folder / "run_benchmark.py").encode() in command
except (FileNotFoundError, ProcessLookupError):
    alive = False
if info.get("boot_id") and info["boot_id"] != boot_id:
    alive = False
if info["state"] == "RUNNING" and not alive:
    info.update(state="INTERRUPTED", detected=datetime.now().astimezone().isoformat(timespec="seconds"),
                interruption_reason="The recorded runner is absent or belongs to an earlier system boot.")
    if "--refresh" in sys.argv:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(info, indent=2) + "\n")
        temporary.replace(path)
        (folder / "STATUS.txt").write_text(
            f"PushT GCS cost RTX 5090: INTERRUPTED\nLast heartbeat: {info['updated']}\n"
            f"Detected: {info['detected']}\nCompleted: {info['completed']}/36\n"
            f"Interrupted run: {info['current']}, cycle {info['current_cycle']}\n"
            f"Reason: {info['interruption_reason']}\n")
print(json.dumps(info, indent=2))
