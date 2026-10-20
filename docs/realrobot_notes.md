# Real-robot Push-T closed loop: notes (2026-10-20)

Branch `realrobot-pusht-pipeline`, script `examples/closed_loop_test.py`. Everything marked *experimental* is new, off by
default (same behaviour as before) and only tried in a few single runs.

## Commands used

Common: `JAX_PLATFORMS=cpu python examples/closed_loop_test.py --real --algo dial --pipeline --blend --cam-correct --adaptive-settle
--read-s 0.1 --bg-link-check --confirm --go-file <file> --out runs/closed_loop_test/<new name>` (always a new `--out`).
Two gates: create `<file>.down` after the start checks, then `<file>` (hand on the enable button).

| run | extra flags |
|---|---|
| original speed, measure every 2 chunks | `--measure-every 2 --max-cycles 64 --max-time 75` |
| slow (half speed) | `--config configs/pusht_real_slow.yaml --dial-noise 0.5 --measure-every N` |
| latency test (original speed, stop after every chunk) | `--measure-every 1 --fast-stop --min-tags 3 --max-reproj-px 2.0 --pose-bin build_interp_drain/bin/aprilcube_pose --max-cycles 40 --max-time 60 --plateau-chunks 15 --plateau-mm 1.0` |

The run restarts `aprilcube_pose` on the NUC with `--pose-bin` (default `build_interp/bin/aprilcube_pose`, i.e. the OLD slow camera):
pass `--pose-bin build_interp_drain/bin/aprilcube_pose`. ROBOT_STATE must be published (state_pub) and the multicast route
(`224.0.0.0/4` on the wired interface) must exist on both machines, otherwise the pre-flight reads 0 Hz.

## Flags

| flag | default | what it does | status |
|---|---|---|---|
| `--config` | `configs/pusht_real.yaml` | scene / cost / vmax config; the planner uses the scene xml of that config | stable |
| `--dial-noise` | 1.0 | DIAL `noise_level` (absolute) | stable |
| `--fast-stop` | off | at a measurement stop: no settle wait, median of the first 3 consecutive stable frames (<0.5 mm, <0.3 deg) with image time >= stop - `--fast-pre-s`; falls back to the newest 3 frames after 0.6 s | experimental |
| `--fast-pre-s` | 0.05 | how long before the stop frames may be used | experimental |
| `--min-tags` | 2 | drop T frames with fewer detected tags | experimental |
| `--max-reproj-px` | 0 (off) | drop T frames with a larger reprojection error | experimental |
| `--plateau-chunks K`, `--plateau-mm` | 0 (off), 1.0 | stop when the planning-state distance varied < plateau-mm over the last K chunks | experimental |
| `--converge-continuous K` | 0 (off) | `--blend`: stop when the camera-corrected state is within tolerance at K consecutive hand-overs | experimental |
| `--min-cam-hz` | 32 (was 38) | bring-up camera check | stable |

Also: the T-frame silence soft stop uses the raw T_SHAPE_POSE stream (not only frames that passed the gates);
`t_pose_raw.csv` has the extra columns `num_tags,reproj_px`; the cleanup restarts the camera with the run's `--pose-bin`.

## Latency findings

- Camera: `recv - stamp` of T_SHAPE_POSE was ~180 ms with `build_interp` (frame queue in `aprilcube_pose`; the stamp is taken when the
  frame is fetched, processing is serial and slower than 60 fps, so ~10 frames queue up). Evidence: latency floor ~166 ms, it fell
  to ~20 ms while the rate was > 60 fps (queue draining), non-interp build had 13-20 ms. After the drain patch
  (`build_interp_drain`): median ~24-29 ms, p90 ~34 ms, ~45 Hz.
- Not clock offset: NUC - 5090 offset +2.5 ms (LCM ping-pong, RTT 0.65 ms); ROBOT_STATE transport ~0.4 ms.
- Stop -> pose per measurement (`--fast-stop`, drain camera): mean 0.114 s (0.079-0.228); before: 0.46 s (settle wait 0.41 + read 0.05).
- One cycle with `--measure-every 1`: 0.93 s = 0.50 walk + 0.11 read + 0.31 plan.

## Open issue

After the recalibration the runs stall 18-25 mm short of the goal (latest: 24.9 mm / 1.45 deg after 40 chunks, plateau not met:
28.6 -> 25.6 mm over the last 15 chunks). The goal itself was checked with the T placed in the sticker outline: the live reading differs
from the hard-coded goal by 0.5 mm / 0.26 deg, and the residual points from the final T pose toward that reading, so it is not a goal
offset. The planner stall (DIAL stops pushing, planned waypoints move ~1 mm per chunk) is under investigation; not yet explained.
With `--min-tags 4` a run was soft-stopped near the goal because only 3-4 tags stay visible there.
