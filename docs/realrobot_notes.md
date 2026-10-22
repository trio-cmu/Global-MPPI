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

---

## 从零运行（给 Zhongqi）

写于 2026-10-22，按我们这台机器上**实际存在**的脚本和参数写。不确定的地方都标了 **待确认**，没有验证过的不要当成已验证的步骤。
实验性参数（`--auto-start`、`--max-fallbacks`、`--fast-stop`、`--min-tags`、`--max-reproj-px`、`--plateau-*`）的说明见上面的 Flags 表。

### a. 要拉的仓库、分支、位置

| 机器 | 仓库 | 分支 | 我们这台机器上的位置 |
|---|---|---|---|
| 5090 | `git@github.com:trio-cmu/Global-MPPI.git`（`git clone --recurse-submodules`，`ksos-tools` 是子模块） | `realrobot-pusht-pipeline` | `~/projects/Global-MPPI` |
| NUC | `git@github.com:trio-cmu/trio_infra.git` | `main`（`ba58f94`） | `~/projects/trio_infra` |
| NUC | 同一个 trio_infra（驱动用的 git worktree） | 我们现在用 `noee-z-upright-blend`；其中的 blend 提交 `7e628c7` **已经是 `origin/main` 的祖先** | `~/projects/trio_infra_upright`（二进制 `build_upright/bin/xy_driver`） |

- `trio_infra` 的 `main` 已经包含：相机帧队列 drain 补丁（`b1858fa`）、`script/t_shape_pose_drain.sh`、`script/setup_lcm_route.sh`、`docs/camera_frame_drain.md`、`--interp-flange`、2026-10-20 的相机到法兰重新标定（`config/calibration.yaml`）、xy_driver 的 `--upright` / 计划限制 / `--max-lead-mm` / `--blend`。
- **待确认**：从一个全新的 `main` 克隆能不能直接编出我们现在用的两个二进制。我们的二进制是在分支 `camera-frame-drain` 和 `noee-z-upright-blend` 上编的，没有从 fresh `main` 验证过。
- **xy_driver 分步归位（转角 >30° 时拆成多步）还没有提交**：它只是 `~/projects/trio_infra_upright` 工作区里的未提交改动（`upright.hpp`、`xy_driver.cpp`、`test_upright.cpp`），**没有在真机上验证过**，默认二进制 `build_upright/bin/xy_driver` 里**没有**它。推到哪个分支：**待确认（还没决定）**。
- 5090 上还有两个外部依赖，`Global-MPPI` 的代码按固定路径去读（不在 Global-MPPI 仓库里）：
  - `~/trio_sim`：trio_infra 的另一个克隆（我们这里是 detached HEAD `acf03b6`，且有很多**未提交的本地改动**：`config/lcm_relay.yaml`、`src/pusht_align/{frames.py,pusht_goal.yaml,pusht_starts.yaml,render_run.py,tshape.py,verify_alignment.py,pusher_start.yaml}`）。代码用到其中的 `src/pusht_align/frames.py`、`pusher_start_pushside_near.yaml`、`pusher_start.yaml`、`pusht_starts.yaml`、`pusht_goal.yaml`（`closed_loop_test.py` 里的 `TRIO_SIM`）。**待确认**：哪些本地改动是必须的、`main` 里的 `src/pusht_align` 能不能替代它。
  - `~/projects/trio_infra/src/mppi_bridge/lcmtypes_py`：**生成出来的** LCM Python 类型（不在 git 里），用 `script/generate_lcm_python.sh` 生成（需要先编出 `lcm-gen`，见该脚本开头的提示）。`closed_loop_test.py` 里的 `TRIO_INFRA`。

### b. 首次构建

**NUC 上**
1. 相机（带 drain 的版本，目录 `build_interp_drain/`，二进制 `build_interp_drain/bin/aprilcube_pose`）。`docs/camera_frame_drain.md` 里的命令（**文档自己写明"没有按原样跑过"，待确认**）：
   ```bash
   cd ~/projects/trio_infra
   cmake -S . -B build_interp_drain -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_IGNORE_PREFIX_PATH="$HOME/miniforge3" -DCMAKE_IGNORE_PATH="$HOME/miniforge3/lib;$HOME/miniforge3/bin"
   cmake --build build_interp_drain --target aprilcube_pose --parallel 10
   ```
   Orbbec SDK 要先编好（`script/build_pose_estimation.sh` 里有编 `thrid-party/OrbbecSDK_v2` 的步骤）。文档里的 `git switch camera-frame-drain` 一步现在不需要了（补丁已在 `main`）；**待确认**。
2. xy_driver（我们的 `build_upright` 是用这套参数配置的，见其 `CMakeCache.txt`）：
   ```bash
   cd ~/projects/trio_infra_upright
   cmake -S . -B build_upright -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_IGNORE_PREFIX_PATH="$HOME/miniforge3" -DCMAKE_IGNORE_PATH="$HOME/miniforge3/lib;$HOME/miniforge3/bin"
   cmake --build build_upright --target xy_driver --parallel 2
   ```
   需要系统里装好 libfranka（我们的 `Franka_DIR=/usr/lib/cmake/Franka`）。`script/build_franka_driver.sh` 是同样的参数但输出到 `build/`。
3. Python 环境：`~/miniforge3/envs/cam_calib`（带 `lcm`，`robot_state_readonly_pub.py` 用它）。**待确认**：这个环境怎么建（我们没有记录）。

**5090 上**
```bash
cd ~/projects/Global-MPPI
git submodule update --init --recursive
conda env create -f environment.yml       # 名字 global-mppi；README 的 Installation 一节有完整步骤（含 pip install -e . 和 ksos-tools）
conda activate global-mppi
pip install -e .
```
- 我们这个环境的实际版本：Python 3.12.14，jax 0.5.3（`jax[cuda12]`），mujoco / mujoco-mjx 3.13.0，lcm 1.5.2（pip），驱动 595.99.02，RTX 5090。
- 我们这个环境里**没有 matplotlib**，所以 `run_real.sh` 画图用 `/usr/bin/python3`（有 numpy 和 matplotlib）；你可以在环境里装 matplotlib，或用 `FIGPY=/path/to/python` 覆盖。`environment.yml` 里写了 matplotlib，**待确认**为什么我们的环境里没有。
- 真机还需要 `lcm`（pip）和上面的 `~/trio_sim`、`~/projects/trio_infra/src/mppi_bridge/lcmtypes_py`。

**ssh 免密（5090 控制 NUC，脚本里全部用 `ssh -o BatchMode=yes`）**
```bash
ssh-keyscan -H 192.168.50.2 >> ~/.ssh/known_hosts
ssh-copy-id -i ~/.ssh/id_ed25519.pub junlong@192.168.50.2     # 输一次 NUC 密码
ssh -o BatchMode=yes junlong@192.168.50.2 true                  # 检查
```
（这几行来自 `examples/open_loop_test.py` 的 `SSH_HELP`。）两台机器还需要各自能 `git pull` GitHub（SSH key）。

### c. 每次运行的完整顺序

1. **两台机器各做一次（每次开机）**：LCM 组播走直连网线而不是 WiFi。
   ```bash
   cd ~/projects/trio_infra && sudo ./script/setup_lcm_route.sh   # 5090 和 NUC 都要
   ip route get 239.255.76.67      # 5090 应经 enp129s0，NUC 应经 enp88s0；经 wlp128s20f3 / wlo1 就是没设好
   ```
   没设好的症状：预检里 `T_SHAPE_POSE` 和 `ROBOT_STATE` 都是 0 Hz。
2. **机器人**：上电；在 Desk 里开 FCI；末端设置选 **No End Effector**。同一时间只能有一个 libfranka 连接。手放在使能键上（半按中间位置）。
3. **摆放**：T 放在 `start_2`（`pusht_starts.yaml`），位置偏差 ≤3 mm、角度 ≤3°（启动检查，写死在代码里）；末端 yaw 离 59.7° 在 30° 以内、tilt 接近 0（`xy_driver --upright` 超过 30° 会拒绝）。
4. **NUC 上要先有 ROBOT_STATE 和相机**（预检要求 `T_SHAPE_POSE`、`ROBOT_STATE` 都有输出；运行脚本启动时会自己停掉再重启它们，结束时恢复）。手动起：
   ```bash
   cd ~/projects/trio_infra; export LCM_DEFAULT_URL="udpm://239.255.76.67:7667?ttl=1"
   nohup ~/miniforge3/envs/cam_calib/bin/python src/extrinsic_check/robot_state_readonly_pub.py > log/state_pub_$(date +%Y%m%d_%H%M).log 2>&1 &
   nohup ./script/t_shape_pose_drain.sh > log/t_shape_pose_$(date +%Y%m%d_%H%M).log 2>&1 &
   ```
   （`script/openloop_cmds.md` 有更多命令。**不要**在工具 shell 里用 `pkill -f`。）
5. **5090 上运行**：
   ```bash
   conda activate global-mppi
   cd ~/projects/Global-MPPI
   nvidia-smi                                   # 确认 GPU 可用、没有别的进程占满显存
   ./script/run_real.sh <ps|dial|mppi|globalmppi> <run名>     # run 名必须是新的
   ```
   - 流程：预检（链路、相机频率、tag 数、起点偏差，失败就不动机器人）→ 5 秒倒计时（Ctrl-C 可取消，此时还没动）→ 归位到起点、下降 → 闭环（`--measure-every 1`，每个 chunk 后停下读姿态）→ 收尾（抬起、恢复相机和 state_pub）→ 出图并追加汇总行。
   - JIT 编译约 28 s（GPU），在预检前完成。
   - 额外参数直接加在后面，例如 `--xy-driver-bin <相对 NUC 仓库的路径>`（默认 `../trio_infra_upright/build_upright/bin/xy_driver`）。
6. **输出**在 `runs/closed_loop_test/<run名>/`：`args.json`（参数和停止原因、最终误差）、`cycles.csv`（每个 chunk 一行）、`t_pose_raw.csv`（每个相机帧，列 `recv_epoch_s, stamp_epoch_s, x_base, y_base, theta_base_rad, num_tags, reproj_px, fallback`）、`xy_state_raw.csv`、`xy_cmd_log.csv`、`trace.csv`、`plans.npz`、`closed_loop_*.png`，以及 `fig_trajectory.png`、`fig_error_vs_time.png`、`fig_cycle_timeline.png`、`fig_plan_time.png`。
7. **`runs/closed_loop_test/summary.csv`**（每个 run 追加一行；`runs/` 不进 git）：`final_pos_mm`/`final_ang_deg`（`final_source` 为 `last measured` 表示没有停止后的最终读数，只有最后一次测量）、`reached_10mm_3deg`、`duration_s`、`plan_mean_ms`/`plan_p95_ms`、`fallbacks`、`truncated_by_max_time`、`cycles`、`stop_reason`。`plateau` 表示距离 15 个 chunk 内变化不到 1 mm 就停了，不是收敛。多个 run 的并排图用 `global_mppi/closed_loop/compare_runs.py`。

### d. 故障排查

- **`T_SHAPE_POSE` 0 Hz**（预检第二行）。我们碰到过三种：
  1. 组播路由没设（两个通道都是 0 Hz）→ 见 c.1。
  2. `state_pub` 没在跑（它在机器人断电后会因 `libfranka UDP receive: Timeout` 崩掉）。插值相机要靠 ROBOT_STATE 才出位姿，所以 ROBOT_STATE 没了 T_SHAPE_POSE 也是 0 Hz → 重新起 state_pub。
  3. 相机进程活着但没输出（日志里 `Failed to restart stream`）→ 重启相机（`t_shape_pose_drain.sh`）。
  相机频率检查要求 ≥32 fps（`--min-cam-hz`），下降后还要求每帧至少 3 个 tag。
- **链路抖动检查**：NUC 上 `enp89s0`（连机器人的网口）的 `carrier_changes` 必须**等于** `~/linkwatch_baseline.txt` 里最后一行 `carrier_changes=` 的值（NUC 重启后计数会清零，所以重启后一定要重记）。
  - 先确认链路是好的：在 NUC 上 `ping -c 60 172.16.0.2` 0% 丢包，`tail ~/linkwatch.log` 看 `Link is Down/Up` 的时间。
  - 更新基线：在 NUC 上运行 `~/linkwatch_sample.sh`（它给 `~/linkwatch_baseline.txt` 追加一行当前值）。**只有确认抖动已经查清、可以接受时才追加，不要为了让检查通过而追加。**
  - 谁定期调用 `linkwatch_sample.sh`：**待确认**（文件里有周期性采样行，但我们没查到调度它的 cron/systemd）。
- **归位被拒**（`Refusing to start: --upright turn of X deg exceeds 30 deg`）：手引末端，让 yaw 离 59.7° 在 30° 以内、tilt 接近 0，再重跑。分步归位还没有进默认二进制（见 a）。
- **tag 丢失**：`T pose lost`、`T_SHAPE_POSE silent for 1.0 s`。含义：测量停止时没有合格帧（被 `--min-tags 3` / `--max-reproj-px 2.0` 丢掉）或相机连续 1 s 没有任何 ≥2 个 tag 的帧（NUC 对少于 2 个 tag 的帧不发消息）。`run_real.sh` 默认带 `--max-fallbacks 2`（读不到位姿时用预测继续，最多连续 2 次，日志和 `t_pose_raw.csv` 标 `fallback=1`）。我们的 PS 两次都是笔跑得很远（离 T 中心 130–200 mm）时丢帧，其他三个算法没有；原因（手臂遮挡相机？）**待确认**，没有验证。

### e. 我们这台机器特有的东西（Zhongqi 要改）

| 内容 | 我们的值 | 在哪改 |
|---|---|---|
| NUC 地址和用户 | `junlong@192.168.50.2` | `examples/open_loop_test.py` 的 `NUC` |
| 机器人 IP | `172.16.0.2` | `open_loop_test.py` 的 `ROBOT_IP` |
| NUC 到机器人的网口（链路检查看它） | `enp89s0` | `open_loop_test.py` 的 `NUC_IFACE`；基线文件也是这个口 |
| 直连网线的网口 | 5090 `enp129s0`，NUC `enp88s0`（其余是 WiFi） | `setup_lcm_route.sh` 自动按 `192.168.50.0/24` 找 |
| NUC 上的仓库和 Python | `~/projects/trio_infra`、`~/miniforge3/envs/cam_calib/bin/python` | `open_loop_test.py` 的 `NUC_REPO`、`NUC_PY` |
| xy_driver 二进制 | `../trio_infra_upright/build_upright/bin/xy_driver` | `--xy-driver-bin` |
| 5090 上的外部目录 | `~/trio_sim`、`~/projects/trio_infra` | `closed_loop_test.py` 的 `TRIO_SIM`、`TRIO_INFRA` |
| 5090 的 Python | `/home/junlong/miniforge3/envs/global-mppi/bin/python`；画图 `/usr/bin/python3` | `run_real.sh` 的 `PYTHON=`、`FIGPY=` 环境变量 |
| 相机 | `build_interp_drain/bin/aprilcube_pose`，`--cam-delay-ms 60`，848x530@60，标定 `config/calibration.yaml`（2026-10-20 重标，只对我们这个相机和夹具有效） | `--pose-bin`、`--cam-delay-ms` |
| 目标和起点 | `configs/pusht_real.yaml`：`goal_base`（0.39983, 0.11279, 1.55985 rad），`start_2`；笔起点 `pusher_start_pushside_near` | 换了桌面或标定要重新记录 |
| yaw | `--yaw-deg=59.7`（在 `closed_loop_test.py` 里传给 xy_driver） | `--yaw-deg` |
| 组播 | `239.255.76.67`，`ttl=1` | `LCM_DEFAULT_URL` |
| 其他 | NUC 上 xy_driver 的 stdin 管道 `~/xy_in`，NUC 的 Xwayland `:2`（只用于 viser 叠图） | `open_loop_test.py` |
