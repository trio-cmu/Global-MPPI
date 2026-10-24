# Real-robot Push-T closed loop: notes (2026-10-20)

> **Zhongqi: start with [`ZHONGQI_QUICKSTART.md`](ZHONGQI_QUICKSTART.md)** (copy-paste setup for your own accounts on the 5090 and the NUC). This file has the details and the reasons.

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
| NUC | `git@github.com:trio-cmu/trio_infra.git` | 要的是 `main`（`ba58f94`）。**我们这台 NUC 实际在 `positions-1015`（`db37dca`），比 `main` 落后 15 个提交**（2026-10-24 只读检查），工作区还有未提交改动（`config/calibration.yaml`，内容和 `main` 的 2026-10-20 重标版**逐字相同**；`data/*_robot.yaml`；被跟踪的 `build/` 里的文件） | `~/projects/trio_infra` |
| NUC | 同一个 trio_infra（驱动用的 git worktree） | 我们现在用 `noee-z-upright-blend`；其中的 blend 提交 `7e628c7` **已经是 `origin/main` 的祖先** | `~/projects/trio_infra_upright`（二进制 `build_upright/bin/xy_driver`） |

- `trio_infra` 的 `main` 已经包含：相机帧队列 drain 补丁（`b1858fa`）、`script/t_shape_pose_drain.sh`、`script/setup_lcm_route.sh`、`docs/camera_frame_drain.md`、`--interp-flange`、2026-10-20 的相机到法兰重新标定（`config/calibration.yaml`）、xy_driver 的 `--upright` / 计划限制 / `--max-lead-mm` / `--blend`。
- **待确认**：从一个全新的 `main` 克隆能不能直接编出我们现在用的两个二进制。我们的二进制是在分支 `camera-frame-drain` 和 `noee-z-upright-blend` 上编的，没有从 fresh `main` 验证过。
- **xy_driver 分步归位（转角 >30° 时拆成多步）还没有提交**：它只是 `~/projects/trio_infra_upright` 工作区里的未提交改动（`upright.hpp`、`xy_driver.cpp`、`test_upright.cpp`），**没有在真机上验证过**，默认二进制 `build_upright/bin/xy_driver` 里**没有**它。推到哪个分支：**待确认（还没决定）**。
- 5090 上还有两个外部依赖，`Global-MPPI` 的代码按固定路径去读（不在 Global-MPPI 仓库里）：
  - `~/trio_sim`：trio_infra 的另一个克隆（我们这里是 detached HEAD `acf03b6`，且有很多**未提交的本地改动**：`config/lcm_relay.yaml`、`src/pusht_align/{frames.py,pusht_goal.yaml,pusht_starts.yaml,render_run.py,tshape.py,verify_alignment.py,pusher_start.yaml}`）。代码用到其中的 `src/pusht_align/frames.py`、`pusher_start_pushside_near.yaml`、`pusher_start.yaml`、`pusht_starts.yaml`、`pusht_goal.yaml`（`closed_loop_test.py` 里的 `TRIO_SIM`）。**待确认**：哪些本地改动是必须的、`main` 里的 `src/pusht_align` 能不能替代它。
  - `~/projects/trio_infra/src/mppi_bridge/lcmtypes_py`：**生成出来的** LCM Python 类型（不在 git 里，已 gitignore），用 `script/generate_lcm_python.sh` 生成。**不需要先编 `lcm-gen`**：`main` 里**提交了** `build/third_party/lcm/lcmgen/lcm-gen`（x86-64 二进制，来自 Zhongqi 的机器），2026-10-24 在全新克隆里直接运行脚本，生成了 `camera_driver_lcm` 和 `franka_driver_lcm`。没有这个目录，真机运行一开始就会 `ModuleNotFoundError: No module named 'camera_driver_lcm'`（`--sim` 不需要它）。`closed_loop_test.py` 里的 `TRIO_INFRA`。
  - **`pusher_start_pushside_near.yaml` 不在 `trio_infra` 的 `main` 里**，但 `closed_loop_test.py` 默认从 `~/trio_sim/src/pusht_align/` 读它，缺了就在启动时 `FileNotFoundError`（`--sim` 也一样）。同样内容的文件在本仓库的 `configs/pusher_start_pushside_near.yaml`（和我们的 `~/trio_sim` 里那份逐字相同）：`cp configs/pusher_start_pushside_near.yaml ~/trio_sim/src/pusht_align/`。`--sim` 只读 `TRIO_SIM` 里的这一个文件（`frames.py` 等只在真机路径里导入），补上它就通；真机路径（`frames.py`、`pusher_start.yaml`、`pusht_starts.yaml`、`pusht_goal.yaml`）没有用 `main` 的版本验证过。

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
- **依赖声明不全（全新克隆检查，2026-10-24）**：`pyproject.toml` 和 `ksos-tools` 都**没有**声明 `lcm`（真机用）和 `newton_sos`（Global-MPPI 默认的 `newton-rs` 求解器，只写在 `ksos-tools/ksos_env.yml` 里；我们的版本是 0.2.3），要自己 `pip install lcm newton_sos`。`environment.yml` 只装 conda 部分（cuda、cudnn、cvxpy、eigenpy、matplotlib 等），jax / mujoco 靠随后的 `pip install -e .`；`environment.yml` 里有 matplotlib，但我们是纯 pip 装的环境，所以没有。

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
   - **`run_real.sh` 不能做 dry-run**（它带 `--auto-start`，和 `--dry-run` 冲突，报 `--auto-start needs --real (and not --dry-run)`）。dry-run 直接调用 python（不动机器人）：`python examples/closed_loop_test.py --real --dry-run --algo globalmppi --pipeline --blend --measure-every 1 --cam-correct --max-cycles 2 --adaptive-settle --read-s 0.1 --bg-link-check --fast-stop --min-tags 3 --max-reproj-px 2.0 --pose-bin build_interp_drain/bin/aprilcube_pose --out <新目录>`。没有真机时 `--sim` 可以走完整个规划。
   - **规划器的 cost 默认读 `--config` 里的 `cost` 段**（`w_pos` 50、`w_ori` 15、`w_approach` 10；2026-10-24 起，提交 `f59292d`）。此前的真机 run（到 `real_globalmppi_cpu_b1` 为止）规划器用的是 `PushT` 默认值，**`w_approach` = 0**，和 yaml 不符；现在的 run 和它们**不可直接比较**。`--no-config-cost` 恢复旧行为。启动日志会打印 `== planner cost: ...` 和 `== jax devices: ...`，先看这两行。
   - **用 CPU 跑**：`JAX_PLATFORM_NAME=cpu ./script/run_real.sh ...`。`JAX_PLATFORMS=cpu` 不行，因为脚本里 `unset JAX_PLATFORMS`。Global-MPPI 在 CPU 上基本不可用：每个 KSOS 规划步的 rollout 约 107 s（`real_globalmppi_cpu_b1`，负载下）。
   - **globalmppi**：每个规划步做 5 次重启（rollout → sigma 标定 → KSOS → MPPI），GPU 上约 5–10 s（负载下），而一个 chunk 只有 0.5 s；`--ksos-every N` 只每 N 个周期做一次完整求解（默认 1）。`cycles.csv` 末尾多了 `ksos_*` 和 `t_*_s` 列（迭代数、收敛比例、各段耗时），run 结束时会打印 `== KSOS:` 汇总行。脚本开头把 `OPENBLAS_NUM_THREADS` 设成 1（默认线程池下 sigma 标定慢约 1000 倍，b3 第一步 201 s）。
   - GPU 被别的任务占着时可能报 `cuSolver internal error`：设 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
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
  - 谁定期调用 `linkwatch_sample.sh`：2026-10-24 在 NUC 上只读查过：**没有**用户 crontab、`/etc/cron.d` 里没有它（root 的 crontab 没查）、`systemctl --user` 里没有 timer、也没有它的进程；`~/linkwatch.log` 最后写入是 2026-10-19，所以现在**没有任何东西在定期采样**，基线文件只在有人手动追加时变。
- **归位被拒**（`Refusing to start: --upright turn of X deg exceeds 30 deg`）：手引末端，让 yaw 离 59.7° 在 30° 以内、tilt 接近 0，再重跑。30° 是 `xy_driver.cpp` 里的 `kUprightMaxRotDeg`（`main` 和我们的二进制一致）。分步归位还没有进默认二进制（见 a）。
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

### f. 全新克隆检查结果（2026-10-24，没有机器人）

在临时目录克隆 `Global-MPPI`（`realrobot-pusht-pipeline`，`f59292d`，含子模块 `ksos-tools` `640ae32`）和 `trio_infra`（`origin/main`，`ba58f94`），`~/` 指到临时目录（`~/trio_sim` 指向 `trio_infra` 克隆）。NUC 只读。

**git 里缺的 / 要注意的**
- `pusher_start_pushside_near.yaml` 不在 `trio_infra` `main`（见 a）；缺了就是启动即 `FileNotFoundError`。
- `lcmtypes_py` 不在 git，要自己生成（见 a，`lcm-gen` 是提交了的）。
- `trio_infra` 的 `main` **跟踪了一个 205 个文件的 `build/` 目录**（`.gitignore` 里写明"tracked build/"）：`CMakeCache.txt` 指向 `/home/zhongqi/Documents/camera_driver`，里面有 `build/bin/aprilcube_pose`、`t_shape_pose_lcm_subscriber`、`lcm-gen` 二进制。不要把 `build/` 当成自己的构建目录（`script/build_franka_driver.sh` 默认输出到这里，会和别人机器上的 `CMakeCache.txt` 冲突）；文档里的 `build_interp_drain/`、`build_upright/` 是被 gitignore 的本地目录，**fresh clone 里不存在**，要按 b 自己编。
- `build_interp_drain/bin/aprilcube_pose`、`build_upright/bin/xy_driver`（含 `--upright` 和 `--blend`）都不在 git，要在 NUC 上编；`main` 的源码里有 `--upright`、`--blend`、`--max-lead-mm`、`--interp-flange`，但没有分步归位。
- `lcm` 和 `newton_sos` 没写进依赖（见 b）。
- 环境变量：`LCM_DEFAULT_URL`（`closed_loop_test.py` 和 `run_real.sh` 自己设默认值）、`JAX_PLATFORM_NAME=cpu`（CPU）、`XLA_PYTHON_CLIENT_PREALLOCATE=false`（GPU 被占时）、`PYTHON` / `FIGPY`（`run_real.sh`）。

**验证过的**：全新克隆里 `--sim --algo mppi` 和 `--sim --algo globalmppi` 都能跑（补上 `pusher_start_pushside_near.yaml` 之后）；`globalmppi` 日志里有 KSOS 迭代数（约 33，全部收敛）。真机 dry-run 在缺 `camera_driver_lcm` 处就停了（补上之后没有再跑，因为 NUC 这次只读）。

**NUC 实际状态（只读）**
- `~/projects/trio_infra`：`positions-1015` / `db37dca`，比 `main` 落后 15 个提交（`b1858fa` drain 补丁、`t_shape_pose_drain.sh`、2026-10-20 重标在 `main` 里；NUC 的 `config/calibration.yaml` 是未提交的本地改动，和 `main` 相同）。有未跟踪的 `Log/`、`build-local/`、`build/bin/franka_lcm_server`、`build/bin/xy_driver`。
- 二进制：`~/projects/trio_infra/build_interp_drain/bin/aprilcube_pose`（2026-10-20）、`~/projects/trio_infra_upright/build_upright/bin/xy_driver`（2026-10-17）、`~/projects/trio_infra/build/bin/{aprilcube_pose,franka_lcm_server,t_shape_pose_lcm_subscriber}`（2026-10-06，旧）。`trio_infra_upright` 在 `noee-z-upright-blend`（`7e628c7`），`upright.hpp` / `xy_driver.cpp` / `test_upright.cpp` 有未提交改动，`build_upright/` 未跟踪（和 a 里写的一致）。
- 进程：`robot_state_readonly_pub.py`（`cam_calib` 环境）和 `build_interp_drain/bin/aprilcube_pose … --width 848 --height 530 --fps 60` 在跑；没有 `xy_driver`、`franka_lcm_server`。
- `~/miniforge3/envs/cam_calib` 存在（`lcm`、`yaml`、`numpy` 可导入），仍然没有它的建立记录；`lcmtypes_py` 已生成；组播路由经 `enp88s0`；`~/xy_in` 目前不存在；NUC 上还有 `~/projects/Global-MPPI-main`（没查内容）；`lcm-gen` 不在 PATH 里。
