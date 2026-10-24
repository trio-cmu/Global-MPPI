# 真机运行快速上手（给 Zhongqi，用你自己的账号）

只复制粘贴用的短版。细节、原因和排查见 [`realrobot_notes.md`](realrobot_notes.md)（"从零运行"一节）。
写于 2026-10-24。**没有在你的账号下完整走过一遍**：全新克隆里只验证了 `--sim`；真机路径里标了"未验证"的地方请先小心。

## 0. 先记住的规则

- **NUC 上同一时间只能有一个人的 `state_pub` / 相机（`aprilcube_pose`）/ `xy_driver` 在跑**：相机和 libfranka 连接都是独占的，LCM 总线上的 `ROBOT_STATE`、`T_SHAPE_POSE` 也会混在一起。开始前先看：
  `ssh <你的NUC用户名>@192.168.50.2 'ps -eo user,etime,args | grep -E "robot_state_readonly|aprilcube_pose|xy_driver|franka_lcm_server" | grep -v grep'`
  有别人的就请对方先停（不要在别人的进程上 `pkill -f`）。`run_real.sh` 只能停自己账号下的进程。
- 手放在使能键上，T 放在 `start_2`（≤3 mm、≤3°），末端 yaw 离 59.7° 在 30° 以内、tilt 接近 0。
- 每次开机两台机器各做一次 LCM 路由（见 §1、§2）。run 名必须是新的。

## 1. NUC（你自己的 NUC 账号）

```bash
mkdir -p ~/projects && cd ~/projects
git clone git@github.com:trio-cmu/trio_infra.git          # 分支 main
cd trio_infra && git log -1 --oneline                      # ba58f94 或更新
sudo ./script/setup_lcm_route.sh && ip route get 239.255.76.67   # 应经 enp88s0
./script/generate_lcm_python.sh                            # 生成 LCM Python 类型；lcm-gen 已提交在 build/ 里，不用先编
```

**不在 git 里、要自己做的：**

| 内容 | 怎么做 |
|---|---|
| 相机二进制 `build_interp_drain/bin/aprilcube_pose` | `cmake -S . -B build_interp_drain -DCMAKE_BUILD_TYPE=Release -DCMAKE_IGNORE_PREFIX_PATH="$HOME/miniforge3" -DCMAKE_IGNORE_PATH="$HOME/miniforge3/lib;$HOME/miniforge3/bin"` 然后 `cmake --build build_interp_drain --target aprilcube_pose --parallel 10`；Orbbec SDK 要先编好（`script/build_pose_estimation.sh` 里有步骤）。来源文档 `docs/camera_frame_drain.md` 自己写明"没有按原样跑过" |
| `xy_driver`（带 `--upright`、`--blend`） | 同样的 `cmake` 参数，目录 `build_upright`，`--target xy_driver --parallel 2`；需要系统里的 libfranka。`main` 的 `src/franka_driver` 和我们在用的分支一致，但**没有从全新 `main` 编过** |
| Python 环境 `~/miniforge3/envs/cam_calib` | 里面要有 `lcm`、`pyyaml`、`numpy` 和 `pylibfranka`（`robot_state_readonly_pub.py` 用）。**怎么建没有记录**，请找 junlong |
| 链路基线 `~/linkwatch_baseline.txt` | 你的主目录里没有。先确认链路正常（`ping -c 60 172.16.0.2` 0% 丢包，`dmesg \| grep enp89s0` 没有最近的 Link Down），再记下当前值：`touch ~/linkwatch.log; echo "$(date -Is) carrier_changes=$(cat /sys/class/net/enp89s0/carrier_changes)" >> ~/linkwatch_baseline.txt`。预检要求 `carrier_changes` 等于这个文件最后一行的值；NUC 重启后计数会清零，要重记 |

**5090 到 NUC 的 ssh 免密**（在 5090 上用你的账号做一次，`<NUC用户名>` 是你的 NUC 账号）：
```bash
ssh-keyscan -H 192.168.50.2 >> ~/.ssh/known_hosts
ssh-copy-id -i ~/.ssh/id_ed25519.pub <NUC用户名>@192.168.50.2
ssh -o BatchMode=yes <NUC用户名>@192.168.50.2 true
```

## 2. 5090（你自己的账号）

```bash
cd ~ && mkdir -p projects && cd projects
git clone --recurse-submodules -b realrobot-pusht-pipeline git@github.com:trio-cmu/Global-MPPI.git
cd Global-MPPI && git submodule update --init --recursive
conda env create -f environment.yml && conda activate global-mppi
pip install -e . && pip install -e ./ksos-tools
pip install lcm newton_sos            # 没有写进依赖，但真机和 Global-MPPI 默认求解器要用

# 代码按固定路径找 trio_infra：~/projects/trio_infra（生成的 LCM 类型）和 ~/trio_sim（pusht_align）
git clone git@github.com:trio-cmu/trio_infra.git ~/projects/trio_infra
ln -s ~/projects/trio_infra ~/trio_sim                       # --sim 这样验证过
cp configs/pusher_start_pushside_near.yaml ~/projects/trio_infra/src/pusht_align/   # 不在 trio_infra 的 main 里，缺了启动就 FileNotFoundError
(cd ~/projects/trio_infra && ./script/generate_lcm_python.sh)
sudo ~/projects/trio_infra/script/setup_lcm_route.sh && ip route get 239.255.76.67    # 应经 enp129s0
```

**写死了 junlong 的地方，用环境变量换成你自己的**（不用改任何源文件；不设就会连到 junlong 的 NUC 账号和路径）。在 `~/.bashrc` 里加，或每次运行前 export：

```bash
export NUC=<你的NUC用户名>@192.168.50.2          # 必须改（默认 junlong@192.168.50.2）
export PYTHON=$(which python)                      # 5090 上 global-mppi 环境的 python（默认 /home/junlong/miniforge3/envs/global-mppi/bin/python）
# 下面这些只在和默认不同时才设。注意加单引号：不加的话 ~ 会在 5090 上被展开成 5090 的主目录
export NUC_REPO='~/projects/trio_infra'            # NUC 上 trio_infra 的位置
export NUC_PY='~/miniforge3/envs/cam_calib/bin/python'   # NUC 上带 lcm / pylibfranka 的 python
export XY_DRIVER_BIN=build_upright/bin/xy_driver   # 相对 NUC_REPO；默认 ../trio_infra_upright/build_upright/bin/xy_driver 是 junlong 的 worktree
export FIGPY=/usr/bin/python3                      # 画图用的 python，要有 numpy 和 matplotlib（默认这个）
```

- `NUC`、`NUC_REPO`、`NUC_PY`、`XY_DRIVER_BIN`、`NUC_DISPLAY` 由 `examples/open_loop_test.py` 读取（`XY_DRIVER_BIN` 也是 `closed_loop_test.py` 里 `--xy-driver-bin` 的默认值，命令行参数优先）。
- `--pose-bin`：`run_real.sh` 固定传 `build_interp_drain/bin/aprilcube_pose`（相对 NUC 仓库），你编在别处才要在命令后面加 `--pose-bin <路径>` 覆盖。

不用改（会自动落到你自己的 NUC 主目录）：`~/linkwatch_baseline.txt`、`~/linkwatch.log`、`~/xy_in`、`{NUC_REPO}/log/`。
不用改但要对：机器人 IP `172.16.0.2`（`ROBOT_IP`）、网口 `enp89s0`（`NUC_IFACE`，NUC 到机器人）、`enp129s0` / `enp88s0`（只是路由检查的提示）。`NUC_DISPLAY`（默认 `:2`，junlong 的 Xwayland）只在 `open_loop_test.py` 的叠图里用，`run_real.sh` 不用。

## 3. 运行

```bash
conda activate global-mppi && cd ~/projects/Global-MPPI      # 并且已按 §2 export 了 NUC / PYTHON（等）
git pull                                   # 每次先拉最新（Global-MPPI 在 realrobot-pusht-pipeline；NUC 的 trio_infra 在 main）
nvidia-smi                                 # GPU 有没有被别人占满

./script/run_real.sh ps         <run名>
./script/run_real.sh dial       <run名>
./script/run_real.sh mppi       <run名>
./script/run_real.sh globalmppi <run名>     # 要 GPU

JAX_PLATFORM_NAME=cpu ./script/run_real.sh mppi <run名> ...     # CPU 变体（不要用 JAX_PLATFORMS，脚本会 unset 它）
./script/run_real.sh mppi <run名> --no-config-cost ...          # 旧行为：规划器 w_approach = 0
./script/run_real.sh globalmppi <run名> --ksos-every 5 ...      # 每 5 个周期才做完整 KSOS（默认每个周期）
```

- `run_real.sh` 已经把 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 设好（GPU 被占时否则可能 `cuSolver internal error`），可以用同名环境变量覆盖。
- 流程：预检（链路、相机、tag、起点偏差，失败就不动机器人）→ 5 秒倒计时（Ctrl-C 取消）→ 归位、下降 → 闭环 → 收尾 → 出图并追加 `runs/closed_loop_test/summary.csv`。结果在 `runs/closed_loop_test/<run名>/`。
- 启动日志先看这两行：`== jax devices: ...`（确认 GPU/CPU）和 `== planner cost: w_pos 50, w_ori 15, w_approach 10`。
- 不动机器人的检查：`python examples/closed_loop_test.py --sim --algo mppi --max-cycles 3 --out /tmp/x`，真机但不发指令用 `--real --dry-run ...`（`run_real.sh` 不能 dry-run，命令见 `realrobot_notes.md`）。

## 4. 已知限制

- **归位 yaw 限制 30°**：`xy_driver --upright` 的转角超过 30° 会拒绝（`Refusing to start: --upright turn of X deg exceeds 30 deg`）。**分步归位（拆成多步）不在 git 里**，也没在真机上验证过。解决：手引末端，让 yaw 离 59.7° 在 30° 以内、tilt 接近 0，再跑。
- **规划器 cost 默认读配置的 `cost` 段**（`w_approach = 10`，提交 `f59292d`）。2026-10-24 之前的真机 run 用的是 `w_approach = 0`，**和现在的 run 不可直接比较**；要复现旧行为加 `--no-config-cost`。
- **Global-MPPI 需要 GPU**：每个规划步 5–10 s（GPU，负载下），而一个 chunk 只有 0.5 s；CPU 上每个 KSOS 步的 rollout 就要约 107 s，基本不能用。第一个规划步还要额外几十秒编译。
- 预检失败时：`ROBOT_STATE` 0 Hz 通常是 `state_pub` 因 `libfranka UDP receive: Timeout` 崩了（重启它）；`ROBOT_STATE` 和 `T_SHAPE_POSE` 都是 0 Hz 通常是组播路由没设（§1、§2 的 `setup_lcm_route.sh`）；`carrier_changes` 对不上基线说明链路抖过，先查清再更新基线，不要为了通过检查而追加。
