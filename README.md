# DE-P-Car

DE-P-Car 将 [DE-P](https://github.com/ZhangJunhe2005/DE-P) 的局部规划思想迁移到 ROS Noetic 下的地面阿克曼小车。工程使用缩放至原模型 `1/3` 的 [hifzhil/car-simulator](https://github.com/hifzhil/car-simulator) Urban Car、深度相机、360° VLP-16、在线 SLAM，以及参考 [FAR Planner](https://github.com/MichaelFYang/far_planner) 构建的动态可见图导航。

当前开发主线是 **DEPCarNet V4.3 Fusion + P6 shadow**：网络同时生成 15 组最多 6 把的前进/倒车 Ackermann 机动序列，并在闭环 DAgger 状态上学习完整序列排序。运行时仍由确定性局部控制链驾驶，V4.3 只发布 shadow 对照结果；连续车身碰撞检查、动态 reachability veto 和停车换挡安全合同始终拥有最终否决权。

> 当前资格边界：V4.3 离线正式验收和 P6 实现审计均为 PASS，但 Gazebo 跨地图闭环资格仍在验证。模型没有 active、真实车辆或 production 权限。

## 当前进展

| 阶段 | 状态 | 当前成果 |
| --- | --- | --- |
| P0～P2 | 完成 | 1/3 Urban Car 物理合同、双向 Ackermann 基线和深度/LiDAR/状态多模态数据合同。 |
| P3 V4 | 完成 | 45,380 个静态样本；train 35,940、validation 9,440；转角、窄路、倒车和三点掉头覆盖审计 PASS。 |
| P4～P5 V2 | 完成 | 路线走廊编码、15 候选双向 rollout、Candidate/Score 两阶段训练与 P6 shadow 基线。 |
| V3～V4.2 | 完成诊断 | 将挡位从外部状态机迁入模型，发展为统一的多把前进/倒车机动序列；保留完整失败报告和门禁链。 |
| P5 V4.3 | 完成 | 80 个闭环 episode、9,305 个重观测样本；精确 signed Hybrid-A* 教师序列；24 epoch Fusion 训练；正式 acceptance PASS。 |
| P6 V4.3 | 进行中 | 在线 SLAM、FAR-style 可见图、滚动路线事务、有向失败支路、实时重锚和 RViz 任意目标 shadow 入口已经实现。 |
| P7/P8 | 未开始 | 动态场景独立测试、holdout 资格和真实部署签发尚未完成。 |

V4.3 正式产物已随仓库提供：

- architecture：`dep_car_multimodal_v43_guarded_contextual_residual_closed_loop_hybrid_sequence_ackermann_15x6`
- checkpoint：`models/dep_car/p5_closed_loop_v43/fusion_closed_loop_sequence.best.pth`
- checkpoint SHA-256：`c89d5401774477caf11159495ee3d5e8eb3fbe6c95fe742ee0c8d528f0f535ac`
- validation：1,997 个样本，十二项离线门禁全部 PASS
- 资格范围：`P6_SHADOW_ONLY`

详细证据见：

- [`V4.3 acceptance`](models/dep_car/p5_closed_loop_v43/fusion_closed_loop_sequence.best.acceptance.json)
- [`V4.3 checkpoint contract`](models/dep_car/p5_closed_loop_v43/fusion_closed_loop_sequence.best.contract.json)
- [`P6 shadow implementation audit`](reports/p6_v43_shadow_implementation_audit.json)
- [`V4.3 数据完整性审计`](reports/p3_v7_v43_independent_integrity_audit.json)

## 运行时架构

```text
rear wheel joints + IMU
            |
            v
       EKF odometry ----> Slam Toolbox ----> online /map
            |                                  |
            |                        observed wall polygons
            |                                  |
360° LiDAR + depth --------> dynamic visibility graph (FAR-style)
            |                                  |
            +---------> rolling route corridor + local carrot
                                               |
                           DEPCarNet V4.3 shadow sequences
                                               |
                         deterministic local Ackermann control
                                               |
                  swept-footprint/dynamic hard veto + stop-before-shift
                                               |
                                          Urban Car
```

职责边界：

- 在线导航不启动 `map_server`，也不在完整 OccupancyGrid 上调用栅格 A*。
- 可见图只提供连通方向和路线管道，不直接输出油门、方向盘或强制离散 waypoint。
- DE-P 局部规划器根据深度、360° LiDAR、车辆状态和路线走廊产生 Ackermann 轨迹。
- V4.3 将 `FORWARD/REVERSE/STOP` 和最多六把的控制序列放在同一个候选内，不再由高层状态机分别挑选前进/倒车银行。
- shadow 模式下，V4.3 的序列会被记录和比较，但不会取得底盘控制权限。
- 进入死路后，有向失败支路记录入口和末端；退出连接器按当前 SLAM/LiDAR 实时重锚，breadcrumb 不重放历史油门或转角。
- 终点默认是 position-only，RViz 箭头方向不会强迫车辆在终点反复微调。

## 从全新克隆复现 RViz 验证

以下流程面向 Ubuntu 20.04、ROS Noetic 和 Gazebo 11。当前 checkpoint 的参考推理环境为 Python 3.10、PyTorch 2.7.0 + CUDA 12.8；默认启动配置需要 NVIDIA GPU。

### 1. 克隆当前开发分支

仓库为私有仓库时，需要先完成 GitHub 身份验证。

```bash
git clone \
  --branch agent/p5-p6-static-planning \
  https://github.com/ZhangJunhe2005/DE-P-Car.git
cd DE-P-Car
```

用于回退的已发布标签：`checkpoint-v43-p6-shadow-20260826`。

### 2. 安装 ROS 系统依赖

这些包安装在宿主机，不需要进入 Conda 环境。

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git \
  python3-numpy python3-opencv python3-yaml python3-pil \
  ros-noetic-effort-controllers \
  ros-noetic-joint-trajectory-controller \
  ros-noetic-gazebo-plugins \
  ros-noetic-velodyne-description \
  ros-noetic-velodyne-gazebo-plugins \
  ros-noetic-map-server \
  ros-noetic-robot-localization \
  ros-noetic-slam-toolbox
```

### 3. 获取锁定版本的第三方仓库

RViz/P6 最小复现只需要 Urban Car 和 FAR 上游来源：

```bash
bash scripts/fetch_locked_repositories.sh --runtime
```

如果还要重新生成 Arena 数据或核对原 DE-P 来源：

```bash
bash scripts/fetch_locked_repositories.sh --all
```

脚本只接受 [`third_party.lock.yaml`](third_party.lock.yaml) 和 [`dep_source.lock.yaml`](dep_source.lock.yaml) 中固定的 commit。已有目录版本错误或存在本地修改时会停止，不会覆盖用户文件。

### 4. 创建策略推理环境

```bash
conda env create -f environment-policy.yml
conda activate yopo

python -m pip install \
  torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

conda deactivate
```

启动器会通过当前 Conda 安装位置自动寻找 `envs/yopo/bin/python`。若环境使用其他名称或路径，可显式设置：

```bash
export DEP_CAR_POLICY_PYTHON=/absolute/path/to/python
```

### 5. 构建 Catkin 工作空间

```bash
source /opt/ros/noetic/setup.bash
bash scripts/bootstrap_workspace.sh
source catkin_ws/devel/setup.bash
```

`bootstrap_workspace.sh` 会把本仓库 ROS 包和锁定的 `car-simulator` 包链接到 `catkin_ws/src`，然后使用系统 Python 3 构建。

### 6. 执行无 Gazebo 复现预检

```bash
bash scripts/verify_rviz_reproduction.sh
```

它会检查：

- Urban Car 与 FAR checkout commit；
- V4.3 checkpoint SHA-256、contract、acceptance 和 authority；
- 仓库自带的最小冻结地图夹具；
- ROS 节点图、禁止的真值/A* 依赖和 shadow 权限边界；
- RViz interactive 命令的 dry-run。

最终应输出：

```text
RViz reproduction preflight PASS
```

### 7. 启动 RViz 任意目标验证

不需要手动进入 Conda 环境：

```bash
bash scripts/run_p6_v43_shadow.sh \
  --stage interactive
```

Gazebo、SLAM 和 RViz 启动后：

1. 等待 `/map` 出现且 RobotModel/点云不再报 TF 错误。
2. 在 RViz 工具栏选择 `2D Nav Goal`。
3. 在已经观察到或希望继续探索的区域点击并拖动。
4. 可以在同一次运行中连续发布多个目标；最新目标会取消旧任务。

RViz 图例：

| 显示项 | 含义 |
| --- | --- |
| `FAR Visibility Route`（青色） | 在线可见图选择的全局连通方向。 |
| `FAR Visibility Graph` | 当前已观察墙面形成的稀疏可见边。 |
| `Local Route Corridor`（蓝绿色） | 实际交给局部规划器的滚动路线管道。 |
| `Local Subgoal`（红色） | 当前短前视 carrot，不要求精确压过。 |
| `Controlling Candidate`（橙色） | 确定性控制链正在执行的 Ackermann 轨迹。 |
| `Policy Selected Path` | V4.3 shadow 推荐，用于与实际控制候选比较。 |
| `Navigation Memory` | breadcrumb、拓扑边、有向失败支路和退出连接器。 |

### 切换冻结地图

先列出当前 manifest 中通过文件、哈希和起点鲁棒性预检的场景：

```bash
bash scripts/run_p6_v43_shadow.sh \
  --stage list
```

仓库默认的 `data/p6_static/reproduction_manifest.json` 只包含一张可直接复现的精简地图。本机已经生成完整 P6 地图集时，可切换 manifest 并列出地图 seed、UUID、场景类型和场景 ID：

```bash
bash scripts/run_p6_v43_shadow.sh \
  --stage list \
  --scenario-manifest data/p6_static/scenario_manifest.json
```

从输出的 `scenarios` 中选择一个 development 场景，例如：

```bash
bash scripts/run_p6_v43_shadow.sh \
  --stage interactive \
  --scenario-manifest data/p6_static/scenario_manifest.json \
  --scenario p6_48667c45aa2e32f1
```

启动前会校验 manifest 身份、`map.world`/`map.yaml` SHA-256、解码后的占用图哈希、地图 seed/UUID，以及冻结的 27 组起点扰动证据。未通过预检的场景列在 `excluded` 中，不能启动。Holdout 场景默认封存；只应在最终独立验收时显式增加 `--allow-holdout`。

`--scenario-manifest` 只更换 Gazebo 测试输入，不修改 V4.3 checkpoint、FAR/DE-P 导航算法或 shadow 权限边界。完整多地图 corpus 体积较大，不随 Git 仓库分发；新克隆的仓库可直接运行默认精简地图，其他地图需先由数据生成流程构建或从受信任的同版本数据集恢复。

### 固定双目标回放

仓库还提供同一张冻结地图上的坐标合同回放：

```bash
bash scripts/run_p6_v43_shadow.sh \
  --stage replay \
  --sequence logged_t_junction_turnaround
```

固定目标使用稳定的 `odom` 坐标，在发布时转换到当前在线 `map`。宿主机预检和在线 SLAM 占用图都会拒绝墙内目标，并以 `INVALID_REPLAY_GOAL` 快速结束，而不是等待完整超时。该入口用于故障复现，不等同于跨地图资格 PASS。

## 常见问题

### `spawn_car` 或 Urban Car 包找不到

确认先后执行：

```bash
bash scripts/fetch_locked_repositories.sh --runtime
bash scripts/bootstrap_workspace.sh
source catkin_ws/devel/setup.bash
```

### `PyTorch policy interpreter not found`

创建名为 `yopo` 的环境，或设置 `DEP_CAR_POLICY_PYTHON`。ROS 节点使用系统 Python，只有模型推理节点使用该解释器。

### authority 或 checkpoint hash mismatch

不要编辑 committed checkpoint、contract 或 runtime 文件后继续使用旧 authority。先恢复对应 Git commit；开发者修改运行时代码后必须重新审计并签发 shadow authority。

### RViz 显示 `Unknown frame map`

在线 SLAM 启动前短暂出现属于正常初始化。若持续存在，检查：

```bash
rosnode list
rostopic hz /dep_car/scan
rostopic hz /odometry/filtered
rosrun tf tf_echo map odom
```

### `global_far_mapping_wait`

系统只会短暂确认未知区域路线稳定性。停在原地无法增加观察，因此超时后应转入局部安全探索；若长期不退出，请保留 `logs/memory_navigation/` 和 `reports/memory_navigation/` 供诊断。

### ROS/Gazebo 端口占用

每次只运行一个 interactive/replay 实例。正常使用 `Ctrl+C` 退出并等待清理完成，再启动下一次验证。

## 数据与训练复现

大规模数据集不随 Git 仓库发布。历史与当前入口仍保留：

```bash
# P3 V4 静态/转角数据
bash scripts/run_p3_v4_corner_curriculum.sh --stage prepare --workers 8
bash scripts/run_p3_v4_corner_curriculum.sh --stage collect --workers 8

# V3/V4 统一挡位与序列训练入口
bash scripts/run_p5_joint_gear_v3.sh --help
bash scripts/run_p5_hybrid_sequence_v4.sh --help

# V4.3 闭环 DAgger 数据、训练和验收
bash scripts/run_p5_closed_loop_v43.sh --help
```

V4.3 数据权威摘要：

- 80 个 episode；
- 9,305 个 closed-loop re-observed 样本；
- 2,386 个多动作样本；
- 2,857 个含倒车序列样本；
- 2,284 个 reverse-then-forward 样本；
- test split 保持封存；
- 教师 Hybrid A* 仅用于离线标签，不是在线运行时依赖。

正式训练配置位于 [`p5_closed_loop_v43.yaml`](dep_car/config/p5_closed_loop_v43.yaml)。长时间数据生成和训练任务应由宿主机用户启动；默认使用 8 个 DataLoader/CPU worker，Gazebo 采集建议使用 4 个隔离实例。

## 测试

当前提交的完整 Python 测试：

```bash
PYTHONPATH=$PWD/dep_car/src \
  /path/to/yopo/bin/python -m pytest -q
```

最近一次结果：

```text
505 passed
```

P6 V4.3 权限与入口审计：

```bash
bash scripts/run_p6_v43_shadow.sh --stage audit
bash scripts/run_p6_v43_shadow.sh --stage interactive --dry-run
```

## 仓库内容与边界

Git 仓库包含：

- DE-P-Car Python/ROS 源码、配置、测试与启动脚本；
- V4.3 正式 best checkpoint、contract、acceptance 和 P6 shadow authority；
- RViz 默认验证所需的一张冻结地图及 scenario manifest；
- 小型审计报告和第三方 commit 锁文件。

以下内容由 `.gitignore` 排除：

- P3/P5 大规模训练数据、rosbag 和提取缓存；
- 失败实验、pilot、optimizer state 和中间 checkpoint；
- 完整 P6 多地图 corpus、ROS/Gazebo 日志与 Catkin 构建产物；
- 第三方仓库工作树。

若要运行八地图矩阵、holdout 或重新训练，需要自行生成对应数据。仓库自带的最小场景只保证 README 中的默认 RViz/replay 入口具备输入文件，不代表跨地图鲁棒性资格。

本项目仍是仿真研究工程。P6 holdout、P7 动态场景和 P8 独立资格签发完成前，不应部署到真实车辆或宣称 production-qualified。
