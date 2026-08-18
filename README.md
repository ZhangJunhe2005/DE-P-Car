# DE-P-Car

DE-P-Car 将 [DE-P](https://github.com/ZhangJunhe2005/DE-P) 的局部规划思想迁移到 ROS Noetic 下的地面阿克曼小车。工程使用缩放后的 `hifzhil/car-simulator` Urban Car、Arena 随机地图、深度相机和 360° VLP-16，并围绕前进、倒车、三点掉头、窄通道及 90° 转角重新建立了数据、模型和运行时安全合同。

当前正式模型是 **DEPCarNetV2 Fusion**。它接收深度、LiDAR BEV、9D 车辆状态、挡位和车体坐标系下的路线走廊，生成 3×5 条满足双向 Ackermann rollout 的候选轨迹。全局 A* 只提供连通方向、路线走廊和挡位提示；局部模型负责轨迹生成与排序；连续车身 hard-veto、动态 reachability veto 和 stop-before-shift 始终拥有最终安全权限。

> 当前阶段：P0～P5 已完成，P6 静态 Gazebo shadow 验证和场景鲁棒性加固正在进行。现有 checkpoint 为 `TRAINED_UNQUALIFIED`，尚未获得 active 或 production 资格。

## 当前进展

| 阶段 | 状态 | 主要成果 |
| --- | --- | --- |
| P0 | 完成 | Urban Car 统一缩放至上游模型的 1/3；车辆尺寸、轴距、轮胎、碰撞体、质量/惯量及规划 footprint 保持一致。 |
| P1 | 完成 | signed bicycle rollout、双向 Hybrid A* 数据标注、换挡惩罚、停车换挡状态机、连续五圆车身碰撞检查。 |
| P2 | 完成 | `StaticAckermannSampleV2` 多模态合同：metric depth、原始点云引用、6 通道 360° LiDAR BEV、IMU、9D state、gear/route/candidate 和完整 provenance。 |
| P3 V4 | 完成 | 45,380 个样本，其中 train 35,940、validation 9,440；680,700 条候选；七类 maneuver 覆盖；正式重审 PASS。 |
| P4 | 完成 | `DEPCarNetV2`、路线走廊编码、双向物理 rollout、路线管道/内侧墙净空/后续可行性损失及 FP32 物理精度岛。 |
| P5 | 完成 | Fusion Candidate Capacity 与 Score Calibration 两阶段训练完成；Candidate 正式验收 PASS，Score 获得 P6 shadow-only PASS。 |
| P6 | 进行中 | RViz 任意目标与固定 seed 场景两种入口已可用；正在对掉头、直角转弯、死胡同恢复和前后受限场景做闭环加固。 |
| P7/P8 | 未开始 | 动态场景正式测试、独立 holdout 验收和 production qualification 尚未签发。 |

P3 V4 重审结果：

- 样本数：45,380
- validation 可行候选中位数：15/15
- validation zero-feasible rate：3.347%
- `SHARP_TURN`：4,108 帧
- `THREE_POINT_TURN`：2,965 帧
- validation 前进/倒车请求：6,851 / 2,589
- test split 保持封存，未参与训练或调参

P5 V2 验收结果：

- Candidate Capacity capable rate：85.62%
- Candidate zero-hard-feasible rate：1.01%
- Score checkpoint SHA-256：`587f7bbd227ab8fd167d2c3996b3e87a92db8cca51a7361dde9c997373a28212`
- Score 资格范围：`P6_SHADOW_ONLY`
- 已知风险：`REVERSE_EXIT` 仍需在 P6 场景中继续验证

## 运行时结构

```text
Arena map + map_server
          |
          v
Topological/Hybrid A*  ---->  route corridor + connectivity + gear hint
                                      |
Depth + 360 LiDAR + vehicle state ---> DEPCarNetV2 Fusion
                                      |
                                      v
                              15 local candidates
                                      |
                     continuous footprint/dynamic hard veto
                                      |
                     gear supervisor + Urban Car adapter
                                      |
                                      v
                                    Gazebo
```

职责边界：

- 全局 A* 不直接控制车辆，只防止局部规划器选择错误绕行方向或进入死胡同。
- V2 局部模型使用一段路线走廊，而不是依赖单个强制 waypoint 完成转角。
- RViz `2D Nav Goal` 默认只约束终点位置，不强迫车辆在终点反复微调箭头朝向。
- 真实墙面和动态障碍始终由 hard-veto 否决；模型不能恢复已被安全层拒绝的候选。
- 车辆优先寻找掉头空间并恢复正向行驶；只在窄路无法掉头时有限倒车逃逸，避免长距离全程倒车。

## 环境

已验证环境：

- Ubuntu 20.04
- ROS Noetic / Gazebo 11
- Python 3.8
- `yopo` Conda 环境用于 PyTorch/CUDA 模型训练与推理
- NVIDIA GPU（当前开发机为 RTX 5070 Ti Laptop）

ROS 系统依赖应在宿主机终端安装，不需要进入 Conda 环境：

```bash
sudo apt-get install \
  ros-noetic-effort-controllers \
  ros-noetic-joint-trajectory-controller \
  ros-noetic-gazebo-plugins \
  ros-noetic-velodyne-description \
  ros-noetic-velodyne-gazebo-plugins \
  ros-noetic-map-server
```

第三方仓库版本记录在 [`third_party.lock.yaml`](third_party.lock.yaml)。大体积第三方源码、数据集、模型权重和运行日志不会提交到 Git；新机器需要按锁文件准备依赖，并将对应 checkpoint 放到合同指定路径。

构建工作空间：

```bash
cd /home/zjh/DE-P-Car
conda deactivate 2>/dev/null || true
source /opt/ros/noetic/setup.bash
bash scripts/bootstrap_workspace.sh
source catkin_ws/devel/setup.bash
```

## P6 静态 Gazebo 验证

### 入口一：RViz 任意目标

```bash
cd /home/zjh/DE-P-Car
source /opt/ros/noetic/setup.bash
source catkin_ws/devel/setup.bash

bash scripts/run_p6_static.sh \
  --stage interactive \
  --config dep_car/config/p6_static_route_v2.yaml \
  --root data/p6_static \
  --cohort development \
  --maximum-scenarios 1 \
  --modality fusion \
  --learned-route-authority
```

在 RViz 中使用 `2D Nav Goal` 发布新目标。默认采用 position-only 到达判定；只有显式加入 `--require-goal-heading` 时才约束箭头朝向。

`interactive` 当前固定运行在 shadow 权限：网络生成、过滤并排序候选，但确定性分支仍掌握底盘控制。它适合观察路线、候选、挡位和 hard-veto 行为，不能作为 active 资格证明。

### 入口二：固定场景复现

列出冻结场景：

```bash
bash scripts/run_p6_static.sh \
  --stage list \
  --config dep_car/config/p6_static_route_v2.yaml \
  --root data/p6_static \
  --cohort development
```

运行一个固定 seed、地图、起点和终点的 shadow episode：

```bash
bash scripts/run_p6_static.sh \
  --stage shadow \
  --config dep_car/config/p6_static_route_v2.yaml \
  --root data/p6_static \
  --scenario p6_d8de47006c6886d3 \
  --modality fusion \
  --learned-route-authority \
  --rerun
```

该 `DEAD_END_ESCAPE` 回归场景在当前 runtime 下已达到：0 碰撞、0 非法换挡、0 zero-feasible 消息，并成功倒车到达目标。

最近的 P6 修复包括：

- 位置到达提前减速、主动制动和终点保持，避免到点后前后振荡。
- 目标在车后时的三点掉头/有限倒车状态机，防止无意义的全程倒车。
- 使用路线走廊而非 A* 折线强制牵引局部轨迹。
- 90° 转角内侧墙 soft-clearance 偏好：只影响排序，不修改 hard feasibility。
- LiDAR 局部栅格的自车未知区与五圆 hard-veto footprint 对齐，修复前后有障碍时从第 0 帧误判静态碰撞的问题；真实占据点不会被清除。

每次运行时代码发生改变后，旧 P6 active authority 会因 runtime hash 不一致自动失效。应重新完成 shadow gate-suite 和审计，不能绕过合同直接启用 active。

## P3 V4 数据流程

数据集不纳入 Git。以下长任务应由宿主机用户启动，并使用 8 个 worker：

```bash
# 生成 P3 V4 任务清单
bash scripts/run_p3_v4_corner_curriculum.sh \
  --stage prepare --workers 8

# 正式并行采集
bash scripts/run_p3_v4_corner_curriculum.sh \
  --stage collect --workers 8

# 查看任务状态
bash scripts/run_p3_v4_corner_curriculum.sh \
  --stage status --workers 8
```

corner01 中两个经认证的非法目标已由 corner02 补充任务替代。合并后的正式 bundle 位于本地 `data/p3_v4/bundle_v1/`，其 authority 不提交到 Git，因为它绑定本机数据文件内容。

正式审计报告保存在 [`reports/p3_v4_corner_reaudit.json`](reports/p3_v4_corner_reaudit.json)。

## P5 V2 Fusion 训练

正式训练只授权 Fusion。Depth-only 和 LiDAR-only 保留为有界诊断 pilot，不再启动三套长期正式训练。

```bash
# Candidate 入口与 authority dry-run
bash scripts/run_p5_route_v2.sh --stage dry-run

# 第一阶段：Candidate Capacity
bash scripts/run_p5_route_v2.sh --stage candidate_capacity

# 全 validation 候选能力验收
bash scripts/run_p5_route_v2.sh --stage candidate_acceptance

# Score Head 入口检查
bash scripts/run_p5_route_v2.sh --stage score-dry-run

# 第二阶段：Score Calibration
bash scripts/run_p5_route_v2.sh --stage score_calibration
```

训练支持 `--resume`。正式长任务应在宿主机运行；代码级 CUDA/AMP 数值验证和短 pilot 可使用：

```bash
bash scripts/run_p5_route_v2.sh \
  --stage pilot \
  --modality fusion \
  --maximum-samples 512 \
  --maximum-steps 16

bash scripts/run_p5_route_v2.sh \
  --stage score-pilot \
  --maximum-samples 512 \
  --maximum-steps 16
```

模型采用 FP32 物理精度岛：CNN/MLP 可以使用 AMP，但 rollout、运动学约束、swept-footprint loss 和 hard-veto 保持 FP32。DataLoader 使用 pinned memory、persistent workers 和预取；各模态只解析自身需要的传感器数据。

## 测试

完整工程检查：

```bash
bash scripts/verify_project.sh
```

P6 V2 相关测试：

```bash
PYTHONPATH=$PWD/dep_car/src \
  /home/zjh/miniconda3/envs/yopo/bin/python -m pytest -q \
  tests/test_p6_runtime.py \
  tests/test_p6_route_entry.py \
  tests/test_p6_route_losses_v2.py \
  tests/test_p6_route_model_v2.py
```

当前 P6 测试结果为 43 passed。

## 仓库内容与资格边界

Git 仓库包含源码、ROS 包、配置、入口脚本、测试和小型审计报告。以下内容由 `.gitignore` 排除：

- `data/p3_*`、`data/p6_static` 等生成数据
- `models/dep_car/p5_route_v2` 等训练权重
- rosbag、Gazebo/ROS 日志、训练日志与缓存
- `catkin_ws` 构建产物
- 第三方仓库工作树

审计报告记录的是开发与 shadow 证据，不代表真实车辆安全认证。当前系统仅允许在仿真中继续验证；在 P6 holdout、P7 动态场景和 P8 独立资格签发完成前，不应部署到真实车辆或宣称 production-qualified。

上游身份与第三方版本分别见 [`dep_source.lock.yaml`](dep_source.lock.yaml) 和 [`third_party.lock.yaml`](third_party.lock.yaml)。
