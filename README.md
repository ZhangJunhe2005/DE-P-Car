# DE-P-Car

DE-P-Car 将 [DE-P](https://github.com/ZhangJunhe2005/DE-P) 的局部规划思想迁移到 ROS Noetic 下的地面阿克曼小车。工程使用缩放后的 `hifzhil/car-simulator` Urban Car、Arena 随机地图、深度相机和 360° VLP-16，并围绕前进、倒车、三点掉头、窄通道及 90° 转角重新建立了数据、模型和运行时安全合同。

当前正式模型是 **DEPCarNetV2 Fusion**。它接收深度、LiDAR BEV、9D 车辆状态、挡位和车体坐标系下的路线走廊，生成 3×5 条满足双向 Ackermann rollout 的候选轨迹。P6 保留冻结地图 Hybrid A* 后端用于模型回归；M6 则以在线 SLAM 的动态多边形可见图负责迷宫连通方向，不启动 `map_server` 或栅格 A*。两种后端都只提供路线走廊，局部模型负责轨迹生成与排序；连续车身 hard-veto、动态 reachability veto 和 stop-before-shift 始终拥有最终安全权限。

> 当前阶段：P0～P5 已完成，P6 静态 Gazebo shadow 验证和场景鲁棒性加固正在进行；M0～M6 已实现在线建图、动态可见图寻路、稀疏历史记忆和死路恢复后端。M6 静态实现审计已 PASS，跨地图 Gazebo 到达率尚未验收。现有 checkpoint 为 `TRAINED_UNQUALIFIED`，尚未获得 active 或 production 资格。

## 当前进展

| 阶段 | 状态 | 主要成果 |
| --- | --- | --- |
| P0 | 完成 | Urban Car 统一缩放至上游模型的 1/3；车辆尺寸、轴距、轮胎、碰撞体、质量/惯量及规划 footprint 保持一致。 |
| P1 | 完成 | signed bicycle rollout、双向 Hybrid A* 数据标注、换挡惩罚、停车换挡状态机、连续五圆车身碰撞检查。 |
| P2 | 完成 | `StaticAckermannSampleV2` 多模态合同：metric depth、原始点云引用、6 通道 360° LiDAR BEV、IMU、9D state、gear/route/candidate 和完整 provenance。 |
| P3 V4 | 完成 | 45,380 个样本，其中 train 35,940、validation 9,440；680,700 条候选；七类 maneuver 覆盖；正式重审 PASS。 |
| P4 | 完成 | `DEPCarNetV2`、路线走廊编码、双向物理 rollout、路线管道/内侧墙净空/后续可行性损失及 FP32 物理精度岛。 |
| P5 | 完成 | Fusion Candidate Capacity 与 Score Calibration 两阶段训练完成；Candidate 正式验收 PASS，Score 获得 P6 shadow-only PASS。 |
| P6 | 进行中 | RViz 任意目标与固定 seed 场景两种入口已可用；冻结地图 A* 回归后端和在线 SLAM/FAR-style 可见图/记忆后端可切换。M6 静态实现完成，跨地图 Gazebo 资格仍待验收。 |
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

工程保留两个互不混用的导航后端。原 P6 后端用于继续复现实验与模型验收：

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

M0～M6 在线导航后端不启动 `map_server` 或 Hybrid A*：

```text
rear wheel joints + IMU -> EKF odom -> online Slam Toolbox -> /map
                                   |
360 LiDAR -> local hard-safety grid + planar scan
                                   |
RViz goal -> observed-wall polygons -> dynamic sparse visibility graph
                                   |
                    known route first / attemptable unknown route
                                   |
                     DEPCarNetV2/local candidate planner
                                   |
dead end -> directed failed-branch memory + bounded exit transaction
                                   |
      current SLAM/LiDAR validates rolling reverse connector; FAR replans
                                   |
              reach branch entry -> select another branch -> resume
```

它不订阅 `/base_pose_ground_truth`，也不在在线地图上做 OccupancyGrid 单元级 A*。正常导航从 `/map` 提取已观察墙面的膨胀轮廓和稀疏可见边：优先使用完全位于已知自由区的路线，没有已知路线时才选择带未知代价的可尝试路线。可尝试路线的起始方向在两个 SLAM 修订中稳定后，即可像上游 FAR 一样进入 `FAR_ATTEMPTABLE_NAVIGATION`，边行驶、边获得墙角后的新视点、边滚动更新可见图；它不需要先在原地等待整条路线变成已知空间。新路线暂时不稳定时保留仍未被新障碍切断的旧路线，避免重规划造成启停和方向闪回。

M6.15 将失败支路保存为“入口 → 末端”的有向禁入语义，而不是画进占据栅格的实体虚拟墙：再次向末端驶入会被 FAR 拒绝，但车辆已经位于支路内部时仍可沿反方向搜索出口。重复静态失败会启动唯一的 `FAR_DEAD_END_EGRESS` 事务；breadcrumb 只提供曾经真实驶过的支路入口锚点，不重放历史油门或转角。局部规划器依据当前 SLAM、LiDAR 和车辆状态，以短前视滚动目标闭环执行前进/倒车，FAR 同时在后台寻找入口外的新方向；到达真实岔路、可掉头点或已重新取得稳定 FAR 路线后才释放脱困权限。SLAM 修正会重锚并重新验证尚未执行的连接段，hard-veto 始终具有最高优先级。具体阿克曼轨迹仍由 DEPCarNetV2/确定性局部候选产生，可见图只把前方约 2.5 m 路线管道交给局部规划器。

M6 的路线层参考 [FAR Planner](https://github.com/MichaelFYang/far_planner) 的动态多边形可见图架构，ROS 适配器在本工程内独立实现，以便直接消费 `nav_msgs/OccupancyGrid` 并保持 Urban Car/DE-P 安全合同。上游参考版本固定在 `third_party.lock.yaml`；网络可用时可执行 `bash scripts/fetch_far_planner_upstream.sh` 获取对应 commit，用于来源核对，不会替换本工程运行节点。

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
  ros-noetic-map-server \
  ros-noetic-robot-localization \
  ros-noetic-slam-toolbox
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

## M0～M6 在线 SLAM、可见图和记忆导航验证

先检查双后端隔离、无真值依赖和运行依赖：

```bash
bash scripts/run_far_navigation.sh --stage audit
```

入口一仍使用 RViz `2D Nav Goal`。小车会根据不断增长的在线地图重建可见图，把蓝绿色 `FAR Visibility Route` 的局部路线管道交给 V2；不会调用栅格 A*：

```bash
bash scripts/run_far_navigation.sh \
  --stage interactive
```

远目标刚发布时允许出现一个有上限的 `FAR_MAPPING_WAIT`，用于排除单帧方向假象；字节内容相同的后续 SLAM 地图可确认候选的时间稳定性，而不触发重复可见图重建。若确认观察没有到达，超时后也会进入 `LOCAL_SAFE_EXPLORATION`，不会无限等待。一旦日志出现 `motion_authorized=True`，即使整条路线仍穿过未知区，也应持续处于 `FAR_ATTEMPTABLE_NAVIGATION/GOAL_SEEK` 并向下一个可见图顶点移动。搜索途中若可见图暂时返回 `NO_ROUTE` 且没有可复用的已走拓扑，同样切换为 `LOCAL_SAFE_EXPLORATION`：局部规划器利用实时 LiDAR 的完整车身扫掠净空继续移动、转弯或执行必要的阿克曼掉头，同时等待在线地图和 FAR 路线恢复。`NO_ROUTE` 本身不再触发“前进 0.8 m—停车等待地图增长”的死锁；只有局部候选也被障碍和 hard veto 全部否决时才会停车或进入恢复。

RViz 中青色 `FAR Visibility Route` 是拓扑绕行结果，绿色 `Local Route
Corridor` 与红色 `Local Subgoal` 是交给 DE-P 的局部路线事务，橙色
`Controlling Candidate` 才是当前真正执行的阿克曼轨迹。FAR 路线有效时，旧的
Bug/TangentBug 边界跟随不能改写其绕行侧；进入前进/倒车组合掉头后，路线与
命令按同一时间戳原子交接并冻结到整个 manoeuvre 完成。局部候选无需贴着 FAR
折线，但应持续朝向同一条路线管道，且始终经过 hard veto。

入口二冻结 Gazebo world、seed、起点以及一个位于隔墙后的不可达探针目标，用来观察“静态停滞—记录失败支路—沿面包屑倒车—寻找最近可行拓扑点—选择其他出口—自动继续原任务”的 M5 闭环。由于探针本身不可达，这是一项持续压力测试，不以到达探针为成功条件：

```bash
bash scripts/run_far_navigation.sh \
  --stage fixed
```

精确复现已知 T 型路口双目标日志只使用这一条冻结输入；地图、坐标和场景标签不会进入运行时策略：

```bash
bash scripts/run_far_navigation.sh \
  --stage replay \
  --policy-mode shadow \
  --headless
```

鲁棒性测试不复用同一张地图。开发矩阵会在达到数量上限前优先选择不同的 `map_seed/map_uuid`，并把报告写入 `reports/memory_navigation/`。改变 `--selection-seed` 可获得另一组仍可复现的地图顺序：

```bash
# 先确认将要运行的 8 张不同随机地图
bash scripts/run_far_navigation.sh \
  --stage matrix --cohort development \
  --maximum-scenarios 8 --selection-seed 20260820 --dry-run

# 正式逐场景运行；每个 episode 独占 ROS/Gazebo master
bash scripts/run_far_navigation.sh \
  --stage matrix --cohort development \
  --maximum-scenarios 8 --selection-seed 20260820
```

封存的 holdout 地图只在最终验收时显式开启，避免调参污染：

```bash
bash scripts/run_far_navigation.sh \
  --stage matrix --cohort holdout --allow-holdout
```

只检查命令和冻结合同而不启动 Gazebo：

```bash
bash scripts/run_far_navigation.sh --stage interactive --dry-run
bash scripts/run_far_navigation.sh --stage fixed --dry-run
```

可通过 `--scenario <id>` 更换冻结世界，也可用 `--goal-x/--goal-y` 覆盖固定目标。默认 `policy-mode=shadow`；若只想隔离验证路线层与确定性控制链，可加 `--policy-mode disabled`。RViz 的 `FAR Visibility Route/Graph` 分别显示当前选中路线和稀疏可见边，`Navigation Memory` 显示 breadcrumb、已走拓扑边、有向失败支路、脱困锚点和青色滚动出口连接。自动 replay/matrix 报告还会记录 `far_dead_end_egress_transactions`、完成原因、最大回退目标距离、最大横向误差和 SLAM 重锚次数。`run_memory_navigation.sh` 保留为兼容别名。

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
