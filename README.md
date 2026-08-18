# DE-P-Car

DE-P-Car 是把最新 DE-P Route-A V4.9.1 的候选规划、确定性安全、动态让行与恢复生命周期迁移到 Urban Car Ackermann 底盘的 ROS Noetic 工程。当前源码已经形成可构建闭环；冻结 UAV 权重被迁移为小车模型的初始化，但不会被误报为已经训练好的车辆策略。

## 已完成

- 固定并验证 `ZhangJunhe2005/DE-P` 当前分支头 `cbcc61d…`；85 个上游测试通过。
- 克隆并锁定 car-simulator、Arena-Rosnav-3D、arena-tools。
- 正式 P4 `DEPCarNetV1`：metric depth+validity 与 6 通道 360° LiDAR BEV 双编码器、9D state、gear-conditioned 3x5 logical queries、独立 candidate/score tower；只 exact 迁移 V4.8.3 的 246 个 depth-backbone tensor，禁止迁移 UAV head/PVA。
- 3 speed × 5 steering、gear-aligned 双向可微 bicycle rollout、161 时刻 FiveCircleContinuousSweptFootprintV3、revision 4 FP32 physical loss/hard-veto、全 15 候选运动学裕度、动态 reachability hard veto、1.0/1.2/1.4 ordered retiming。
- 数据标注保留带档位 Hybrid A*；P6 在线全局层改为快速拓扑走廊，只负责连通性和避免死胡同。正反向 signed lattice、连续车身 hard veto、局部双挡仲裁、到点制动和 stop-before-shift 掌握实际运动权限，网络只能在确定性安全边界内按挡位工作。
- Urban Car ros_control adapter，包含 Ackermann 内外轮转角、速度 PI 与 0.35 s freshness watchdog。
- Urban Car 采用统一 `1/3` 线性缩放：车身、车轮、轴距、碰撞体、质量/惯量和规划 footprint 保持一致；VLP-16 移到车顶中央，避免后向视场被车身遮挡，测量量程不缩放。
- `StaticAckermannSampleV2` revision 2 多模态合同：metric depth、rosbag 引用式原始点云、6 通道 360° BEV、IMU、9D 车辆/路线状态、gear-conditioned route/candidate、状态插值、measurement-stamped TF 和完整 provenance。
- P3 Pilot 与 V3 增量补强已经完成；23,236 条 source 全部保留，3,218 条初始足迹不可行帧经认证 curation 隔离。冻结的 `bundle_v2_curated` 共 20,018 条开发样本：16,394 train、3,624 validation，按 map UUID 隔离为 31/5 张地图，test 不参与 P4/P5 调参。七类 maneuver、MISSION/RECOVERY、正反挡和三点掉头数据均进入后续门禁。
- P4 训练视图已就绪：8-worker DataLoader、地图字节/语义权威、同挡 route 修复、三模态路径、严格 Candidate Capacity→Score Calibration 参数分区和 UNQUALIFIED checkpoint 合同。P5 v2 已完成 512 样本三模态 CUDA/AMP 短验证，P3 全量重审、P4 CUDA 验收和三模态正式 dry-run 已重新签发 PASS。
- P4 safety authority 已升级为 signed SDF（known-free 正、occupied/unknown 负）与 mean+CVaR+worst barrier；训练 index 使用逐 NPZ 内容 SHA256，P5 CLI 对 footprint、data、config、checkpoint lineage 全部 fail-closed。
- Gazebo odometry 发布 `map -> dummy` 动态 TF；VLP-16 点云和 640×480 深度图已接入预配置 RViz。
- arena-tools 固定 seed/UUID wrapper，自动生成 ROS map + SDF/Gazebo world；不需要手工 Blender 转换。
- Arena-compatible crossing/head-on/multi-agent 动态场景与 evaluation-only GT 边界。
- 16 个 catkin 包以 `-j8` 构建成功；P4 机器验收、候选/评分两阶段的 fail-closed、checkpoint 和数据权威工具均已实现并通过本轮正式复签。

P0～P4 已按 P5 v2 合同重新验收；首轮 P5 三模态 Candidate 训练仅作隔离诊断，随后三模态 Candidate Capacity 与 Score Calibration 已完成。三个 Score checkpoint 保持 `TRAINED_UNQUALIFIED`，当前仅允许 P6 shadow 验证，尚未获得 active/production 资格。

## 一次性准备

当前机器已验证这些依赖可用；在新的 Ubuntu 20.04/Noetic 机器上缺少依赖时，用有 sudo 权限的终端执行：

```bash
sudo apt-get install ros-noetic-effort-controllers \
  ros-noetic-joint-trajectory-controller \
  ros-noetic-gazebo-plugins \
  ros-noetic-velodyne-description \
  ros-noetic-velodyne-gazebo-plugins \
  ros-noetic-map-server
```

然后构建：

```bash
cd /home/zjh/DE-P-Car
conda deactivate 2>/dev/null || true
source /opt/ros/noetic/setup.bash
bash scripts/bootstrap_workspace.sh
source catkin_ws/devel/setup.bash
```

## 启动 Urban Car 静态场景

```bash
cd /home/zjh/DE-P-Car
conda deactivate 2>/dev/null || true
source /opt/ros/noetic/setup.bash
source catkin_ws/devel/setup.bash

MAP_DIR=$PWD/data/arena_maps/dep_car_map_0000_e149ae3d
roslaunch dep_car_bringup urban_sim.launch \
  world:=$MAP_DIR/map.world \
  map_yaml:=$MAP_DIR/map.yaml \
  gazebo_model_path:=$PWD/data/arena_maps \
  gui:=true enable_rviz:=true
```

RViz 会自动使用 `map` 作为 Fixed Frame，并显示 `/map`、车辆模型、`/velodyne_points` 和 `/camera/depth/image_raw`。用 `2D Nav Goal` 发布 `/move_base_simple/goal`。在线拓扑 A* 发布不含硬挡位命令的连通走廊与局部子目标；局部规划器决定前进/倒车候选并只发布 `/dep_car/cmd_ackermann`，实际底盘输出由 `urban_car_adapter.py` 统一完成。按一次 `Ctrl+C` 即可清理 Gazebo、Gazebo GUI 和 RViz。

## 动态测试

动态 world 已生成在 `data/dynamic_eval/worlds/`。Arena actor mesh 位于锁定的 Arena 仓库，因此启动时把两个 model root 都传入：

```bash
ACTOR_MODELS=$PWD/third_party/arena-rosnav-3D/simulator_setup/worlds/small_warehouse/models
export GAZEBO_PLUGIN_PATH=$PWD/third_party/ActorCollisionsPlugin/build:${GAZEBO_PLUGIN_PATH:-}
roslaunch dep_car_bringup urban_sim.launch \
  world:=$PWD/data/dynamic_eval/worlds/crossing.world \
  map_yaml:=$PWD/data/arena_maps/dep_car_map_0000_e149ae3d/map.yaml \
  gazebo_model_path:=$PWD/data/arena_maps:$ACTOR_MODELS
```

`/gazebo/model_states` 与 Pedsim GT 只允许 `dep_car_evaluation` 读取，perception/planner 只使用 VLP-16、静态地图和车辆状态。

## 数据与模型

生成地图：

```bash
QT_QPA_PLATFORM=offscreen /usr/bin/python3 \
  ros/dep_car_dataset/scripts/generate_static_maps.py \
  --type indoor --count 20 --seed 50000
```

离线 synthetic 2-D pilot（仅验证数据合同，不可用于 production qualification）：

```bash
PYTHONPATH=$PWD/dep_car/src /usr/bin/python3 \
  ros/dep_car_dataset/scripts/generate_static_dataset.py \
  --samples-per-map 4
```

正式数据使用 V2 同步采集器。保持 Gazebo 与规划栈运行，在另一个终端启动：

```bash
source /opt/ros/noetic/setup.bash
source catkin_ws/devel/setup.bash
roslaunch dep_car_dataset multimodal_collector.launch \
  output:=$PWD/data/static_multimodal_v2 \
  map_uuid:=e149ae3d-f90c-5563-89d7-a8b5cda05eec \
  map_hash:=11496511e0ee104b133581bcee86d99d2b52c9372bfe83b5dd28db79ead10dbd \
  simulator_seed:=49100
```

车辆正在执行有效路线时保存一个严格同步样本：

```bash
rosservice call /dep_car_multimodal_collector/capture
```

推荐先录原始 episode，再确定性离线抽取，便于修改预处理后复现：

```bash
rosrun dep_car_dataset record_multimodal_episode.sh /tmp/dep_car_episode.bag

rosrun dep_car_dataset extract_multimodal_bag.py /tmp/dep_car_episode.bag \
  --output data/static_multimodal_v2 \
  --map-uuid e149ae3d-f90c-5563-89d7-a8b5cda05eec \
  --map-hash 11496511e0ee104b133581bcee86d99d2b52c9372bfe83b5dd28db79ead10dbd \
  --simulator-seed 49100

/usr/bin/python3 tools/audit_multimodal_dataset.py data/static_multimodal_v2
```

正式数据以 rosbag 为原始权威源，NPZ 通过 bag SHA256、topic、消息索引和时间戳回指原始点云，不再为每个样本重复保存整帧点云；如需现场排错可给离线抽取命令追加 `--embed-raw-lidar`。BEV、Range Image、局部安全栅格统一使用 measurement-stamped `velodyne→chassis` TF 和车体自滤波；odom、IMU、joint state 则插值到 LiDAR 参考时间。训练/验证/测试由 map UUID 分组，禁止同一地图跨 split。旧 V1 或 V2 revision 1 样本不得用于最终车型训练。

正式 P4 初始化在 `models/dep_car/dep_car_net_v1_depth_v483_init.pth`。其 contract 的 `production_qualified=false`，默认 ROS loader 会拒绝运行。旧 `dep_car_v1_v491_transfer_init.pth` 是 8D LiDAR range-image prototype，仅保留历史兼容，不得用于 P5。完成车辆数据训练和独立 qualification 后才能签发 production contract。

P4 实现验收在 `yopo` Conda 环境执行；它不会启动正式训练：

```bash
PYTHONPATH=$PWD/dep_car/src conda run -n yopo python tools/verify_p4.py --threads 8
```

P5 三模态入口 dry-run 会核对 candidate stage、数据 split、初始化 lineage、P3/P4 门禁和 CUDA，但不会训练：

```bash
bash scripts/run_p5_training.sh --stage dry-run --modality all --workers 8
```

P5 已获单独批准，长时间计算仍由宿主机用户启动。严格按 Candidate Capacity→候选验收→Score Calibration 执行 Depth-only、LiDAR-only、Fusion 三组实验；所有训练输出都固定为 `UNQUALIFIED`。Score Calibration 禁止直接从 transfer initialization 或一步 smoke 启动，只接受通过 `tools/accept_p5_candidate.py` 门槛的 candidate checkpoint。完整命令、断点恢复方式和回传材料见 `reports/p5_training_launch_guide.md`。Fusion 缺失 depth/LiDAR 的独立鲁棒性门槛和三组实验矩阵汇总属于 P5 验收，不在 P4 中冒充已完成。

P3 V3 增量补强与 curation 已完成。`bundle_v2_curated` 使用 8 worker 从已认证 source 构建：逐帧复核 production signed-SDF 的初始车身足迹，显式隔离不合法起始状态并签发 curation authority；没有删除或改写任何原始 NPZ。宿主机命令与历史证据见 `reports/p3_v3_data_reinforcement_implementation.md`。

P6 `DEPCarNetV1` depth+BEV+9D+gear adapter 与 shadow 控制边界已经接入。场景冻结、起点扰动审计、交互式 RViz 目标、固定场景复现和 shadow 报告入口见 `reports/p6_static_validation_launch_guide.md`；active 仍必须等待完整 shadow gate-suite 审计签发。

## 验证与审计

```bash
bash scripts/verify_project.sh
```

接口事实、迁移边界与已知限制见 [`reports/`](reports/)；上游身份见 [`dep_source.lock.yaml`](dep_source.lock.yaml) 与 [`third_party.lock.yaml`](third_party.lock.yaml)。
