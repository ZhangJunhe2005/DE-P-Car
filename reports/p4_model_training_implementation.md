# P4 正式网络与损失实现说明

状态：`P4_IMPLEMENTATION_PASS / P5_FORMAL_BLOCKED_BY_P3_AND_VALIDATION_COVERAGE`。

P4 已完成网络、双向可微物理、连续车体扫掠、训练数据视图、损失、两阶段训练入口与安全工件合同。没有启动 P5 正式训练，也没有授予 Gazebo 或 production qualification。机器验收见 [p4_model_implementation_acceptance.json](p4_model_implementation_acceptance.json)，阶段汇总见 [p4_acceptance.json](p4_acceptance.json)。

## 网络合同

正式架构为 `dep_car_multimodal_v1_ackermann_3x5`：

- 深度输入 `[B,2,96,160]`：metric depth/10 m 与 validity；复用已认证 DE-P MobileNetV3 depth backbone。
- LiDAR 输入 `[B,6,160,160]`：P3 冻结的 360° BEV 六通道与独立 CNN。
- 状态输入为 9D 物理量 `[v,a,delta,yaw_rate,gx,gy,sin(epsi),cos(epsi),kappa_ref]`。
- `requested_gear` 只能由确定性 `GearSupervisor` 给出 `-1/+1`；网络不预测挡位。
- 3 个速度锚点 × 5 个转角锚点形成 15 个 logical queries，顺序固定为 speed-major/steering-minor。
- 每个候选预测 `steering_mid/steering_end/speed_end/duration` 四个 residual；独立 score head 输出 lower-is-better cost。
- depth-only、lidar-only、fusion 均有正式路径；缺失模态不会污染被冻结 encoder 的 BatchNorm/optimizer 状态。
- 左右镜像通过 group averaging 强制等变。

网络实际导入的三个第三方 depth-backbone 源文件也纳入 P4 implementation aggregate，防止仅权重形状相同、运行时代码却已变化。

## 双向 Ackermann 运动学

`AckermannRolloutV1` 的合同已升级为 `AckermannRolloutV2GearAlignedLongitudinalLimits`：

- 输出 `[B,15,11,6]` 的 `[t,x,y,yaw,v,delta]`，canonical horizon 为 1.0 s。
- effective wheelbase `0.3863798396 m`；前进上限 `2.5 m/s`，倒车上限 `0.5 m/s`。
- 转角、转角速率、驱动加速度、制动减速度分别限于 `0.61240774 rad`、`0.75 rad/s`、`1.5 m/s²`、`2.0 m/s²`。
- 加减速限制在 requested-gear 对齐坐标中应用，因此正反挡都有正确的“加速/制动”物理意义。
- 未停车请求反挡会被拒绝；换挡帧先清零速度和加速度，不允许网络跨挡积分。
- PyTorch rollout、NumPy lattice 与训练数据状态投影使用同一合同。

## 连续 swept footprint V3

安全合同为 `FiveCircleContinuousSweptFootprintV3`：

- 缩放车体为 `0.49667 m × 0.27667 m`，五圆纵向中心为 `[-0.19867,-0.09933,0,0.09933,0.19867] m`。
- 每圆半径 `0.266979 m`，包含 `0.12 m` safety margin。
- 11 个 rollout 源时刻之间各插值 16 个 SE(2) 子步，共查询 161 个时刻，防止车辆在相邻离散姿态间穿过薄障碍。
- runtime hard veto 扣除半个栅格对角线；训练的双线性 signed-SDF 查询扣除一个完整栅格对角线。
- P3 旧 NPZ 中的 legacy feasible/clearance 只作诊断。P4/P5 使用不可变地图、每帧 `chassis_to_map` 和当前连续 footprint 在线重算。

## 训练数据权威

冻结开发索引包含 9,290 个样本：8,268 train、1,022 validation，来自 24 张 train 地图与 3 张 validation 地图。正式 loader 绑定：

- training index SHA-256：`962242859a862c87123e201ed506afa3a0018c4c551ffcd68e469ab4a890494d`
- NPZ 内容 aggregate：`e7bbe901877bae81e04117a99c0c935087c3098372cf60b48358857a466ff1c2`
- 地图合同 aggregate：`9d0251f764c1983a5ff73db67af481ab891b913b7984cd4de4cd967e774d1fa2`

loader 对实际读取的 PNG/YAML 字节与语义重新验权，拒绝非零 map origin yaw。P4 tiny-overfit 只选择 train 样本；validation NPZ 原始字节会为索引权威做哈希，validation 地图会做权威校验，但不会解析 validation 样本语义或参与 tiny-overfit。test NPZ、test 地图和 test 语义全部保持封存。

## 可微目标 revision 3

目标 ID 为 `dep_car_objective_v3_signed_sdf_cvar_continuous_swept_route_capacity_score`：

- `L_safe`：known-free 为正、occupied/unknown 为负的 signed SDF；按 mean + worst-10% CVaR + worst barrier 聚合 161×5 个连续 footprint 查询。
- `L_guide`：同挡路线的横向误差、车辆航向、有向 progress 与 endpoint。
- `L_kin`：挡位符号、速度、转角、转角速率、gear-aligned 加减速度和横向加速度。
- `L_comfort`：纵向 jerk 与 steering acceleration。
- `L_diversity`：按正/倒挡可达包络归一化后约束候选覆盖。
- `L_anchor`：限制 residual 偏离 canonical lattice。
- `L_score`：仅在硬可行候选中学习 lower-is-better 排序。

`candidate_capacity` 与 `score_calibration` 的 train/freeze 模块名单互斥且由 `training.yaml` 冻结。

## 初始化与安全加载

P4 初始化只 exact 迁移 V4.8.3 depth backbone 的 246 个 tensor；UAV state/head/PVA、旧 8D LiDAR range-image prototype 和 partial transfer 全部禁止。

- 源 checkpoint SHA-256：`22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60`
- P4 初始化 SHA-256：`3c570d5ed3715ab019dacfc29b0ae6e0e405d0dc7d27cd861b625877f7f43c6c`
- 初始化 contract SHA-256：`c20ae24ea856888c273a58772732d013289816ddeab9dbbe8077cdcffc95ea96`
- transfer manifest SHA-256：`5703ef1255d4b7a4b43b7f2f46a72f2f74fc4096d35e637e5b76f7879227ab2b`
- P4 implementation aggregate：`c040cee4f7bd0ba43b19731e3721d3fe8d3344407fbacff9c299b5c985807077`

迁移工具先固定并校验同一份源字节快照，再进行受控的 legacy pickle 加载；P4/P5 工件统一使用 `weights_only=True` 从已经哈希的同一字节快照加载。重复迁移得到完全相同的初始化 checkpoint。

## P4 验收结果

最终 CPU/8-thread 验收状态为 PASS，errors 为空：

- 16 个真实 train 样本覆盖 16 张地图、七类 maneuver 和前/倒挡。
- depth-only、lidar-only、fusion 均输出有限的 `[16,15,11,6]` 轨迹。
- depth、LiDAR、state、speed/steering embedding 与 candidate head 均收到有限非零梯度；candidate 阶段 score 无梯度泄漏。
- Candidate tiny-overfit：总损失 `1.31167 → 1.01960`；独立 direct-residual oracle 达 `1.00319`，收敛门槛通过。
- Score tiny-overfit：`2.72388 → 2.17344`；candidate 分支保持冻结，oracle regret 与 top-1 指标改善。
- checkpoint save/reload 最大输出误差 `0.0`。
- 项目测试 `240 passed`，上游 DE-P 测试 `85 passed`，catkin `-j8` 构建通过。

## P5 门禁与下一步

正式 P5 dry-run 目前必须输出 `BLOCKED`，原因只有真实数据/覆盖问题：

- P3 V3 在 9,290 个 train+validation 样本、139,350 条候选上的总体 zero-feasible rate 为 `19.22497%`，未过 `<10%`。
- `NARROW_CORRIDOR=55.93407%`、`SHARP_TURN=32.4375%`、`THREE_POINT_TURN=36.33333%`，均未过逐模式 `<25%`。
- validation candidate context 为 `MISSION=210 / RECOVERY=6 / UNKNOWN=806`；RECOVERY 未达 20，且 UNKNOWN 不允许混入正式资格。

入口允许同时满足 `max_steps <= 10` 与 `max_samples <= 32` 的 bounded smoke；任何超参覆盖都会永久写入 smoke lineage。候选和评分阶段各 1 step 的 fusion smoke 已跑通，输出均为 `TRAINED_UNQUALIFIED`；外部 candidate acceptance 对该 smoke 按预期返回 FAIL。正式训练不得通过修改 YAML、单独设置一个 cap、扩大 smoke 上限或使用现有输出覆盖来绕过门禁。

下一步应先执行 P3 V3 增量补强/重新索引，使四项几何门槛和两项 validation context 门槛全部通过，再单独批准 P5 的 Depth-only、LiDAR-only、Fusion 正式训练。
