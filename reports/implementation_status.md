# 实施状态

截至 2026-08-16：`P0～P2 PASS`、`P3 原始采集完成但 V3 资格重审 FAIL`、`P4 IMPLEMENTATION PASS`、`P5 FORMAL BLOCKED`、`P6～P8 未执行`。

## 已通过

- `P0_PASS_V3`：缩放 Urban Car 的轴距、速度、转角、转角速率、制动与传感器时间/坐标语义已冻结。
- `P1_PASS_V2`：双向 signed lattice、Hybrid A*、deterministic gear supervisor、stop-before-shift、倒车退出与三点掉头基线通过。
- `P2_PASS_REVISION_2`：深度、VLP-16、IMU、车辆状态、measurement-stamped TF、rosbag 引用与 BEV preprocessing 权威链通过。
- `P3_COLLECTION_COMPLETE`：30 张隔离地图、150/150 episode 与原始 bag/hash 可追溯；P5 开发索引冻结为 8,268 train + 1,022 validation，test 未用于 P4/P5 调参。
- `P4_IMPLEMENTATION_PASS`：双编码器、9D state、外部挡位条件、15 logical queries、双向可微 Ackermann、连续 161 时刻五圆 swept footprint、revision 3 目标、两阶段参数分区和安全 checkpoint 合同全部实现。
- `P4_VERIFICATION_PASS`：P4 机器报告 errors 为空；当前 248 个项目测试、85 个上游测试和 catkin `-j8` 构建通过。候选与评分各 1-step bounded smoke 也已跑通且保持 `UNQUALIFIED`。
- `P3_V3_REINFORCEMENT_CODE_READY`：180-task targeted wave、连续几何预检、base bag 非破坏性重抽取、独立 bundle/index、动态 V3 审计和 P5 authority proposal 已实现。full-access 单 episode 验证产生 60 个已知 context 样本（34 reverse / 26 forward），断点复用通过；大规模运行由用户启动。

## 当前数据门禁

P3 V3 只审计冻结的 train+validation 开发权威，不打开 test。9,290 个样本、139,350 条以当前 Ackermann lattice 重新生成的候选完成连续 signed-SDF swept-footprint 复核，读取失败为 0，结果为：

- overall zero-feasible `19.22497%`，门槛 `<10%`；
- NARROW_CORRIDOR `55.93407%`，门槛 `<25%`；
- SHARP_TURN `32.4375%`，门槛 `<25%`；
- THREE_POINT_TURN `36.33333%`，门槛 `<25%`；
- 可行候选数总体中位数为 14，通过 `>=2`。

validation candidate context 为 `MISSION=210 / RECOVERY=6 / UNKNOWN=806`。正式 P5 还要求 RECOVERY 至少 20 且不允许 UNKNOWN。

因此：

- `P5_FORMAL_BLOCKED_BY_P3_AND_COVERAGE`：正式 dry-run 返回 BLOCKED，且只报告上述真实问题；不能通过改 YAML、改超参或伪造 sidecar 绕过。
- 仅允许 `max_steps<=10` 且 `max_samples<=32` 的 bounded smoke；工件永久记录 smoke lineage，并被 candidate acceptance 拒绝。
- 先补强 train/validation 的窄通道、急转、三点掉头候选表达能力，并用权威规则消除 UNKNOWN/补足 RECOVERY；全部门槛通过后才可批准 P5 正式训练。

## 未完成边界

- P5 尚未执行 Depth-only、LiDAR-only、Fusion 三组正式训练，也尚未签发三组实验矩阵汇总。
- Fusion 正式 checkpoint 的 depth-missing 与 lidar-missing 独立鲁棒性门槛将在 P5 验收中补齐；P4 只验证三条前向路径和训练时的单模态 dropout 实现。
- P6 learned-policy ROS adapter 尚未实现。当前 `local_planner_node.py` 仍是确定性 baseline/历史 `LidarDEPCarV1` 接口，不能加载 `DEPCarNetV1` 冒充 learned closed loop。
- P6 动静态 Gazebo、P7 动态场景和 P8 扩展数据/production qualification 均未执行。

权威报告：

- [P3 V3 开发集重审](p3_development_reaudit_v3.json)
- [P4 机器验收](p4_model_implementation_acceptance.json)
- [P4 阶段汇总](p4_acceptance.json)
