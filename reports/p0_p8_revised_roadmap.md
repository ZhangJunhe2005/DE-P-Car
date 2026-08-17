# DE-P-Car P0～P8 修订路线图

本路线图采用“先物理与数据资格、再训练、最后闭环签发”的顺序。P4 的实现通过不等于 P3 数据合格，也不等于 P5 模型性能或 P6 Gazebo 闭环通过。

| 阶段 | 当前状态 | 已完成/目标 | 进入下一阶段的门槛 |
|---|---|---|---|
| P0 物理与传感器冻结 | PASS | 1/3 Urban Car 尺寸/质量/惯量、轴距、速度、转角/转角速率、加减速、制动、相机/LiDAR TF 与时间语义形成版本合同 | 标定与 TF/频率/盲区审计通过 |
| P1 双向运动学基线 | PASS | signed bicycle lattice、Hybrid A* 挡位路线、stop-before-shift、换挡惩罚、连续碰撞检查、倒车退出和三点掉头 | 无网络也能安全完成前进、倒车与 F→R→F |
| P2 多模态数据合同 | PASS REVISION 2 | rosbag 原始权威、深度、VLP-16、BEV、IMU、9D state、挡位路线、逐时刻 TF、同步/哈希/重放 | 样本与 bag/map/preprocessing 权威链通过 |
| P3 Pilot 数据 | REQUALIFICATION REQUIRED | 30 张隔离地图、150/150 episode 已采集；P5 dev index 为 9,290 样本（8,268 train、1,022 validation） | V3 连续 footprint 重审当前 overall 19.22497%，且窄通道/急转/三点掉头失败；validation RECOVERY=6、UNKNOWN=806，需增量补强和重建索引 |
| P4 正式网络与损失 | PASS | Depth+validity/LiDAR BEV 双编码器、9D state、外部 gear、3×5 queries、双向可微 rollout、连续 swept footprint V3、revision 3 loss、两阶段冻结与安全工件 | 当前 248 项目测试、85 上游测试、catkin -j8、真实 train tiny-overfit、三模态、梯度、roundtrip 与机器验收通过 |
| P5 两阶段训练 | BLOCKED | 入口、8-worker 默认、正式超参冻结、数据/map/index TOCTOU 防护、bounded smoke、candidate acceptance 已准备 | P3 与 validation context 全部 PASS 后，依次做 Depth-only/LiDAR-only/Fusion Candidate Capacity，再做 Score Calibration；补齐 fusion 缺失传感器与三组实验矩阵验收 |
| P6 静态 Gazebo 闭环 | NOT STARTED | 先实现 `DEPCarNetV1` ROS adapter，再 shadow mode，最后允许网络控制 | 未见地图前进/倒车/窄通道/三点掉头无碰撞、无非法换挡，优于 deterministic baseline |
| P7 动态场景 | NOT STARTED | tracking、Kalman prediction、reachability veto，覆盖 crossing/head-on/multi-agent 和倒车后方障碍 | 动态让行、紧急停车与恢复生命周期通过 |
| P8 扩展数据与资格签发 | NOT STARTED | 扩到至少 80 张地图、至少 10 张 validation 地图，冻结独立测试与 checkpoint contract | 独立测试报告通过后才可 `production_qualified=true` |

## 冻结原则

- gear 由确定性状态机掌权，网络只在给定挡位下生成候选。
- 轨迹可行性按完整缩放车体、safety margin 与连续扫掠判断，不用中心点或稀疏姿态代替。
- split 以 map UUID 隔离；P3/P4/P5 开发期间 test 保持封存。
- P3 先证明候选集合表达得出安全动作，P5 才评价网络能否生成并选对动作。
- 所有 P5 输出仍为 `UNQUALIFIED`；训练本身不授予 Gazebo 或 production 资格。
- P8 的 80/10 地图规模是最终资格要求，不冒充当前 P5 数据质量门槛，也不能替代 P3 V3 修复。

当前最短关键路径：`P3 V3 增量补强 → 重建 dev index/context → P3 PASS → 批准 P5 三模态正式训练`。
