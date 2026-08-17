# DE-P V4.9.1 → DE-P-Car 迁移矩阵

| 能力 | 处理方式 | DE-P-Car 实现 |
|---|---|---|
| MobileNetV3 backbone | 权重迁移 | 冻结 V4.8.3 backbone 全部兼容张量转入 LiDAR 模型 |
| independent candidate/score head | 部分权重迁移 | feature 通道与 score tower/head 迁移；车辆状态通道重新初始化 |
| 3×5 candidate grid | 语义迁移 | 3 speed × 5 steering |
| UAV PVA output | 重写 | speed/steering offsets → bicycle rollout |
| static safety | 重写几何 | 五圆 swept footprint + OccupancyGrid authority |
| dynamic tracking | 2-D 适配 | map-difference LiDAR observations + CV Kalman |
| bounded reachability | 2-D 适配 | time-aligned actor disks + covariance inflation |
| hard veto before ranking | 原样保留 | 静态/动态碰撞候选永不被 learned score 恢复 |
| ordered retiming | 原样保留 | 1.0 → 1.2 → 1.4，首个可行档立即采用 |
| dynamic yield | 原样保留 | 不累积静态死锁证据 |
| mission/recovery lifecycle | 原样保留 | 临时目标、mission reacquire、fresh re-arm |
| yaw scan recovery | 废弃 | VLP-16 360° + bounded deterministic reverse |
| UAV checkpoint | 禁止直接运行 | 只生成 `INITIALIZATION_ONLY_RETRAINING_REQUIRED` 迁移初始化 |

