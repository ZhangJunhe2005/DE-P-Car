# DE-P-Car 目标架构

正常规划链为：VLP-16 → 16×440 range/mask → V4.8.3-lineage MobileNetV3 → 3×5 Ackermann 参数 → bicycle rollout → swept footprint 静态硬检查 → 2-D Kalman reachability 动态硬检查 → risk ranking → ordered retiming → Urban Car adapter。

全局引导由 OccupancyGrid + mission goal → Hybrid A* → 4 m local subgoal 提供。网络不直接追最终任务点。

恢复链明确区分 `DYNAMIC_YIELD` 与 `STATIC_DEADLOCK`。动态阻塞只停车等待；静态死锁才允许低速、有限距离、通过完整 footprint 检查的 deterministic reverse。倒车完成后必须重新获得 mission-conditioned 可行规划才 re-arm。Ackermann 版本不使用 UAV 的原地 60/90/120° yaw scan。

Gazebo/Pedsim GT 只允许进入 `dep_car_evaluation`，正式 perception/planner 输入仅来自 LiDAR、地图与车辆状态。

