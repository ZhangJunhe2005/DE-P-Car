# P3 Pilot 数据生成实现说明

状态：`P3_PASS`。40 张地图和 150 项任务清单已由宿主机生成；150/150 项 Gazebo 采集完成，最终 10252 个样本和 150 个权威 rosbag 已通过完整审计。

当前任务清单 SHA256：`c46190e4ddfcd5eb96739b51dfe832a8db885581375641186f9f3c7f7782f39c`。清单无 quota deficit，30 张入选地图严格分为 train 24、validation 3、test 3，每图 5 个任务。

最终验收报告位于 `data/p3_pilot/run/p3_pilot_audit.json`，状态为 `PASS`。关键结果：总体零可行率 0.06262、每模式最高 0.21871、中位可行候选数 15、Oracle 路线误差 P90 0.80614 m、倒车样本比例 0.44596、非法换挡 0；所有 bag 的实际 SHA256 均与样本声明相同。

本轮修复冻结了三个运行时合同：局部点云栅格不再与 footprint 重复膨胀并改为 0.05 m；scaled Urban Car 的 receding candidate horizon 改为 1.0 s；静态恢复期间每帧发布并重新硬校验完整 15 条 reverse bank。恢复样本通过 `CandidateContextV2` 与普通 mission 样本区分。

## 默认规模

- arena-tools 确定性生成 40 张候选地图，从中选取 30 张：train 24、validation 3、test 3。
- 150 个任务，每张入选地图最多 5 个 episode。
- 七类任务配额：NORMAL 36、SHARP_TURN 22、NARROW_CORRIDOR 22、U_TURN 16、DEAD_END_ESCAPE 16、REVERSE_EXIT 22、THREE_POINT_TURN 16。
- 每个 episode 最多运行 18 秒，LiDAR 10 Hz、抽取 stride=1；目标是最终保留 10k～30k 个 revision-2 样本。
- rosbag 使用 LZ4，但 640×480 原始深度仍可能占用数十 GiB。启动前应检查宿主机磁盘；编排器低于 20 GiB 可用空间会安全停止。

## 运行结构

1. `generate_static_maps.py --resume` 生成或验证地图 provenance。
2. `generate_pilot_tasks.py` 先用 Hybrid A* 离线证明任务可行，再写入带 SHA256 的分桶任务清单。
3. `run_pilot_collection.py` 默认启动 8 条并行流水线；每条流水线依次启动独立 Gazebo、录制权威 rosbag、发布目标、保存 episode 结果、关闭仿真并离线抽取样本。
4. `collection_state.json` 原子更新；完成任务自动跳过，失败任务可显式重试，旧 artifact 会移入 `previous_attempts/`，不会被直接删除。
5. `audit_p3_pilot_dataset.py` 检查样本合同、任务绑定、map split、bag 实体与 SHA256、各机动桶覆盖、倒车比例和 Candidate Expressiveness/Oracle-of-15。
6. `reextract_pilot_samples.py` 可从不可变 rosbag 离线恢复此前因候选上下文不匹配而拒绝的 recovery 样本，无需重新启动 Gazebo。

采集状态会同时绑定配置和任务清单的 SHA256。一次采集开始后不要直接修改这两个文件；如需改变配置，应使用新的 `--root` 建立独立 Pilot，避免混合不同数据合同。

并行采集默认使用 8 个 worker，对应 ROS master 端口 `11321～11328` 和 Gazebo master 端口 `11351～11358`。各 worker 使用独立进程组、ROS 日志和端口；同一张地图的任务固定落在同一个 worker，避免同图并发。worker 每隔 1 秒错峰启动，离线抽取和最终审计同样使用 8 worker。数值库在每个 worker 内限制为单线程，避免 8 条流水线再各自派生大量 BLAS 线程。

已经存在且 SHA/配置/地图引用均匹配的任务清单会被 `prepare` 自动复用，不再重复执行耗时的 Hybrid A* 搜索。只有显式传入 `--force-prepare` 才会重新生成清单。

## 宿主机命令

先只准备地图与任务清单，不启动 Gazebo：

```bash
cd /home/zjh/DE-P-Car
bash scripts/run_p3_pilot.sh --stage prepare
bash scripts/run_p3_pilot.sh --stage collect --dry-run
```

dry-run 输出中的 `parallel_workers` 应为 `8`，并显示互不重复的 ROS/Gazebo 端口。

确认 dry-run 后，建议先真实采集一个 episode，验证宿主机上的 Gazebo、rosbag 和离线抽取闭环：

```bash
bash scripts/run_p3_pilot.sh --stage collect --maximum-tasks 1 --fail-fast
```

该任务成功后会被记入断点状态，随后正式启动全量耗时采集时会自动跳过：

```bash
cd /home/zjh/DE-P-Car
bash scripts/run_p3_pilot.sh --stage collect
```

上式默认等价于传入 `--workers 8`。`collection_state.json` 会记录请求的 worker 数和宿主机可见 CPU 线程数。

中断后执行同一条命令即可续跑。只重试失败任务：

```bash
bash scripts/run_p3_pilot.sh --stage collect --retry-failed
```

单个 episode 失败时编排器会继续收集其余任务，最终以非零状态退出并保留失败原因；执行上面的重试命令即可只处理失败项。

全部完成后执行正式审计（默认重新计算每个 rosbag 的 SHA256）：

```bash
bash scripts/run_p3_pilot.sh --stage audit
```

若提取合同升级而原始 bag 保持不变，可用 8 worker 只重提取受影响的恢复任务：

```bash
bash scripts/run_p3_pilot.sh --stage reextract --workers 8
```

也可以一次完成准备、采集和审计：

```bash
bash scripts/run_p3_pilot.sh --stage all
```

该入口会主动退出 conda 环境并加载 ROS Noetic/catkin，不需要在 `yopo` 环境中执行。

## 关键输出

- `data/p3_pilot/task_manifest.json`：不可变任务合同。
- `data/p3_pilot/run/collection_state.json`：断点续跑状态。
- `data/p3_pilot/run/bags/`：权威原始 rosbag，不能在训练资格签发前删除。
- `data/p3_pilot/run/samples/`：`StaticAckermannSampleV2 revision 2` 派生样本。
- `data/p3_pilot/run/p3_pilot_audit.json`：P3 进入 P4 的最终门槛报告。
