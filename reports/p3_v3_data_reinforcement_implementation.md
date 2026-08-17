# P3 V3 增量补强与索引重建实现

状态：`WAVE_COMPLETE / BUNDLE_V1_AUDIT_FAIL / CURATED_V2_CODE_READY`。180 个
episode 已全部完成，base 重抽取与 23,236 样本的 bundle_v1 已封存。正式 V1
重审失败；修复后的 curated bundle_v2 构建仍由用户在宿主机启动。

## 为什么日志里有 PASS 却不能 proposal

报告包含两个不同层级：

- `validation_coverage.status: PASS` 只表示 validation 的动作、前后挡和
  `MISSION/RECOVERY` 数量达到覆盖门槛。
- 顶层 `status` 才是 proposal 的放行状态。32 样本命令的顶层为 `SMOKE`；
  23,236 样本正式命令的顶层为 `FAIL`。

正式失败项为总体零可行率 `19.4698%`、`SHARP_TURN=28.3667%` 和
`THREE_POINT_TURN=26.4028%`。全量只读根因诊断进一步确认：3,218 条样本在
候选轨迹 `t=0` 时车身占地已经与训练地图的安全足迹相交，任何速度或转角候选
都无法挽救。把这些不合法起始状态显式隔离后，原候选格点剩余
`1306 / 20018 = 6.5241%`，总体和逐模式门槛均可通过。它们不会被删除或改写，
仍保留在原 source 目录作为恢复/碰撞诊断资料。

修复使用 `dep_car/config/p3_v3_curated.yaml` 新建
`data/p3_v3/bundle_v2_curated`。构建器会用 8 个进程对所有来源样本复算
production signed-SDF `t=0` 足迹，生成带哈希的 `curation_authority.json`，再
只把合法起始状态复制进新训练索引；旧 bundle_v1 和所有源 NPZ 保持不变。

## 为什么需要补强

原 P3 并不缺少深度或 LiDAR 数据。需要补强的是在当前车辆安全合同下的候选表达分布与标签权威：

- P4 从离散姿态/旧 footprint 升级为完整缩放车体、一个栅格对角线 allowance 和 161 时刻连续扫掠。
- 现有 9,290 个开发样本中有 1,786 帧的当前 15 候选全部不可行；总体 `19.22497%`，窄通道 `55.93407%`、急转 `32.4375%`、三点掉头 `36.33333%`。
- 806 个 validation NPZ 在 `candidate_context` 加入合同前被抽取，所以字段为 UNKNOWN。原 bag 仍包含 candidates 与 route-command，可用当前 extractor 确定性恢复，不需要猜标签。

旧 bag/NPZ 不删除、不改写。新 V3 bundle 由“base bag 非破坏性重抽取 + targeted wave”组成。

## 实现组件

- `dep_car/config/p3_v3_incremental.yaml`：冻结 8 worker、180 个任务、split/mode 配额、连续几何预检和最终门槛。
- `tools/generate_p3_v3_wave.py`：在 31 train + 5 validation 地图上并行生成提案；test 地图在打开 YAML/PNG 前排除。每条路线抽 7 个 checkpoint，并在停车/名义速度下以当前 15 lattice + production signed-SDF 连续 footprint 验证每个 probe 至少两条可行候选。
- `tools/reextract_p3_v3_base.py`：仅重抽取原 135 个 train/validation bag 到新目录；恢复 `MISSION/RECOVERY`、验证 requested gear、逐文件 SHA-256、原子状态和断点续跑；test bag 不打开。
- `run_pilot_collection.py`：复用已有 8 路独立 ROS/Gazebo 管线采集 wave01，保留 rosbag、日志和失败重试。
- `tools/build_p3_v3_bundle.py`：将 base_reextracted 与 wave01 做独立、逐字节验证的 copy；拒绝路径冲突、UNKNOWN、非 drive gear 和未完成 source authority；随后以 8 worker 重建 `P3TrainingIndexV2` 并签发 bundle authority。
- `tools/audit_p3_v3_bundle.py`：验证 bundle/source/index/map/content 哈希后，调用 P4/P5 同一连续几何 evaluator 全量重审；同时检查 validation 的 context、七类 maneuver 和前/倒挡覆盖。有限审计不能覆盖正式报告。
- `tools/propose_p3_v3_training_authority.py`：只有 bundle 审计 PASS 才生成 P5 config 变更提案；不会修改 `training.yaml`，也不会启动训练。
- `scripts/run_p3_v3_reinforcement.sh`：宿主机统一入口。

## Wave01 容量

正式 wave 为 180 个 episode：144 train、36 validation。

| 模式 | Train | Validation | 总数 |
|---|---:|---:|---:|
| NARROW_CORRIDOR | 54 | 12 | 66 |
| SHARP_TURN | 36 | 9 | 45 |
| THREE_POINT_TURN | 36 | 12 | 48 |
| DEAD_END_ESCAPE guard | 18 | 3 | 21 |

按原数据约 68 帧/episode 估算将新增约 12,000 帧，合并后约 21,000 帧，仍低于原 30,000 上限。若新增 zero-feasible 控制在约 3% 以下，总体才有足够余量越过 `<10%`；因此 100 个 episode 在数学上并不稳妥。

## 宿主机分阶段命令

先验证配置；此步骤不生成任务、不启动 Gazebo：

```bash
cd /home/zjh/DE-P-Car
bash scripts/run_p3_v3_reinforcement.sh --stage validate --workers 8
```

生成带连续几何预检的正式 180-task manifest。该步骤会使用 8 个 CPU 进程，可能耗时较长：

```bash
bash scripts/run_p3_v3_reinforcement.sh --stage prepare --workers 8
```

重抽取 base 开发 bag。一次单任务 smoke 已完成，正式命令会自动复用该任务并继续剩余 134 个：

```bash
bash scripts/run_p3_v3_reinforcement.sh --stage base-reextract --workers 8 --dry-run
bash scripts/run_p3_v3_reinforcement.sh --stage base-reextract --workers 8
```

新 wave 仍按以前偏好的三步执行：

```bash
# 1. 只检查 8 路 ROS/Gazebo 命令
bash scripts/run_p3_v3_reinforcement.sh --stage collect --workers 8 --dry-run

# 2. 真实采集一个 episode
bash scripts/run_p3_v3_reinforcement.sh --stage collect --workers 8 \
  --maximum-tasks 1 --fail-fast

# 3. 正式采集剩余任务；完成项自动跳过
bash scripts/run_p3_v3_reinforcement.sh --stage collect --workers 8
```

失败任务修复后只重试 FAILED。先 dry-run 验证选择集合，再真实重试；若只剩少量失败项，worker 数会自动收缩，不会重跑 COMPLETE 项：

```bash
bash scripts/run_p3_v3_reinforcement.sh --stage collect --workers 2 \
  --retry-failed --dry-run
bash scripts/run_p3_v3_reinforcement.sh --stage collect --workers 2 \
  --retry-failed
```

也可用 `--task-id TASK_ID --retry-failed` 精确重跑一个失败 episode。episode 计时只在传感器、`map -> chassis` TF、全局路线、路线挡位命令和首个候选轨迹全部就绪后开始，避免并行 Gazebo 启动抖动导致有效帧不足。

原始 bundle_v1 已经完成，下列旧命令仅保留为历史流程说明：

```bash
bash scripts/run_p3_v3_reinforcement.sh --stage status
bash scripts/run_p3_v3_reinforcement.sh --stage bundle --workers 8 --dry-run
bash scripts/run_p3_v3_reinforcement.sh --stage bundle --workers 8
```

当前应从相同的已认证 source 构建新的 curated bundle_v2。第一条只验证来源和
命令，不复制数据；第二条才会用 8 个进程做 `t=0` 复核、显示进度并复制合法
样本：

```bash
cd /home/zjh/DE-P-Car

# 1. 来源/配置 dry-run
bash scripts/run_p3_v3_reinforcement.sh \
  --stage bundle --config dep_car/config/p3_v3_curated.yaml \
  --workers 8 --dry-run

# 2. 正式构建 bundle_v2_curated（不覆盖 bundle_v1）
bash scripts/run_p3_v3_reinforcement.sh \
  --stage bundle --config dep_car/config/p3_v3_curated.yaml \
  --workers 8
```

构建结束后先做 32 样本 smoke；它写入独立报告，不覆盖正式 P3 报告。随后启动
全量 8-worker 审计：

```bash
bash scripts/run_p3_v3_reinforcement.sh --stage audit --workers 8 \
  --config dep_car/config/p3_v3_curated.yaml --maximum-samples 32
bash scripts/run_p3_v3_reinforcement.sh --stage audit --workers 8 \
  --config dep_car/config/p3_v3_curated.yaml
```

只有正式审计 PASS 后才生成 P5 authority 提案：

```bash
bash scripts/run_p3_v3_reinforcement.sh --stage proposal \
  --config dep_car/config/p3_v3_curated.yaml
```

此 proposal 仍需单独批准后才会应用到 `training.yaml`。

## 资源与执行边界

- 当前磁盘剩余约 143 GiB；预计 base 重抽取、wave bag/NPZ 和独立 bundle 合计还需约 55～75 GiB，运行中仍受 20 GiB 安全下限保护。
- P3 主要消耗 CPU、Gazebo 和磁盘，不使用训练 GPU。P5 正式训练才使用 CUDA。
- Codex 环境内凡需 CUDA、ROS/Gazebo 多进程、DataLoader/process-pool 或受限 IPC 的真实执行，必须先申请 full-access；不能因为沙盒失败就静默降为单线程并把性能问题留给宿主机。
- test NPZ、test bag 和 test 地图几何在整个 V3 开发流程保持封存。
