# P3 修正车体几何后的最小增量补强方案

> 历史文档，已被 [p3_development_v3_remediation_plan.md](p3_development_v3_remediation_plan.md) 取代。本文基于旧 V2 全 split 统计（10,252 样本、10.8076%），不得再作为 P5 解锁依据；当前权威是 V3 train+validation-only 连续 swept-footprint 重审。

状态：方案冻结草案，尚未启动 Gazebo，尚未生成或修改任何数据  
适用对象：`StaticAckermannSampleV2` P3 Pilot 数据及 P4/P5 训练视图  
依据报告：`reports/p3_footprint_v2_reaudit.json`

## 1. 目标与边界

本方案用于修复 `FootprintConfig` 更正后，P3 候选表达能力不再满足原资格门槛的问题。处理方式是建立独立、可追溯的增量数据 wave，而不是改写、替换或删除原来的 10,252 个 NPZ。

必须同时满足以下边界：

1. 原始 `data/p3_pilot` 全目录保持只读；原 task manifest、bag、episode、NPZ 和验收报告均不得被重新生成或移动。
2. 数据生成策略、任务筛选规则和候选预检参数只能使用 train 数据制定，validation 用于冻结后的验证；test 不参与任务选择、阈值调整、失败重试或 wave 扩容决策。
3. 不放宽原 P3 门槛。最终仍要求 corrected-footprint 总体 zero-feasible rate `< 0.10`、各模式 `< 0.25`、feasible candidate 中位数 `>= 2`。
4. 所有新增数据使用新的 config、seed、task manifest、collection state 和 work root。每个 wave 一经生成即不可变。
5. 不通过重复采集同一 COMPLETE 任务或只增加简单 NORMAL 样本来“刷低”总体比例。新增任务仍需保持模式、地图、左右转和路线几何的覆盖。

## 2. 已确认的问题

修正 footprint 后的全量审计结果为：

- 样本：10,252；候选：153,780；读取失败：0。
- 总体 zero-feasible：1,108 / 10,252 = `10.8076%`，未满足 `< 10%`。
- `SHARP_TURN`：457 / 1,678 = `27.2348%`，未满足 `< 25%`。
- feasible candidate 中位数仍为 15，满足 `>= 2`。
- 12,961 个候选标签发生 legacy 与 corrected 之间的变化。

下述补强预算只使用 train+validation 统计，不使用 test：

- development 总样本 `N = 9,290`，zero-feasible `Z = 971`，比例 `10.4521%`。
- development `SHARP_TURN` 样本 `N_s = 1,600`，zero-feasible `Z_s = 441`，比例 `27.5625%`。

现有质量重试机制也不能直接用于本次修复。`--retry-zero-feasible-rate-above` 读取的是采集时的旧 footprint episode 计数。在 train+validation 上：

- 阈值 0.10 时，旧机制会选中 22 个任务，corrected 审计实际有 46 个任务超标，漏掉 25 个。
- 阈值 0.25 时，旧机制漏掉 9 个 corrected 实际超标任务。

此外，重跑 COMPLETE 任务会调用 `archive_previous_artifacts()`，移动原 bag、episode 和 NPZ，因此不能对原根目录使用。

## 3. 数学下界

设新增 development 样本数为 `M`，其中 corrected zero-feasible 数为 `E`；新增 sharp 样本数为 `K`，其中 zero-feasible 数为 `E_s`。

### 3.1 理论零失败下界

若假设所有新增样本均至少有一个 corrected-feasible candidate，即 `E = 0`：

```text
971 / (9290 + M) < 0.10
M > 420
```

所以总体至少新增 `421` 个 zero-free 样本。

同理，对于 `SHARP_TURN`：

```text
441 / (1600 + K) < 0.25
K > 164
```

所以其中至少需要 `165` 个 zero-free sharp 样本。

当前采集合同规定每个成功 episode 至少提取 20 个样本。因此，在“每个新增样本都 non-zero”的理想条件下，任务数量下界为：

- 总成功任务至少 `ceil(421 / 20) = 22`。
- 其中 sharp 成功任务至少 `ceil(165 / 20) = 9`。

这只是数学下界，不是生产预算。任何新增 zero-feasible 样本都会使该预算失效。

### 3.2 带余量的保守预算

正式增量批次预先冻结以下质量目标：

- 增量批次 corrected overall zero-feasible rate `<= 5%`。
- 增量批次 corrected sharp zero-feasible rate `<= 15%`。
- 合并后的 development guard target：overall `<= 9.5%`、sharp `<= 24%`。

总体样本预算由下式得到：

```text
(971 + 0.05 M) / (9290 + M) <= 0.095
M >= 1966
```

sharp 样本预算由下式得到：

```text
(441 + 0.15 K) / (1600 + K) <= 0.24
K >= 634
```

因此向上取整后的正式预算为：

- 新增样本至少 `2,000`。
- 其中 `SHARP_TURN` 至少 `650`。
- 按每任务最低 20 个样本计算，准备 `100` 个成功任务，其中至少 `33` 个 sharp。

现有 150 个任务平均产生 68.35 个样本。若新任务保持相近产量，100 个任务预计增加约 6,835 个样本，合并后约 17,087 个，仍低于原 P3 的 30,000 样本上限。

## 4. 冻结的任务 quota

100 个任务的正式 quota 如下：

| 模式 | 任务数 |
|---|---:|
| NORMAL | 19 |
| SHARP_TURN | 33 |
| NARROW_CORRIDOR | 12 |
| U_TURN | 8 |
| DEAD_END_ESCAPE | 8 |
| REVERSE_EXIT | 12 |
| THREE_POINT_TURN | 8 |
| 合计 | 100 |

该分配提高 sharp 比例，同时近似保留非 sharp 模式的原始相对分布，不把总体门槛建立在单纯增加 NORMAL/U_TURN 样本之上。

为减少一次性 Gazebo 成本，可以采用两个不可变 wave：

| Wave | NORMAL | SHARP | NARROW | U_TURN | DEAD_END | REVERSE | THREE_POINT | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wave01 | 6 | 10 | 4 | 2 | 2 | 4 | 2 | 30 |
| wave02 | 13 | 23 | 8 | 6 | 6 | 8 | 6 | 70 |
| 累计 | 19 | 33 | 12 | 8 | 8 | 12 | 8 | 100 |

wave01 是运行与产量确认波，不单独获得正式资格。wave02 必须使用 wave01 之前已经冻结的几何、候选预检和验收规则；不能根据 test 结果决定是否启动或改变 wave02。

## 5. 地图与 split 规则

现有 40 张地图按 UUID 固定划分为：31 train、5 validation、4 test。原 Pilot 已使用 24 train、3 validation、3 test；尚未使用的地图为 7 train、2 validation、1 test。

增量规则如下：

1. wave01 优先使用尚未使用的 7 train + 2 validation 地图，显式排除那 1 张未用 test 地图。按每图最多 5 个任务，9 张图可容纳 45 个任务，足够 wave01 的 30 个任务。
2. 100 任务的保守容量需要允许使用全部 31 train + 5 validation 地图。36 张图、每图最多 5 个任务，总容量为 180。
3. config 应固定：`selected_count: 36`，`required_split_maps: {train: 31, validation: 5, test: 0}`。由于 required 数量之和等于 selected 数量，生成器不得从 remaining maps 补入 test。
4. 生成器和审计器仍必须显式检查 `allowed_splits == [train, validation]`；不能只依赖 selected_count 的间接保证。
5. 新增 validation 样本只参与 validation，不进入训练。原 test 样本保持封存，且不新增 test 任务。

建议目录结构：

```text
data/p3_footprint_v2_increment/
  wave01/
    task_manifest.json
    run/
  wave02/
    task_manifest.json
    run/
```

每个 wave 的 manifest 必须记录：

- wave ID、独立 task seed、独立 task ID prefix。
- 原 P3 task manifest SHA-256。
- `p3_footprint_v2_reaudit.json` SHA-256。
- 冻结的 `FootprintConfig` contract SHA-256。
- `allowed_splits: [train, validation]` 与 `test_excluded: true`。
- task proposal、Hybrid A*、candidate preflight 的版本和参数。
- 使用的 map UUID 与 occupancy SHA-256。
- config SHA-256、manifest 自身 SHA-256。

不得向已有 manifest 追加任务。若需要新任务，创建新 wave、使用新 seed 和新 manifest。

## 6. 原数据只读规则与禁用命令

禁止对 `data/p3_pilot` 使用以下 collection 选项：

```text
--rerun-complete
--rerun-all-complete
--retry-zero-feasible-rate-above <value>
```

也禁止以下行为：

- 对原根目录执行 `--force-prepare`。
- 重新运行原 COMPLETE task ID。
- 删除或移动原 bag、episode、NPZ、collection state。
- 通过修改 `feasible` 或 `static_clearance` 数组给原 NPZ 换标签。
- 为增量任务执行 `--stage prepare`，因为地图应复用现有不可变 map authority。
- 将新的 task manifest 与原 `data/p3_pilot/run/collection_state.json` 混用。
- 用 test 的模式、地图或任务结果决定 retry、seed、quota、预检阈值或停止时机。

允许的 retry 仅限新 wave 根目录中的基础设施失败任务：

```text
--retry-failed
```

COMPLETE 但质量不足的增量任务不重跑、不替换；它进入该 wave 的完整审计记录。需要继续补量时使用相同冻结策略创建下一 wave。

## 7. 需要补齐的工具能力

以下为后续实施需要的最小代码改造，本文件本身不实施这些改造。

### 7.1 增量 task generator

在 `generate_pilot_tasks.py` 中增加：

- `allowed_splits` 强约束，遇到 test map 立即失败。
- `exclude/base-manifest`，使 wave01 可优先选择原 Pilot 未使用的开发地图。
- `wave_id`、`task_prefix` 和独立 seed，避免 sample/task ID 冲突。
- `--workers 8` 的确定性并行 proposal。随机种子按任务槽固定，最终由主进程按稳定顺序归并，保证输出不随 worker 调度变化。
- corrected-footprint candidate route preflight：对 Hybrid A* 路径抽样，在相应挡位、转角状态和 retime factors 下生成 3×5 lattice，转换到 map 坐标并调用修正后的 swept-footprint evaluator。
- 预检至少要求每个受检 route pose 存在一条安全候选、route feasible-count 中位数 `>= 2`，并对小幅位姿扰动保持安全；具体扰动合同只能用 train 冻结。
- sharp 左/右转、heading-change、steering、clearance 和地图覆盖检查，防止只保留宽阔空间中的简单 sharp 样本。
- preflight rejection ledger，用于审计生成器，但失败 proposal 不成为训练样本。

### 7.2 增量数据审计

新增只读增量审计工具，要求：

- 从 global immutable map occupancy、candidate pose 和 `chassis_to_map` 重新计算 corrected 标签。
- 拒绝任何 test 样本或 test task。
- 校验 config、manifest、map occupancy、bag、sample、preprocessing、footprint contract 哈希。
- 检查任务和 sample ID 与 base/其他 wave 不重复。
- 按 split、mode、map、task 输出样本数、zero-feasible、feasible-count 和 clearance。
- 不修改 NPZ；不提供 footprint 或验收阈值的命令行覆盖参数。
- 增量累计达到至少 2,000 样本、650 sharp，且批次 overall `<= 5%`、sharp `<= 15%` 才能进入 bundle 审计。

### 7.3 只读 dataset bundle

新增 `P3DatasetBundle` 描述和审计工具：

- bundle 只保存 base 与各 wave 的根路径和不可变哈希，不复制、不移动 NPZ。
- 验证 map UUID split、sample ID、task ID、manifest、config、preprocessing 和 geometry contract 一致。
- 默认只构建 train+validation view；test 必须通过显式 final-evaluation 开关才能读取。
- checkpoint provenance 绑定 `dataset_bundle_sha256`，不能只绑定原单一 task manifest SHA-256。
- final bundle 报告聚合 base 与全部合格 wave，并重新执行所有 P3 门槛。

### 7.4 corrected 标签训练视图

原 NPZ 的 `feasible`、`static_clearance` 属于 legacy footprint，而新 wave 会使用 corrected footprint。二者不能直接混合作为同一监督标签。

在不改写 NPZ 的前提下，正式训练必须选择以下一种统一方案：

1. P4/P5 loader 每次按 immutable global map、`chassis_to_map` 和 frozen footprint 重算 corrected 标签；或
2. 生成由 sample path、sample SHA-256 和 footprint SHA-256 绑定的只读 sidecar label overlay。

原 NPZ 字段只能作为 legacy provenance，不得用于正式 corrected feasibility 监督。

## 8. 离线 manifest 准备模板

以下命令不会启动 Gazebo。应在上述 generator 改造和增量 config 完成后执行：

```bash
cd /home/zjh/DE-P-Car

/usr/bin/python3 ros/dep_car_dataset/scripts/generate_pilot_tasks.py \
  --config dep_car/config/p3_footprint_v2_increment_wave01.yaml \
  --maps data/p3_pilot/maps \
  --output data/p3_footprint_v2_increment/wave01/task_manifest.json \
  --validate-only

/usr/bin/python3 ros/dep_car_dataset/scripts/generate_pilot_tasks.py \
  --config dep_car/config/p3_footprint_v2_increment_wave01.yaml \
  --maps data/p3_pilot/maps \
  --output data/p3_footprint_v2_increment/wave01/task_manifest.json \
  --workers 8
```

正式生成前必须确认 validate-only 输出中：

- test maps/tasks 均为 0。
- quota 与本方案一致。
- footprint contract SHA-256 与修正审计报告一致。
- 没有与 base 重复的 task ID。
- manifest 为 non-partial，proposal deficit 为 0。

## 9. 宿主机三步采集命令模板

脚本会退出 conda 并加载 ROS Noetic 与当前 catkin workspace，因此不需要在 YOPO conda 环境中运行。所有命令必须指向新的 wave root。

### 第一步：dry-run

```bash
cd /home/zjh/DE-P-Car

bash scripts/run_p3_pilot.sh \
  --stage collect \
  --config dep_car/config/p3_footprint_v2_increment_wave01.yaml \
  --root data/p3_footprint_v2_increment/wave01 \
  --workers 8 \
  --dry-run
```

### 第二步：采集一个 episode 验证闭环

```bash
cd /home/zjh/DE-P-Car

bash scripts/run_p3_pilot.sh \
  --stage collect \
  --config dep_car/config/p3_footprint_v2_increment_wave01.yaml \
  --root data/p3_footprint_v2_increment/wave01 \
  --workers 8 \
  --maximum-tasks 1 \
  --fail-fast
```

该任务若 COMPLETE，第三步会自动跳过；若属于基础设施失败，第三步的 `--retry-failed` 会重试。

### 第三步：正式完成 wave

```bash
cd /home/zjh/DE-P-Car

bash scripts/run_p3_pilot.sh \
  --stage collect \
  --config dep_car/config/p3_footprint_v2_increment_wave01.yaml \
  --root data/p3_footprint_v2_increment/wave01 \
  --workers 8 \
  --retry-failed
```

采集期间不得对同一 wave 启动第二个 orchestrator，也不得把 `--root` 改回 `data/p3_pilot`。

## 10. 分层验收门槛

### 10.1 单 wave 完整性

- manifest/config/map/bag/sample 哈希全部通过。
- `FAILED = INTERRUPTED = RUNNING = 0`。
- episode completion rate `>= 0.95`。
- illegal shift count `= 0`。
- 单一 preprocessing hash，且等于 P3/P4 冻结值。
- test task、test sample 均为 0。
- NPZ 没有被审计器改写。

### 10.2 增量累计资格

- 新增 train+validation 样本 `>= 2,000`。
- 新增 `SHARP_TURN` 样本 `>= 650`。
- corrected overall zero-feasible rate `<= 0.05`。
- corrected `SHARP_TURN` zero-feasible rate `<= 0.15`。
- corrected feasible candidate 中位数 `>= 2`。
- quota、地图、左右 sharp 和路线几何覆盖通过。

### 10.3 Development bundle guard

在不打开 test 的情况下：

- train+validation corrected overall zero-feasible rate `<= 0.095`。
- train+validation corrected `SHARP_TURN` zero-feasible rate `<= 0.24`。
- train+validation 每个模式均 `< 0.25`。
- feasible candidate 中位数 `>= 2`。

guard 未通过时只能基于 train 调整下一版策略，或使用已经冻结的策略增加新 wave；不得查看 test 后再决定修改。

### 10.4 最终一次性资格

生成器、训练视图和所有 wave 冻结后，才允许显式打开 test 做最终 bundle 审计。最终必须满足原 P3 全部门槛：

- corrected overall zero-feasible rate `< 0.10`。
- corrected 每个 maneuver mode zero-feasible rate `< 0.25`。
- feasible candidate 中位数 `>= 2`。
- 样本总数在 `[10,000, 30,000]`。
- 地图总数 `>= 20`，validation maps `>= 2`，test maps `>= 2`。
- 每个非 NORMAL 模式样本 `>= 500`。
- oracle route error p90 `<= 0.85 m`。
- reverse sample fraction `>= 0.12`。
- episode completion rate `>= 0.95`。
- illegal shift count `= 0`。
- 全部样本具有同一个冻结 preprocessing hash。
- base 与 wave 的原始 authority、bundle SHA-256、corrected label contract 全部通过。

只有 final bundle 报告 PASS 后，P5 正式训练才可以使用该 bundle；P4 代码和单元测试可以在此之前继续，但不得将未合格数据签发为正式训练 authority。
