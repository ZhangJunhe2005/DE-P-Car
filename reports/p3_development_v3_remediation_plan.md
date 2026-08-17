# P3 V3 开发数据增量补强方案

状态：`IMPLEMENTED_AWAITING_LARGE_RUN`。权威输入为 [p3_development_reaudit_v3.json](p3_development_reaudit_v3.json)，不使用 test 选择任务、调阈值或决定扩容。脚本与宿主机命令见 [p3_v3_data_reinforcement_implementation.md](p3_v3_data_reinforcement_implementation.md)。

## 当前缺口

冻结开发权威包含 24 张 train 地图、3 张 validation 地图和 9,290 个样本。当前 lattice + 连续 signed-SDF footprint 对 139,350 条候选的复核结果：

| 门禁 | 当前值 | 目标 |
|---|---:|---:|
| overall zero-feasible | 19.22497% | <10% |
| NARROW_CORRIDOR | 55.93407% | <25% |
| SHARP_TURN | 32.43750% | <25% |
| THREE_POINT_TURN | 36.33333% | <25% |
| overall median feasible candidates | 14 | >=2 |
| validation RECOVERY context | 6 | >=20 |
| validation UNKNOWN context | 806 | 0 |

NORMAL、REVERSE_EXIT 与 U_TURN 的逐模式 zero-feasible 已通过。DEAD_END_ESCAPE 为 24.24520%，非常接近 25% 边界，应作为 guard bucket 一并监控，但不能用它替代三个失败模式。

## 修复原则

1. 不改写现有 NPZ/bag，不放宽 `<10% / <25% / median>=2` 门槛，不降低车体或 safety margin。
2. 所有候选预检必须调用与 P4/P5 相同的 canonical lattice、gear-aligned 状态投影、161 时刻五圆扫掠和 signed-SDF/bilinear evaluator。
3. 新 wave 只允许 train/validation 地图；map UUID 继续决定 split。test NPZ、test YAML/PNG 和 test 指标在开发重审中保持封存。
4. UNKNOWN 不能按猜测重命名为 RECOVERY。当前实现从原始 bag 的 candidates 与 route-command 重新抽取 `MISSION/RECOVERY`；旧 NPZ 不修改，无法重抽取或无法证明的样本不进入新 bundle。
5. 每个 wave 都先生成任务清单和 candidate preflight 报告，再采集；不采集预检已经 zero-feasible 的起始状态。

## 建议执行波次

### Wave A：修复标签与索引权威

- 为每个开发样本补齐可追溯 `candidate_context`，记录派生规则版本与源字段 hash。
- 对 validation 强制 `MISSION>=20`、`RECOVERY>=20`、`UNKNOWN=0`；七类 maneuver 各不少于 50 帧，前/倒挡各不少于 100 帧。
- 重新生成内容寻址的 training index；旧 index 保持只读并在新 bundle 中记录父 hash。

### Wave B：候选表达能力补强

- 优先新增 NARROW_CORRIDOR、SHARP_TURN、THREE_POINT_TURN；同时给 DEAD_END_ESCAPE 留 guard 样本。
- 每个任务在采集前对多个起点/航向/挡位运行 15 候选 V3 preflight，只接受至少 2 条连续 footprint 可行候选的任务。
- 窄通道覆盖不同宽度、偏置起点和双侧障碍；急转覆盖左右转、不同入弯速度；三点掉头覆盖 F→R→F 与相反转向，并保留停车换挡帧。
- train 用于选择生成参数；冻结参数后再生成 validation wave。不能根据 validation 失败逐样本重试到通过。

### Wave C：完整开发重审

- 合并 base_reextracted + 合格 wave 成新的 immutable development bundle。
- 重建 index/content/map aggregates，并执行 `tools/audit_p3_v3_bundle.py`；连续几何仍委托给 `tools/audit_p3_footprint_upgrade.py` 的冻结 evaluator。
- 审计必须遍历新 index 的 100% 样本，sample failure 为 0，且报告代码/地图/数据/rollout/geometry hash。
- 正式 `tools/train_dep_car.py --dry-run` 必须只显示所有 P3、coverage、index、map、training-YAML gate 为 PASS；仍不打开 test。

## P5 解锁条件

只有同时满足以下条件，才可把 `training.yaml` 中 `p5_formal_training_allowed` 改为 true 并清空对应 blocked gates：

- overall zero-feasible `<0.10`；七类 maneuver 各 `<0.25`；median feasible candidates `>=2`；
- validation MISSION/RECOVERY、七类 maneuver、前/倒挡覆盖达到冻结阈值，UNKNOWN 为 0；
- 新 index、NPZ 内容、地图合同和 P4 implementation aggregate 全部匹配；
- 正式 dry-run 无 synthetic/implementation/rollout/hash 伪失败；
- test 仍未访问。

通过这些条件只授权 P5 正式训练，不授权 P6 Gazebo 控制或 production deployment。
