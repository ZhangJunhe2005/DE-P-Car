# P5 v2 Candidate Capacity 训练迭代

日期：2026-08-17

状态：`FORMAL_REAUTHORIZATION_PASS / READY_TO_START_CANDIDATE_CAPACITY`

## 修改原因

首轮 40-epoch 三模态 Candidate Capacity 虽然完成，但不得进入 Score
Calibration。其 validation 运动学违规率为：

| 模态 | v1 运动学违规率 |
|---|---:|
| depth_only | 48.7542% |
| lidar_only | 64.7451% |
| fusion | 63.6253% |

原实现让 FP16 rollout/差分结果进入 1e-6 hard-veto，并只通过 top-3
capacity 项约束运动学；因此 candidate loss 很低并不代表全部 15 个候选可执行。
这些 v1 checkpoint 保留在 `models/dep_car/p5/` 作为诊断对照，但合同已失效，
不可转签为 PASS。

## v2 合同

- Depth/LiDAR CNN、状态编码器和 MLP 在启用 CUDA AMP 时继续使用混合精度。
- residual bounding、Ackermann rollout、signed-SDF/swept-footprint loss、
  运动学约束和 hard veto 固定为 FP32；梯度可跨越 FP32 cast 回传。
- 全部 15 个候选都增加 10% operating-margin 运动学损失；hard veto 仍使用
  校准后的真实物理上限，不放宽安全门槛。
- 单独统计 opposite motion、speed、steering、steering-rate、acceleration、
  deceleration 和 lateral-acceleration 七类违规。
- 每个 epoch 同时更新可恢复的 `last` checkpoint，并按“资格失败数、最大门槛
  超限、最坏运动学违规率、最坏 zero-feasible、最坏可行候选均值、loss”选择
  `.best.pth`。验收和 Score Calibration 默认读取 best，断点恢复只读取 last。
- v2 正式输出与 v1 隔离在 `models/dep_car/p5_v2/`。

## 512 样本 GPU 矩阵

三组均使用 RTX 5070 Ti、CUDA AMP、8 个 DataLoader worker、512 个 train 和
512 个 validation 上限、1 epoch/32 optimizer steps。所有 loss/gradient 有限，
last/best/contract/history/metrics 均成功生成。

| 模态 | validation loss | 运动学违规率 | zero-feasible | 平均可行候选 |
|---|---:|---:|---:|---:|
| depth_only | 1.31945 | 8.7891% | 2.8571% | 11.1020 |
| lidar_only | 1.31399 | 8.7891% | 2.6531% | 11.2837 |
| fusion | 1.23794 | 8.3537% | 2.6531% | 11.4490 |

违规分项中仅 forward lateral-acceleration 非零；另外六类均为 0。这个有限运行
用于验证实现与下降方向，不具备 40 epoch/10,000 step、完整 validation 分组覆盖，
因此所有 pilot 工件永久为 `UNQUALIFIED`。

P4 CUDA tiny-overfit 预检也通过：Candidate 从 `1.08914` 降至 terminal-window
`0.81175`，direct residual oracle floor 为 `0.76104`，可约 gap ratio 为
`0.15456 < 0.20`；Score 的 entropy-normalized gap ratio 为 `0.05471 < 0.20`，
top-1 oracle accuracy 从 `0.0625` 升到 `0.4375`。Candidate/Score 梯度分区、
checkpoint round-trip 和三模态 forward 均通过。该报告只写入 `/tmp`，不替代
正式 P4 复签。

## 正式训练前的门禁

本次修改改变了 loss/implementation 哈希，旧 P3/P4 签发按设计 fail-closed。
现已对全部 20,018 条 development 样本重跑 P3 V3 审计、重新生成 proposal、
完成 CUDA P4 机器验收和三模态 dry-run，并重新签发 P4。不能通过复制报告或
手改 SHA256 复现授权。完整顺序见 `p5_training_launch_guide.md`。
P4 汇总器也会直接比较 P3 报告内封存的 implementation aggregate；旧报告不再
可能在 P4 报告中被误标成 `p5_gate_eligible=true`。

最终 `reports/p4_acceptance.json` 为 `DEPCarP4AcceptanceV2 / PASS`：

- P3：20,018/20,018，zero-feasible `0.0652413`，test 未访问；
- 三模态 dry-run：全部 `DRY_RUN_READY / formal_training_authorized=true`；
- 当前代际：`p5_v2_fp32_margin_best`，正式 checkpoint 数量为 0；
- 旧 `models/dep_car/p5/` v1 checkpoint 已记录哈希并隔离为 diagnostic-only；
- 下一状态：`READY_TO_START_CANDIDATE_CAPACITY`。
