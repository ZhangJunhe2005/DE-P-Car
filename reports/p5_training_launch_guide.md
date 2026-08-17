# P5 两阶段三模态训练启动说明

签发时间：2026-08-16

## 当前边界

- `reports/p4_acceptance.json` 为 `DEPCarP4AcceptanceV2 / PASS`。
- P3 V3 curated 开发集共 20,018 条：train 16,394、validation 3,624；train 31 张地图、validation 5 张地图。
- `depth_only`、`lidar_only`、`fusion` 三个 Candidate Capacity dry-run 均为 `DRY_RUN_READY`，且使用 CUDA、8 个 DataLoader worker 和 8 个 PyTorch CPU thread。
- 当前没有 `models/dep_car/p5/*.pth`，正式 P5 训练尚未启动。
- P5 训练输出仍是 `UNQUALIFIED`；通过本阶段训练和候选验收不等于获得 Gazebo 或生产部署资格。

所有长时间训练均由宿主机用户启动。建议三种模态串行运行，避免三个任务同时争用一张 GPU。每个 epoch 会向终端输出一行 JSON，并同步保存到 `logs/p5/`。

## 第 1 步：Candidate Capacity

在宿主机终端运行：

```bash
cd /home/zjh/DE-P-Car

bash scripts/run_p5_training.sh \
  --stage candidate_capacity --modality depth_only --workers 8

bash scripts/run_p5_training.sh \
  --stage candidate_capacity --modality lidar_only --workers 8

bash scripts/run_p5_training.sh \
  --stage candidate_capacity --modality fusion --workers 8
```

默认输出：

- `models/dep_car/p5/depth_only_candidate_capacity.pth`
- `models/dep_car/p5/lidar_only_candidate_capacity.pth`
- `models/dep_car/p5/fusion_candidate_capacity.pth`

训练器默认执行 40 epoch，启用 CUDA AMP；每个 checkpoint 同时生成 contract、history、metrics 和状态为 `PENDING_EXTERNAL_ACCEPTANCE` 的候选验收 sidecar。不要手工修改这些文件。

如训练中断，源 checkpoint 与新输出必须是不同路径，例如：

```bash
bash scripts/run_p5_training.sh \
  --stage candidate_capacity --modality fusion \
  --resume models/dep_car/p5/fusion_candidate_capacity.pth \
  --output models/dep_car/p5/fusion_candidate_capacity_resumed.pth \
  --workers 8
```

后续验收和评分训练应通过 `--source` 指向最终的 resumed checkpoint。

## 第 2 步：逐模态候选验收

三个 Candidate Capacity 任务各自完成后运行：

```bash
bash scripts/run_p5_training.sh \
  --stage accept_candidate --modality depth_only

bash scripts/run_p5_training.sh \
  --stage accept_candidate --modality lidar_only

bash scripts/run_p5_training.sh \
  --stage accept_candidate --modality fusion
```

命令必须退出码为 0，并在输出中显示 `"gate_passed": true`，才可以训练对应模态的 Score Calibration。验收会重新计算真实 validation 指标及正反挡覆盖，不接受 smoke、部分 epoch、哈希不一致或未达到 40 epoch/10,000 step 的候选权重。

如果最终候选使用了自定义或 resumed 路径，需显式传入：

```bash
bash scripts/run_p5_training.sh \
  --stage accept_candidate --modality fusion \
  --source models/dep_car/p5/fusion_candidate_capacity_resumed.pth
```

## 第 3 步：Score Calibration

只有对应候选验收 PASS 后才运行：

```bash
bash scripts/run_p5_training.sh \
  --stage score_calibration --modality depth_only --workers 8

bash scripts/run_p5_training.sh \
  --stage score_calibration --modality lidar_only --workers 8

bash scripts/run_p5_training.sh \
  --stage score_calibration --modality fusion --workers 8
```

默认输出为 `models/dep_car/p5/<modality>_score_calibration.pth`。该阶段会在启动时实时复核候选 checkpoint、contract、metrics 和 acceptance sidecar；Candidate 分支被冻结，只训练 Score 分支。

如果候选采用自定义路径：

```bash
bash scripts/run_p5_training.sh \
  --stage score_calibration --modality fusion \
  --source models/dep_car/p5/fusion_candidate_capacity_resumed.pth \
  --workers 8
```

Score 阶段中断时同样使用不同的新输出路径：

```bash
bash scripts/run_p5_training.sh \
  --stage score_calibration --modality fusion \
  --resume models/dep_car/p5/fusion_score_calibration.pth \
  --output models/dep_car/p5/fusion_score_calibration_resumed.pth \
  --workers 8
```

## 结果回传

每一阶段运行结束后，请保留终端最后的 JSON，并把对应 `logs/p5/*.log`、checkpoint 同名 `.contract.json`、`.history.json`、`.metrics.json` 和 `.candidate_acceptance.json`（候选阶段）交给验收流程。不要提前启动 P6 Gazebo learned-policy 控制。
