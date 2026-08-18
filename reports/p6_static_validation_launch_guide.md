# P6 静态 Gazebo 验证启动指南

## 当前资格边界

- 三个 P5 Score Head checkpoint 均保持原 SHA-256，不需要为 ROS 适配重新训练。
- P6 首先运行 `shadow`：DEPCarNetV1 生成并排序 15 条双向 Ackermann 候选，但确定性规划器仍控制车辆。
- 每条学习候选必须重新通过运动学、静态连续车体和动态预测 hard veto。
- 只有 `shadow` 审计签发 PASS sidecar 后，同一 checkpoint 与同一运行时代码才可进入 Gazebo `active`。
- P6 active 只授权仿真，不代表 P8 production-qualified。

## 1. 准备并冻结场景

先检查命令，不生成地图：

```bash
cd /home/zjh/DE-P-Car
bash scripts/run_p6_static.sh --stage prepare --workers 8 --dry-run
```

正式生成 40 张固定 seed 地图并冻结 35 个场景；这是长任务，由宿主机终端执行：

```bash
cd /home/zjh/DE-P-Car
bash scripts/run_p6_static.sh --stage prepare --workers 8
```

独立校验地图、UUID、seed、world/map 哈希和三模态 checkpoint：

```bash
bash scripts/run_p6_static.sh --stage validate
```

对冻结的 35 个起点执行 `±0.02 m / ±0.035 rad` 共 27 点扰动审计，并把证据写入场景清单：

```bash
bash scripts/run_p6_static.sh --stage start_audit
```

本轮结果为 34/35 鲁棒；`p6_cb4a93cedae5f755` 被保留作诊断但不会进入默认 interactive 或 gate-suite。新场景在冻结前即执行同一门禁。

生成器支持断点复用；重复执行 `prepare` 会校验并跳过身份一致的已有地图。

## 2. RViz 任意目标交互验证

启动一个 development 场景，自动使用 Fusion checkpoint 和 shadow 权限：

```bash
bash scripts/run_p6_static.sh \
  --stage interactive \
  --cohort development \
  --maximum-scenarios 1 \
  --modality fusion
```

在 RViz 选择 `2D Nav Goal` 后设置任意目标。橙色路径是实际控制候选，蓝色路径是经过 hard veto 的学习候选。按 `Ctrl+C` 会依次发送 SIGINT、SIGTERM；只有超时未退出的进程才会升级到 SIGKILL。

未显式指定 `--scenario` 时，interactive 会优先选择起点最鲁棒、净空最大的 `NORMAL` development 场景。在线全局层发布的是宽度膨胀后的 2-D 拓扑走廊，用于保证连通性和避免死胡同；它不再把 Hybrid A* 的逐点转角和挡位当作控制指令。局部规划器以走廊子目标为软引导，对前进/倒车候选实施连续 footprint hard veto，并在首选挡位整组不可行时测试另一挡位。

全局状态含义：`INVALID_START` 表示实测车身起点非法，`START_BLOCKED` 表示起点合法但 10 个短 Ackermann 原语均不可行，`INVALID_GOAL` 表示终点车身放不下，`READY_CORRIDOR` 表示可达走廊已发布。远目标不再等待 5 秒 Hybrid A* 精确航向搜索。

## 3. 固定场景 shadow 验证

先 dry-run 一个可复现场景：

```bash
bash scripts/run_p6_static.sh \
  --stage shadow \
  --cohort development \
  --maximum-scenarios 1 \
  --modality fusion \
  --dry-run
```

再真实运行一个场景并显示 Gazebo/RViz：

```bash
bash scripts/run_p6_static.sh \
  --stage shadow \
  --cohort development \
  --maximum-scenarios 1 \
  --modality fusion \
  --gui --rviz
```

运行满足审计门槛的最小确定性场景集（至少 14 个，并覆盖全部七种动作）：

```bash
bash scripts/run_p6_static.sh \
  --stage shadow \
  --gate-suite \
  --modality fusion
```

已成功的报告会自动跳过；需要重跑时加 `--rerun`。单独固定某个场景可使用 `--scenario <scenario_id>`。

## 4. 审计与 active

shadow 长任务完成后签发审计：

```bash
bash scripts/run_p6_static.sh --stage audit --modality fusion
```

仅当输出 `status: PASS` 后，才运行 holdout active 最小资格集：

```bash
bash scripts/run_p6_static.sh \
  --stage active \
  --gate-suite \
  --modality fusion
```

默认 active 不允许确定性 fallback；模型未加载、sidecar 不匹配、运行时代码被修改或没有安全候选时，小车制动并报告原因。
