# P5 AMP encoder/token dtype regression

> 历史 v1 报告：仅修复 encoder/token 拼接 dtype，未解决 FP16 物理计算与
> 全 15 候选运动学约束。当前权威为 `p5_v2_candidate_iteration.md`。

Date: 2026-08-16

Status: **PASS**

## Failure reproduced

The first formal `depth_only` Candidate Capacity run failed before its first
optimizer update with:

```text
index_copy_(): self and source expected to have the same dtype,
but got (self) Float and (source) Half
```

CUDA autocast produced FP16 Depth/LiDAR encoder features, while the learned
missing-modality tokens correctly remained FP32 checkpoint parameters.
`Tensor.index_copy` does not perform dtype promotion.

## Repair

`tools/train_dep_car.py::_amp_encoder_output_fp32` keeps the sensor convolution
kernels under CUDA AMP, but casts the two encoder outputs to FP32 before the
model joins present features with FP32 missing-modality tokens. Gradients cross
the cast. Model parameters, checkpoint dtypes, inference behavior and the
signed P4 model implementation contract are unchanged.

Trainer SHA-256 after repair:

```text
0ec39f0e3f951673d34c088f99fc94dc061284fca7211aa9b8d02ba20eda0562
```

## GPU regression matrix

All runs used the frozen P3 V3 curated authority, CUDA AMP, eight DataLoader
workers, two train/two validation samples and exactly one optimizer step.
Artifacts were written only beneath
`/tmp/dep_car_p5_amp_smoke.OxKu2b` and are permanently smoke-limited and
`UNQUALIFIED`.

| Stage | depth_only | lidar_only | fusion |
|---|---|---|---|
| Candidate Capacity | PASS, step 1 | PASS, step 1 | PASS, step 1 |
| Score Calibration | PASS, step 1 | PASS, step 1 | PASS, step 1 |

No formal checkpoint was written beneath `models/dep_car/p5/`. The three
formal entry dry-runs were regenerated after the repair and all report
`DRY_RUN_READY`; the canonical P4 acceptance was then reissued with the new
trainer and dry-run hashes.

## Automated regression

`tests/test_p5_train_entry.py` covers mixed Depth-only, LiDAR-only and Fusion
rows under CPU autocast and verifies hook cleanup plus FP32 token preservation.
The complete DE-P-Car Python suite passes: `263 passed`.
