# P5 Score Head optimized preflight and launch guide

Status: `READY_FOR_HOST_LONG_TRAINING`

This Score-only execution path leaves the accepted Candidate trainer,
`training.yaml`, and the P4 implementation aggregate unchanged.  The optimized
trainer has a separate performance contract and records its own trainer/data
hashes in every Score checkpoint and contract.

## Completed gates

- Candidate external acceptance: `PASS` for `depth_only`, `lidar_only`, and
  `fusion`.
- Formal CUDA dry-run: `DRY_RUN_READY` for all three modalities with batch 64.
- Dataset equivalence: the effective original/optimized tensors were identical
  for all three modalities on sampled real validation frames.
- CUDA AMP numerical equivalence: maximum absolute difference was `0.0` for
  inputs, all five network outputs, all objective terms, hard feasibility, and
  every kinematic constraint component.
- GPU pilot: four training batches plus 256-frame validation completed for all
  three modalities and wrote independently attested best/last artifacts.
- Relevant regression suite: `107 passed`.
- Sealed test split: never opened.

Canonical reports:

- `reports/p5_score_optimized_depth_only_dry_run.json`
- `reports/p5_score_optimized_lidar_only_dry_run.json`
- `reports/p5_score_optimized_fusion_dry_run.json`
- `reports/p5_score_optimized_numerics.json`

## Batch-size decision

Equal-sample Fusion smoke benchmark (472 geometry-valid training frames plus
490 validation frames):

| Batch | Elapsed (s) | Processed samples/s | Validation oracle regret |
|---:|---:|---:|---:|
| 16 | 6.584 | 146.1 | 0.0669 |
| 64 | 4.527 | 212.5 | 0.0661 |
| 128 | 4.615 | 208.5 | 0.0977 |

Batch 64 is the signed formal setting.  It improved end-to-end short-run
throughput by about 45% over batch 16; batch 128 provided no further gain and
had worse short-pilot ranking regret.

## Formal long-training commands

Run these sequentially on the host.  Each command uses CUDA, AMP, 8 loader
workers, prefetch factor 4, batch 64, and the full 40-epoch development view.

```bash
cd /home/zjh/DE-P-Car

bash scripts/run_p5_score_optimized.sh \
  --stage score_calibration --modality depth_only
```

```bash
cd /home/zjh/DE-P-Car

bash scripts/run_p5_score_optimized.sh \
  --stage score_calibration --modality lidar_only
```

```bash
cd /home/zjh/DE-P-Car

bash scripts/run_p5_score_optimized.sh \
  --stage score_calibration --modality fusion
```

Do not run the three jobs concurrently on one GPU.  Progress is printed after
each completed epoch and is also written under `logs/p5_score_v1/`.

Formal outputs are written under `models/dep_car/p5_score_v1/`.  That directory
is intentionally ignored by Git because it contains large learned artifacts.

## Resume example

Resume from the last fully written artifact into a new output path.  Never use
the same path for source and output.

```bash
cd /home/zjh/DE-P-Car

bash scripts/run_p5_score_optimized.sh \
  --stage score_calibration \
  --modality fusion \
  --resume models/dep_car/p5_score_v1/fusion_score_calibration.pth \
  --output models/dep_car/p5_score_v1/fusion_score_calibration_resume.pth
```

After all three long jobs complete, inspect the best/last histories and run the
Score ranking acceptance analysis before proceeding to P6 Gazebo shadow mode.
