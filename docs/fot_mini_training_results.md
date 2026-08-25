# FoT Mini 512 Training Results

This report records the compact reproduction run completed on the LDS A100
server. It makes the current training stage auditable without committing
downloaded datasets, model caches, full logs, or checkpoint binaries.

## Scope

- Cover images: 4,500 train / 500 validation images from COCO2017-val.
- Motion bank: 937 train / 104 validation ground-truth flows from MPI Sintel.
- Resolution: 512 x 512.
- Model: TemplateEmbedding and MotionCaptureNet with 32 base channels.
- Frozen channel simulator: `stabilityai/sd-vae-ft-mse`.
- Optimization: AdamW, BF16, batch size 4, four frames, five epochs.
- Hardware: one NVIDIA A100-SXM4-80GB.

The exact configuration is stored in
[`configs/fot_mini_512.json`](../configs/fot_mini_512.json), and the full
machine-readable metrics are in
[`experiments/fot-mini-512/metrics.json`](../experiments/fot-mini-512/metrics.json).

## Epoch Results

| Epoch | Global step | Train total | Validation total | Best validation |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,125 | 1.4495 | 0.8482 | 0.8482 |
| 2 | 2,250 | 0.6527 | 0.6037 | 0.6037 |
| 3 | 3,375 | 0.4583 | 0.6964 | 0.6037 |
| 4 | 4,500 | 0.3547 | 0.4711 | 0.4711 |
| 5 | 5,625 | 0.2938 | 0.3997 | 0.3997 |

Final validation components:

| Component | Value |
| --- | ---: |
| Fidelity MSE | 0.0002843 |
| Fidelity LPIPS | 0.0619023 |
| Motion loss | 0.3489479 |
| Recovery L1 | 0.0319033 |
| Recovery LPIPS | 0.1233320 |

All 39 repository tests passed after data integration. Training completed all
5,625 steps without an OOM, NaN, or traceback.

## Checkpoints

Checkpoint binaries remain outside Git because they are generated artifacts.
On LDS they are located at:

```text
/data/lvzhengshu/FOT/checkpoints/fot-mini-512/best.pt
/data/lvzhengshu/FOT/checkpoints/fot-mini-512/last.pt
```

SHA-256:

```text
best.pt  670b6f8f354917e7f9ffffbbc8f3ef20363bdbbada73344d2925af90478e6c59
last.pt  d45e261f4704f8f179b090335d878fdd88117b60813dc01ee33c87dc7039ea73
```

Both checkpoint files were reopened on CPU and verified to contain epoch 5
state (`epoch=4` in the zero-based checkpoint field), global step 5,625, and
best metric 0.3996627938747406.

## Interpretation and Next Stage

These numbers validate the compact training pipeline on held-out COCO images
and held-out Sintel flows. They are not formal truth-recovery results on real
I2V videos and should not be compared directly with the paper's benchmark.

The next stage is to keep an independent source-image test set out of all model
selection, protect those images with `best.pt`, generate I2V videos, recover the
sources, and report PSNR, SSIM, LPIPS, and CLIP Similarity.
