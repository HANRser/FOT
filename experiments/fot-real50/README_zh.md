# FoT real50 轻量复现测试结果

## 测试范围

本次测试使用 `Graphs` 分支中冻结的 `fot_real50` 数据集，共 50 张独立现实照片，
包括 Animal、Camera、Face、Human-Environment 和 Multi-Human 五类，每类 10 张。
模型输入统一为 512×512 RGB PNG。

测试使用轻量训练的最佳 checkpoint：

```text
checkpoints/fot-mini-512/best.pt
SHA-256: 670b6f8f354917e7f9ffffbbc8f3ef20363bdbbada73344d2925af90478e6c59
epoch: 5（checkpoint 内部从 0 计数为 4）
global step: 5625
```

图像经过保护、Stable Video Diffusion 14 帧生成、运动估计和真值恢复。50 张图片分成
10 批，每批 5 张，以断点续跑方式完成。I2V 使用固定随机种子 0，指标为
PSNR、SSIM、LPIPS 和 CLIP image similarity。

## 完整性审计

结果审计通过，没有跳过测试图片：

- 输入：50 张；
- Protected、Forged、Recovered、Confidence：各 50 张；
- `run.json`：50 条成功记录，0 条错误；
- 10 个批次各包含 5 张，批次并集为全部 50 张；
- 无缺失、额外、重复或不可读取的输出文件；
- 完整指标报告均包含 50 张图片。

机器可读审计结果见 [`raw/result_audit.json`](raw/result_audit.json)。

## 总体指标

| 阶段 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP similarity ↑ |
| --- | ---: | ---: | ---: | ---: |
| Protected vs. Original | 35.5194 | 0.9391 | 0.0792 | 0.9605 |
| Recovered vs. Original | 19.6923 | 0.5655 | 0.4290 | 0.8749 |

保护图与原图的整体差异较小，说明轻量 checkpoint 基本保持了输入图像的视觉质量。
恢复质量存在明显场景差异，说明当前轻量训练已跑通完整保护和恢复链路，但还没有达到
稳定的高质量恢复水平。

## 分场景恢复指标

| 场景 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP similarity ↑ |
| --- | ---: | ---: | ---: | ---: |
| Animal | 19.3098 | 0.5499 | 0.4383 | 0.8996 |
| Camera | 22.9283 | 0.6865 | 0.3645 | 0.9019 |
| Face | 22.6511 | 0.6752 | 0.3863 | 0.8428 |
| Human-Environment | 18.8795 | 0.5336 | 0.3960 | 0.8692 |
| Multi-Human | 14.6928 | 0.3820 | 0.5597 | 0.8612 |

Camera 和 Face 的像素恢复指标最好；Multi-Human 最弱。这个趋势与多人非刚性运动、
遮挡和空间异质性更难建模的预期一致。各类别的保护与恢复完整指标见
[`category_summary.csv`](category_summary.csv)。

## 运行情况

- 50 张推理累计耗时：2007.05 秒；
- 单张平均耗时：40.14 秒；
- 单张范围：26.62–53.19 秒；
- FoT 进程记录的峰值 CUDA 分配：15.99 GiB；
- 全程无 OOM、无推理错误。

上述时间不包括每批模型重新加载、CPU 指标计算和等待共享 GPU 空闲的时间。

## 结果文件

- `raw/formal/metrics_protected.json/csv`：50 张保护图逐图指标及汇总；
- `raw/formal/metrics_recovered.json/csv`：50 张恢复图逐图指标及汇总；
- `raw/formal/run.json`：checkpoint、数据清单哈希、逐图耗时和显存记录；
- `raw/chunks/chunk_001` 至 `chunk_010`：每批 5 张的独立指标；
- `raw/data_validation.json`：测试集与实际 COCO 训练副本的泄漏和重复检查；
- `raw/result_audit.json`：输出完整性审计。

完整的 200 张生成结果保留在 LDS：

```text
/data/lvzhengshu/FOT/outputs/fot-real50/formal/
```

仓库只保存小体积的指标、审计和运行元数据，不提交生成 PNG、模型权重、数据集副本或
模型缓存。

## 结论与边界

本次结果证明轻量复现已完成独立现实图片上的端到端运行和量化测试。它属于基础复现，
不能直接等同于论文的完整训练规模或正式基准结果。后续若要提高恢复质量，应优先扩大
运动训练覆盖、加强 Multi-Human 与 Human-Environment 场景，并与论文设置下的攻击、
生成器和消融实验进行统一比较。
