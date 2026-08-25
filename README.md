# Flow of Truth 复现：第一步 - Scatter Operator

本目录目前实现论文第 3.4 节公式 (5) 的可微像素前向散射。论文只声明
`S(image, motion)` 是可微 scatter operator，并未公开碰撞或空洞处理细节；
因此实现提供 `sum` 和 `mean` 两种策略，便于后续与作者代码对齐。

## 坐标约定

- 输入图像：`image [B,C,H,W]`
- 前向运动场：`motion [B,2,H,W]`
- `motion[:,0] = dx`，向右为正；`motion[:,1] = dy`，向下为正
- 源像素 `(x,y)` 到目标坐标 `(x+dx,y+dy)`
- 非整数目标坐标按双线性权重 splat 到四邻域
- 越界贡献丢弃；没有任何贡献的位置用 `fill_value` 填充

## 使用

```python
import torch
from scatter_operator import scatter_operator

image = torch.rand(2, 3, 256, 256, device="cuda")
motion = torch.randn(2, 2, 256, 256, device="cuda") * 4.0

# 默认 mean：碰撞像素按累计权重归一化，适合生成自然图像。
warped = scatter_operator(image, motion)

# sum：严格的双线性加权 forward splat。
warped_sum = scatter_operator(image, motion, reduction="sum")

# weight/mask 可用于识别空洞，并在后续 loss 中屏蔽无效区域。
result = scatter_operator(image, motion, return_aux=True)
warped, weight, valid_mask = result
```

`grid_sample` 是反向采样（为每个目标像素查找源坐标），不能直接表达一对多的
前向散射。本实现借鉴它的双线性插值思想，但通过 `Tensor.scatter_add` 聚合四邻域
贡献，完全由 PyTorch 算子组成，可直接在 CUDA 上执行和反向传播。

## 验证

安装与你的出租 GPU 驱动匹配的官方 PyTorch 后运行：

```powershell
python -m pytest -q
```

测试覆盖恒等运动、整数平移、亚像素双线性权重、像素碰撞、CPU 数值梯度检查，
并在 CUDA 可用时额外执行 GPU 前向与反向传播测试。

## 第二步：Template Embedding

`template_embedding.py` 实现论文第 3.3 节中的可学习模板
`T [C,H,W]`、残差 U-Net Encoder `E(I_0,T)`，以及公式 (4) 的
MSE + LPIPS 图像保真损失。

```python
import torch
from template_embedding import ImageFidelityLoss, TemplateEmbedding

device = "cuda"
model = TemplateEmbedding(
    channels=3,
    height=256,
    width=256,
    base_channels=32,
    residual_scale=0.05,
).to(device)
criterion = ImageFidelityLoss(lambda_mse=1.0, lambda_lpips=1.0).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

original = torch.rand(4, 3, 256, 256, device=device)  # 输入范围 [0,1]
embedded = model(original)
losses = criterion(original, embedded)

optimizer.zero_grad(set_to_none=True)
losses.total.backward()
optimizer.step()  # Encoder 与 T 会被联合更新

print(losses.total.item(), losses.mse.item(), losses.lpips.item())
```

安装 LPIPS：

```bash
pip install lpips
```

LPIPS 的预训练参数被冻结，但保留对输入的梯度。ResUNet 最后一层以零初始化，
因此初始输出严格等于原图；第一步先更新输出层，之后梯度会继续传入深层特征与 T。

## 第三步：Motion Capture 的概率损失与置信图

`motion_capture.py` 实现第 3.5 节公式 (6) 的 Mixture-of-Laplace NLL 和
公式 (8) 的 Confidence Map：

```python
from motion_capture import confidence_map, mixture_of_laplace_nll

# flow_pred/flow_gt: [B,2,H,W]
# alpha/beta2: [B,1,H,W]；论文指定 beta1 固定为 0。
loss_motion = mixture_of_laplace_nll(
    flow_pred,
    flow_gt,
    alpha,
    beta1=0.0,
    beta2=beta2,
    valid_mask=valid_mask,       # 可选 [B,1,H,W]
    alpha_is_logits=False,
)

confidence = confidence_map(
    alpha,
    beta1=0.0,
    beta2=beta2,
    normalization="minmax",
)

loss_total = losses.total + lambda_motion * loss_motion
```

如果运动估计头直接输出未激活的 alpha logit，推荐设置
`alpha_is_logits=True`，避免在模型外手动 sigmoid 后再取对数造成数值损失。
`confidence_map` 默认对每个样本做空间 min-max 归一化；论文只写了 `Norm`，
没有进一步规定具体归一化形式，因此也提供 `max` 和 `none` 选项。

## 第四步：Flow Reversal

`flow_reversal.py` 实现第 3.6 节公式 (10) 的双线性 backward warp，以及
公式 (11) 的多帧置信度加权融合：

```python
from flow_reversal import backward_warp, confidence_weighted_fusion

warped_frames = []
effective_confidences = []

for frame_t, flow_0_to_t, confidence_t in zip(frames, flows, confidences):
    result = backward_warp(
        frame_t,                 # [B,C,H,W]
        flow_0_to_t,             # [B,2,H,W]，像素单位
        return_valid_mask=True,
    )
    warped_frames.append(result.image)
    effective_confidences.append(
        confidence_t * result.valid_mask.to(confidence_t.dtype)
    )

truth = confidence_weighted_fusion(
    warped_frames,
    effective_confidences,
)  # [B,C,H,W]
```

也可以使用一步式接口：

```python
from flow_reversal import reverse_and_fuse

truth = reverse_and_fuse(frames, flows, confidences)
```

这里 `F_0_to_t(x,y)` 表示原图像素 `(x,y)` 在第 t 帧中的位移，因此回溯时
使用 `frame_t(x + dx, y + dy)`。越界采样默认填 0，一步式接口会自动将越界
mask 乘入置信度，防止这些 padding 像素污染最终融合结果。

## I2V 模拟：冻结 VAE 压缩与重建

`i2v_simulation.py` 使用 Diffusers 的
`stabilityai/sd-vae-ft-mse` AutoencoderKL 模拟 I2V 的压缩重建伪影：

```bash
pip install diffusers transformers accelerate safetensors
```

```python
import torch

from i2v_simulation import FrozenVAEReconstructor
from scatter_operator import scatter_operator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_simulator = FrozenVAEReconstructor(
    torch_dtype=torch.float32,
    enable_slicing=True,
).to(device)

# 独立调试用假输入。正式训练时替换为 Template Encoder 的输出 I_0^T。
embedded = torch.rand(1, 3, 256, 256, device=device)
motion = torch.zeros(1, 2, 256, 256, device=device)

reconstructed = vae_simulator.reconstruct(embedded)

# hat(I_0^T) 随后进入已实现的 Scatter Operator。
forged = scatter_operator(reconstructed, motion, reduction="mean")

print(reconstructed.shape, reconstructed.min().item(), reconstructed.max().item())
print(forged.shape)
```

VAE 参数被冻结且始终保持 eval，但没有使用 `torch.no_grad()`：这确保图像损失或
运动损失仍可穿过 VAE 回传到 Template Encoder。默认使用 posterior mode 得到
确定性重建；设置 `sample_posterior=True` 可模拟随机潜变量采样。

项目还提供了可直接运行的完整检查脚本：

```bash
python simulation_debug.py
```

如果计算节点不能连接 Hugging Face，请先在可联网节点下载：

```bash
hf download stabilityai/sd-vae-ft-mse \
  --local-dir /data/$USER/models/sd-vae-ft-mse
```

然后在离线计算节点运行：

```bash
python simulation_debug.py \
  --model /data/$USER/models/sd-vae-ft-mse \
  --local-files-only
```

## 完整复现入口

本仓库现已补齐计划书中的端到端工程：`template/`、`i2v/`、`flow/`、
`recovery/`、`demo/`、训练与评估脚本。底层论文公式仍保留在仓库根目录，目录版
模块是稳定导入接口。

```bash
# 1. 安装与你的 CUDA 匹配的 PyTorch，再安装其余依赖
pip install -r requirements.txt

# 2. 不下载大模型，先验证完整计算图和形状
python simulation_debug.py

# 3. 基础复现数据：COCO2017-val + Sintel GT flow（约 4.1 GB 下载）
bash scripts/download_fot_mini.sh /data/lvzhengshu/FOT

# 4. 联合训练 Template 与 Motion Capture（论文统一使用 512x512）
python train.py \
  --data data/processed/fot-mini/train_images.txt \
  --val-data data/processed/fot-mini/val_images.txt \
  --flow-data data/processed/fot-mini/train_flows.txt \
  --val-flow-data data/processed/fot-mini/val_flows.txt \
  --output-dir checkpoints/fot-mini-512 \
  --size 512 \
  --epochs 20 \
  --batch-size 4 \
  --num-frames 4 \
  --local-files-only

# 5. 离线轻量 Demo
python run_demo.py --mock

# 6. 使用训练权重运行正式 SVD + Motion Capture Demo
python run_demo.py \
  --checkpoint checkpoints/fot-mini-512/best.pt \
  --local-files-only

# 7. 评估恢复结果
python evaluate.py original.png recovered.png --lpips
```

论文规模训练使用 118K COCO 图像和 85K 个光流样本。本仓库的 `fot-mini`
配置只面向基础方法复现：确定性划分 4,500 张训练图、500 张验证图，以及
Sintel 训练集约 1,041 个真实光流场；在线仿射流仍可在不传 `--flow-data` 时
使用。数据清单和 `metadata.json` 由 `prepare_data.py` 生成，正式测试图片不进入
上述训练/验证清单。

批量图片测评要求参考图与恢复图使用相同的相对路径和文件名：

```bash
python evaluate.py \
  --reference-dir data/test \
  --recovered-dir outputs/fot-256/recovered \
  --lpips --clip --local-files-only \
  --output results/fot-256.json
```

命令同时写出 JSON 汇总和逐图片 CSV，包括 PSNR、标准 SSIM、LPIPS 与
CLIP Similarity 的均值、标准差、最小值和最大值。

正式训练链路为：

```text
Original
  -> TemplateEmbedding
  -> FrozenVAEReconstructor
  -> Differentiable Scatter + known Sintel/affine flows
  -> MotionCaptureNet(flow, alpha, beta2)
  -> confidence-guided Flow Reversal
  -> Recovered Truth
```

总损失包含保护图 MSE/LPIPS、Mixture-of-Laplace 运动 NLL，以及恢复图
L1/LPIPS。`train.py` 默认启用 A100 适用的 BF16，逐 epoch 写出原子的
`last.pt` 和 `best.pt`。用下面的命令从中断位置继续：

```bash
python train.py \
  --data data/processed/fot-mini/train_images.txt \
  --val-data data/processed/fot-mini/val_images.txt \
  --flow-data data/processed/fot-mini/train_flows.txt \
  --val-flow-data data/processed/fot-mini/val_flows.txt \
  --output-dir checkpoints/fot-mini-512 \
  --resume checkpoints/fot-mini-512/last.pt \
  --local-files-only
```

`MotionCaptureNet.forward_video` 会分块处理视频，避免 14 帧的 SVD 原生
`576x1024` 输出一次性占满显存。没有 checkpoint 时，正式 Demo 仍可用
torchvision RAFT-Small + 光度置信度作为基线；`--mock` 只用于环境检查，不代表
论文指标。
