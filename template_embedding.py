"""Flow of Truth 第 3.3 节：可学习模板与模板嵌入网络。

论文给出了 T、E(I_0, T) 和图像保真损失的定义，但没有公开 Encoder 的逐层
结构。本实现使用残差 U-Net，并将网络输出解释为对原图的小幅残差。
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_count(channels: int, maximum: int = 8) -> int:
    """选择能够整除通道数的 GroupNorm 分组数。"""
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class LearnableTemplate(nn.Module):
    """论文中的全局可学习模板 T，参数本身严格为 [C,H,W]。"""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if min(channels, height, width) <= 0:
            raise ValueError("channels、height、width 必须为正整数")
        self.channels = channels
        self.height = height
        self.width = width
        self.template = nn.Parameter(torch.empty(channels, height, width))
        nn.init.normal_(self.template, mean=0.0, std=init_std)

    def forward(self, batch_size: int) -> Tensor:
        """共享同一个 T，并展开成可与 batch 图像拼接的 [B,C,H,W]。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        return self.template.unsqueeze(0).expand(batch_size, -1, -1, -1)


class ResidualBlock(nn.Module):
    """两层卷积残差块；GroupNorm 对小 batch 比 BatchNorm 更稳定。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class UpBlock(nn.Module):
    """上采样后与同尺度 Encoder 特征拼接，再进行残差融合。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fuse = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # 指定 skip 的尺寸，因此输入 H/W 不必严格为 8 的倍数。
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.project(x)
        return self.fuse(torch.cat((x, skip), dim=1))


class ResUNetEncoder(nn.Module):
    """实现论文中的 E：输入 [I_0,T]，输出 I_0^T。

    ``image`` 和 ``template`` 在通道维拼接。U-Net 预测有界残差，最终输出为
    ``clamp(image + residual_scale * tanh(delta), output_range)``。
    """

    def __init__(
        self,
        image_channels: int = 3,
        template_channels: int = 3,
        base_channels: int = 32,
        residual_scale: float = 0.05,
        output_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        if min(image_channels, template_channels, base_channels) <= 0:
            raise ValueError("通道数必须大于 0")
        if residual_scale <= 0:
            raise ValueError("residual_scale 必须大于 0")
        if output_range is not None and output_range[0] >= output_range[1]:
            raise ValueError("output_range 必须满足 min < max")

        self.image_channels = image_channels
        self.template_channels = template_channels
        self.residual_scale = residual_scale
        self.output_range = output_range

        b = base_channels
        self.stem = ResidualBlock(image_channels + template_channels, b)
        self.down1 = ResidualBlock(b, 2 * b, stride=2)
        self.down2 = ResidualBlock(2 * b, 4 * b, stride=2)
        self.down3 = ResidualBlock(4 * b, 8 * b, stride=2)
        self.bottleneck = ResidualBlock(8 * b, 8 * b)
        self.up2 = UpBlock(8 * b, 4 * b, 4 * b)
        self.up1 = UpBlock(4 * b, 2 * b, 2 * b)
        self.up0 = UpBlock(2 * b, b, b)
        self.head = nn.Conv2d(b, image_channels, kernel_size=3, padding=1)

        # 初始时 E(I,T)=I，避免训练开始即产生明显视觉扰动。
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image: Tensor, template: Tensor) -> Tensor:
        if image.ndim != 4 or template.ndim != 4:
            raise ValueError("image 与 template 都必须是 [B,C,H,W]")
        if image.shape[0] != template.shape[0] or image.shape[2:] != template.shape[2:]:
            raise ValueError("image 与 template 的 B、H、W 必须一致")
        if image.shape[1] != self.image_channels:
            raise ValueError(f"image 通道数应为 {self.image_channels}")
        if template.shape[1] != self.template_channels:
            raise ValueError(f"template 通道数应为 {self.template_channels}")
        if image.device != template.device:
            raise ValueError("image 与 template 必须位于同一设备")

        x0 = self.stem(torch.cat((image, template.to(image.dtype)), dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.bottleneck(self.down3(x2))
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        residual = self.residual_scale * torch.tanh(self.head(x))
        embedded = image + residual

        if self.output_range is not None:
            embedded = embedded.clamp(*self.output_range)
        return embedded


class TemplateEmbedding(nn.Module):
    """联合持有可学习 T 与 Encoder E 的完整第 3.3 节模块。"""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        *,
        base_channels: int = 32,
        residual_scale: float = 0.05,
        output_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        template_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.learnable_template = LearnableTemplate(
            channels, height, width, init_std=template_init_std
        )
        self.encoder = ResUNetEncoder(
            image_channels=channels,
            template_channels=channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
            output_range=output_range,
        )

    @property
    def template(self) -> nn.Parameter:
        """直接访问形状 [C,H,W] 的参数 T。"""
        return self.learnable_template.template

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError("image 必须是 [B,C,H,W]")
        expected = (
            self.learnable_template.channels,
            self.learnable_template.height,
            self.learnable_template.width,
        )
        if image.shape[1:] != expected:
            raise ValueError(
                f"image 的 [C,H,W] 应为 {expected}，实际为 {tuple(image.shape[1:])}"
            )
        template = self.learnable_template(image.shape[0])
        return self.encoder(image, template)


class FidelityLossOutput(NamedTuple):
    total: Tensor
    mse: Tensor
    lpips: Tensor


class ImageFidelityLoss(nn.Module):
    """公式 (4)：lambda_1 * MSE + lambda_2 * LPIPS。

    默认使用 ``lpips`` 包的预训练网络。传入 ``lpips_model`` 可用于离线环境、
    单元测试或替换为项目指定的感知网络。LPIPS 参数被冻结，但关于输入图像的
    计算图会保留，因此梯度仍会传回 Encoder 和模板 T。
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_lpips: float = 1.0,
        *,
        lpips_net: str = "alex",
        lpips_model: Optional[nn.Module] = None,
        input_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        if lambda_mse < 0 or lambda_lpips < 0:
            raise ValueError("损失权重不能为负")
        if input_range[0] >= input_range[1]:
            raise ValueError("input_range 必须满足 min < max")

        if lpips_model is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError(
                    "计算公式 (4) 需要 lpips：请执行 `pip install lpips`"
                ) from exc
            lpips_model = lpips.LPIPS(net=lpips_net)

        self.lambda_mse = float(lambda_mse)
        self.lambda_lpips = float(lambda_lpips)
        self.input_range = input_range
        self.lpips_model = lpips_model.eval()
        for parameter in self.lpips_model.parameters():
            parameter.requires_grad_(False)

    def _to_lpips_range(self, image: Tensor) -> Tensor:
        low, high = self.input_range
        # 官方 LPIPS 网络默认接收 [-1,1]；不要 detach，以保留输入梯度。
        return 2.0 * (image - low) / (high - low) - 1.0

    def forward(self, original: Tensor, embedded: Tensor) -> FidelityLossOutput:
        if original.shape != embedded.shape:
            raise ValueError("original 与 embedded 的形状必须相同")
        if original.ndim != 4:
            raise ValueError("original 与 embedded 必须是 [B,C,H,W]")

        mse = F.mse_loss(embedded, original)
        original_lpips = self._to_lpips_range(original).float()
        embedded_lpips = self._to_lpips_range(embedded).float()
        # 即使外部调用 criterion.train()，预训练 LPIPS 仍保持推理行为。
        self.lpips_model.eval()
        perceptual = self.lpips_model(original_lpips, embedded_lpips).mean()
        total = self.lambda_mse * mse + self.lambda_lpips * perceptual
        return FidelityLossOutput(total=total, mse=mse, lpips=perceptual)

