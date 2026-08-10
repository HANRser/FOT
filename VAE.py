"""Flow of Truth 第 3.4 节的 VAE 压缩与重建模拟。

该模块使用冻结的 ``stabilityai/sd-vae-ft-mse`` AutoencoderKL 模拟 I2V
生成器中常见的特征压缩与重建伪影。重建输出可直接传给 Scatter Operator。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from torch import Generator, Tensor, nn


DEFAULT_VAE_ID = "stabilityai/sd-vae-ft-mse"


class FrozenVAEReconstructor(nn.Module):
    """冻结的 AutoencoderKL 图像重建器。

    VAE 权重不参与优化，但前向过程不会放入 ``torch.no_grad()``。这样
    ``I_0^T -> VAE -> loss`` 的梯度仍能穿过 VAE 回传到 Template Encoder。

    Args:
        model_id: Hugging Face 模型 ID，默认使用论文复现指定的 ft-MSE VAE。
        torch_dtype: 加载权重的数据类型。默认 None 即模型原始 float32。
        cache_dir: Hugging Face 模型缓存目录。
        local_files_only: True 时只从本地缓存加载，集群离线运行时很有用。
        sample_posterior: 是否从 posterior 采样；默认 False 使用 mode，保证重建
            可复现。设为 True 可加入 VAE 随机性。
        clamp_output: 是否把重建结果严格截断到 [0,1]。
        enable_slicing: 启用 Diffusers VAE slicing，降低多 batch 解码峰值显存。
        enable_tiling: 启用 VAE tiling，适用于高分辨率且显存紧张的情况。
        vae: 可选的外部 VAE，仅用于依赖注入/单元测试；正式使用无需传入。
    """

    def __init__(
        self,
        model_id: str = DEFAULT_VAE_ID,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        sample_posterior: bool = False,
        clamp_output: bool = True,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        vae: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if vae is None:
            try:
                from diffusers import AutoencoderKL
            except ImportError as exc:
                raise ImportError(
                    "缺少 diffusers，请执行：pip install diffusers transformers "
                    "accelerate safetensors"
                ) from exc

            load_kwargs = {"local_files_only": local_files_only}
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
            if cache_dir is not None:
                load_kwargs["cache_dir"] = str(cache_dir)
            try:
                vae = AutoencoderKL.from_pretrained(model_id, **load_kwargs)
            except OSError as exc:
                raise RuntimeError(
                    f"无法加载 VAE：{model_id!r}。如果计算节点不能访问 "
                    "huggingface.co，请先在可联网节点下载模型，再把 model_id "
                    "改为本地模型目录并设置 local_files_only=True。原始错误："
                    f"{exc}"
                ) from exc

        self.vae = vae
        self.model_id = model_id
        self.sample_posterior = sample_posterior
        self.clamp_output = clamp_output

        # 冻结参数但不禁用 autograd；requires_grad=False 只是不计算参数梯度。
        self.vae.requires_grad_(False)
        self.vae.eval()

        if enable_slicing:
            if not hasattr(self.vae, "enable_slicing"):
                raise TypeError("传入的 VAE 不支持 enable_slicing()")
            self.vae.enable_slicing()
        if enable_tiling:
            if not hasattr(self.vae, "enable_tiling"):
                raise TypeError("传入的 VAE 不支持 enable_tiling()")
            self.vae.enable_tiling()

    @property
    def spatial_scale_factor(self) -> int:
        """VAE 的空间下采样倍数；sd-vae-ft-mse 为 8。"""
        config = getattr(self.vae, "config", None)
        block_channels = getattr(config, "block_out_channels", None)
        if block_channels is None:
            return 1
        return 2 ** (len(block_channels) - 1)

    def train(self, mode: bool = True) -> "FrozenVAEReconstructor":
        """允许外层模块切换状态，但始终强制冻结 VAE 保持 eval。"""
        super().train(mode)
        self.vae.eval()
        return self

    def _vae_parameter(self) -> Optional[nn.Parameter]:
        return next(iter(self.vae.parameters()), None)

    def _validate_image(self, image: Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image 应为 [B,3,H,W]，实际为 {tuple(image.shape)}")
        if not image.is_floating_point():
            raise TypeError("image 必须是浮点张量，且数值范围应为 [0,1]")
        if image.shape[0] <= 0:
            raise ValueError("batch 不能为空")

        factor = self.spatial_scale_factor
        if image.shape[-2] % factor != 0 or image.shape[-1] % factor != 0:
            raise ValueError(
                f"H 和 W 必须能被 VAE 下采样倍数 {factor} 整除，"
                f"实际为 {tuple(image.shape[-2:])}"
            )

        parameter = self._vae_parameter()
        if parameter is not None and image.device != parameter.device:
            raise ValueError(
                f"image 位于 {image.device}，VAE 位于 {parameter.device}；"
                "请先将模块和输入移动到同一设备"
            )

    def reconstruct(
        self,
        image: Tensor,
        *,
        sample_posterior: Optional[bool] = None,
        generator: Optional[Generator] = None,
    ) -> Tensor:
        """执行 ``[0,1] -> encode -> decode -> [0,1]`` 重建。

        Args:
            image: 嵌入模板后的图像 ``I_0^T``，[B,3,H,W]，范围 [0,1]。
            sample_posterior: 覆盖构造函数中的 posterior 采样设置。
            generator: 随机采样使用的 PyTorch Generator。

        Returns:
            重建图像 ``hat(I_0^T)``，[B,3,H,W]，默认范围严格为 [0,1]。
        """
        self._validate_image(image)
        original_dtype = image.dtype
        parameter = self._vae_parameter()
        vae_dtype = parameter.dtype if parameter is not None else image.dtype

        # Stable Diffusion VAE 的图像输入范围为 [-1,1]。
        normalized = image.to(dtype=vae_dtype) * 2.0 - 1.0
        posterior = self.vae.encode(normalized, return_dict=True).latent_dist

        should_sample = (
            self.sample_posterior
            if sample_posterior is None
            else sample_posterior
        )
        if should_sample:
            latents = posterior.sample(generator=generator)
        else:
            latents = posterior.mode()

        # 这是直接的 VAE round-trip，中间没有扩散模型，因此不应用
        # vae.config.scaling_factor（0.18215 只用于 diffusion latent）。
        decoded = self.vae.decode(latents, return_dict=True).sample
        reconstructed = (decoded + 1.0) * 0.5
        if self.clamp_output:
            reconstructed = reconstructed.clamp(0.0, 1.0)
        return reconstructed.to(dtype=original_dtype)

    def forward(self, image: Tensor) -> Tensor:
        """等价于 ``reconstruct(image)``，方便插入 nn.Module 管线。"""
        return self.reconstruct(image)
