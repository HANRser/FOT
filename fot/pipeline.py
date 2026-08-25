from __future__ import annotations

from typing import NamedTuple

import torch.nn.functional as F
from torch import Tensor

from template_embedding import TemplateEmbedding
from .i2v import I2VGenerator
from .recovery import photometric_confidence, recover


class PipelineOutput(NamedTuple):
    protected: Tensor
    video: Tensor
    flows: Tensor
    confidences: Tensor
    recovered: Tensor


class FlowOfTruthPipeline:
    def __init__(
        self,
        template: TemplateEmbedding,
        i2v: I2VGenerator,
        flow_estimator=None,
        *,
        motion_model=None,
    ) -> None:
        if flow_estimator is None and motion_model is None:
            raise ValueError("flow_estimator 与 motion_model 至少需要提供一个")
        self.template = template
        self.i2v = i2v
        self.flow_estimator = flow_estimator
        self.motion_model = motion_model

    def __call__(self, image: Tensor, *, num_frames: int = 14) -> PipelineOutput:
        protected = self.template(image)
        video = self.i2v.generate(protected, num_frames=num_frames)
        if video.ndim != 5 or video.shape[0] != protected.shape[0] or video.shape[2] != 3:
            raise ValueError("I2V 输出必须为 [B,T,3,H,W]，且 batch 与输入一致")

        # SVD defaults to its native 576x1024 output even when the protected image
        # has a smaller training resolution. Optical flow and photometric confidence
        # must be computed in the video's coordinate system.
        flow_reference = protected
        if protected.shape[-2:] != video.shape[-2:]:
            flow_reference = F.interpolate(
                protected,
                size=video.shape[-2:],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        flow_reference = flow_reference.to(device=video.device, dtype=video.dtype)

        if self.motion_model is not None:
            prediction = self.motion_model.forward_video(flow_reference, video)
            flows = prediction.flow
            confidences = prediction.confidence
        else:
            flows = self.flow_estimator(flow_reference, video)
            confidences = photometric_confidence(flow_reference, video, flows)
        recovered = recover(video, flows, confidences)

        # Return recovered truth at the source-image resolution so it can be used
        # directly by the recovery loss and image-quality metrics.
        if recovered.shape[-2:] != image.shape[-2:]:
            recovered = F.interpolate(
                recovered,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return PipelineOutput(protected, video, flows, confidences, recovered)
