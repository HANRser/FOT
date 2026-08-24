from __future__ import annotations

from typing import NamedTuple, Optional
import torch
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
    def __init__(self, template: TemplateEmbedding, i2v: I2VGenerator, flow_estimator) -> None:
        self.template, self.i2v, self.flow_estimator = template, i2v, flow_estimator

    def __call__(self, image: Tensor, *, num_frames: int = 14) -> PipelineOutput:
        protected = self.template(image)
        video = self.i2v.generate(protected, num_frames=num_frames)
        flows = self.flow_estimator(protected, video)
        confidences = photometric_confidence(protected, video, flows)
        return PipelineOutput(protected, video, flows, confidences, recover(video, flows, confidences))
