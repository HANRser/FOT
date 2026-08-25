import torch
import torch.nn.functional as F
from template_embedding import TemplateEmbedding
from fot.i2v import I2VGenerator
from fot.motion_model import MotionCaptureNet
from fot.pipeline import FlowOfTruthPipeline


class ZeroFlow(torch.nn.Module):
    def forward(self, reference, video):
        return video.new_zeros(video.shape[0],video.shape[1],2,*video.shape[-2:])


class ResizingI2V:
    def generate(self, image, *, num_frames):
        frame = F.interpolate(
            image, size=(48, 64), mode="bilinear", align_corners=False
        )
        return frame.unsqueeze(1).expand(-1, num_frames, -1, -1, -1)


def test_mock_pipeline_shapes_and_gradient():
    image=torch.rand(1,3,32,32)
    model=TemplateEmbedding(3,32,32,base_channels=4)
    out=FlowOfTruthPipeline(model,I2VGenerator(mock=True),ZeroFlow())(image,num_frames=3)
    assert out.video.shape == (1,3,3,32,32)
    assert out.flows.shape == (1,3,2,32,32)
    assert out.confidences.shape == (1,3,1,32,32)
    out.recovered.mean().backward()
    assert model.template.grad is not None


def test_pipeline_aligns_native_video_resolution_and_restores_source_size():
    image = torch.rand(1, 3, 32, 32)
    model = TemplateEmbedding(3, 32, 32, base_channels=4)
    out = FlowOfTruthPipeline(model, ResizingI2V(), ZeroFlow())(
        image, num_frames=2
    )

    assert out.protected.shape == (1, 3, 32, 32)
    assert out.video.shape == (1, 2, 3, 48, 64)
    assert out.flows.shape == (1, 2, 2, 48, 64)
    assert out.confidences.shape == (1, 2, 1, 48, 64)
    assert out.recovered.shape == image.shape


def test_pipeline_accepts_trainable_motion_capture_model():
    image = torch.rand(1, 3, 32, 32)
    template = TemplateEmbedding(3, 32, 32, base_channels=4)
    motion_model = MotionCaptureNet(base_channels=4)
    out = FlowOfTruthPipeline(
        template,
        I2VGenerator(mock=True),
        motion_model=motion_model,
    )(image, num_frames=2)

    assert out.flows.shape == (1, 2, 2, 32, 32)
    assert out.confidences.shape == (1, 2, 1, 32, 32)
    assert out.recovered.shape == image.shape
