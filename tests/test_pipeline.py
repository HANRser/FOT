import torch
from template_embedding import TemplateEmbedding
from fot.i2v import I2VGenerator
from fot.pipeline import FlowOfTruthPipeline


class ZeroFlow(torch.nn.Module):
    def forward(self, reference, video):
        return video.new_zeros(video.shape[0],video.shape[1],2,*video.shape[-2:])


def test_mock_pipeline_shapes_and_gradient():
    image=torch.rand(1,3,32,32)
    model=TemplateEmbedding(3,32,32,base_channels=4)
    out=FlowOfTruthPipeline(model,I2VGenerator(mock=True),ZeroFlow())(image,num_frames=3)
    assert out.video.shape == (1,3,3,32,32)
    assert out.flows.shape == (1,3,2,32,32)
    assert out.confidences.shape == (1,3,1,32,32)
    out.recovered.mean().backward()
    assert model.template.grad is not None
