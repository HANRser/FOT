import torch
from template_embedding import TemplateEmbedding
from fot.i2v import I2VGenerator
from fot.pipeline import FlowOfTruthPipeline


class ZeroFlow(torch.nn.Module):
    def forward(self, reference, video): return video.new_zeros(video.shape[0],video.shape[1],2,*video.shape[-2:])


if __name__ == "__main__":
    x=torch.rand(1,3,64,64); p=FlowOfTruthPipeline(TemplateEmbedding(3,64,64,base_channels=8),I2VGenerator(mock=True),ZeroFlow())
    out=p(x,num_frames=4); print({k:tuple(v.shape) for k,v in out._asdict().items()}); out.recovered.mean().backward(); print("gradient_ok",p.template.template.grad is not None)
