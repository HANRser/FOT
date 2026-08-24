"""Gradio demo. Uses SVD and RAFT by default; --mock avoids large downloads."""
import argparse, tempfile
import numpy as np, torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from template_embedding import TemplateEmbedding
from fot.i2v import I2VGenerator
from fot.flow import RAFTEstimator
from fot.pipeline import FlowOfTruthPipeline


class ZeroFlow(torch.nn.Module):
    def forward(self, reference, video): return video.new_zeros(video.shape[0],video.shape[1],2,*video.shape[-2:])


def build(mock=False, size=256):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe=FlowOfTruthPipeline(TemplateEmbedding(3,size,size).to(device), I2VGenerator(device=str(device),mock=mock), ZeroFlow().to(device) if mock else RAFTEstimator().to(device))
    def run(image):
        x=pil_to_tensor(image.convert("RGB").resize((size,size))).float().div(255).unsqueeze(0).to(device)
        with torch.inference_mode(): out=pipe(x,num_frames=8 if mock else 14)
        conf=out.confidences.mean(1).repeat(1,3,1,1)
        return to_pil_image(out.protected[0].cpu()), to_pil_image(out.video[0,-1].cpu()), to_pil_image(out.recovered[0].cpu()), to_pil_image(conf[0].cpu())
    return run


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mock",action="store_true"); p.add_argument("--share",action="store_true"); args=p.parse_args()
    import gradio as gr
    with gr.Blocks(title="Flow of Truth") as demo:
        gr.Markdown("# Flow of Truth I2V 回溯取证")
        inp=gr.Image(type="pil",label="Original Image"); btn=gr.Button("保护、生成并恢复")
        outs=[gr.Image(label=x) for x in ("Protected Image","Forged Frame","Recovered Truth","Confidence Map")]
        btn.click(build(args.mock),inp,outs)
    demo.launch(share=args.share)
if __name__=="__main__": main()
