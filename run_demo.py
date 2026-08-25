"""Gradio demo for checkpoint-backed FoT inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from fot.checkpoint import load_checkpoint
from fot.flow import RAFTEstimator
from fot.i2v import I2VGenerator
from fot.motion_model import MotionCaptureNet
from fot.pipeline import FlowOfTruthPipeline
from template_embedding import TemplateEmbedding


class ZeroFlow(torch.nn.Module):
    def forward(self, reference, video):
        return video.new_zeros(
            video.shape[0], video.shape[1], 2, *video.shape[-2:]
        )


def build(
    mock: bool = False,
    size: int = 256,
    *,
    checkpoint: str | None = None,
    template_channels: int = 32,
    motion_channels: int = 32,
    motion_chunk_size: int = 1,
    local_files_only: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    template = TemplateEmbedding(
        3, size, size, base_channels=template_channels
    ).to(device)
    i2v = I2VGenerator(
        device=str(device), mock=mock, local_files_only=local_files_only
    )

    if checkpoint is not None:
        motion_model = MotionCaptureNet(
            base_channels=motion_channels,
            video_chunk_size=motion_chunk_size,
        ).to(device)
        metadata = load_checkpoint(
            checkpoint,
            template=template,
            motion_model=motion_model,
            map_location=device,
        )
        template.eval()
        motion_model.eval()
        pipeline = FlowOfTruthPipeline(
            template, i2v, motion_model=motion_model
        )
        print(
            f"loaded checkpoint={checkpoint} epoch={metadata['epoch']} "
            f"step={metadata['global_step']}"
        )
    else:
        flow_estimator = ZeroFlow().to(device) if mock else RAFTEstimator().to(device)
        pipeline = FlowOfTruthPipeline(template, i2v, flow_estimator)

    def run(image):
        if image is None:
            raise ValueError("请先上传图片")
        x = (
            pil_to_tensor(image.convert("RGB").resize((size, size)))
            .float()
            .div(255)
            .unsqueeze(0)
            .to(device)
        )
        with torch.inference_mode():
            output = pipeline(x, num_frames=8 if mock else 14)
        confidence = output.confidences.mean(1).repeat(1, 3, 1, 1)
        return (
            to_pil_image(output.protected[0].cpu()),
            to_pil_image(output.video[0, -1].cpu()),
            to_pil_image(output.recovered[0].cpu()),
            to_pil_image(confidence[0].cpu()),
        )

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--template-channels", type=int, default=32)
    parser.add_argument("--motion-channels", type=int, default=32)
    parser.add_argument("--motion-chunk-size", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint and not Path(args.checkpoint).is_file():
        raise SystemExit(f"checkpoint 不存在：{args.checkpoint}")

    import gradio as gr

    with gr.Blocks(title="Flow of Truth") as demo:
        gr.Markdown("# Flow of Truth I2V 回溯取证")
        image = gr.Image(type="pil", label="Original Image")
        button = gr.Button("保护、生成并恢复")
        outputs = [
            gr.Image(label=label)
            for label in (
                "Protected Image",
                "Forged Frame",
                "Recovered Truth",
                "Confidence Map",
            )
        ]
        button.click(
            build(
                args.mock,
                args.size,
                checkpoint=args.checkpoint,
                template_channels=args.template_channels,
                motion_channels=args.motion_channels,
                motion_chunk_size=args.motion_chunk_size,
                local_files_only=args.local_files_only,
            ),
            image,
            outputs,
        )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
