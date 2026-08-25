import torch

from fot.checkpoint import load_checkpoint, save_checkpoint
from fot.motion_model import MotionCaptureNet
from template_embedding import TemplateEmbedding


def test_checkpoint_round_trip(tmp_path):
    template = TemplateEmbedding(3, 16, 16, base_channels=4)
    motion = MotionCaptureNet(base_channels=4)
    optimizer = torch.optim.AdamW(
        list(template.parameters()) + list(motion.parameters()), lr=1e-4
    )
    original_template = template.template.detach().clone()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        template=template,
        motion_model=motion,
        optimizer=optimizer,
        epoch=3,
        global_step=17,
        best_metric=0.25,
        config={"size": 16},
    )

    with torch.no_grad():
        template.template.zero_()
    metadata = load_checkpoint(
        path,
        template=template,
        motion_model=motion,
        optimizer=optimizer,
    )

    assert torch.equal(template.template, original_template)
    assert metadata == {
        "epoch": 3,
        "global_step": 17,
        "best_metric": 0.25,
        "config": {"size": 16},
    }

