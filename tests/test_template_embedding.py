import pytest

torch = pytest.importorskip("torch")
from torch import nn

from template_embedding import ImageFidelityLoss, LearnableTemplate, TemplateEmbedding


class FakeLPIPS(nn.Module):
    """不下载预训练权重，用于验证损失组合及梯度逻辑。"""

    def forward(self, x, y):
        return (x - y).abs().mean(dim=(1, 2, 3), keepdim=True)


def test_learnable_template_has_required_shape_and_shared_batch():
    module = LearnableTemplate(3, 16, 20)
    batch_template = module(4)
    assert module.template.shape == (3, 16, 20)
    assert batch_template.shape == (4, 3, 16, 20)
    assert module.template.requires_grad
    torch.testing.assert_close(batch_template[0], batch_template[3])


def test_embedding_output_shape_range_and_identity_initialization():
    model = TemplateEmbedding(3, 17, 23, base_channels=8)
    image = torch.rand(2, 3, 17, 23)
    embedded = model(image)
    assert embedded.shape == image.shape
    assert embedded.min() >= 0.0
    assert embedded.max() <= 1.0
    torch.testing.assert_close(embedded, image)


def test_gradient_reaches_encoder_and_template_after_head_is_nonzero():
    model = TemplateEmbedding(3, 16, 16, base_channels=8)
    nn.init.normal_(model.encoder.head.weight, std=1e-3)
    image = torch.rand(1, 3, 16, 16)
    model(image).mean().backward()
    assert model.template.grad is not None
    assert torch.isfinite(model.template.grad).all()
    assert model.encoder.head.weight.grad is not None


def test_image_fidelity_loss_matches_formula_and_backpropagates():
    original = torch.zeros(2, 3, 8, 8)
    embedded = torch.full_like(original, 0.25, requires_grad=True)
    criterion = ImageFidelityLoss(
        lambda_mse=2.0,
        lambda_lpips=0.5,
        lpips_model=FakeLPIPS(),
    )
    losses = criterion(original, embedded)

    # MSE=0.25^2；映射至 [-1,1] 后绝对差为 0.5。
    torch.testing.assert_close(losses.mse, torch.tensor(0.0625))
    torch.testing.assert_close(losses.lpips, torch.tensor(0.5))
    torch.testing.assert_close(losses.total, torch.tensor(0.375))
    losses.total.backward()
    assert embedded.grad is not None
    assert torch.isfinite(embedded.grad).all()

