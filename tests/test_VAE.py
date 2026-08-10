from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from VAE import FrozenVAEReconstructor


class FakePosterior:
    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value

    def sample(self, generator=None):
        del generator
        return self.value + 0.2


class FakeVAE(nn.Module):
    """恒等 VAE，用于离线验证归一化、冻结和梯度，不下载模型。"""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(block_out_channels=(16, 32, 64, 64))
        self.last_encoded = None

    def encode(self, image, return_dict=True):
        assert return_dict
        self.last_encoded = image.detach().clone()
        return SimpleNamespace(latent_dist=FakePosterior(image * self.scale))

    def decode(self, latents, return_dict=True):
        assert return_dict
        return SimpleNamespace(sample=latents * self.scale)


def test_vae_is_frozen_and_stays_in_eval_mode():
    fake = FakeVAE()
    model = FrozenVAEReconstructor(vae=fake)
    assert not model.vae.training
    assert all(not parameter.requires_grad for parameter in model.vae.parameters())

    model.train()
    assert model.training
    assert not model.vae.training


def test_reconstruct_converts_ranges_and_preserves_shape():
    fake = FakeVAE()
    model = FrozenVAEReconstructor(vae=fake)
    image = torch.full((2, 3, 16, 24), 0.25)
    reconstructed = model.reconstruct(image)

    torch.testing.assert_close(fake.last_encoded, torch.full_like(image, -0.5))
    torch.testing.assert_close(reconstructed, image)
    assert reconstructed.shape == image.shape
    assert reconstructed.min() >= 0.0
    assert reconstructed.max() <= 1.0


def test_input_gradient_passes_through_frozen_vae():
    model = FrozenVAEReconstructor(vae=FakeVAE())
    image = torch.full((1, 3, 16, 16), 0.4, requires_grad=True)
    model.reconstruct(image).mean().backward()

    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert all(parameter.grad is None for parameter in model.vae.parameters())


def test_sampling_posterior_is_optional():
    model = FrozenVAEReconstructor(vae=FakeVAE())
    image = torch.full((1, 3, 16, 16), 0.4)
    deterministic = model.reconstruct(image)
    sampled = model.reconstruct(image, sample_posterior=True)
    assert not torch.allclose(sampled, deterministic)


def test_rejects_invalid_shape_and_non_multiple_of_eight():
    model = FrozenVAEReconstructor(vae=FakeVAE())
    with pytest.raises(ValueError, match=r"\[B,3,H,W\]"):
        model.reconstruct(torch.rand(1, 1, 16, 16))
    with pytest.raises(ValueError, match="倍数 8"):
        model.reconstruct(torch.rand(1, 3, 15, 16))
