import pytest

torch = pytest.importorskip("torch")
from torch.autograd import gradcheck

from scatter_operator import ScatterResult, scatter_operator


def test_zero_motion_is_identity():
    image = torch.randn(2, 3, 5, 7)
    motion = torch.zeros(2, 2, 5, 7)
    actual = scatter_operator(image, motion)
    torch.testing.assert_close(actual, image)


def test_integer_translation_moves_right_and_leaves_hole():
    image = torch.arange(1.0, 7.0).reshape(1, 1, 2, 3)
    motion = torch.zeros(1, 2, 2, 3)
    motion[:, 0] = 1.0

    result = scatter_operator(image, motion, return_aux=True)
    assert isinstance(result, ScatterResult)
    expected = torch.tensor([[[[0.0, 1.0, 2.0], [0.0, 4.0, 5.0]]]])
    torch.testing.assert_close(result.image, expected)
    torch.testing.assert_close(result.mask, expected != 0)


def test_fractional_motion_uses_bilinear_weights_in_sum_mode():
    # 只有左上角源像素非零，向右下各移动半个像素后均分到四邻域。
    image = torch.zeros(1, 1, 2, 2)
    image[0, 0, 0, 0] = 4.0
    motion = torch.full((1, 2, 2, 2), 0.5)
    actual = scatter_operator(image, motion, reduction="sum")
    expected = torch.ones(1, 1, 2, 2)
    torch.testing.assert_close(actual, expected)


def test_collision_reduction_sum_and_mean():
    image = torch.tensor([[[[2.0, 4.0]]]])
    motion = torch.zeros(1, 2, 1, 2)
    # motion 的维度顺序是 [B, 2, H, W]：修改 x=1 位置的 dx。
    motion[0, 0, 0, 1] = -1.0  # 两个源像素都映射到目标 x=0。

    summed = scatter_operator(image, motion, reduction="sum")
    averaged = scatter_operator(image, motion, reduction="mean")
    torch.testing.assert_close(summed, torch.tensor([[[[6.0, 0.0]]]]))
    torch.testing.assert_close(averaged, torch.tensor([[[[3.0, 0.0]]]]))


def test_gradcheck_image_and_subpixel_motion():
    # 避开整数边界，gradcheck 才不会跨越 floor 的不可微点。
    image = torch.randn(1, 1, 2, 2, dtype=torch.double, requires_grad=True)
    motion = torch.full(
        (1, 2, 2, 2), 0.2, dtype=torch.double, requires_grad=True
    )
    fn = lambda img, flow: scatter_operator(img, flow, reduction="sum")
    assert gradcheck(fn, (image, motion), eps=1e-6, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前环境没有 CUDA")
def test_cuda_forward_and_backward():
    image = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
    motion = (torch.randn(2, 2, 32, 32, device="cuda") * 0.25).requires_grad_()
    output = scatter_operator(image, motion, reduction="mean")
    output.square().mean().backward()
    assert torch.isfinite(image.grad).all()
    assert torch.isfinite(motion.grad).all()
