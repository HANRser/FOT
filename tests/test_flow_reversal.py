import pytest

torch = pytest.importorskip("torch")

from flow_reversal import (
    backward_warp,
    confidence_weighted_fusion,
    reverse_and_fuse,
)


def test_zero_flow_is_identity_for_batch():
    frame = torch.randn(2, 3, 5, 7)
    flow = torch.zeros(2, 2, 5, 7)
    actual = backward_warp(frame, flow)
    torch.testing.assert_close(actual, frame, atol=1e-6, rtol=1e-6)


def test_positive_dx_samples_from_right_and_reports_invalid_border():
    frame = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    flow = torch.zeros(1, 2, 1, 4)
    flow[:, 0] = 1.0
    result = backward_warp(frame, flow, return_valid_mask=True)
    expected = torch.tensor([[[[1.0, 2.0, 3.0, 0.0]]]])
    torch.testing.assert_close(result.image, expected)
    torch.testing.assert_close(
        result.valid_mask, torch.tensor([[[[True, True, True, False]]]])
    )


def test_subpixel_flow_uses_bilinear_interpolation():
    frame = torch.tensor([[[[0.0, 2.0]]]])
    flow = torch.zeros(1, 2, 1, 2)
    flow[0, 0, 0, 0] = 0.5
    actual = backward_warp(frame, flow)
    torch.testing.assert_close(actual[0, 0, 0, 0], torch.tensor(1.0))


def test_backward_warp_has_frame_and_flow_gradients():
    frame = torch.randn(1, 1, 3, 3, requires_grad=True)
    flow = torch.full((1, 2, 3, 3), 0.2, requires_grad=True)
    backward_warp(frame, flow).square().mean().backward()
    assert torch.isfinite(frame.grad).all()
    assert torch.isfinite(flow.grad).all()


def test_confidence_fusion_handles_batch_and_lists():
    image1 = torch.tensor([2.0, 10.0]).reshape(2, 1, 1, 1)
    image2 = torch.tensor([6.0, 20.0]).reshape(2, 1, 1, 1)
    confidence1 = torch.tensor([0.75, 0.20]).reshape(2, 1, 1, 1)
    confidence2 = torch.tensor([0.25, 0.80]).reshape(2, 1, 1, 1)
    actual = confidence_weighted_fusion(
        [image1, image2], [confidence1, confidence2]
    )
    expected = torch.tensor([3.0, 18.0]).reshape(2, 1, 1, 1)
    torch.testing.assert_close(actual, expected)


def test_fusion_accepts_stacked_tensors_and_zero_weight_fallback():
    images = torch.tensor([1.0, 3.0]).reshape(1, 2, 1, 1, 1)
    confidences = torch.zeros(1, 2, 1, 1, 1)
    zeros = confidence_weighted_fusion(images, confidences)
    mean = confidence_weighted_fusion(images, confidences, fallback="mean")
    torch.testing.assert_close(zeros, torch.zeros(1, 1, 1, 1))
    torch.testing.assert_close(mean, torch.full((1, 1, 1, 1), 2.0))


def test_reverse_and_fuse_masks_out_of_bounds_samples():
    frame1 = torch.tensor([[[[1.0, 2.0]]]])
    frame2 = torch.tensor([[[[4.0, 8.0]]]])
    flow1 = torch.zeros(1, 2, 1, 2)
    flow2 = torch.zeros(1, 2, 1, 2)
    flow2[:, 0] = 1.0  # 第二帧在最后一个目标像素处越界。
    confidence = torch.ones(1, 1, 1, 2)

    actual = reverse_and_fuse(
        [frame1, frame2], [flow1, flow2], [confidence, confidence]
    )
    # x=0 融合 frame1[0]=1 与 frame2[1]=8；x=1 只保留第一帧。
    expected = torch.tensor([[[[4.5, 2.0]]]])
    torch.testing.assert_close(actual, expected)

