import torch

from fot.motion_model import MotionCaptureNet


def test_motion_capture_pair_shapes_ranges_and_gradient():
    model = MotionCaptureNet(base_channels=4)
    reference = torch.rand(2, 3, 32, 40)
    frame = torch.rand_like(reference)
    prediction = model(reference, frame)

    assert prediction.flow.shape == (2, 2, 32, 40)
    assert prediction.alpha_logits.shape == (2, 1, 32, 40)
    assert prediction.beta2.shape == (2, 1, 32, 40)
    assert prediction.confidence.shape == (2, 1, 32, 40)
    assert torch.all((prediction.confidence >= 0) & (prediction.confidence <= 1))

    prediction.flow.mean().backward()
    assert model.flow_head.weight.grad is not None


def test_motion_capture_video_shapes():
    model = MotionCaptureNet(base_channels=4)
    reference = torch.rand(2, 3, 32, 40)
    video = torch.rand(2, 3, 3, 32, 40)
    prediction = model.forward_video(reference, video)

    assert prediction.flow.shape == (2, 3, 2, 32, 40)
    assert prediction.alpha_logits.shape == (2, 3, 1, 32, 40)
    assert prediction.beta2.shape == (2, 3, 1, 32, 40)
    assert prediction.confidence.shape == (2, 3, 1, 32, 40)

