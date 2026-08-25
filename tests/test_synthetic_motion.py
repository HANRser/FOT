import torch

from fot.synthetic_motion import (
    AffineMotionConfig,
    make_synthetic_video,
    sample_affine_flows,
)


def test_identity_motion_is_first_frame():
    image = torch.rand(2, 3, 16, 20)
    config = AffineMotionConfig(
        max_translation=4,
        max_rotation_degrees=3,
        min_scale=0.98,
        max_scale=1.02,
        include_identity=True,
    )
    flows = sample_affine_flows(image, 3, config)

    assert flows.shape == (2, 3, 2, 16, 20)
    assert torch.equal(flows[:, 0], torch.zeros_like(flows[:, 0]))


def test_synthetic_video_shapes_and_identity_values():
    image = torch.rand(1, 3, 16, 20)
    video, flows, masks = make_synthetic_video(image, 2)

    assert video.shape == (1, 2, 3, 16, 20)
    assert flows.shape == (1, 2, 2, 16, 20)
    assert masks.shape == (1, 2, 1, 16, 20)
    assert torch.allclose(video[:, 0], image, atol=1e-6)
    assert torch.equal(masks[:, 0], torch.ones_like(masks[:, 0]))

