import struct

import numpy as np
import torch
from PIL import Image

from fot.data import ImageMotionDataset, read_flo, resize_flow


def write_flo(path, flow):
    height, width = flow.shape[:2]
    with path.open("wb") as stream:
        stream.write(struct.pack("<fii", 202021.25, width, height))
        stream.write(np.asarray(flow, dtype="<f4").tobytes())


def test_read_and_resize_flo_scales_vectors(tmp_path):
    path = tmp_path / "sample.flo"
    flow = np.zeros((2, 4, 2), dtype=np.float32)
    flow[..., 0] = 2
    flow[..., 1] = 3
    write_flo(path, flow)

    loaded = read_flo(path)
    resized, valid = resize_flow(loaded, (4, 2))

    assert loaded.shape == (2, 2, 4)
    assert resized.shape == (2, 4, 2)
    assert torch.allclose(resized[0], torch.ones(4, 2))
    assert torch.allclose(resized[1], torch.full((4, 2), 6.0))
    assert valid.bool().all()


def test_image_motion_dataset_uses_manifests_and_identity(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(image_path)
    flow_path = tmp_path / "flow.flo"
    write_flo(flow_path, np.zeros((6, 8, 2), dtype=np.float32))
    (tmp_path / "images.txt").write_text("image.png\n", encoding="utf-8")
    (tmp_path / "flows.txt").write_text("flow.flo\n", encoding="utf-8")

    dataset = ImageMotionDataset(
        tmp_path / "images.txt",
        tmp_path / "flows.txt",
        size=16,
        num_frames=2,
        randomize_flows=False,
    )
    image, flows, masks = dataset[0]

    assert image.shape == (3, 16, 16)
    assert flows.shape == (2, 2, 16, 16)
    assert masks.shape == (2, 1, 16, 16)
    assert torch.equal(flows[0], torch.zeros_like(flows[0]))
