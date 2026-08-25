import numpy as np
import pytest
from PIL import Image

from fot.evaluation import ImageEvaluator, aggregate, pair_directories, save_report


def test_identical_image_metrics():
    image = Image.fromarray(np.full((16, 20, 3), 128, dtype=np.uint8))
    result = ImageEvaluator(device="cpu")(image, image)
    assert result["psnr"] >= 100
    assert result["ssim"] == pytest.approx(1.0)


def test_directory_pairing_aggregation_and_report(tmp_path):
    reference_dir = tmp_path / "reference"
    recovered_dir = tmp_path / "recovered"
    reference_dir.mkdir()
    recovered_dir.mkdir()
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    image.save(reference_dir / "a.png")
    image.save(recovered_dir / "a.png")

    pairs = pair_directories(reference_dir, recovered_dir)
    assert [pair[0] for pair in pairs] == ["a.png"]
    rows = [{"name": "a.png", "psnr": 100.0, "ssim": 1.0}]
    summary = aggregate(rows)
    json_path, csv_path = save_report(tmp_path / "metrics.json", rows, summary)
    assert json_path.is_file()
    assert csv_path.is_file()
    assert summary["ssim"]["mean"] == 1.0

