import argparse, json
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from fot.metrics import evaluate


def load(path):
    return pil_to_tensor(Image.open(path).convert("RGB")).float().div(255).unsqueeze(0)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("reference"); p.add_argument("recovered"); p.add_argument("--lpips",action="store_true"); a=p.parse_args()
    ref,rec=load(a.reference),load(a.recovered)
    if ref.shape != rec.shape: raise SystemExit("两幅图像尺寸必须一致")
    model=None
    if a.lpips:
        import lpips
        model=lpips.LPIPS(net="alex").eval()
    print(json.dumps(evaluate(ref,rec,model),indent=2,ensure_ascii=False))
