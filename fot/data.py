from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, size: int = 256):
        self.files = [p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if not self.files: raise ValueError(f"未在 {root} 找到图像")
        self.transform = v2.Compose([v2.Resize((size, size)), v2.ToImage(), v2.ToDtype(__import__('torch').float32, scale=True)])
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return self.transform(Image.open(self.files[i]).convert("RGB"))
