from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224) -> Callable:
    return T.Compose(
        [
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
            T.ToTensor(),
            AddGaussianNoise(0.0, 0.02),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> Callable:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


@dataclass
class DatasetMeta:
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    class_counts: torch.Tensor


class ImageFolderFlat(Dataset):
    def __init__(self, root: str | Path, transform: Callable | None = None):
        self.root = Path(root)
        self.transform = transform
        self.class_names = sorted([x.name for x in self.root.iterdir() if x.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.samples: List[Tuple[Path, int]] = []
        for cname in self.class_names:
            cdir = self.root / cname
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                for p in cdir.glob(ext):
                    self.samples.append((p, self.class_to_idx[cname]))

        if not self.samples:
            raise ValueError(f"No image samples found under: {self.root}")

        self.samples.sort(key=lambda x: str(x[0]))
        counts = np.zeros(len(self.class_names), dtype=np.int64)
        for _, y in self.samples:
            counts[y] += 1
        self.meta = DatasetMeta(
            class_to_idx=self.class_to_idx,
            idx_to_class={v: k for k, v in self.class_to_idx.items()},
            class_counts=torch.tensor(counts, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y, str(p)
