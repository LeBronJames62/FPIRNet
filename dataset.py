from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import list_images, load_image


def _expand_roots(roots: Sequence[str | Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        files.extend(list_images(root, recursive=True))
    files = sorted(set(path.resolve() for path in files))
    if not files:
        raise RuntimeError(f"No images found under: {roots}")
    return files


def _subset(items: List[Path], maximum: int | None, seed: int) -> List[Path]:
    if maximum is None or int(maximum) >= len(items):
        return items
    generator = random.Random(int(seed))
    return sorted(generator.sample(items, int(maximum)), key=str)


def _crop(image: torch.Tensor, size: int, training: bool) -> torch.Tensor:
    pad_h = max(0, size - image.shape[-2])
    pad_w = max(0, size - image.shape[-1])
    if pad_h or pad_w:
        mode = "reflect" if pad_h < image.shape[-2] and pad_w < image.shape[-1] else "replicate"
        image = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
    _, height, width = image.shape
    if training:
        top = 0 if height == size else random.randint(0, height - size)
        left = 0 if width == size else random.randint(0, width - size)
    else:
        top = max(0, (height - size) // 2)
        left = max(0, (width - size) // 2)
    return image[:, top:top + size, left:left + size]


def _augment(image: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        image = torch.flip(image, dims=[2])
    if random.random() < 0.5:
        image = torch.flip(image, dims=[1])
    turns = random.randint(0, 3)
    return torch.rot90(image, turns, dims=[1, 2]) if turns else image


class CleanImageDataset(Dataset):
    """Load clean crops. Turbulence is synthesized online on the GPU."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        patch_size: int,
        training: bool,
        max_images: int | None = None,
        repeat: int = 1,
        subset_seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.files = _subset(_expand_roots(roots), max_images, subset_seed)
        self.patch_size = int(patch_size)
        self.training = bool(training)
        self.repeat = max(1, int(repeat))
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.files) * self.repeat

    def __getitem__(self, index: int) -> Dict[str, Any]:
        base_index = index % len(self.files)
        path = self.files[base_index]
        clean = _crop(load_image(path), self.patch_size, self.training)
        if self.training and self.augment:
            clean = _augment(clean)
        return {"clean": clean, "path": str(path), "index": base_index}


def build_loader(cfg, training: bool):
    from torch.utils.data import DataLoader
    dataset = CleanImageDataset(
        roots=cfg["roots"], patch_size=int(cfg.get("patch_size", 192)),
        training=training, max_images=cfg.get("max_images"),
        repeat=int(cfg.get("repeat", 1)), subset_seed=int(cfg.get("subset_seed", 42)),
        augment=bool(cfg.get("augment", training)),
    )
    workers = int(cfg.get("num_workers", 2))
    return DataLoader(
        dataset, batch_size=int(cfg.get("batch_size", 1)), shuffle=training,
        num_workers=workers, pin_memory=bool(cfg.get("pin_memory", True)),
        drop_last=training,
        persistent_workers=bool(workers > 0 and cfg.get("persistent_workers", False)),
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if workers else None,
    )
