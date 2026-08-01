from __future__ import annotations
import functools
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.datasets as dsets


class BEVDataset(Dataset):
    LABEL_MAP: Dict[int, str] = {
        0: "airplane", 1: "automobile", 2: "bird",  3: "cat",
        4: "deer",     5: "dog",        6: "frog",  7: "horse",
        8: "ship",     9: "truck"
    }

    def __init__(
        self,
        root: str = "data/",
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_size: int = 128,
        size: Optional[int] = None,
    ):
        super().__init__()

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.root       = Path(root)
        self.split      = split
        self.transform  = transform or self._default_transform()
        self.cache_size = cache_size

        is_train = split == "train"
        self._data = dsets.CIFAR10(
            root=str(self.root),
            train=is_train,
            download=False,   # already downloaded
        )
        self._size = size if size else len(self._data)
        self._cache = functools.lru_cache(maxsize=cache_size)(self._load_sample)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self._cache(idx)
        return self.transform(image), label

    def _load_sample(self, idx: int):
        return self._data[idx]

    @staticmethod
    def _default_transform() -> T.Compose:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            T.Resize((256, 256)),
        ])

    def class_name(self, label: int) -> str:
        return self.LABEL_MAP[label]

    def __repr__(self) -> str:
        return (
            f"BEVDataset(split={self.split}, "
            f"size={len(self)}, "
            f"cache={self.cache_size})"
        )