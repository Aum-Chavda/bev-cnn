from dataclasses import dataclass
from typing import Tuple


@dataclass
class BEVConfig:
    input_size: Tuple[int, int] = (64, 64)
    in_channels: int = 3
    base_channels: int = 64
    num_stages: int = 4
    strides: Tuple[int, ...] = (1, 2, 2, 2)
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 20
    device: str = "cuda"
    data_dir: str = "data/"
    checkpoint_dir: str = "checkpoints/"

    def __post_init__(self):
        if len(self.strides) != self.num_stages:
            raise ValueError(
                f"Expected {self.num_stages} strides, got {len(self.strides)}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")