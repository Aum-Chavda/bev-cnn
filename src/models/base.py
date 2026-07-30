from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn
from src.utils.config import BEVConfig


class BaseBackbone(ABC, nn.Module):
    """
    Abstract base class for all BEV backbone networks.
    Any backbone MUST implement:
      - forward()
      - get_output_channels()
    """

    def __init__(self, config: BEVConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            dict of feature maps e.g. {"C2": ..., "C3": ..., "C4": ...}
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_channels(self) -> Dict[str, int]:
        """
        Returns number of channels at each output stage.
        e.g. {"C2": 64, "C3": 128, "C4": 256}
        """
        raise NotImplementedError

    def num_parameters(self) -> int:
        """Total trainable parameters — concrete method, free for all subclasses."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_size={self.config.input_size}, "
            f"params={self.num_parameters():,})"
        )