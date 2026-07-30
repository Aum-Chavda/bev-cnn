from typing import Dict, List
import torch
import torch.nn as nn
from src.utils.config import BEVConfig
from src.models.base import BaseBackbone
from src.models.blocks import ConvBNReLU, ResidualBlock
from src.models.registry import ModelRegistry

@ModelRegistry.register("bevresnet")
class BEVResNet(BaseBackbone):
    """
    OOP: inherits from BaseBackbone, fulfills the ABC contract
    DS&A: nn.Sequential is a linked list of layers
    """

    def __init__(self, config: BEVConfig):
        super().__init__(config)

        c = config.base_channels  # 64

        # stem — first layer, reduces spatial size quickly
        self.stem = ConvBNReLU(config.in_channels, c, kernel_size=7, stride=2, padding=3)

        # four stages, each doubles channels and halves spatial size
        self.stage1 = self._make_stage(c,     c,     stride=config.strides[0])
        self.stage2 = self._make_stage(c,     c * 2, stride=config.strides[1])
        self.stage3 = self._make_stage(c * 2, c * 4, stride=config.strides[2])
        self.stage4 = self._make_stage(c * 4, c * 8, stride=config.strides[3])

    def _make_stage(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        """Builds one stage: two residual blocks."""
        return nn.Sequential(
            ResidualBlock(in_ch, out_ch, stride=stride),
            ResidualBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns multi-scale feature maps — fulfills BaseBackbone contract."""
        x  = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}

    def get_output_channels(self) -> Dict[str, int]:
        c = self.config.base_channels
        return {"C2": c, "C3": c * 2, "C4": c * 4, "C5": c * 8}
