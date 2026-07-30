from typing import Optional
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """
    Conv2d + BatchNorm2d + ReLU — the atomic building block.
    OOP: composition (holds three nn.Module objects inside one)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Two ConvBNReLU blocks + a skip connection.
    OOP: composition of ConvBNReLU objects
    DS&A: think of it as a node that has two paths — main and shortcut
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.main = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, stride=stride),
            ConvBNReLU(out_channels, out_channels, stride=1),
        )

        # projection shortcut — needed when shape changes
        self.shortcut: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        return nn.functional.relu(self.main(x) + identity, inplace=True)