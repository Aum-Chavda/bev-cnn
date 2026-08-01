# BEV CNN Feature Extractor

A Bird's-Eye View CNN feature extractor built from scratch in PyTorch.
Multi-scale feature maps for autonomous perception tasks.

## Architecture
Input (B, 3, 256, 256) -> Stem -> Stage1 -> Stage2 -> Stage3 -> Stage4
C2: (B, 64, 128, 128) | C3: (B, 128, 64, 64) | C4: (B, 256, 32, 32) | C5: (B, 512, 16, 16)

## What I built
- Phase 1: BEVConfig, BaseBackbone ABC, ConvBNReLU, ResidualBlock, BEVResNet, ModelRegistry
- Phase 2: BEVDataset (CIFAR-10, LRU cache), build_dataloader (pin_memory, drop_last)

## Setup
uv venv && uv python pin 3.11
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
Download CIFAR-10 to data/ folder manually

## Run tests
uv run pytest tests/ -v

## Status
- [x] Phase 1 - Backbone
- [x] Phase 2 - Data Pipeline
- [ ] Phase 3 - Trainer + Callbacks
- [ ] Phase 4 - Ablation + Visualisation
