# BEV CNN Feature Extractor

A Bird's-Eye View (BEV) CNN feature extractor built from scratch in PyTorch.
Implements production-grade ML engineering patterns including OOP architecture,
abstract base classes, factory registry, callback-based training, and FP16 mixed
precision training on a GTX 1650 Ti (4GB VRAM).

## Motivation
Autonomous perception systems — from self-driving cars to robotic grasping —
require robust feature extraction from top-down spatial representations.
This project builds a ResNet-style backbone that extracts multi-scale BEV
features, directly applicable to downstream tasks like 3D object detection
and path planning.

## Architecture

Input (B, 3, 32, 32) — CIFAR-10 as BEV proxy  
    → GPU resize to (B, 3, 256, 256)  
    → Stem — ConvBNReLU, stride=2  
    → Stage 1 — 2x ResidualBlock → C2 (B, 64,  128, 128)  
    → Stage 2 — 2x ResidualBlock → C3 (B, 128, 64,  64)  
    → Stage 3 — 2x ResidualBlock → C4 (B, 256, 32,  32)  
    → Stage 4 — 2x ResidualBlock → C5 (B, 512, 16,  16)  

Multi-scale outputs (C2–C5) are compatible with FPN-style detection heads.

## Key Engineering Decisions

**OOP Architecture**
- `BaseBackbone(ABC, nn.Module)` — abstract contract enforced at instantiation
- `ModelRegistry` — factory pattern (hash map) for model lookup by string name
- `Callback` (Protocol) — structural subtyping, no inheritance required
- `Trainer.from_config()` — @classmethod alternative constructor

**Training**
- FP16 mixed precision via `torch.amp.autocast` + `GradScaler` — fits 4GB VRAM
- GPU-side resize via `F.interpolate` — avoids CPU bottleneck
- `heapq`-based top-k checkpoint saving — O(log k) per checkpoint
- Observer pattern callbacks — `EarlyStopping`, `PrintLogger`

**Data Pipeline**
- `BEVDataset(Dataset)` — implements `__len__` and `__getitem__` contract
- `functools.lru_cache` on `_load_sample` — O(1) repeated access
- `build_dataloader` factory — `pin_memory` auto-detected from CUDA availability

**Testing**
- 31 tests across all modules
- Positive, negative, and contract testing
- `pytest.raises` for exception validation
- Shape assertions on every tensor output

## Project Structure

src/
├── models/
│ ├── base.py # BaseBackbone ABC
│ ├── blocks.py # ConvBNReLU, ResidualBlock
│ ├── resnet.py # BEVResNet — multi-scale feature extractor
│ └── registry.py # ModelRegistry — factory pattern
├── data/
│ └── dataset.py # BEVDataset + build_dataloader
├── training/
│ ├── trainer.py # Trainer + CheckpointEntry
│ └── callbacks.py # Callback Protocol, PrintLogger, EarlyStopping
└── utils/
├── config.py # BEVConfig dataclass
└── metrics.py # MetricTracker
tests/
└── test_models.py # 31 passing tests
## Setup

```bash
# requires Python 3.11, uv, CUDA 12.8+
uv venv && uv python pin 3.11
uv sync
# download CIFAR-10 manually to data/ folder
```

## Run Tests

```bash
uv run pytest tests/ -v
```

## Skills Demonstrated
- PyTorch: nn.Module, autograd, DataLoader, AMP, hooks
- OOP: ABC, Protocol, dataclass, factory, observer, composition
- DS&A: hash map, heap, LRU cache, linked list (nn.Sequential)
- Python: type annotations, dunders, @staticmethod, @classmethod, @decorator
- Software Engineering: bottom-up testing, separation of concerns, clean architecture

## Status
- [x] Phase 1 — OOP backbone (BEVConfig, BaseBackbone, ConvBNReLU, ResidualBlock, BEVResNet, ModelRegistry)
- [x] Phase 2 — Data pipeline (BEVDataset, LRU cache, build_dataloader, GPU resize)
- [x] Phase 3 — Training (Trainer, FP16, heapq checkpoints, Callback Protocol, EarlyStopping)
- [ ] Phase 4 — Ablation + forward hooks + feature map visualisation