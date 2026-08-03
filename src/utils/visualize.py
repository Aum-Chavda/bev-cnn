from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn


class FeatureMapHook:
    """
    Captures intermediate feature maps via PyTorch forward hooks.
    OOP      : context manager via __enter__ / __exit__
    PyTorch  : register_forward_hook — called after every forward pass
    DS&A     : dict as ordered map from layer name to feature tensor
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model       = model
        self.layer_names = set(layer_names)
        self._features:  Dict[str, torch.Tensor] = {}
        self._handles:   List[torch.utils.hooks.RemovableHook] = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(self._hook_fn(name))
                self._handles.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook_fn(self, name: str):
        def hook(module, input, output):
            self._features[name] = output.detach().cpu()
        return hook

    @property
    def features(self) -> Dict[str, torch.Tensor]:
        return self._features


class ModelSummary:
    """
    Prints layer-by-layer parameter count.
    DS&A: BFS traversal over named_modules() graph.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def summary(self) -> str:
        lines = [f"{'Layer':<40} {'Params':>10} {'Trainable':>10}"]
        lines.append("-" * 62)
        total = 0
        trainable = 0
        for name, module in self.model.named_modules():
            if not list(module.children()):  # leaf modules only
                params    = sum(p.numel() for p in module.parameters())
                is_train  = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total    += params
                trainable += is_train
                if params > 0:
                    lines.append(f"{name:<40} {params:>10,} {is_train:>10,}")
        lines.append("-" * 62)
        lines.append(f"{'Total':<40} {total:>10,} {trainable:>10,}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def save_feature_maps(
    features: Dict[str, torch.Tensor],
    output_dir: str = "outputs/",
    max_channels: int = 4,
) -> None:
    """
    Saves feature map grids as text summary.
    Each stage: shape, mean activation, max activation.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = []
    for name, feat in features.items():
        B, C, H, W = feat.shape
        mean_act = feat.mean().item()
        max_act  = feat.max().item()
        report.append(
            f"{name}: shape={list(feat.shape)} "
            f"mean={mean_act:.4f} max={max_act:.4f}"
        )

    report_path = out / "feature_map_report.txt"
    report_path.write_text("\n".join(report))
    print(f"Feature map report saved to {report_path}")
    for line in report:
        print(line)