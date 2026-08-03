from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from src.models.base import BaseBackbone
from src.training.callbacks import Callback
from src.utils.metrics import MetricTracker


@dataclass
class CheckpointEntry:
    """
    DS&A: heapq needs comparable objects.
    We negate val_loss so heapq (min-heap) gives us the WORST checkpoint to evict.
    """
    val_loss: float
    epoch: int
    path: Path

    def __lt__(self, other: CheckpointEntry) -> bool:
        return self.val_loss < other.val_loss


class Trainer:
    """
    OOP  : composition — holds model, optimizer, scaler, callbacks
    DS&A : heapq for top-k checkpoint management
    PyTorch: autocast (FP16), GradScaler, train/eval mode switching
    """

    def __init__(
        self,
        model: BaseBackbone,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        epochs: int = 20,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/",
        keep_top_k: int = 3,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.epochs         = epochs
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_top_k     = keep_top_k
        self.callbacks      = callbacks or []

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # PyTorch optimizer and FP16 scaler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else torch.amp.GradScaler("cpu")          # FP16 gradient scaling
        self.criterion = nn.CrossEntropyLoss()

        # DS&A: max-heap of size keep_top_k using negated loss
        # stores worst checkpoints so we can evict them when a better one arrives
        self._checkpoint_heap: List[CheckpointEntry] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(self) -> None:
        """Main training loop."""
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc   = self._val_epoch()

            logs = {
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            }

            self._fire_callbacks("on_epoch_end", epoch, logs)
            self._save_checkpoint(epoch, val_loss)
            self.scheduler.step()

            # check early stopping
            for cb in self.callbacks:
                if hasattr(cb, "should_stop") and cb.should_stop:
                    print("Training stopped early.")
                    return

    @classmethod
    def from_config(cls, model, train_loader, val_loader, config) -> Trainer:
        """
        OOP: @classmethod — alternative constructor.
        Lets you do: trainer = Trainer.from_config(model, train_dl, val_dl, cfg)
        instead of passing every argument manually.
        """
        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=config.lr,
            epochs=config.epochs,
            device=config.device,
            checkpoint_dir=config.checkpoint_dir,
        )

    # ------------------------------------------------------------------ #
    #  Private methods                                                     #
    # ------------------------------------------------------------------ #

    def _train_epoch(self):
        self.model.train()  # enables dropout, batchnorm in train mode
        loss_tracker = MetricTracker("train_loss")
        acc_tracker  = MetricTracker("train_acc")

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            images = torch.nn.functional.interpolate(
    images, size=(256, 256), mode='bilinear', align_corners=False
)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # FP16 forward pass — uses less VRAM on your GTX 1650 Ti
            with torch.amp.autocast(device_type=self.device):
                features = self.model(images)
                # use C5 (most abstract features) for classification
                pooled = features["C5"].mean(dim=[2, 3])  # global average pool
                logits = self._classify(pooled)
                loss   = self.criterion(logits, labels)

            # FP16 backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            loss_tracker.update(loss.item(), n=len(labels))
            acc_tracker.update(acc, n=len(labels))

            self._fire_callbacks("on_batch_end", batch_idx, {
                "train_loss": loss.item(),
                "train_acc":  acc,
            })

        return loss_tracker.compute(), acc_tracker.compute()

    def _val_epoch(self):
        self.model.eval()  # disables dropout, batchnorm uses running stats
        loss_tracker = MetricTracker("val_loss")
        acc_tracker  = MetricTracker("val_acc")

        with torch.no_grad():  # no gradient computation during validation
            for images, labels in self.val_loader:
                images = images.to(self.device)
                images = torch.nn.functional.interpolate(
    images, size=(256, 256), mode='bilinear', align_corners=False
)
                labels = labels.to(self.device)

                features = self.model(images)
                pooled   = features["C5"].mean(dim=[2, 3])
                logits   = self._classify(pooled)
                loss     = self.criterion(logits, labels)

                acc = (logits.argmax(dim=1) == labels).float().mean().item()
                loss_tracker.update(loss.item(), n=len(labels))
                acc_tracker.update(acc, n=len(labels))

        return loss_tracker.compute(), acc_tracker.compute()

    def _classify(self, pooled: torch.Tensor) -> torch.Tensor:
        """Linear classifier on top of pooled features."""
        if not hasattr(self, "_classifier"):
            in_features = pooled.shape[1]
            self._classifier = nn.Linear(in_features, 10).to(self.device)
        return self._classifier(pooled)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        DS&A: heapq top-k pattern.
        Keep only the best keep_top_k checkpoints by val_loss.
        """
        path = self.checkpoint_dir / f"epoch_{epoch:03d}_loss_{val_loss:.4f}.pt"
        torch.save({
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }, path)

        entry = CheckpointEntry(val_loss=val_loss, epoch=epoch, path=path)
        heapq.heappush(self._checkpoint_heap, entry)

        # evict worst checkpoint if heap exceeds top-k
        if len(self._checkpoint_heap) > self.keep_top_k:
            worst = heapq.heappop(self._checkpoint_heap)
            if worst.path.exists():
                worst.path.unlink()  # delete file

    def _fire_callbacks(self, event: str, *args) -> None:
        """Calls event method on every callback — observer pattern."""
        for cb in self.callbacks:
            getattr(cb, event)(*args)