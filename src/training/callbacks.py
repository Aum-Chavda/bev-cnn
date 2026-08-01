from __future__ import annotations
from typing import Protocol, runtime_checkable, Dict, Any


@runtime_checkable
class Callback(Protocol):
    """
    OOP: Protocol — defines an interface without inheritance.
    Any class that has these methods IS a Callback, no need to subclass.
    This is called structural subtyping or duck typing made explicit.
    """
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None: ...
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None: ...


class PrintLogger:
    """Prints metrics at end of each epoch."""

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        metrics = " | ".join(f"{k}: {v:.4f}" for k, v in logs.items())
        print(f"Epoch {epoch:03d} → {metrics}")

    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        pass  # silent during batch


class EarlyStopping:
    """
    Stops training if val_loss doesn't improve for `patience` epochs.
    DS&A: tracks a running minimum — O(1) per epoch.
    """

    def __init__(self, patience: int = 3, monitor: str = "val_loss"):
        self.patience  = patience
        self.monitor   = monitor
        self.best      = float("inf")
        self.counter   = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current = logs.get(self.monitor, float("inf"))
        if current < self.best:
            self.best    = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch} — no improvement for {self.patience} epochs")
                self.should_stop = True

    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        pass