from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class MetricTracker:
    """
    Tracks running average of a metric over batches.
    OOP: dataclass with methods — hybrid between data container and behaviour.
    DS&A: running sum — O(1) update, O(1) average, no need to store all values.
    """
    name: str
    _total: float = field(default=0.0, repr=False)
    _count: int   = field(default=0,   repr=False)

    def update(self, value: float, n: int = 1) -> None:
        self._total += value * n
        self._count += n

    def compute(self) -> float:
        if self._count == 0:
            return 0.0
        return self._total / self._count

    def reset(self) -> None:
        self._total = 0.0
        self._count = 0

    def __repr__(self) -> str:
        return f"{self.name}: {self.compute():.4f}"