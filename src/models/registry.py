from typing import Dict, Type
from src.models.base import BaseBackbone


class ModelRegistry:
    """
    Factory pattern — maps string names to backbone classes.
    DS&A: a hash map (dict) used as a lookup table.
    OOP: classmethod, decorator pattern
    """

    _registry: Dict[str, Type[BaseBackbone]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a backbone class under a string name."""
        def decorator(backbone_cls: Type[BaseBackbone]):
            cls._registry[name] = backbone_cls
            return backbone_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseBackbone]:
        if name not in cls._registry:
            raise KeyError(
                f"Model '{name}' not found. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def available(cls):
        return list(cls._registry.keys())