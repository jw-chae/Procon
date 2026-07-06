"""Registry for modular component registration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class Registry:
    """Simple registry for component classes."""
    name: str
    _items: Dict[str, Callable[..., Any]]

    def __init__(self, name: str) -> None:
        self.name = name
        self._items = {}

    def register(self, key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def _wrap(obj: Callable[..., Any]) -> Callable[..., Any]:
            if key in self._items:
                raise KeyError(f"{self.name} registry already has key: {key}")
            self._items[key] = obj
            return obj
        return _wrap

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            raise KeyError(f"{self.name} registry missing key: {key}. Available: {list(self._items.keys())}")
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items.keys())


# Global registries
FEATURE_EXTRACTORS = Registry("feature_extractors")
MEMORY_BUILDERS = Registry("memory_builders")
SCORERS = Registry("scorers")
INFERENCE_BACKENDS = Registry("inference_backends")
DATASETS = Registry("datasets")
BACKBONES = Registry("backbones")
