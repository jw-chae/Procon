"""Engine for building components from configuration."""
from __future__ import annotations

from typing import Any, Dict

import torch

from skipcore.utils.registry import FEATURE_EXTRACTORS, INFERENCE_BACKENDS, MEMORY_BUILDERS, SCORERS


def build_feature_extractor(cfg: Dict[str, Any], backbone_cfg: Dict[str, Any] | None = None):
    """Build feature extractor from config."""
    cls = FEATURE_EXTRACTORS.get(cfg["type"])
    args = dict(cfg)
    args.pop("type")
    if backbone_cfg is not None:
        args.setdefault("backbone_cfg", backbone_cfg)
    return cls(**args)


def build_memory_builder(cfg: Dict[str, Any]):
    """Build memory builder from config."""
    cls = MEMORY_BUILDERS.get(cfg["type"])
    args = dict(cfg)
    args.pop("type")
    return cls(**args)


def build_inference_backend(cfg: Dict[str, Any], bank: torch.Tensor):
    """Build inference backend from config."""
    cls = INFERENCE_BACKENDS.get(cfg["type"])
    args = dict(cfg)
    args.pop("type")
    if "device" not in args:
        args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cls(bank=bank, **args)


def build_scorer(
    cfg: Dict[str, Any],
    backend=None,
    bank: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    stats: Dict[str, Any] | None = None,
):
    """Build scorer from config."""
    cls = SCORERS.get(cfg["type"])
    args = dict(cfg)
    args.pop("type")
    return cls(backend=backend, bank=bank, positions=positions, stats=stats, **args)
