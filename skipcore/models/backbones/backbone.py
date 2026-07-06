"""Backbone factory."""
from __future__ import annotations

from typing import Any, Dict

from skipcore.utils.registry import BACKBONES


def build_backbone(cfg: Dict[str, Any]):
    """Build backbone from configuration."""
    backbone_type = cfg.get("type", "dinov2_multilayer")
    cls = BACKBONES.get(backbone_type)
    args = {k: v for k, v in cfg.items() if k != "type"}
    return cls(**args)
