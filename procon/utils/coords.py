"""Coordinate utilities for patch positions."""
from __future__ import annotations

from typing import Tuple

import torch


def make_patch_positions(grid_shape: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create normalized (0-1) patch position coordinates."""
    h, w = grid_shape
    if h <= 1:
        ys = torch.zeros(h, device=device, dtype=dtype)
    else:
        ys = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype)
    if w <= 1:
        xs = torch.zeros(w, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([yy, xx], dim=-1).reshape(h * w, 2)
    return coords
