"""Robust distance-fusion operators for ConsensusCore.

Given per-bank nearest-neighbor distances ``D`` of shape ``[B, M]`` (B memory
banks, M query patches), fuse them into a single distance vector of shape
``[M]``.

The central hypothesis is that a single coreset memory can falsely project an
anomalous patch onto normal memory through one unstable anchor. Minimum fusion
reproduces that single-anchor failure mode, so it is intentionally *not* the
default. Median / upper-quantile / trimmed-mean fusion require a patch to be
consistently supported by most banks before it is accepted as normal.
"""
from __future__ import annotations

import torch


def fuse_distances(
    distances: torch.Tensor,
    mode: str = "median",
    quantile: float = 0.75,
    trim_ratio: float = 0.1,
) -> torch.Tensor:
    """Fuse per-bank distances ``[B, M]`` into ``[M]``.

    Args:
        distances: Tensor of shape ``[B, M]`` with one row per memory bank.
        mode: One of ``mean``, ``median``, ``quantile``, ``trimmed_mean``,
            ``max`` or ``min``. ``min`` is provided only for the single-anchor
            ablation and should not be used for the proposed method.
        quantile: Quantile in ``[0, 1]`` used when ``mode == "quantile"``.
        trim_ratio: Fraction trimmed from each tail for ``trimmed_mean``.

    Returns:
        Fused distance tensor of shape ``[M]``.
    """
    if distances.ndim == 1:
        return distances
    if distances.shape[0] == 1:
        return distances[0]

    mode = str(mode).lower()
    if mode == "mean":
        return distances.mean(dim=0)
    if mode == "median":
        return distances.median(dim=0).values
    if mode == "max":
        return distances.max(dim=0).values
    if mode == "min":
        # Single-anchor failure mode; kept only for ablation parity.
        return distances.min(dim=0).values
    if mode == "quantile":
        q = float(min(max(quantile, 0.0), 1.0))
        return torch.quantile(distances, q, dim=0)
    if mode == "trimmed_mean":
        b = distances.shape[0]
        k = int(round(b * float(trim_ratio)))
        sorted_d, _ = distances.sort(dim=0)
        if 2 * k >= b:
            # Trimming would remove everything; fall back to the median.
            return distances.median(dim=0).values
        return sorted_d[k:b - k].mean(dim=0)
    raise KeyError(f"Unsupported fusion mode: {mode}")
