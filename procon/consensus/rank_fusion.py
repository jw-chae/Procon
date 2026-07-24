"""Per-image rank normalization for heterogeneous expert fusion.

Direct addition of expert score maps is dangerous: failed recipes inflated the
*normal* score floor, so adding raw scores can drown the ranking signal. Rank
normalization keeps only each expert's *ordering* within an image, mapping every
patch to ``[0, 1]`` where 1 is the most anomalous. This removes score-scale
mismatch between heterogeneous experts (NN distance vs reconstruction residual).
"""
from __future__ import annotations

import torch


def rank_normalize_map(score_map: torch.Tensor) -> torch.Tensor:
    """Per-image rank normalization of a 1-D patch-score vector ``[P]``.

    Returns ``rank(S) / (P - 1)`` in ``[0, 1]``; ties are broken by the stable
    argsort order, which is sufficient for ranking-based metrics.
    """
    if score_map.ndim != 1:
        score_map = score_map.reshape(-1)
    p = score_map.shape[0]
    if p <= 1:
        return torch.zeros_like(score_map)
    order = torch.argsort(score_map, dim=0)
    ranks = torch.empty_like(score_map)
    ranks[order] = torch.arange(p, device=score_map.device, dtype=score_map.dtype)
    return ranks / float(p - 1)
