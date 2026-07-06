"""INP-inspired intrinsic prototype refinement for ConsensusCore.

This module borrows the INP-Former philosophy that a test image contains its own
intrinsic normal information -- *without training any decoder*. After external
consensus scoring, low-score patches of a single test image are treated as normal
candidates, a small set of prototypes is built from them, and the internal
distance to those prototypes is fused back into the score.
"""
from __future__ import annotations

import torch


def _kmeans(x: torch.Tensor, k: int, iters: int = 10, seed: int = 0) -> torch.Tensor:
    """Lightweight Lloyd's k-means; returns ``[k, D]`` centroids."""
    n = x.shape[0]
    k = max(1, min(k, n))
    g = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randperm(n, generator=g)[:k].to(x.device)
    centroids = x[init].clone()
    for _ in range(iters):
        d = torch.cdist(x, centroids)
        assign = d.argmin(dim=1)
        new_centroids = centroids.clone()
        for j in range(k):
            sel = assign == j
            if sel.any():
                new_centroids[j] = x[sel].mean(dim=0)
        if torch.allclose(new_centroids, centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def intrinsic_prototype_refine(
    patch_feats: torch.Tensor,
    s_ext: torch.Tensor,
    select_quantile: float = 0.7,
    num_prototypes: int = 8,
    alpha: float = 0.5,
    fusion: str = "add",
    seed: int = 0,
    return_components: bool = False,
):
    """Refine a single image's patch scores with intrinsic prototypes.

    Args:
        patch_feats: Patch features of one image, shape ``[P, D]``.
        s_ext: External consensus patch scores for that image, shape ``[P]``.
        select_quantile: Patches with ``s_ext <= Q_p`` are normal candidates.
        num_prototypes: Number of intrinsic prototypes ``K``.
        alpha: Internal-score weight.
        fusion: ``add`` (conservative) or ``max`` (aggressive).
        seed: RNG seed for prototype initialization.
        return_components: also return the internal score ``s_int``.

    Returns:
        Refined patch scores ``[P]``, or ``(s_final, s_int)`` if
        ``return_components`` is set.
    """
    p = patch_feats.shape[0]
    if p == 0:
        return (s_ext, torch.zeros_like(s_ext)) if return_components else s_ext

    thresh = torch.quantile(s_ext, float(select_quantile))
    omega = s_ext <= thresh
    if int(omega.sum().item()) < 2:
        return (s_ext, torch.zeros_like(s_ext)) if return_components else s_ext

    candidates = patch_feats[omega].float()
    protos = _kmeans(candidates, num_prototypes, seed=seed)
    s_int = torch.cdist(patch_feats.float(), protos).min(dim=1).values

    fusion = str(fusion).lower()
    if fusion == "max":
        s_final = torch.maximum(s_ext, float(alpha) * s_int)
    else:
        s_final = s_ext + float(alpha) * s_int
    return (s_final, s_int) if return_components else s_final
