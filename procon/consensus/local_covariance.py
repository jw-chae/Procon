"""Local geometry residuals for V9 extension (spec V26 / V27).

V9 reconstructs ``z`` as a softmax-weighted *mean* of nearby anchors, implicitly
assuming the local normal manifold is isotropic. These variants instead model the
local normal geometry explicitly:

* V26 local PCA: fit a rank-``r`` tangent subspace from the ``k_pca`` nearest
  anchors and score the off-subspace residual ``||(z-mu) - U U^T (z-mu)||``.
  Anomalies that deviate orthogonally to the normal tangent plane are amplified.
* V27 local Mahalanobis: whiten ``(z-mu)`` by the local covariance (estimated in
  the PCA-reduced subspace for stability) and score the Mahalanobis norm.

Everything is computed from the frozen memory anchors -- no training.
"""
from __future__ import annotations

from typing import Optional

import torch


def _local_neighbors(queries: torch.Tensor, bank: torch.Tensor, k: int):
    """Return ``([P, k, D]`` neighbor coords, ``[P, k]`` distances)."""
    kk = min(k, bank.shape[0])
    d2 = torch.cdist(queries, bank) ** 2
    knn_d2, knn_idx = torch.topk(d2, kk, dim=1, largest=False)
    return bank[knn_idx], knn_d2


def local_pca_residual_bank(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k_pca: int = 10,
    rank: int = 2,
) -> torch.Tensor:
    """Off-subspace residual against a local PCA tangent plane (V26).

    Args:
        queries: ``[P, D]`` test patch features.
        bank: ``[M, D]`` memory anchors.
        k_pca: neighbors used to fit the local subspace.
        rank: subspace rank ``r``.

    Returns:
        Residual ``[P]``.
    """
    neighbors, _ = _local_neighbors(queries, bank, k_pca)  # [P, k, D]
    mu = neighbors.mean(dim=1)                              # [P, D]
    centered = neighbors - mu.unsqueeze(1)                  # [P, k, D]
    # Batched SVD: principal directions are the right singular vectors.
    # centered = U S Vh  with Vh: [P, min(k,D), D].
    r = max(1, min(rank, centered.shape[1], centered.shape[2]))
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        _, _, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
        vh = vh.to(queries.device)
    basis = vh[:, :r, :]                                    # [P, r, D]
    delta = (queries - mu).unsqueeze(1)                    # [P, 1, D]
    coeff = torch.matmul(delta, basis.transpose(1, 2))     # [P, 1, r]
    proj = torch.matmul(coeff, basis).squeeze(1)           # [P, D]
    residual = torch.linalg.vector_norm((queries - mu) - proj, dim=1)
    return residual


def local_mahalanobis_bank(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k_maha: int = 10,
    rank: int = 4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Local Mahalanobis residual in a PCA-reduced subspace (V27).

    The full ``D x D`` covariance is unstable in high dimensions, so the
    neighborhood is reduced to its top-``rank`` PCA directions and the
    Mahalanobis distance is computed there (plus the orthogonal residual energy).

    Returns:
        Mahalanobis-style score ``[P]``.
    """
    neighbors, _ = _local_neighbors(queries, bank, k_maha)  # [P, k, D]
    mu = neighbors.mean(dim=1)                               # [P, D]
    centered = neighbors - mu.unsqueeze(1)                   # [P, k, D]
    k = centered.shape[1]
    r = max(1, min(rank, k, centered.shape[2]))
    try:
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        _, s, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
        s = s.to(queries.device)
        vh = vh.to(queries.device)
    basis = vh[:, :r, :]                                     # [P, r, D]
    # Per-direction variance from singular values (sample covariance).
    var = (s[:, :r] ** 2) / max(k - 1, 1)                    # [P, r]
    delta = queries - mu                                    # [P, D]
    coeff = torch.matmul(delta.unsqueeze(1),
                         basis.transpose(1, 2)).squeeze(1)   # [P, r]
    maha_in = (coeff ** 2 / (var + eps)).sum(dim=1)         # within-subspace
    # Orthogonal residual energy, normalized by the smallest retained variance.
    recon = torch.matmul(coeff.unsqueeze(1), basis).squeeze(1)
    ortho = torch.linalg.vector_norm(delta - recon, dim=1) ** 2
    ortho_var = var[:, -1] + eps
    score = maha_in + ortho / ortho_var
    return torch.sqrt(torch.clamp(score, min=0.0))
