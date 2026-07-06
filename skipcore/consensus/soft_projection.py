"""Decoder-free INP-style soft projection for ConsensusCore (spec V8-V10).

Instead of scoring a test patch ``z`` by its distance to the *single* nearest
memory anchor, we reconstruct a normal approximation ``z_hat`` from a softmax-
weighted combination of its ``k`` nearest anchors and score the reconstruction
residual ``||z - z_hat||``. The memory bank thus acts as a *local normal
dictionary* rather than a set of isolated nearest references -- a training-free
approximation of INP-Former's projection, without any decoder.

For bank ``M_b`` and patch ``z``:
    N_k = kNN(z, M_b)
    w_j = softmax(-||z - m_j||^2 / tau)
    z_hat = sum_j w_j m_j
    r_b(z) = ||z - z_hat||
    H_b(z) = -sum_j w_j log w_j        (weight entropy)

A diffuse weight distribution (high entropy) can reconstruct anomalies too well,
so entropy is exposed as an optional diagnostic / penalty signal.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def auto_tau(queries: torch.Tensor, bank: torch.Tensor, k: int) -> float:
    """Estimate tau as the median squared distance to the selected neighbors.

    Computed on a subsample of queries for speed. Avoids overly large tau, which
    would flatten the weights and reconstruct anomalies too well.
    """
    n = queries.shape[0]
    if n == 0:
        return 1.0
    idx = torch.randperm(n, device=queries.device)[: min(n, 512)]
    sub = queries[idx].float()
    d2 = torch.cdist(sub, bank.float()) ** 2  # [s, M]
    kk = min(k, bank.shape[0])
    knn_d2 = torch.topk(d2, kk, dim=1, largest=False).values
    tau = float(torch.median(knn_d2).item())
    return max(tau, 1e-8)


def soft_projection_bank(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k: int = 5,
    tau: Optional[float] = None,
    return_entropy: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-patch reconstruction residual against one bank.

    Args:
        queries: ``[P, D]`` test patch features.
        bank: ``[M, D]`` memory anchors.
        k: number of local neighbors used for reconstruction.
        tau: softmax temperature; ``None`` -> :func:`auto_tau`.
        return_entropy: also return per-patch weight entropy.

    Returns:
        ``(residual [P], entropy [P] or None)``.
    """
    kk = min(k, bank.shape[0])
    if tau is None:
        tau = auto_tau(queries, bank, kk)

    # cdist needs fp32; keep large banks in fp16 to save GPU memory but compute
    # the distance in fp32 for numerical stability and half-precision support.
    q32 = queries.float()
    b32 = bank.float()
    d2_full = torch.cdist(q32, b32) ** 2  # [P, M]
    knn_d2, knn_idx = torch.topk(d2_full, kk, dim=1, largest=False)  # [P, k]
    # Softmax over negative squared distance.
    logits = -knn_d2 / tau
    w = torch.softmax(logits, dim=1)  # [P, k]
    neighbors = b32[knn_idx]  # [P, k, D]
    z_hat = (w.unsqueeze(-1) * neighbors).sum(dim=1)  # [P, D]
    residual = torch.linalg.vector_norm(q32 - z_hat, dim=1)  # [P]

    entropy = None
    if return_entropy:
        entropy = -(w * torch.log(w + 1e-12)).sum(dim=1)  # [P]
    return residual, entropy


def soft_projection_banks_vectorized(
    queries: torch.Tensor,
    banks: list,
    taus: list,
    k: int = 5,
    query_chunk: int = 0,
) -> torch.Tensor:
    """Bank-vectorized soft-projection residuals for ONE image, all banks.

    Equivalent to looping :func:`soft_projection_bank` over ``banks`` (same ``k``
    and the supplied per-bank ``taus``) and stacking the residuals, but stacks
    the banks into a ``[B, M, D]`` tensor and does a single batched
    ``cdist``/``topk``/gather. ``taus`` must be precomputed in the legacy
    per-bank order so the global RNG (``auto_tau``'s ``randperm``) is consumed
    identically; this function itself draws no randomness.

    The batched cdist uses a different fp32 reduction order than the per-bank
    cdist, so the result is *numerically equivalent* (residual diff ~1e-6) but
    not bit-identical. Use only where that tolerance is acceptable.

    Args:
        queries: ``[P, D]`` test patch features.
        banks: list of ``B`` tensors, each ``[M, D]`` (all the same ``M``).
        taus: list of ``B`` floats (legacy-order temperatures).
        k: neighbours used for reconstruction.
        query_chunk: if > 0, process queries in chunks of this size to bound the
            ``[B, chunk, M]`` distance tensor (needed for large ``M``).

    Returns:
        ``[B, P]`` residual tensor.
    """
    B = len(banks)
    P = queries.shape[0]
    M = banks[0].shape[0]
    kk = min(k, M)
    q32 = queries.float()                                    # [P, D]
    bstack = torch.stack([b.float() for b in banks], 0)      # [B, M, D]
    tau_t = torch.tensor(taus, device=q32.device,
                         dtype=q32.dtype).view(B, 1, 1)
    D = q32.shape[1]

    if query_chunk and query_chunk < P:
        out = q32.new_empty((B, P))
        for s in range(0, P, query_chunk):
            e = min(P, s + query_chunk)
            qc = q32[s:e].unsqueeze(0).expand(B, e - s, D)   # [B, c, D]
            d2 = torch.cdist(qc, bstack) ** 2                # [B, c, M]
            knn_d2, knn_idx = torch.topk(d2, kk, dim=2, largest=False)
            w = torch.softmax(-knn_d2 / tau_t, dim=2)        # [B, c, k]
            nb = torch.gather(
                bstack.unsqueeze(1).expand(B, e - s, M, D), 2,
                knn_idx.unsqueeze(-1).expand(B, e - s, kk, D))
            z_hat = (w.unsqueeze(-1) * nb).sum(dim=2)        # [B, c, D]
            out[:, s:e] = torch.linalg.vector_norm(qc - z_hat, dim=2)
        return out

    qb = q32.unsqueeze(0).expand(B, P, D)                    # [B, P, D]
    d2 = torch.cdist(qb, bstack) ** 2                        # [B, P, M]
    knn_d2, knn_idx = torch.topk(d2, kk, dim=2, largest=False)  # [B, P, k]
    w = torch.softmax(-knn_d2 / tau_t, dim=2)               # [B, P, k]
    nb = torch.gather(
        bstack.unsqueeze(1).expand(B, P, M, D), 2,
        knn_idx.unsqueeze(-1).expand(B, P, kk, D))           # [B, P, k, D]
    z_hat = (w.unsqueeze(-1) * nb).sum(dim=2)               # [B, P, D]
    return torch.linalg.vector_norm(qb - z_hat, dim=2)      # [B, P]


def soft_projection_bank_multi(
    queries: torch.Tensor,
    bank: torch.Tensor,
    k_list: list,
    tau_scales: list,
    tau0: Optional[float] = None,
) -> dict:
    """Compute soft-projection residuals for several (k, tau) scales at once.

    Efficiency: the top-``max(k_list)`` neighbors are computed *once* per bank,
    and smaller-k results are derived by slicing. tau scales reuse the same
    neighbor set. ``tau0`` defaults to :func:`auto_tau` at ``max(k_list)``.

    Args:
        queries: ``[P, D]`` test patch features.
        bank: ``[M, D]`` memory anchors.
        k_list: neighborhood sizes, e.g. ``[3, 5, 7]``.
        tau_scales: temperature scale factors, e.g. ``[0.5, 1.0, 2.0]``.
        tau0: base temperature; ``None`` -> auto at ``max(k_list)``.

    Returns:
        Dict mapping ``(k, scale)`` -> residual tensor ``[P]``.
    """
    k_max = min(max(k_list), bank.shape[0])
    if tau0 is None:
        tau0 = auto_tau(queries, bank, k_max)

    d2_full = torch.cdist(queries, bank) ** 2  # [P, M]
    knn_d2, knn_idx = torch.topk(d2_full, k_max, dim=1, largest=False)  # [P,kmax]
    neighbors_max = bank[knn_idx]  # [P, kmax, D]

    out: dict = {}
    for k in k_list:
        kk = min(k, k_max)
        d2_k = knn_d2[:, :kk]          # [P, k]
        nb_k = neighbors_max[:, :kk]   # [P, k, D]
        for scale in tau_scales:
            tau = max(tau0 * float(scale), 1e-8)
            w = torch.softmax(-d2_k / tau, dim=1)        # [P, k]
            z_hat = (w.unsqueeze(-1) * nb_k).sum(dim=1)  # [P, D]
            res = torch.linalg.vector_norm(queries - z_hat, dim=1)  # [P]
            out[(int(k), float(scale))] = res
    return out

