"""V9-aware (soft-projection-aware) memory selection (spec V20 / V21 / V22).

Standard k-center coreset selection optimizes nearest-neighbor coverage, which is
not necessarily optimal for V9, where anchors act as a local reconstruction
basis. These builders instead pick anchors that minimize the *soft-projection
reconstruction residual* of normal training patches.

* V20 reconstruction-greedy: from a larger k-center candidate pool, greedily add
  the candidate that most reduces sampled normal reconstruction residual.
* V21 + diversity: add a min-distance diversity bonus to avoid redundant anchors.
* V22 reconstruction-pruned k-center: build a large k-center memory, score each
  anchor by its reconstruction contribution, keep the top fraction.

All operate on frozen features; nothing is trained.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from skipcore.memory.bank import MemoryBank
from skipcore.memory.builders.approx_greedy_coreset import ApproxGreedyCoresetBuilder
from skipcore.utils.seed import set_seed


def _kcenter_indices(patches: torch.Tensor, ratio: float, seed: int,
                     proj_dim: int, device: Optional[str]) -> torch.Tensor:
    """Run greedy k-center and return the selected row indices into ``patches``."""
    set_seed(seed)
    builder = ApproxGreedyCoresetBuilder(
        percentage=ratio, seed=seed,
        dimension_to_project_features_to=proj_dim, device=device, dtype="fp32",
    )
    bank = builder(patches, None)
    emb = bank.embeddings.to(torch.float32)
    # Map the selected embeddings back to original indices via nearest match.
    d = torch.cdist(emb, patches.to(torch.float32))
    return d.argmin(dim=1)


def _softproj_residual(x: torch.Tensor, anchors: torch.Tensor,
                       k: int, tau: Optional[float]) -> torch.Tensor:
    """Per-sample soft-projection residual of ``x`` against ``anchors`` -> [N]."""
    kk = min(k, anchors.shape[0])
    d2 = torch.cdist(x, anchors) ** 2
    knn_d2, knn_idx = torch.topk(d2, kk, dim=1, largest=False)
    if tau is None:
        tau = float(torch.median(knn_d2).item()) or 1e-8
    w = torch.softmax(-knn_d2 / tau, dim=1)
    nb = anchors[knn_idx]
    z_hat = (w.unsqueeze(-1) * nb).sum(dim=1)
    return torch.linalg.vector_norm(x - z_hat, dim=1)


def build_recgreedy_memory(
    patches: torch.Tensor,
    ratio: float,
    seed: int,
    candidate_ratio: float = 0.05,
    sample_patches: int = 20000,
    k: int = 5,
    tau: Optional[float] = None,
    diversity_lambda: float = 0.0,
    proj_dim: int = 196,
    device: Optional[str] = None,
    dtype: str = "fp32",
) -> MemoryBank:
    """Reconstruction-greedy memory (V20) with optional diversity bonus (V21)."""
    dev = torch.device(device if device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    n = patches.shape[0]
    target = max(1, int(round(n * ratio)))

    # Candidate pool from a larger k-center selection.
    cand_idx = _kcenter_indices(patches, candidate_ratio, seed, proj_dim, device)
    cand = patches[cand_idx].to(dev, torch.float32)          # [C, D]
    n_cand = cand.shape[0]
    target = min(target, n_cand)

    # Sampled normal patches to estimate reconstruction residual.
    rng = np.random.default_rng(seed)
    s = min(sample_patches, n)
    sub_idx = torch.from_numpy(rng.choice(n, size=s, replace=False))
    sub = patches[sub_idx].to(dev, torch.float32)            # [S, D]

    # Greedy selection by marginal reconstruction-residual reduction.
    selected: List[int] = []
    # Seed with the candidate whose single-anchor projection helps most: use the
    # medoid (closest to the sample mean) as a cheap start.
    start = int(torch.cdist(cand, sub.mean(0, keepdim=True)).argmin().item())
    selected.append(start)

    kk = max(1, min(k, target))
    while len(selected) < target:
        cur = cand[selected]                                  # [m, D]
        base_res = _softproj_residual(sub, cur, kk, tau).mean()
        sel_set = set(selected)
        pool = [c for c in range(n_cand) if c not in sel_set]
        if not pool:
            break
        # Evaluate a random subset of candidates each step for tractability.
        if len(pool) > 192:
            pool = rng.choice(pool, size=192, replace=False).tolist()
        gains = []
        for c in pool:
            trial = torch.cat([cur, cand[c:c + 1]], dim=0)
            res = _softproj_residual(sub, trial, min(kk + 1, trial.shape[0]),
                                     tau).mean()
            gain = float(base_res - res)
            if diversity_lambda > 0.0:
                div = float(torch.cdist(cand[c:c + 1], cur).min())
                gain = gain + diversity_lambda * div
            gains.append((gain, c))
        gains.sort(key=lambda t: -t[0])
        # Batch-add the best few candidates per step to keep runtime bounded.
        n_add = max(1, min(target // 40, target - len(selected), len(gains)))
        for _, c in gains[:n_add]:
            selected.append(c)

    final = cand_idx[torch.tensor(selected, dtype=torch.long)]
    emb = patches[final]
    if dtype == "fp16":
        emb = emb.to(torch.float16)
    method = "recgreedy_diverse" if diversity_lambda > 0 else "recgreedy"
    return MemoryBank(embeddings=emb, positions=None,
                      metadata={"method": method, "count": str(emb.shape[0])})


def build_rec_pruned_kcenter(
    patches: torch.Tensor,
    final_ratio: float,
    seed: int,
    large_ratio: float = 0.05,
    k: int = 5,
    tau: Optional[float] = None,
    proj_dim: int = 196,
    device: Optional[str] = None,
    dtype: str = "fp32",
) -> MemoryBank:
    """Reconstruction-pruned k-center (V22): keep top-contribution anchors."""
    dev = torch.device(device if device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    n = patches.shape[0]
    large_idx = _kcenter_indices(patches, large_ratio, seed, proj_dim, device)
    anchors = patches[large_idx].to(dev, torch.float32)       # [L, D]
    L = anchors.shape[0]
    keep = max(1, min(int(round(n * final_ratio)), L))

    # Sample normal patches; measure each anchor's leave-one-out contribution.
    rng = np.random.default_rng(seed)
    s = min(20000, n)
    sub = patches[torch.from_numpy(rng.choice(n, size=s, replace=False))]
    sub = sub.to(dev, torch.float32)

    kk = min(k, L)
    d2 = torch.cdist(sub, anchors) ** 2                       # [S, L]
    knn_d2, knn_idx = torch.topk(d2, kk, dim=1, largest=False)
    if tau is None:
        tau = float(torch.median(knn_d2).item()) or 1e-8
    # Contribution proxy: total softmax weight each anchor receives.
    w = torch.softmax(-knn_d2 / tau, dim=1)                   # [S, k]
    contrib = torch.zeros(L, device=dev)
    contrib.scatter_add_(0, knn_idx.reshape(-1), w.reshape(-1))

    top = torch.topk(contrib, keep, largest=True).indices
    final = large_idx[top.cpu()]
    emb = patches[final]
    if dtype == "fp16":
        emb = emb.to(torch.float16)
    return MemoryBank(embeddings=emb, positions=None,
                      metadata={"method": "rec_pruned_kcenter",
                                "count": str(emb.shape[0])})
