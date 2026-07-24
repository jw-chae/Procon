"""Construction of multiple perturbed coreset memory banks for ConsensusCore.

Each bank is an independently perturbed *discrete normal projector*. Diversity is
induced through one of the following perturbation sources (section 3 of the spec):

    seed       -- different greedy k-center seeds + random-projection seeds
    ratio      -- different memory ratios cycled across banks
    bootstrap  -- image-level bootstrap (sample training images with replacement)

The proposed default is ``seed`` diversity: same frozen features, same memory
ratio, different k-center / random-projection seeds.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from procon.memory.bank import MemoryBank
from procon.memory.builders.approx_greedy_coreset import ApproxGreedyCoresetBuilder
from procon.utils.seed import set_seed


def _kcenter_bank(
    patches: torch.Tensor,
    ratio: float,
    seed: int,
    proj_dim: int,
    device: Optional[str],
    dtype: str = "fp32",
) -> MemoryBank:
    """Build a single greedy-coreset (k-center) memory bank."""
    # Re-seed the global RNG so the builder's internal random projection differs
    # per bank, which is the random-projection perturbation source.
    set_seed(seed)
    builder = ApproxGreedyCoresetBuilder(
        percentage=ratio,
        seed=seed,
        dimension_to_project_features_to=proj_dim,
        device=device,
        dtype=dtype,
    )
    return builder(patches, None)


def _random_bank(patches: torch.Tensor, ratio: float, seed: int, dtype: str = "fp32") -> MemoryBank:
    """Build a memory bank by uniformly sampling ``ratio`` of the patches."""
    n = patches.shape[0]
    k = max(1, int(round(n * ratio)))
    rng = np.random.default_rng(seed)
    idx = torch.from_numpy(rng.choice(n, size=k, replace=False))
    emb = patches[idx]
    if dtype == "fp16":
        emb = emb.to(torch.float16)
    return MemoryBank(embeddings=emb, positions=None, metadata={"method": "random", "seed": str(seed)})


def build_consensus_banks(
    patches: torch.Tensor,
    image_index: Optional[torch.Tensor],
    num_banks: int,
    ratio: float,
    seed: int = 0,
    diversity: str = "seed",
    ratios: Optional[List[float]] = None,
    proj_dim: int = 196,
    device: Optional[str] = None,
    dtype: str = "fp32",
) -> List[MemoryBank]:
    """Construct ``num_banks`` perturbed coreset memory banks.

    Args:
        patches: Pooled normal training patch features ``[N, D]`` (CPU tensor).
        image_index: Optional ``[N]`` tensor mapping each patch to its source
            image; required only for ``diversity == "bootstrap"``.
        num_banks: Number of banks ``B`` to build.
        ratio: Base memory ratio ``r`` (e.g. 0.01).
        seed: Base random seed; bank ``b`` uses ``seed + b``.
        diversity: One of ``seed``, ``ratio``, ``bootstrap``.
        ratios: Ratio list used when ``diversity == "ratio"``; banks cycle through
            it (defaults to ``[0.005, 0.01, 0.02]``).
        proj_dim: Projection dim used inside the greedy-coreset distance step.
        device: Builder device override.
        dtype: Stored embedding dtype.

    Returns:
        List of ``MemoryBank`` objects.
    """
    diversity = str(diversity).lower()
    banks: List[MemoryBank] = []

    if diversity == "ratio":
        ratios = ratios or [0.005, 0.01, 0.02]
        for b in range(num_banks):
            r = float(ratios[b % len(ratios)])
            banks.append(_kcenter_bank(patches, r, seed + b, proj_dim, device, dtype))
        return banks

    if diversity == "bootstrap":
        if image_index is None:
            raise ValueError("bootstrap diversity requires image_index")
        unique_imgs = torch.unique(image_index)
        n_imgs = unique_imgs.numel()
        for b in range(num_banks):
            rng = np.random.default_rng(seed + b)
            chosen = unique_imgs[torch.from_numpy(rng.choice(n_imgs, size=n_imgs, replace=True))]
            mask = torch.isin(image_index, chosen)
            sub = patches[mask]
            banks.append(_kcenter_bank(sub, ratio, seed + b, proj_dim, device, dtype))
        return banks

    # Default: seed / random-projection perturbation.
    for b in range(num_banks):
        banks.append(_kcenter_bank(patches, ratio, seed + b, proj_dim, device, dtype))
    return banks


def build_single_bank(
    patches: torch.Tensor,
    ratio: float,
    seed: int,
    method: str = "kcenter",
    proj_dim: int = 196,
    device: Optional[str] = None,
    dtype: str = "fp32",
) -> MemoryBank:
    """Build one bank for the single / full / random ablation baselines.

    Args:
        method: ``kcenter`` (greedy coreset), ``full`` (all patches) or
            ``random`` (uniform subsample).
    """
    method = str(method).lower()
    if method == "full":
        emb = patches if dtype != "fp16" else patches.to(torch.float16)
        return MemoryBank(embeddings=emb, positions=None, metadata={"method": "full"})
    if method == "random":
        return _random_bank(patches, ratio, seed, dtype)
    return _kcenter_bank(patches, ratio, seed, proj_dim, device, dtype)
