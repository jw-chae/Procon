"""K-Center greedy coreset selection."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from skipcore.memory.bank import MemoryBank
from skipcore.utils.dtype import resolve_dtype
from skipcore.utils.registry import MEMORY_BUILDERS


@MEMORY_BUILDERS.register("kcenter")
class KCenterMemoryBuilder:
    """K-Center greedy algorithm for coreset selection."""
    
    def __init__(
        self,
        K: int,
        seed: int = 0,
        max_samples: Optional[int] = None,
        dtype: str = "fp32",
    ) -> None:
        self.K = K
        self.seed = seed
        self.max_samples = max_samples
        self.dtype = dtype

    def __call__(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> MemoryBank:
        data = patches.to(dtype=torch.float32)
        n = data.shape[0]
        
        if self.max_samples is not None and n > self.max_samples:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(n, self.max_samples, replace=False)
            data = data[indices]
            if positions is not None:
                positions = positions[indices]
            n = data.shape[0]
        
        if self.K >= n:
            embeddings = data.to(dtype=resolve_dtype(self.dtype))
            return MemoryBank(embeddings=embeddings, positions=positions)
        
        device = data.device
        rng = np.random.default_rng(self.seed)
        centers = [int(rng.integers(0, n))]
        min_dists = torch.full((n,), float("inf"), device=device)
        
        for _ in range(self.K - 1):
            center = data[centers[-1]]
            dists = torch.sum((data - center) ** 2, dim=1).sqrt()
            min_dists = torch.minimum(min_dists, dists)
            next_idx = int(torch.argmax(min_dists).item())
            centers.append(next_idx)
        
        indices = torch.tensor(centers, dtype=torch.long)
        embeddings = data[indices].to(dtype=resolve_dtype(self.dtype))
        pos_out = None if positions is None else positions[indices]
        
        return MemoryBank(embeddings=embeddings, positions=pos_out)
