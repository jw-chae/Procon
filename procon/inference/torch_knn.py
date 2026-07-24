"""Pure PyTorch KNN backend."""
from __future__ import annotations

import torch

from procon.utils.registry import INFERENCE_BACKENDS


@INFERENCE_BACKENDS.register("torch_knn")
class TorchKNNBackend:
    """KNN backend using pure PyTorch operations."""
    
    def __init__(self, bank: torch.Tensor, device: str = "cuda", normalize_l2: bool = False, **_: object) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.normalize_l2 = bool(normalize_l2)
        self.bank = bank.to(self.device)
        if self.normalize_l2:
            self.bank = torch.nn.functional.normalize(self.bank, p=2, dim=1)

    def query(self, queries: torch.Tensor, k: int) -> torch.Tensor:
        if self.bank.device != queries.device or self.bank.dtype != queries.dtype:
            self.bank = self.bank.to(device=queries.device, dtype=queries.dtype)
        if self.normalize_l2:
            queries = torch.nn.functional.normalize(queries, p=2, dim=1)
        distances = torch.cdist(queries, self.bank)
        values, _ = torch.topk(distances, k=k, dim=1, largest=False)
        return values
