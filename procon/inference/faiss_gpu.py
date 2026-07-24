"""FAISS GPU backend for fast KNN search."""
from __future__ import annotations

import torch

from procon.utils.registry import INFERENCE_BACKENDS


@INFERENCE_BACKENDS.register("faiss_gpu")
class FaissGPUBackend:
    """KNN backend using FAISS with GPU acceleration."""
    
    def __init__(
        self,
        bank: torch.Tensor,
        device: str = "cuda",
        normalize_l2: bool = False,
        use_float16: bool = True,
        **_: object,
    ) -> None:
        try:
            import faiss
            from faiss.contrib import torch_utils
        except ImportError as exc:
            raise ImportError("faiss[gpu] and faiss.contrib.torch_utils are required for faiss_gpu backend") from exc
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.normalize_l2 = bool(normalize_l2)
        self.use_float16 = bool(use_float16) and torch.cuda.is_available()
        
        if self.normalize_l2:
            bank = torch.nn.functional.normalize(bank, p=2, dim=1)
        target_dtype = torch.float16 if self.use_float16 else torch.float32
        bank_np = bank.detach().cpu().to(dtype=target_dtype).numpy()
        
        dim = bank_np.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = self.use_float16
            self.index = faiss.GpuIndexFlatL2(res, dim, cfg)
        
        self.index.add(bank_np)

    def query(self, queries: torch.Tensor, k: int) -> torch.Tensor:
        if self.normalize_l2:
            queries = torch.nn.functional.normalize(queries, p=2, dim=1)
        target_dtype = torch.float16 if self.use_float16 else torch.float32
        queries_np = queries.detach().cpu().to(dtype=target_dtype).numpy()
        
        distances, _ = self.index.search(queries_np, k)
        # Note: FAISS IndexFlatL2 returns squared L2 distances, keep as-is (no sqrt)
        # This matches patchcore original behavior
        return torch.from_numpy(distances).to(queries.device)
