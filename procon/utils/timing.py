"""Timing and GPU memory measurement utilities."""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class TimingStats:
    """Accumulated timing statistics."""
    extract_ms: List[float] = field(default_factory=list)
    knn_ms: List[float] = field(default_factory=list)
    postprocess_ms: List[float] = field(default_factory=list)
    total_ms: List[float] = field(default_factory=list)
    
    def add_sample(
        self, 
        extract_ms: float, 
        knn_ms: float, 
        postprocess_ms: float, 
        total_ms: float
    ) -> None:
        """Add a timing sample."""
        self.extract_ms.append(extract_ms)
        self.knn_ms.append(knn_ms)
        self.postprocess_ms.append(postprocess_ms)
        self.total_ms.append(total_ms)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with mean values and FPS."""
        import numpy as np
        
        result = {}
        if self.extract_ms:
            result["extract_ms"] = float(np.mean(self.extract_ms))
        if self.knn_ms:
            result["knn_ms"] = float(np.mean(self.knn_ms))
        if self.postprocess_ms:
            result["postprocess_ms"] = float(np.mean(self.postprocess_ms))
        if self.total_ms:
            result["total_ms"] = float(np.mean(self.total_ms))
            result["fps"] = 1000.0 / result["total_ms"]
        return result
    
    def reset(self) -> None:
        """Reset all timing stats."""
        self.extract_ms.clear()
        self.knn_ms.clear()
        self.postprocess_ms.clear()
        self.total_ms.clear()


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0
    
    def __enter__(self) -> "Timer":
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
    
    def mark(self) -> float:
        """Mark current time and return elapsed ms since start."""
        if self.sync_cuda:
            torch.cuda.synchronize()
        current = time.perf_counter()
        return (current - self.start_time) * 1000


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)


def get_gpu_memory_peak_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_gpu_memory_stats() -> None:
    """Reset GPU memory stats and clear cache."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def get_bank_memory_mb(embeddings: torch.Tensor) -> float:
    """Get memory bank size in MB."""
    return embeddings.element_size() * embeddings.nelement() / (1024 ** 2)
