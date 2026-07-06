"""Base class for KNN backends."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class KNNBackendBase(ABC):
    """Abstract base class for KNN backends."""
    
    @abstractmethod
    def query(self, queries: torch.Tensor, k: int) -> torch.Tensor:
        """Query k nearest neighbors.
        
        Args:
            queries: [N, D] query features
            k: Number of neighbors
            
        Returns:
            [N, k] distances to k nearest neighbors
        """
        pass
