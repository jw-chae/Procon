"""Memory bank for storing coreset embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch


@dataclass
class MemoryBank:
    """Container for memory bank embeddings and metadata."""
    embeddings: torch.Tensor
    positions: Optional[torch.Tensor]
    stats: Optional[Dict[str, torch.Tensor]] = None
    metadata: Dict[str, str] = None

    def save(self, path: str | Path) -> None:
        """Save memory bank to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        meta = {} if self.metadata is None else self.metadata
        torch.save({
            "embeddings": self.embeddings,
            "positions": self.positions,
            "stats": self.stats,
            "metadata": meta,
        }, path)

    @staticmethod
    def load(path: str | Path) -> "MemoryBank":
        """Load memory bank from file."""
        data = torch.load(path, map_location="cpu")
        return MemoryBank(
            embeddings=data["embeddings"],
            positions=data.get("positions"),
            stats=data.get("stats"),
            metadata=data.get("metadata", {}),
        )
