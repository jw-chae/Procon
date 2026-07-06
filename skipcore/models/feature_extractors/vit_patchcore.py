"""ViT-based PatchCore feature extractor."""
from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F

from skipcore.models.backbones.backbone import build_backbone
from skipcore.utils.registry import FEATURE_EXTRACTORS


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim: int) -> None:
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim: int) -> None:
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


@FEATURE_EXTRACTORS.register("vit_patchcore")
class ViTPatchCoreExtractor:
    """Feature extractor for PatchCore using ViT backbone."""
    
    def __init__(
        self,
        backbone_cfg: dict,
        token_source: str = "patch",
        normalize: str | None = None,
        pretrain_embed_dimension: int | None = None,
        target_embed_dimension: int | None = None,
    ) -> None:
        self.backbone = build_backbone(backbone_cfg)
        self.token_source = token_source
        self.normalize = normalize
        self.last_grid_shape: Optional[Tuple[int, int]] = None
        self.enable_cam = False
        self.last_cam_tokens: Optional[torch.Tensor] = None

        if (pretrain_embed_dimension is None) != (target_embed_dimension is None):
            raise ValueError("Both pretrain_embed_dimension and target_embed_dimension must be set together.")
        self.use_mapping = pretrain_embed_dimension is not None
        self.mapper = MeanMapper(pretrain_embed_dimension) if self.use_mapping else None
        self.aggregator = Aggregator(target_embed_dimension) if self.use_mapping else None

    def _select_tokens(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.token_source == "patch":
            return patch_tokens
        if self.token_source == "cls":
            return cls_token.unsqueeze(1)
        if self.token_source == "patch+cls":
            return torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        raise KeyError(f"Unsupported token_source: {self.token_source}")

    def _apply_mapping(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_mapping:
            return tokens
        b, n, d = tokens.shape
        flat = tokens.reshape(b * n, d)
        flat = self.mapper(flat)
        flat = self.aggregator(flat)
        return flat.reshape(b, n, -1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.backbone, "extract_tokens"):
            raise RuntimeError("ViT backbone does not implement extract_tokens")
        cls_token, patch_tokens = self.backbone.extract_tokens(x)
        tokens = self._select_tokens(cls_token, patch_tokens)

        patch_count = patch_tokens.shape[1]
        grid = int(math.sqrt(patch_count))
        self.last_grid_shape = (grid, grid) if grid * grid == patch_count else None

        tokens = self._apply_mapping(tokens)
        if self.enable_cam:
            self.last_cam_tokens = tokens
            self.last_cam_tokens.retain_grad()
        if self.normalize == "l2":
            tokens = F.normalize(tokens, p=2, dim=-1)
        return tokens
