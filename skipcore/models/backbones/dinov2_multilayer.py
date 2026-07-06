"""
DINOv2 Multi-layer Backbone for SkipCore.

Implements multi-layer feature extraction from DINOv2 ViT models,
with layer fusion and random projection for dimensionality reduction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skipcore.utils.registry import BACKBONES


def _make_random_projection(in_dim: int, out_dim: int, seed: int = 42) -> nn.Linear:
    """Create a fixed random projection matrix (Gaussian)."""
    rng = np.random.RandomState(seed)
    weight = rng.randn(out_dim, in_dim).astype(np.float32) / np.sqrt(in_dim)
    proj = nn.Linear(in_dim, out_dim, bias=False)
    proj.weight.data = torch.from_numpy(weight)
    proj.weight.requires_grad_(False)
    return proj


@BACKBONES.register("dinov2_multilayer")
class DINOv2MultiLayerBackbone:
    """
    DINOv2 backbone with multi-layer feature extraction and fusion.
    
    Args:
        model_name: DINOv2 model variant (dinov2_vits14, dinov2_vitb14, etc.)
        pretrained: Whether to load pretrained weights
        layers: List of layer indices to extract from (negative indices supported)
        layer_fusion: How to fuse multi-layer features (weighted_sum, concat, sum)
        layer_weights: Weight scheme for weighted_sum mode
        layer_norm_mode: Per-layer normalization (l2, layernorm, None)
        proj_type: Projection method (rp for random projection, none)
        proj_dim: Target dimension after projection
        proj_seed: Random seed for reproducible random projection
        final_norm: Whether to L2-normalize fused features
    """
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        pretrained: bool = True,
        layers: List[int] = [-1, -2, -3, -4],
        layer_fusion: str = "weighted_sum",
        layer_weights: Union[str, List[float]] = "uniform",
        layer_norm_mode: Optional[str] = "l2",
        proj_type: str = "rp",
        proj_dim: int = 256,
        proj_seed: int = 42,
        final_norm: bool = True,
    ) -> None:
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        self.blocks = getattr(self.model, "blocks", None)
        if self.blocks is None:
            raise RuntimeError("DINOv2 model does not have 'blocks' attribute")
        n_blocks = len(self.blocks)
        
        # Normalize layer indices
        self.layer_indices = []
        for idx in layers:
            if idx < 0:
                idx = n_blocks + idx
            if 0 <= idx < n_blocks:
                self.layer_indices.append(idx)
            else:
                raise ValueError(f"Layer index {idx} out of range for {n_blocks} blocks")
        self.layer_indices = sorted(set(self.layer_indices))
        
        # Register hooks
        self._layer_outputs: Dict[int, torch.Tensor] = {}
        for idx in self.layer_indices:
            self.blocks[idx].register_forward_hook(self._make_hook(idx))
        
        self.layer_fusion = layer_fusion
        self.n_layers = len(self.layer_indices)
        
        # Compute layer weights
        if isinstance(layer_weights, list):
            if len(layer_weights) != self.n_layers:
                raise ValueError(f"layer_weights length {len(layer_weights)} != n_layers {self.n_layers}")
            self.layer_weights = torch.tensor(layer_weights, dtype=torch.float32)
        elif layer_weights == "uniform":
            self.layer_weights = torch.ones(self.n_layers) / self.n_layers
        elif layer_weights == "linear_recent_heavier":
            weights = torch.arange(1, self.n_layers + 1, dtype=torch.float32)
            self.layer_weights = weights / weights.sum()
        else:
            raise ValueError(f"Unknown layer_weights: {layer_weights}")
        
        self.layer_norm_mode = layer_norm_mode
        self.final_norm = final_norm
        self.embed_dim = self.model.embed_dim
        
        if layer_fusion == "concat":
            self.fused_dim = self.embed_dim * self.n_layers
        else:
            self.fused_dim = self.embed_dim
        
        self.proj_type = proj_type
        self.proj_dim = proj_dim
        self.projection = None
        if proj_type == "rp":
            self.projection = _make_random_projection(self.fused_dim, proj_dim, seed=proj_seed)
            self.output_dim = proj_dim
        elif proj_type == "none":
            self.output_dim = self.fused_dim
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")
    
    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self._layer_outputs[layer_idx] = output
        return hook
    
    def _fuse_layers(self, layer_tokens: List[torch.Tensor]) -> torch.Tensor:
        if self.layer_norm_mode == "l2":
            layer_tokens = [F.normalize(t, p=2, dim=-1) for t in layer_tokens]
        elif self.layer_norm_mode == "layernorm":
            layer_tokens = [
                (t - t.mean(dim=-1, keepdim=True)) / (t.std(dim=-1, keepdim=True) + 1e-6)
                for t in layer_tokens
            ]
        
        stacked = torch.stack(layer_tokens, dim=0)
        
        if self.layer_fusion == "concat":
            return torch.cat(layer_tokens, dim=-1)
        elif self.layer_fusion == "sum":
            return stacked.sum(dim=0)
        elif self.layer_fusion == "weighted_sum":
            weights = self.layer_weights.view(-1, 1, 1, 1).to(stacked.device)
            return (stacked * weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown layer_fusion: {self.layer_fusion}")
    
    def extract_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and fuse multi-layer patch tokens."""
        self._layer_outputs.clear()
        
        with torch.no_grad():
            _ = self.model(x)
        
        layer_tokens_list = []
        for idx in self.layer_indices:
            if idx not in self._layer_outputs:
                raise RuntimeError(f"Failed to capture tokens from layer {idx}")
            tokens = self._layer_outputs[idx]
            patch_tokens = tokens[:, 1:, :]
            layer_tokens_list.append(patch_tokens)
        
        fused = self._fuse_layers(layer_tokens_list)
        
        if self.projection is not None:
            self.projection = self.projection.to(device=fused.device, dtype=fused.dtype)
            b, n, d = fused.shape
            fused = self.projection(fused.reshape(b * n, d)).reshape(b, n, -1)
        
        if self.final_norm:
            fused = F.normalize(fused, p=2, dim=-1)
        
        final_idx = self.layer_indices[-1]
        cls_token = self._layer_outputs[final_idx][:, 0, :].clone()
        
        self._layer_outputs.clear()
        
        return cls_token, fused

    def extract_per_layer(
        self,
        x: torch.Tensor,
        layers: List[int],
        projections: Dict[int, nn.Module],
        per_layer_norm: bool = True,
        final_norm: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """Single forward; return each requested layer's projected patch tokens.

        Reproduces, in ONE forward pass, what a set of single-layer group
        extractors (``layer_fusion='concat'`` with one layer each) would
        produce independently. For each requested (negative) layer index:
            patch_tokens -> [per-layer L2] -> RP(projections[layer]) -> [L2]

        ``projections`` maps the (negative) layer index to that group's random
        projection module. This is mathematically identical to running a
        separate :class:`DINOv2MultiLayerBackbone` per layer (verified
        bit-for-bit), but does the heavy DINOv2 forward only once.
        """
        self._layer_outputs.clear()
        with torch.no_grad():
            _ = self.model(x)
        n_blocks = len(self.blocks)
        out: Dict[int, torch.Tensor] = {}
        for layer in layers:
            idx = n_blocks + layer if layer < 0 else layer
            if idx not in self._layer_outputs:
                raise RuntimeError(f"Failed to capture tokens from layer {layer}")
            patch = self._layer_outputs[idx][:, 1:, :]   # [B, P, 768]
            if per_layer_norm:
                patch = F.normalize(patch, p=2, dim=-1)
            proj = projections[layer]
            proj = proj.to(device=patch.device, dtype=patch.dtype)
            b, p, d = patch.shape
            feat = proj(patch.reshape(b * p, d)).reshape(b, p, -1)
            if final_norm:
                feat = F.normalize(feat, p=2, dim=-1)
            out[layer] = feat
        self._layer_outputs.clear()
        return out

    def to(self, device):
        self.model.to(device)
        if self.projection is not None:
            self.projection.to(device)
        return self
    
    @property
    def embed_dimension(self) -> int:
        return self.output_dim
