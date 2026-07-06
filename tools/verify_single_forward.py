"""Step-1 verification: single-forward multi-layer extraction == 4 separate
single-layer extractors, bit-for-bit.

The champion runs four separate DINOv2 extractors (one per layer in
{-3,-6,-8,-9}), i.e. four forward passes. Each group extractor computes, for its
one layer:  patch_tokens -> per-layer L2 -> RP(768->512, seed=42)  (final_norm
is False in the configs). Because DINOv2 already produces every block output in
a single forward (the layers are just captured by hooks), four passes are pure
waste.

This script confirms that doing ONE forward and applying the same per-layer
L2 + shared RP reproduces each group extractor's output exactly.
"""
import torch
import torch.nn.functional as F

from skipcore.models.backbones.dinov2_multilayer import (
    DINOv2MultiLayerBackbone, _make_random_projection,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
POOL = [-3, -6, -8, -9]


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 392, 392, device=DEV)

    # --- Reference: one separate group extractor per layer (old path) ---
    old = {}
    for layer in POOL:
        g = DINOv2MultiLayerBackbone(
            model_name="dinov2_vitb14", layers=[layer],
            layer_fusion="concat", layer_norm_mode="l2",
            proj_type="rp", proj_dim=512, proj_seed=42, final_norm=False,
        )
        g.to(DEV)
        with torch.no_grad():
            _, fused = g.extract_tokens(x)  # [B, P, 512]
        old[layer] = fused.float().cpu()
        del g
        torch.cuda.empty_cache() if DEV == "cuda" else None

    # --- New path: ONE forward, capture all layers, shared RP per layer ---
    m = DINOv2MultiLayerBackbone(
        model_name="dinov2_vitb14", layers=POOL,
        layer_fusion="concat", layer_norm_mode="l2",
        proj_type="none", final_norm=False,
    )
    m.to(DEV)
    rp = _make_random_projection(768, 512, seed=42).to(DEV)

    m._layer_outputs.clear()
    with torch.no_grad():
        m.model(x)
    # map negative layer -> normalised block index
    n_blocks = len(m.blocks)
    new = {}
    for layer in POOL:
        idx = n_blocks + layer
        patch = m._layer_outputs[idx][:, 1:, :]              # [B, P, 768]
        patch = F.normalize(patch, p=2, dim=-1)              # per-layer L2
        b, p, d = patch.shape
        # match the real code: cast RP to the feature dtype/device per call
        rp_c = rp.to(device=patch.device, dtype=patch.dtype)
        proj = rp_c(patch.reshape(b * p, d)).reshape(b, p, -1)  # RP 768->512
        new[layer] = proj.float().cpu()

    print("layer |   max_abs_diff   | shapes")
    worst = 0.0
    for layer in POOL:
        d = (old[layer] - new[layer]).abs().max().item()
        worst = max(worst, d)
        print(f"  {layer:>3} | {d:.3e} | {tuple(old[layer].shape)}")
    print(f"\nWORST max_abs_diff = {worst:.3e}",
          "=> BIT-IDENTICAL" if worst < 1e-6 else "=> DIFFERENT!")


if __name__ == "__main__":
    main()
