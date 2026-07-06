#!/usr/bin/env python
"""Dump per-layer and final anomaly maps for qualitative figures (Fig 3 & 4).

Standalone: rebuilds the champion ProCon pipeline (recipe ``p3_drop4_3689`` =
pool {-3,-6,-8,-9}, per-layer independent 1% coreset banks, V9 soft projection,
median over banks, mean over layers) and, for a handful of *defect* test images
of a category, saves:
    - the original RGB image,
    - the ground-truth mask,
    - the four per-layer residual maps S_{-3}, S_{-6}, S_{-8}, S_{-9},
    - the final fused anomaly map S_map (mean over layers).

Everything is written as a single ``.npz`` per (category, sample) plus a small
manifest. Rendering into the composite Fig 3 / Fig 4 panels is done separately
(``render_qualitative_figures.py``), so the numeric maps are reusable.

This does NOT modify the locked runner; it re-uses the same primitives
(``build_feature_extractor``, ``build_consensus_banks``, ``soft_projection_bank``,
``patches_to_map``) so the maps match the champion scoring rule.

Usage:
    python tools/dump_figure_maps.py --dataset mvtec --category transistor \
        --num_samples 3 --output figures/qual_maps
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from skipcore.consensus.banks import build_consensus_banks
from skipcore.consensus.recipes import get_recipe
from skipcore.consensus.soft_projection import soft_projection_bank
from skipcore.data.gpu_transforms import preprocess_images, preprocess_masks
from skipcore.data.loaders import build_loader
from skipcore.engine import build_feature_extractor
from skipcore.postprocess.maps import blur_map, patches_to_map
from skipcore.utils import load_yaml, set_seed, get_dataset_categories  # noqa: F401
import skipcore.data.datasets  # noqa: F401  (register datasets)
from skipcore.models.backbones import dinov2_multilayer  # noqa: F401

VISA_DEFAULT_ROOT = "/media/jjack/Extreme SSD/dataset/visa/visa_mvtec"
DEFAULT_CONFIGS = {
    "mvtec": "configs/mvtec_default.yaml",
    "visa": "configs/visa_default.yaml",
    "realiad": "configs/realiad_default.yaml",
}


def _load_cfg(dataset: str) -> dict:
    cfg = load_yaml(DEFAULT_CONFIGS[dataset])
    cfg["dataset"]["name"] = dataset
    if dataset == "visa":
        cfg["dataset"]["root"] = cfg["dataset"].get("visa_root", VISA_DEFAULT_ROOT)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DEFAULT_CONFIGS), default="mvtec")
    ap.add_argument("--category", required=True)
    ap.add_argument("--recipe", default="p3_drop4_3689")
    ap.add_argument("--num_samples", type=int, default=3,
                    help="number of defect test images to dump")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="figures/qual_maps")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    cfg = _load_cfg(args.dataset)
    cfg["dataset"]["category"] = args.category
    img_size = int(cfg["dataset"].get("img_size", 392))
    resize = cfg["dataset"].get("resize")
    gpu_preprocess = bool(cfg.get("dataset", {}).get("gpu_preprocess", False))
    blur_sigma = float(cfg.get("postprocess", {}).get("blur_sigma", 0))
    blur_torch = bool(cfg.get("postprocess", {}).get("blur_torch", False))

    recipe = get_recipe(args.recipe)
    groups = recipe["v28_groups"]                 # {"l3":[-3],...}
    num_banks = int(recipe.get("num_banks", 5))
    ratio = float(recipe.get("memory_ratio", 0.01))
    sp_k = int(recipe.get("softproj_k", 5))
    proj_dim = int(cfg.get("memory", {}).get(
        "dimension_to_project_features_to", 196))
    builder_device = cfg.get("memory", {}).get("device")
    dtype = str(cfg.get("memory", {}).get("dtype", "fp32"))

    # ---- one frozen extractor per layer group (single layer each) ----
    group_extractors = {}
    for gname, layers in groups.items():
        gcfg = dict(cfg.get("backbone", {}))
        gcfg["layers"] = list(layers)
        gcfg["layer_fusion"] = "concat"
        gx = build_feature_extractor(cfg["features"], backbone_cfg=gcfg)
        if hasattr(gx, "backbone"):
            gx.backbone.to(device)
        group_extractors[gname] = gx
    gnames = list(group_extractors.keys())

    # ---- build per-layer banks from the normal training set ----
    print(f"[{args.category}] building per-layer banks ...")
    group_embs = {}
    grid = None
    for gname, extractor in group_extractors.items():
        train_loader = build_loader(cfg["dataset"], split="train", shuffle=False)
        feats = []
        with torch.no_grad():
            for images, _lbl, _m, _p in train_loader:
                if gpu_preprocess:
                    images = preprocess_images(images, img_size, resize, device)
                else:
                    images = images.to(device)
                patches = extractor(images)               # [b, P, D]
                if extractor.last_grid_shape is not None:
                    grid = extractor.last_grid_shape
                for i in range(patches.shape[0]):
                    feats.append(patches[i].cpu())
        all_patches = torch.cat(feats, dim=0)
        banks = build_consensus_banks(
            all_patches, None, num_banks=num_banks, ratio=ratio,
            seed=args.seed, diversity="seed", proj_dim=proj_dim,
            device=builder_device, dtype=dtype)
        emb_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        group_embs[gname] = [b.embeddings.to(device=device, dtype=emb_dtype)
                             for b in banks]
        del all_patches, feats

    # ---- score selected defect test images, saving per-layer + final maps ----
    out_dir = Path(args.output) / args.dataset / args.category
    out_dir.mkdir(parents=True, exist_ok=True)
    test_loader = build_loader(cfg["dataset"], split="test", shuffle=False)

    saved = 0
    manifest = []
    with torch.no_grad():
        for images, labels, masks, paths in test_loader:
            if gpu_preprocess:
                images_d = preprocess_images(images, img_size, resize, device)
                if masks is not None:
                    masks = preprocess_masks(
                        masks, img_size, resize, device).cpu()
            else:
                images_d = images.to(device)
            for i in range(images_d.shape[0]):
                if saved >= args.num_samples:
                    break
                if int(labels[i]) != 1:      # only defect images
                    continue
                img1 = images_d[i:i + 1]
                per_layer_maps = []
                for gname in gnames:
                    feat = group_extractors[gname](img1)[0]     # [P, D]
                    res = [soft_projection_bank(feat, emb, k=sp_k, tau=None)[0]
                           for emb in group_embs[gname]]
                    stack = torch.stack(res, dim=0)             # [B, P]
                    s_l = torch.median(stack, dim=0).values     # [P] bank median
                    smap = patches_to_map(s_l.unsqueeze(0), grid, img_size)
                    smap = blur_map(smap, blur_sigma, use_torch=blur_torch)
                    per_layer_maps.append(smap.squeeze().float().cpu().numpy())
                # mean over layers = final anomaly map
                s_final = np.mean(np.stack(per_layer_maps, axis=0), axis=0)

                m = masks[i]
                mask_np = (m.squeeze().cpu().numpy() > 0).astype(np.uint8) \
                    if torch.is_tensor(m) else np.zeros_like(s_final)

                stem = Path(str(paths[i])).stem
                defect = Path(str(paths[i])).parent.name
                fname = f"{defect}_{stem}"
                np.savez_compressed(
                    out_dir / f"{fname}.npz",
                    image_path=str(paths[i]),
                    layer_names=np.array(gnames),
                    layer_maps=np.stack(per_layer_maps, axis=0),  # [4, H, W]
                    final_map=s_final,                            # [H, W]
                    gt_mask=mask_np,                              # [H, W]
                )
                manifest.append(f"{fname}  <-  {paths[i]}")
                saved += 1
                print(f"  saved {fname}")
            if saved >= args.num_samples:
                break

    (out_dir / "manifest.txt").write_text("\n".join(manifest))
    print(f"[{args.category}] dumped {saved} samples -> {out_dir}")


if __name__ == "__main__":
    main()
