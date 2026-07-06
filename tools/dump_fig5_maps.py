#!/usr/bin/env python
"""Dump NN / SoftProjection / ProCon anomaly maps for the supplementary Fig 5.

For a handful of *defect* test images of a category, compute three anomaly maps
that form the qualitative progression of the method genealogy, all from the same
frozen DINOv2 ViT-B/14 features at the same 1% coreset budget:

    - ``nn_map``       : NN Memory (V0, hard nearest-neighbor distance to a single
                         1% coreset built on the concat feature).
    - ``softproj_map`` : Soft Projection Memory (V9, decoder-free soft projection
                         residual on the same concat feature, median over banks).
    - ``procon_map``   : ProCon (champion ``p3_drop4_3689``: per-layer independent
                         banks {4,5,7,10}, soft projection, median over banks,
                         mean over layers).

One ``.npz`` per (category, sample) is written with keys ``image_path``,
``gt_mask``, ``nn_map``, ``softproj_map``, ``procon_map``. Rendering into the
composite Fig 5 panels is done by ``render_fig5.py``.

This re-uses the same primitives as the locked runner (``build_feature_extractor``,
``build_consensus_banks``, ``soft_projection_bank``, ``patches_to_map``,
``blur_map``) so the maps faithfully represent each method.

Usage:
    python tools/dump_fig5_maps.py --dataset mvtec --category transistor \
        --num_samples 5 --output figures/fig5_maps
"""
from __future__ import annotations

import argparse
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
from skipcore.utils import load_yaml, set_seed  # noqa: F401
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


def _collect_train_patches(cfg, extractor, img_size, resize, gpu_preprocess,
                           device):
    """Pool all normal training patch features for one extractor -> [N, D]."""
    train_loader = build_loader(cfg["dataset"], split="train", shuffle=False)
    feats = []
    grid = None
    with torch.no_grad():
        for images, _lbl, _m, _p in train_loader:
            if gpu_preprocess:
                images = preprocess_images(images, img_size, resize, device)
            else:
                images = images.to(device)
            patches = extractor(images)                     # [b, P, D]
            if extractor.last_grid_shape is not None:
                grid = extractor.last_grid_shape
            for i in range(patches.shape[0]):
                feats.append(patches[i].cpu())
    return torch.cat(feats, dim=0), grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DEFAULT_CONFIGS), default="mvtec")
    ap.add_argument("--category", required=True)
    ap.add_argument("--recipe", default="p3_drop4_3689")
    ap.add_argument("--num_samples", type=int, default=5,
                    help="number of defect test images to dump")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="figures/fig5_maps")
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
    emb_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # ---------- (A) concat extractor for the V0 / V9 baselines ----------
    # Uses the config's default backbone (concat of the 5 default layers, l2
    # norm) -- exactly the feature that the V0 (NN) and V9 (SoftProj) baselines
    # score on.
    concat_extractor = build_feature_extractor(
        cfg["features"], backbone_cfg=cfg.get("backbone", {}))
    if hasattr(concat_extractor, "backbone"):
        concat_extractor.backbone.to(device)

    print(f"[{args.category}] building concat banks (V0/V9) ...")
    concat_patches, concat_grid = _collect_train_patches(
        cfg, concat_extractor, img_size, resize, gpu_preprocess, device)
    # V0: single 1% coreset bank (hard NN).
    nn_banks = build_consensus_banks(
        concat_patches, None, num_banks=1, ratio=ratio, seed=args.seed,
        diversity="seed", proj_dim=proj_dim, device=builder_device, dtype=dtype)
    nn_emb = nn_banks[0].embeddings.to(device=device, dtype=emb_dtype)
    # V9: consensus banks (soft projection).
    v9_banks = build_consensus_banks(
        concat_patches, None, num_banks=num_banks, ratio=ratio, seed=args.seed,
        diversity="seed", proj_dim=proj_dim, device=builder_device, dtype=dtype)
    v9_embs = [b.embeddings.to(device=device, dtype=emb_dtype) for b in v9_banks]
    del concat_patches

    # ---------- (B) per-layer extractors + banks for ProCon ----------
    print(f"[{args.category}] building per-layer banks (ProCon) ...")
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

    group_embs = {}
    group_grid = None
    for gname, extractor in group_extractors.items():
        patches, grid = _collect_train_patches(
            cfg, extractor, img_size, resize, gpu_preprocess, device)
        group_grid = grid or group_grid
        banks = build_consensus_banks(
            patches, None, num_banks=num_banks, ratio=ratio, seed=args.seed,
            diversity="seed", proj_dim=proj_dim, device=builder_device,
            dtype=dtype)
        group_embs[gname] = [b.embeddings.to(device=device, dtype=emb_dtype)
                             for b in banks]
        del patches

    # ---------- (C) score selected defect test images ----------
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

                # --- NN Memory (V0): hard nearest-neighbor distance ---
                feat_c = concat_extractor(img1)[0]              # [P, D]
                d_nn = torch.cdist(feat_c.float(), nn_emb.float())
                s_nn = d_nn.min(dim=1).values                  # [P]
                nn_map = patches_to_map(s_nn.unsqueeze(0), concat_grid, img_size)
                nn_map = blur_map(nn_map, blur_sigma, use_torch=blur_torch)
                nn_map = nn_map.squeeze().float().cpu().numpy()

                # --- Soft Projection Memory (V9): median over banks ---
                res9 = [soft_projection_bank(feat_c, emb, k=sp_k, tau=None)[0]
                        for emb in v9_embs]
                s_v9 = torch.median(torch.stack(res9, dim=0), dim=0).values
                v9_map = patches_to_map(s_v9.unsqueeze(0), concat_grid, img_size)
                v9_map = blur_map(v9_map, blur_sigma, use_torch=blur_torch)
                v9_map = v9_map.squeeze().float().cpu().numpy()

                # --- ProCon: per-layer soft projection, mean over layers ---
                per_layer_maps = []
                for gname in gnames:
                    feat = group_extractors[gname](img1)[0]     # [P, D]
                    res = [soft_projection_bank(feat, emb, k=sp_k, tau=None)[0]
                           for emb in group_embs[gname]]
                    s_l = torch.median(torch.stack(res, dim=0), dim=0).values
                    smap = patches_to_map(s_l.unsqueeze(0), group_grid, img_size)
                    smap = blur_map(smap, blur_sigma, use_torch=blur_torch)
                    per_layer_maps.append(smap.squeeze().float().cpu().numpy())
                procon_map = np.mean(np.stack(per_layer_maps, axis=0), axis=0)

                m = masks[i]
                mask_np = (m.squeeze().cpu().numpy() > 0).astype(np.uint8) \
                    if torch.is_tensor(m) else np.zeros_like(procon_map)

                stem = Path(str(paths[i])).stem
                defect = Path(str(paths[i])).parent.name
                fname = f"{defect}_{stem}"
                np.savez_compressed(
                    out_dir / f"{fname}.npz",
                    image_path=str(paths[i]),
                    nn_map=nn_map,
                    softproj_map=v9_map,
                    procon_map=procon_map,
                    gt_mask=mask_np,
                )
                manifest.append(f"{fname}  <-  {paths[i]}")
                saved += 1
                print(f"  saved {fname}")
            if saved >= args.num_samples:
                break

    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"[{args.category}] dumped {saved} samples -> {out_dir}")


if __name__ == "__main__":
    main()
