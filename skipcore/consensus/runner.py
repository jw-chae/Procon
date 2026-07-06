"""Per-category build + evaluate pipeline for ConsensusCore.

Reuses the frozen SkipCore feature extractor, greedy-coreset builder, torch KNN
backend, post-processing and metric utilities. No encoder/decoder is trained: the
only learned structure is the set of discrete coreset memories.
"""
from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from skipcore.consensus.banks import build_consensus_banks, build_single_bank
from skipcore.consensus.experts import (
    combine_expert_score,
    combine_multiscale,
    is_multiscale,
    multiscale_config,
    needs_nn,
    needs_soft,
)
from skipcore.consensus.fusion import fuse_distances
from skipcore.consensus.local_covariance import (
    local_mahalanobis_bank,
    local_pca_residual_bank,
)
from skipcore.consensus.oob_pruning import prune_banks_oob
from skipcore.consensus.prototypes import intrinsic_prototype_refine
from skipcore.consensus.reliability import (
    compute_oob_reliability,
    compute_stability_reliability,
    reliability_to_penalty,
)
from skipcore.consensus.soft_projection import (
    auto_tau,
    soft_projection_bank,
    soft_projection_bank_multi,
    soft_projection_banks_vectorized,
)
from skipcore.consensus.v9_memory_selection import (
    build_rec_pruned_kcenter,
    build_recgreedy_memory,
)
from skipcore.data.gpu_transforms import preprocess_images, preprocess_masks
from skipcore.data.loaders import build_loader
from skipcore.engine import build_inference_backend
from skipcore.eval.metrics import image_metrics, pixel_metrics
from skipcore.postprocess.maps import blur_map, patches_to_map
from skipcore.utils import apply_runtime_settings, set_seed


def _topmean(patch_scores: torch.Tensor, ratio: float) -> torch.Tensor:
    """Image-level TopMean: mean of the top-``ratio`` fraction of patch scores."""
    p = patch_scores.shape[-1]
    k = max(1, int(round(p * float(ratio))))
    topk = torch.topk(patch_scores, k, dim=-1).values
    return topk.mean(dim=-1)


def _maybe_oob_prune(banks, all_patches, args, device):
    """Apply OOB-residual coreset pruning if requested; return (banks, audit)."""
    if not bool(getattr(args, "oob_prune", False)):
        return banks, None
    sp_k = int(getattr(args, "softproj_k", 5))
    sp_tau_arg = getattr(args, "softproj_tau", "auto")
    sp_tau = None if sp_tau_arg in (None, "auto") else float(sp_tau_arg)
    banks, audit = prune_banks_oob(
        banks, all_patches, device,
        k=sp_k, tau=sp_tau,
        frac=float(getattr(args, "oob_frac", 0.0)),
        agg=str(getattr(args, "oob_agg", "median")),
        threshold=str(getattr(args, "oob_threshold", "quantile")),
        refill=str(getattr(args, "oob_refill", "none")),
        abs_c=float(getattr(args, "oob_abs_c", 2.0)),
        seed=int(getattr(args, "seed", 0)),
    )
    return banks, audit


def _accumulate_score_stats(stat_acc: dict, smap_np, mask_bin) -> None:
    """Accumulate per-image anomaly/normal pixel-score statistics."""
    flat = smap_np.reshape(-1)
    mflat = mask_bin.reshape(-1)
    normal = flat[mflat == 0]
    anom = flat[mflat == 1]
    if normal.size:
        stat_acc["mu_normal"].append(float(normal.mean()))
        stat_acc["median_normal"].append(float(np.median(normal)))
        stat_acc["q90_normal"].append(float(np.quantile(normal, 0.90)))
        stat_acc["q95_normal"].append(float(np.quantile(normal, 0.95)))
    if anom.size:
        stat_acc["mu_anom"].append(float(anom.mean()))
        stat_acc["median_anom"].append(float(np.median(anom)))
        stat_acc["q90_anom"].append(float(np.quantile(anom, 0.90)))


def extract_patches(
    extractor,
    loader,
    cfg: Dict[str, Any],
    device: torch.device,
    inference_ctx,
    with_index: bool = False,
    with_labels: bool = False,
) -> Dict[str, Any]:
    """Extract patch features from a data loader.

    Returns a dict with ``feats`` (list of per-image ``[P, D]`` CPU tensors),
    and optionally ``image_index``, ``labels``, ``masks``, ``grid``.
    """
    gpu_preprocess = bool(cfg.get("dataset", {}).get("gpu_preprocess", False))
    img_size = int(cfg["dataset"].get("img_size", 224))
    resize = cfg["dataset"].get("resize")

    feats: List[torch.Tensor] = []
    image_index: List[int] = []
    labels: List[int] = []
    masks: List[torch.Tensor] = []
    grid: Optional[Tuple[int, int]] = None
    img_counter = 0

    for batch in tqdm(loader, leave=False):
        images, batch_labels, batch_masks, _paths = batch
        if gpu_preprocess:
            images = preprocess_images(images, img_size, resize, device)
            if with_labels and batch_masks is not None:
                batch_masks = preprocess_masks(batch_masks, img_size, resize, device).cpu()
        else:
            images = images.to(device)

        with inference_ctx():
            patches = extractor(images)  # [b, P, D]

        if extractor.last_grid_shape is not None:
            grid = extractor.last_grid_shape

        b = patches.shape[0]
        for i in range(b):
            feats.append(patches[i].cpu())
            if with_index:
                image_index.extend([img_counter] * patches.shape[1])
            img_counter += 1
        if with_labels:
            labels.extend([int(x) for x in batch_labels])
            if batch_masks is not None:
                for i in range(b):
                    m = batch_masks[i]
                    masks.append(m.squeeze().cpu() if torch.is_tensor(m) else m)

    out: Dict[str, Any] = {"feats": feats, "grid": grid}
    if with_index:
        out["image_index"] = torch.tensor(image_index, dtype=torch.long)
    if with_labels:
        out["labels"] = labels
        out["masks"] = masks
    return out


def _query_banks(
    backends: List[Any],
    queries: torch.Tensor,
    fusion: str,
    quantile: float,
    trim_ratio: float,
) -> torch.Tensor:
    """Query each bank with k=1 and fuse per-patch distances -> ``[M]``."""
    dists = []
    for backend in backends:
        d = backend.query(queries, k=1).squeeze(-1)  # [M]
        dists.append(d)
    stacked = torch.stack(dists, dim=0)  # [B, M]
    return fuse_distances(stacked, mode=fusion, quantile=quantile, trim_ratio=trim_ratio)


def _query_banks_soft(
    bank_embs: List[torch.Tensor],
    penalties: List[torch.Tensor],
    queries: torch.Tensor,
    fusion: str,
    quantile: float,
    trim_ratio: float,
) -> torch.Tensor:
    """Soft per-bank nearest-anchor distance with an additive reliability penalty.

    ``d_b(z) = min_m [ ||z - m|| + penalty_m ]`` where ``penalty_m = lambda(1-r_m)``.
    Distances are fused across banks exactly like the hard path.
    """
    dists = []
    for emb, pen in zip(bank_embs, penalties):
        d = torch.cdist(queries, emb)  # [P, M_b]
        d = d + pen.unsqueeze(0)
        dists.append(d.min(dim=1).values)  # [P]
    stacked = torch.stack(dists, dim=0)  # [B, P]
    return fuse_distances(stacked, mode=fusion, quantile=quantile, trim_ratio=trim_ratio)


def _nn_stack(backends: List[Any], queries: torch.Tensor) -> torch.Tensor:
    """Per-bank NN distances without fusion -> ``[B, P]`` (expert evidence)."""
    return torch.stack(
        [bk.query(queries, k=1).squeeze(-1) for bk in backends], dim=0
    )


def _soft_stack(
    bank_embs: List[torch.Tensor], queries: torch.Tensor, k: int,
    tau: Optional[float],
) -> torch.Tensor:
    """Per-bank soft-projection residuals without fusion -> ``[B, P]``."""
    res = [soft_projection_bank(queries, emb, k=k, tau=tau)[0]
           for emb in bank_embs]
    return torch.stack(res, dim=0)


def _query_banks_softproj(
    bank_embs: List[torch.Tensor],
    queries: torch.Tensor,
    k: int,
    tau: Optional[float],
    fusion: str,
    quantile: float,
    trim_ratio: float,
    return_entropy: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Consensus decoder-free soft projection residual (spec V8/V9).

    Per bank, reconstruct ``z_hat`` from the k nearest anchors and take the
    residual ``||z - z_hat||``; fuse residuals (and optionally entropies) across
    banks like the hard nearest-neighbor path.
    """
    res_list = []
    ent_list = []
    for emb in bank_embs:
        r, h = soft_projection_bank(queries, emb, k=k, tau=tau,
                                    return_entropy=return_entropy)
        res_list.append(r)
        if return_entropy:
            ent_list.append(h)
    res = fuse_distances(torch.stack(res_list, dim=0), mode=fusion,
                         quantile=quantile, trim_ratio=trim_ratio)
    ent = None
    if return_entropy:
        ent = fuse_distances(torch.stack(ent_list, dim=0), mode=fusion,
                             quantile=quantile, trim_ratio=trim_ratio)
    return res, ent


def _query_multiscale_softproj(
    bank_embs: List[torch.Tensor],
    queries: torch.Tensor,
    ms_cfg: dict,
    trim_ratio: float,
) -> torch.Tensor:
    """Multi-scale soft-projection consensus (spec V16-V19).

    For each (k, tau) scale, fuse per-bank residuals across banks at each
    quantile in ``q_list`` to form an expert ``S_{k,tau,q}``; then fuse all
    experts with the expert-level fusion. top-max(k) is computed once per bank.
    """
    k_list = ms_cfg["k_list"]
    tau_scales = ms_cfg["tau_scales"]
    q_list = ms_cfg["q_list"]
    # Per bank: dict[(k,scale)] -> [P]. Stack across banks into [B, P].
    per_bank = [
        soft_projection_bank_multi(queries, emb, k_list, tau_scales)
        for emb in bank_embs
    ]
    experts: List[torch.Tensor] = []
    for k in k_list:
        for scale in tau_scales:
            stack = torch.stack(
                [pb[(int(k), float(scale))] for pb in per_bank], dim=0
            )  # [B, P]
            for q in q_list:
                experts.append(
                    fuse_distances(stack, mode="quantile", quantile=q,
                                   trim_ratio=trim_ratio)
                )
    return combine_multiscale(experts, ms_cfg["expert_fusion"],
                              ms_cfg["expert_q"])


def _query_local_geometry(
    bank_embs: List[torch.Tensor],
    queries: torch.Tensor,
    mode: str,
    fusion: str,
    quantile: float,
    trim_ratio: float,
    k_pca: int,
    rank: int,
    maha_eps: float,
) -> torch.Tensor:
    """Consensus over per-bank local-geometry residuals (V26 / V27)."""
    res = []
    for emb in bank_embs:
        if mode == "local_maha":
            res.append(local_mahalanobis_bank(queries, emb, k_maha=k_pca,
                                               rank=rank, eps=maha_eps))
        else:  # local_pca
            res.append(local_pca_residual_bank(queries, emb, k_pca=k_pca,
                                                rank=rank))
    stack = torch.stack(res, dim=0)
    return fuse_distances(stack, mode=fusion, quantile=quantile,
                          trim_ratio=trim_ratio)


def evaluate_category(
    cfg: Dict[str, Any],
    category: str,
    args: Any,
    device: torch.device,
    extractor,
) -> Dict[str, Any]:
    """Build memory banks for one category and evaluate the test split."""
    cfg = copy.deepcopy(cfg)
    cfg["dataset"]["category"] = category

    runtime = cfg.get("runtime", {})
    use_inference_mode = apply_runtime_settings(runtime)
    inference_ctx = torch.inference_mode if use_inference_mode else torch.no_grad
    img_size = int(cfg["dataset"].get("img_size", 224))

    set_seed(args.seed)

    # ---- Train: extract patches, build banks ----
    train_loader = build_loader(cfg["dataset"], split="train", shuffle=False)
    need_index = args.method == "consensus" and args.bank_diversity == "bootstrap"
    train = extract_patches(
        extractor, train_loader, cfg, device, inference_ctx, with_index=need_index
    )
    all_patches = torch.cat(train["feats"], dim=0)  # [N, D]
    image_index = train.get("image_index")

    proj_dim = int(cfg.get("memory", {}).get("dimension_to_project_features_to", 196))
    builder_device = cfg.get("memory", {}).get("device")
    dtype = str(cfg.get("memory", {}).get("dtype", "fp32"))

    mem_select = getattr(args, "mem_select", None)
    if args.method == "consensus" and mem_select:
        # V9-aware (reconstruction-aware) memory selection, per bank.
        sp_k_mem = int(getattr(args, "softproj_k", 5))
        banks = []
        for b in range(args.num_banks):
            sd = args.seed + b
            if mem_select == "rec_pruned":
                banks.append(build_rec_pruned_kcenter(
                    all_patches, final_ratio=args.memory_ratio, seed=sd,
                    large_ratio=float(getattr(args, "large_memory_ratio", 0.05)),
                    k=sp_k_mem, proj_dim=proj_dim, device=builder_device,
                    dtype=dtype,
                ))
            else:  # recgreedy / recgreedy_diverse
                div = float(getattr(args, "div_lambda", 0.0)) \
                    if mem_select == "recgreedy_diverse" else 0.0
                banks.append(build_recgreedy_memory(
                    all_patches, ratio=args.memory_ratio, seed=sd,
                    candidate_ratio=float(getattr(args, "candidate_ratio", 0.05)),
                    sample_patches=int(getattr(args, "recgreedy_sample_patches",
                                               20000)),
                    k=sp_k_mem, diversity_lambda=div, proj_dim=proj_dim,
                    device=builder_device, dtype=dtype,
                ))
    elif args.method == "consensus":
        banks = build_consensus_banks(
            all_patches,
            image_index,
            num_banks=args.num_banks,
            ratio=args.memory_ratio,
            seed=args.seed,
            diversity=args.bank_diversity,
            ratios=args.ratios,
            proj_dim=proj_dim,
            device=builder_device,
            dtype=dtype,
        )
    else:  # single / full / random ablations
        banks = [
            build_single_bank(
                all_patches,
                ratio=args.memory_ratio,
                seed=args.seed,
                method=args.method,
                proj_dim=proj_dim,
                device=builder_device,
                dtype=dtype,
            )
        ]

    banks, oob_audit = _maybe_oob_prune(banks, all_patches, args, device)
    backends = [build_inference_backend(cfg["inference"], bank.embeddings) for bank in banks]
    bank_sizes = [int(b.embeddings.shape[0]) for b in banks]

    # ---- Optional anchor reliability (soft ConsensusCore: V3 / V4) ----
    use_reliability = bool(getattr(args, "use_anchor_reliability", False))
    soft_penalties: List[torch.Tensor] = []
    if use_reliability:
        rel_type = str(getattr(args, "reliability_type", "stability")).lower()
        rel_lambda = float(getattr(args, "reliability_lambda", 0.03))
        if rel_type == "oob":
            tau_mu = getattr(args, "tau_mu", None)
            tau_sigma = getattr(args, "tau_sigma", None)
            tau_mu = None if tau_mu in (None, "auto") else float(tau_mu)
            tau_sigma = None if tau_sigma in (None, "auto") else float(tau_sigma)
            reliabilities = compute_oob_reliability(banks, device, tau_mu, tau_sigma)
        else:
            delta = getattr(args, "stability_delta", None)
            delta = None if delta in (None, "auto") else float(delta)
            reliabilities = compute_stability_reliability(banks, device, delta)
        soft_penalties = reliability_to_penalty(reliabilities, rel_lambda)

    # ---- Optional decoder-free soft projection setup (V8 / V9 / V10) ----
    use_softproj = bool(getattr(args, "use_soft_projection", False))
    # ---- Optional heterogeneous expert ensemble (V11-V15) ----
    expert_recipe = getattr(args, "expert_recipe", None)
    use_expert = expert_recipe is not None
    # ---- Optional multi-scale soft-projection family (V16-V19) ----
    use_multiscale = use_expert and is_multiscale(expert_recipe)
    ms_cfg = multiscale_config(expert_recipe, args) if use_multiscale else None
    # ---- Optional local-geometry residual (V26 / V27) ----
    score_mode = str(getattr(args, "score_mode", "softproj") or "softproj")
    use_localgeom = score_mode in ("local_pca", "local_maha")
    sp_k = int(getattr(args, "softproj_k", 5))
    sp_tau_arg = getattr(args, "softproj_tau", "auto")
    sp_tau = None if sp_tau_arg in (None, "auto") else float(sp_tau_arg)

    proj_bank_embs: List[torch.Tensor] = []
    need_soft_embs = use_softproj or use_reliability or use_multiscale or \
        use_localgeom or (use_expert and needs_soft(expert_recipe))
    if need_soft_embs:
        proj_bank_embs = [
            b.embeddings.to(device=device, dtype=torch.float32) for b in banks
        ]

    sp_alpha = float(getattr(args, "softproj_alpha", 0.5))
    sp_fusion = str(getattr(args, "softproj_fusion", "direct")).lower()
    entropy_lambda = float(getattr(args, "entropy_lambda", 0.0))
    use_entropy = entropy_lambda > 0.0

    # ---- Optional component/score-map dumping for diagnostics ----
    debug_save = bool(getattr(args, "debug_save_components", False))
    comp_dir = None
    if debug_save and args.output:
        from pathlib import Path as _Path
        comp_dir = _Path(args.output) / "components" / category
        comp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Test: extract patches, score ----
    test_loader = build_loader(cfg["dataset"], split="test", shuffle=False)
    test = extract_patches(
        extractor, test_loader, cfg, device, inference_ctx, with_labels=True
    )
    grid = test["grid"]
    labels = np.asarray(test["labels"], dtype=np.int64)

    blur_sigma = float(cfg.get("postprocess", {}).get("blur_sigma", 0))
    blur_torch = bool(cfg.get("postprocess", {}).get("blur_torch", False))

    image_scores: List[float] = []
    pixel_maps: List[np.ndarray] = []
    pixel_masks: List[np.ndarray] = []
    have_masks = len(test["masks"]) == len(test["feats"])

    # Pixel-score statistics (anomaly vs normal), accumulated over the category.
    stat_acc = {k: [] for k in (
        "mu_anom", "mu_normal", "median_anom", "median_normal",
        "q90_anom", "q90_normal", "q95_normal",
    )}

    for i, feat in enumerate(tqdm(test["feats"], leave=False, desc=f"score {category}")):
        feat = feat.to(device)  # [P, D]
        with inference_ctx():
            s_int = None
            if use_localgeom:
                # Local PCA / Mahalanobis residual consensus (V26 / V27).
                s_final = _query_local_geometry(
                    proj_bank_embs, feat, score_mode,
                    args.fusion, args.quantile, args.trim_ratio,
                    k_pca=int(getattr(args, "pca_neighbors", 10)),
                    rank=int(getattr(args, "pca_rank", 2)),
                    maha_eps=float(getattr(args, "maha_eps", 1e-4)),
                )
                s_ext = s_final
                img_score = _topmean(
                    s_final.unsqueeze(0), args.topmean_ratio
                ).item()
                image_scores.append(img_score)
                if grid is not None and have_masks:
                    smap = patches_to_map(s_final.unsqueeze(0), grid, img_size)
                    smap = blur_map(smap, blur_sigma, use_torch=blur_torch)
                    smap_np = smap.squeeze().float().cpu().numpy()
                    pixel_maps.append(smap_np)
                    m = test["masks"][i]
                    m_np = m.numpy() if torch.is_tensor(m) else np.asarray(m)
                    mask_bin = (m_np > 0).astype(np.int32)
                    pixel_masks.append(mask_bin)
                    _accumulate_score_stats(stat_acc, smap_np, mask_bin)
                continue
            if use_multiscale:
                # Multi-scale soft-projection consensus (V16-V19).
                s_final = _query_multiscale_softproj(
                    proj_bank_embs, feat, ms_cfg, args.trim_ratio
                )
                s_ext = s_final
                img_score = _topmean(
                    s_final.unsqueeze(0), args.topmean_ratio
                ).item()
                image_scores.append(img_score)
                if grid is not None and have_masks:
                    smap = patches_to_map(s_final.unsqueeze(0), grid, img_size)
                    smap = blur_map(smap, blur_sigma, use_torch=blur_torch)
                    smap_np = smap.squeeze().float().cpu().numpy()
                    pixel_maps.append(smap_np)
                    m = test["masks"][i]
                    m_np = m.numpy() if torch.is_tensor(m) else np.asarray(m)
                    mask_bin = (m_np > 0).astype(np.int32)
                    pixel_masks.append(mask_bin)
                    _accumulate_score_stats(stat_acc, smap_np, mask_bin)
                continue
            if use_expert:
                # Compute raw bank evidence once, then combine experts.
                nn_stack = _nn_stack(backends, feat) if needs_nn(expert_recipe) \
                    else None
                soft_stack = _soft_stack(proj_bank_embs, feat, sp_k, sp_tau) \
                    if needs_soft(expert_recipe) else None
                s_ext = (nn_stack if nn_stack is not None else soft_stack)
                # A representative base map for diagnostics (median view).
                s_ext = torch.quantile(s_ext, 0.5, dim=0) if s_ext.shape[0] > 1 \
                    else s_ext[0]
                s_final = combine_expert_score(
                    expert_recipe, args, nn_stack, soft_stack
                )
                img_score = _topmean(s_final.unsqueeze(0), args.topmean_ratio).item()
                image_scores.append(img_score)
                if grid is not None and have_masks:
                    smap = patches_to_map(s_final.unsqueeze(0), grid, img_size)
                    smap = blur_map(smap, blur_sigma, use_torch=blur_torch)
                    smap_np = smap.squeeze().float().cpu().numpy()
                    pixel_maps.append(smap_np)
                    m = test["masks"][i]
                    m_np = m.numpy() if torch.is_tensor(m) else np.asarray(m)
                    mask_bin = (m_np > 0).astype(np.int32)
                    pixel_masks.append(mask_bin)
                    _accumulate_score_stats(stat_acc, smap_np, mask_bin)
                continue

            # Base external consensus score (hard NN, or soft-reliability NN).
            if use_reliability:
                s_ext = _query_banks_soft(
                    proj_bank_embs, soft_penalties, feat,
                    args.fusion, args.quantile, args.trim_ratio,
                )
            else:
                s_ext = _query_banks(
                    backends, feat, args.fusion, args.quantile, args.trim_ratio
                )  # [P]

            s_int = None
            if use_softproj:
                s_rec, s_h = _query_banks_softproj(
                    proj_bank_embs, feat, sp_k, sp_tau,
                    args.fusion, args.quantile, args.trim_ratio,
                    return_entropy=use_entropy,
                )
                s_int = s_rec
                if sp_fusion == "gate_residual":
                    # Keep V1 NN base; boost only where reconstruction is worse.
                    s_final = s_ext + sp_alpha * torch.clamp(s_rec - s_ext, min=0.0)
                elif sp_fusion == "max":
                    s_final = torch.maximum(s_ext, sp_alpha * s_rec)
                else:  # direct (V8 / V9)
                    s_final = s_rec
                if use_entropy and s_h is not None:
                    s_final = s_final + entropy_lambda * s_h
            elif args.use_intrinsic_proto:
                s_final, s_int = intrinsic_prototype_refine(
                    feat, s_ext,
                    select_quantile=args.proto_select_quantile,
                    num_prototypes=args.num_prototypes,
                    alpha=args.proto_alpha,
                    fusion=args.proto_fusion,
                    seed=args.seed,
                    return_components=True,
                )
            else:
                s_final = s_ext

            img_score = _topmean(s_final.unsqueeze(0), args.topmean_ratio).item()
        image_scores.append(img_score)

        if grid is not None and have_masks:
            smap = patches_to_map(s_final.unsqueeze(0), grid, img_size)
            smap = blur_map(smap, blur_sigma, use_torch=blur_torch)
            smap_np = smap.squeeze().float().cpu().numpy()
            pixel_maps.append(smap_np)
            m = test["masks"][i]
            m_np = m.numpy() if torch.is_tensor(m) else np.asarray(m)
            mask_bin = (m_np > 0).astype(np.int32)
            pixel_masks.append(mask_bin)

            # Per-image anomaly/normal pixel-score statistics.
            _accumulate_score_stats(stat_acc, smap_np, mask_bin)

            # Optional component dump.
            if comp_dir is not None:
                ext_map = patches_to_map(s_ext.unsqueeze(0), grid, img_size)
                ext_map = blur_map(ext_map, blur_sigma, use_torch=blur_torch)
                np.savez_compressed(
                    comp_dir / f"img{i:04d}.npz",
                    s_ext=ext_map.squeeze().float().cpu().numpy(),
                    s_int=(patches_to_map(s_int.unsqueeze(0), grid, img_size)
                           .squeeze().float().cpu().numpy()
                           if s_int is not None else np.zeros_like(smap_np)),
                    s_final=smap_np,
                    mask=mask_bin,
                )

    scores = np.asarray(image_scores, dtype=np.float64)
    result: Dict[str, Any] = {"category": category, "bank_size": int(np.mean(bank_sizes))}
    result.update(image_metrics(scores, labels))

    if pixel_maps:
        maps_arr = np.stack(pixel_maps, axis=0)
        masks_arr = np.stack(pixel_masks, axis=0)
        result.update(pixel_metrics(maps_arr, masks_arr, compute_pro=True))

        # Attach mean pixel-score statistics + ranking contrasts.
        stats = {k: (float(np.mean(v)) if v else 0.0) for k, v in stat_acc.items()}
        stats["contrast_mu"] = stats["mu_anom"] - stats["mu_normal"]
        stats["contrast_median"] = stats["median_anom"] - stats["median_normal"]
        result["score_stats"] = stats

    if oob_audit is not None:
        result["oob_audit"] = oob_audit

    # Free GPU memory between categories.
    del backends, banks, soft_penalties, proj_bank_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _group_normalize(stack: torch.Tensor, mode: str,
                     stats: Optional[tuple] = None) -> torch.Tensor:
    """Normalize a group's per-bank residuals ``[B, P]`` by *train-normal* stats.

    ``stats`` is the precomputed ``(center, scale)`` of the group's residuals over
    the normal training patches (zscore: mean/std; robust: median/IQR). Using
    train-normal statistics -- NOT per-test-image statistics -- preserves the
    anomaly signal of an individual image while equalizing group scales.
    """
    if mode in (None, "none") or stats is None:
        return stack
    center, scale = stats
    return (stack - center) / (scale + 1e-8)


def _fit_group_norm_stats(bank_embs: List[torch.Tensor],
                          train_feats: List[torch.Tensor],
                          device: torch.device, mode: str,
                          sp_k: int, sp_tau: Optional[float],
                          max_imgs: int = 64) -> tuple:
    """Compute train-normal residual (center, scale) for a layer group.

    Residuals of a sampled subset of normal training images are pooled across
    all banks; returns scalar tensors so test-time normalization uses a fixed
    train statistic.
    """
    res_all = []
    n = min(len(train_feats), max_imgs)
    step = max(1, len(train_feats) // n)
    for feat in train_feats[::step][:n]:
        f = feat.to(device)
        for emb in bank_embs:
            res_all.append(soft_projection_bank(f, emb, k=sp_k, tau=sp_tau)[0])
    flat = torch.cat(res_all)
    if mode == "robust":
        center = flat.median()
        scale = torch.quantile(flat, 0.75) - torch.quantile(flat, 0.25)
    else:  # zscore
        center = flat.mean()
        scale = flat.std()
    return center.detach(), scale.detach()


def _score_one_image_layergroup(
    feat_by_group: Dict[str, torch.Tensor],
    gnames: List[str],
    group_bank_embs: Dict[str, List[torch.Tensor]],
    group_norm_stats: Dict[str, tuple],
    *,
    sp_k: int,
    sp_tau,
    groupnorm: str,
    layer_combine: str,
    trim_ratio: float,
    pool: str,
    q: float,
    readout: str,
    topmean_ratio: float,
    bank_vectorized: bool = False,
    query_chunk: int = 0,
) -> Tuple[float, torch.Tensor]:
    """Score a single image's per-group features (factored from the V28 loop).

    ``feat_by_group[gname]`` is that group's ``[P, D]`` patch features for ONE
    image (already on the target device). Returns ``(image_score, s_final[P])``.

    This is the *single source of truth* for V28 per-image scoring: both the
    legacy pre-extract path and the streaming path call it, so they are
    guaranteed bit-identical (including the per-call ``auto_tau`` RNG order).

    When ``bank_vectorized`` is True the per-bank soft-projection loop is
    replaced by a single batched call (:func:`soft_projection_banks_vectorized`).
    The per-bank ``tau`` values are still computed in the legacy order (so the
    RNG is consumed identically), but the residual uses a batched cdist whose
    fp32 reduction order differs slightly -- numerically equivalent (~1e-6), not
    bit-identical. Only the ``tau is None`` (auto) case is supported here.
    """
    group_residuals = []
    for gname in gnames:
        feat = feat_by_group[gname]
        embs = group_bank_embs[gname]
        if bank_vectorized and sp_tau is None and len(embs) > 1 \
                and len({int(e.shape[0]) for e in embs}) == 1:
            # tau per bank in legacy order (consumes RNG identically), then a
            # single batched residual over all banks.
            taus = [auto_tau(feat, e, min(sp_k, e.shape[0])) for e in embs]
            stack = soft_projection_banks_vectorized(
                feat, embs, taus, k=sp_k, query_chunk=query_chunk)  # [B, P]
        else:
            res = [soft_projection_bank(feat, emb, k=sp_k, tau=sp_tau)[0]
                   for emb in embs]
            stack = torch.stack(res, dim=0)            # [B, P]
        stack = _group_normalize(stack, groupnorm,
                                 group_norm_stats.get(gname))
        group_residuals.append(stack)
    pooled_stack = torch.cat(group_residuals, dim=0)  # [G*B, P]
    if layer_combine != "none":
        layer_scores = [torch.median(s, dim=0).values
                        for s in group_residuals]      # each [P]
        L = torch.stack(layer_scores, dim=0)           # [G, P]
        if groupnorm == "perimage_robust":
            med = L.median(dim=1, keepdim=True).values
            q1 = torch.quantile(L, 0.25, dim=1, keepdim=True)
            q3 = torch.quantile(L, 0.75, dim=1, keepdim=True)
            L = (L - med) / (q3 - q1 + 1e-8)
        if layer_combine == "mean":
            s_final = L.mean(dim=0)
        elif layer_combine == "median":
            s_final = L.median(dim=0).values
        elif layer_combine == "q75":
            s_final = torch.quantile(L, 0.75, dim=0)
        elif layer_combine == "trimmed_mean":
            G = L.shape[0]
            cut = int(G * trim_ratio)
            Ls = torch.sort(L, dim=0).values
            lo, hi = cut, G - cut
            if hi <= lo:
                s_final = L.mean(dim=0)
            else:
                s_final = Ls[lo:hi].mean(dim=0)
        elif layer_combine == "max":
            s_final = L.max(dim=0).values
        elif layer_combine == "product":
            s_final = torch.exp(torch.log(L.clamp_min(1e-8)).mean(dim=0))
        elif layer_combine == "rank":
            P = L.shape[1]
            denom = max(P - 1, 1)
            ranks = torch.argsort(torch.argsort(L, dim=1), dim=1)
            s_final = (ranks.float() / denom).mean(dim=0)
        else:
            raise ValueError(f"unknown layer_combine={layer_combine}")
    elif pool == "mean":
        s_final = pooled_stack.mean(dim=0)
    elif pool == "max":
        s_final = pooled_stack.max(dim=0).values
    else:  # quantile consensus
        s_final = torch.quantile(pooled_stack, q, dim=0)
    if readout == "max":
        img_score = s_final.max().item()
    else:  # topmean
        img_score = _topmean(s_final.unsqueeze(0), topmean_ratio).item()
    return img_score, s_final


def evaluate_category_layergroup(
    cfg: Dict[str, Any],
    category: str,
    args: Any,
    device: torch.device,
    group_extractors: Dict[str, Any],
) -> Dict[str, Any]:
    """Two-group layer-wise V9 with group-normalized consensus pooling (V28).

    Each layer group g has its own frozen extractor + consensus banks. Per group
    we compute per-bank soft-projection residuals, group-normalize them, then
    pool *all* per-bank-per-group residuals together with Q_q (or mean/max for
    the ablations). Scoring is otherwise standard V9 (soft projection).
    """
    cfg = copy.deepcopy(cfg)
    cfg["dataset"]["category"] = category
    runtime = cfg.get("runtime", {})
    use_inference_mode = apply_runtime_settings(runtime)
    inference_ctx = torch.inference_mode if use_inference_mode else torch.no_grad
    img_size = int(cfg["dataset"].get("img_size", 224))
    set_seed(args.seed)

    proj_dim = int(cfg.get("memory", {}).get("dimension_to_project_features_to",
                                             196))
    builder_device = cfg.get("memory", {}).get("device")
    dtype = str(cfg.get("memory", {}).get("dtype", "fp32"))
    sp_k = int(getattr(args, "softproj_k", 5))
    sp_tau_arg = getattr(args, "softproj_tau", "auto")
    sp_tau = None if sp_tau_arg in (None, "auto") else float(sp_tau_arg)
    pool = str(getattr(args, "v28_pool", "quantile"))
    q = float(getattr(args, "v28_q", 0.5))
    groupnorm = str(getattr(args, "v28_groupnorm", "zscore"))
    trim_ratio = float(getattr(args, "trim_ratio", 0.1))
    # Axis-1 two-level combine: bank-consensus (median) within each layer/group,
    # then combine the per-layer scores with mean/max/product/rank. When 'none',
    # fall back to the original flat pooling over all group*bank residuals.
    layer_combine = str(getattr(args, "layer_combine", "none") or "none")
    # Axis-3 image readout: 'topmean' (mean of top-ratio patches) or 'max'.
    readout = str(getattr(args, "v28_readout", "topmean") or "topmean")
    # (alpha) bank-vectorized soft projection: one batched cdist over the B banks
    # instead of a Python loop. Numerically equivalent (~1e-6), big speed-up.
    bank_vectorized = bool(getattr(args, "bank_vectorized", False))
    sp_query_chunk = int(getattr(args, "sp_query_chunk", 0))

    # --- Per-group: build banks (test is scored later in a streaming pass) ---
    group_bank_embs: Dict[str, List[torch.Tensor]] = {}
    group_norm_stats: Dict[str, tuple] = {}
    group_oob_audit: Dict[str, dict] = {}
    labels = None
    grid = None
    _timing = os.environ.get("SKIPCORE_TIMING", "0") == "1"
    _t = {"train_extract": 0.0, "coreset": 0.0, "test_extract_score": 0.0}
    _t0 = time.time()
    for gname, extractor in group_extractors.items():
        train_loader = build_loader(cfg["dataset"], split="train", shuffle=False)
        train = extract_patches(extractor, train_loader, cfg, device,
                                inference_ctx)
        if _timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t["train_extract"] += time.time() - _t0; _t0 = time.time()
        all_patches = torch.cat(train["feats"], dim=0)
        banks = build_consensus_banks(
            all_patches, None, num_banks=args.num_banks,
            ratio=args.memory_ratio, seed=args.seed, diversity="seed",
            proj_dim=proj_dim, device=builder_device, dtype=dtype,
        )
        banks, gaudit = _maybe_oob_prune(banks, all_patches, args, device)
        if gaudit is not None:
            group_oob_audit[gname] = gaudit
        # Large coresets (e.g. 5-10% budget) are kept in fp16 on GPU to bound
        # memory; soft_projection casts to fp32 internally for the distance.
        emb_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        embs = [
            b.embeddings.to(device=device, dtype=emb_dtype) for b in banks
        ]
        group_bank_embs[gname] = embs
        # Train-normal residual statistics for group normalization (V28).
        if groupnorm in ("zscore", "robust"):
            with inference_ctx():
                group_norm_stats[gname] = _fit_group_norm_stats(
                    embs, train["feats"], device, groupnorm, sp_k, sp_tau,
                )
        # Free the train features once banks (and any norm stats) are built;
        # otherwise they accumulate across the four layer groups and can
        # exhaust host RAM on large datasets.
        del all_patches, train
        if _timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t["coreset"] += time.time() - _t0; _t0 = time.time()

    # --- Streaming test scoring (single pass, no full-feature accumulation) ---
    # Instead of extracting and storing every test image's features for all four
    # groups up front (~12 GB on Real-IAD), iterate the test loader ONCE and run
    # all group extractors on each batch, scoring immediately and discarding the
    # features. The image order (shuffle=False), group order and bank order are
    # identical to the legacy pre-extract path, and feature extraction consumes
    # no global RNG, so the per-image ``auto_tau`` RNG sequence -- and hence the
    # scores -- are bit-identical to the old implementation.
    gnames = list(group_extractors.keys())
    gpu_preprocess = bool(cfg.get("dataset", {}).get("gpu_preprocess", False))
    resize = cfg["dataset"].get("resize")
    blur_sigma = float(cfg.get("postprocess", {}).get("blur_sigma", 0))
    blur_torch = bool(cfg.get("postprocess", {}).get("blur_torch", False))

    image_scores: List[float] = []
    pixel_maps: List[np.ndarray] = []
    pixel_masks: List[np.ndarray] = []
    labels_list: List[int] = []
    stat_acc = {k: [] for k in (
        "mu_anom", "mu_normal", "median_anom", "median_normal",
        "q90_anom", "q90_normal", "q95_normal",
    )}

    test_loader = build_loader(cfg["dataset"], split="test", shuffle=False)
    for batch in tqdm(test_loader, leave=False, desc=f"score {category}"):
        images, batch_labels, batch_masks, _paths = batch
        with inference_ctx():
            if gpu_preprocess:
                images = preprocess_images(images, img_size, resize, device)
                if batch_masks is not None:
                    batch_masks = preprocess_masks(
                        batch_masks, img_size, resize, device).cpu()
            else:
                images = images.to(device)
            group_feats = {}
            for gname in gnames:
                patches = group_extractors[gname](images)  # [b, P, D]
                if group_extractors[gname].last_grid_shape is not None:
                    grid = group_extractors[gname].last_grid_shape
                group_feats[gname] = patches
            b = len(images) if isinstance(images, (list, tuple)) \
                else images.shape[0]
            for j in range(b):
                feat_by_group = {g: group_feats[g][j] for g in gnames}
                img_score, s_final = _score_one_image_layergroup(
                    feat_by_group, gnames, group_bank_embs, group_norm_stats,
                    sp_k=sp_k, sp_tau=sp_tau, groupnorm=groupnorm,
                    layer_combine=layer_combine, trim_ratio=trim_ratio,
                    pool=pool, q=q, readout=readout,
                    topmean_ratio=args.topmean_ratio,
                    bank_vectorized=bank_vectorized,
                    query_chunk=sp_query_chunk,
                )
                image_scores.append(img_score)
                labels_list.append(int(batch_labels[j]))
                if grid is not None and batch_masks is not None:
                    smap = patches_to_map(s_final.unsqueeze(0), grid, img_size)
                    smap = blur_map(smap, blur_sigma, use_torch=blur_torch)
                    smap_np = smap.squeeze().float().cpu().numpy()
                    pixel_maps.append(smap_np)
                    m = batch_masks[j]
                    if torch.is_tensor(m):
                        m = m.squeeze().cpu()
                        m_np = m.numpy()
                    else:
                        m_np = np.asarray(m).squeeze()
                    mask_bin = (m_np > 0).astype(np.uint8)
                    pixel_masks.append(mask_bin)
                    _accumulate_score_stats(stat_acc, smap_np, mask_bin)
            del group_feats, images

    labels = np.asarray(labels_list, dtype=np.int64)
    have_masks = len(pixel_masks) == len(labels) and len(pixel_masks) > 0
    if _timing and torch.cuda.is_available():
        torch.cuda.synchronize()
    _t["test_extract_score"] += time.time() - _t0; _t0 = time.time()

    scores = np.asarray(image_scores, dtype=np.float64)
    total_bank = sum(sum(e.shape[0] for e in embs)
                     for embs in group_bank_embs.values())
    result: Dict[str, Any] = {"category": category,
                              "bank_size": int(total_bank)}
    result.update(image_metrics(scores, labels))
    if pixel_maps:
        maps_arr = np.stack(pixel_maps, axis=0)
        masks_arr = np.stack(pixel_masks, axis=0)
        pixel_maps.clear()
        pixel_masks.clear()
        result.update(pixel_metrics(maps_arr, masks_arr, compute_pro=True))
        stats = {k: (float(np.mean(v)) if v else 0.0)
                 for k, v in stat_acc.items()}
        stats["contrast_mu"] = stats["mu_anom"] - stats["mu_normal"]
        stats["contrast_median"] = stats["median_anom"] - stats["median_normal"]
        result["score_stats"] = stats

    if group_oob_audit:
        # Aggregate per-group audits into one (mean of ratios, sum of counts).
        agg = {}
        keys = next(iter(group_oob_audit.values())).keys()
        for kk in keys:
            vals = [a[kk] for a in group_oob_audit.values()]
            agg[kk] = (sum(vals) if kk in ("removed_count", "total_anchors",
                                           "total_rare", "memory_size")
                       else float(np.mean(vals)))
        agg["per_group"] = group_oob_audit
        result["oob_audit"] = agg

    if _timing:
        _t["metric"] = time.time() - _t0
        print(f"[TIMING {category}] " + " ".join(
            f"{k}={v:.1f}s" for k, v in _t.items()))

    del group_bank_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
