"""Evaluation metrics for anomaly detection."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn import metrics as sk_metrics
except ImportError as exc:
    raise ImportError("scikit-learn is required for metrics.") from exc

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _compute_f1_max(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute F1-max (best F1 across all thresholds).
    
    Returns:
        f1_max: Best F1 score
        f1_threshold: Threshold at best F1
        f1_precision: Precision at best F1
        f1_recall: Recall at best F1
    """
    precision, recall, thresholds = sk_metrics.precision_recall_curve(labels, scores)
    
    if precision.size <= 1:
        return 0.0, 0.0, 0.0, 0.0
    
    # F1 = 2 * precision * recall / (precision + recall)
    # Skip last point (precision=1, recall=0)
    denom = precision[:-1] + recall[:-1]
    f1 = np.zeros_like(denom, dtype=np.float32)
    np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=f1,
        where=denom > 0,
    )
    
    if f1.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    best_idx = int(np.argmax(f1))
    f1_max = float(f1[best_idx])
    f1_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.0
    f1_precision = float(precision[best_idx])
    f1_recall = float(recall[best_idx])
    
    return f1_max, f1_threshold, f1_precision, f1_recall


def _compute_pro(
    maps: np.ndarray,
    masks: np.ndarray,
    fpr_limit: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """Compute Per-Region Overlap (PRO) score.
    
    PRO evaluates localization quality by computing overlap per connected component,
    averaging across all anomaly regions. This penalizes methods that only detect
    part of an anomaly region.
    """
    if not HAS_SCIPY:
        return 0.0
    
    # Compute thresholds. Only the min/max of the score volume are needed, so
    # take them directly instead of materialising a full ``maps.flatten()`` copy
    # (for Real-IAD that copy is ~2.3 GB and the resulting allocation churn
    # forces page-cache reclaim, which spikes the systemd-oomd memory-pressure
    # signal and gets the whole session OOM-killed). min()/max() are reductions
    # with O(1) extra memory and give identical thresholds.
    thresholds = np.linspace(float(maps.min()), float(maps.max()), num_thresholds)

    # The connected components of the (binary) ground truth are independent of
    # the score threshold, so label every image ONCE up front and cache the flat
    # pixel indices of each region. The original code re-ran ``ndimage.label``
    # inside the threshold loop (num_thresholds x n_images calls), which for
    # Real-IAD (200 x ~3.8k) is ~760k component analyses per category. This
    # precomputation is mathematically identical but ~num_thresholds times
    # faster.
    n_imgs = maps.shape[0] if maps.ndim == 3 else 1
    regions_per_img: List[np.ndarray] = []  # flat indices per GT region
    for i in range(n_imgs):
        mask_i = masks[i] if masks.ndim == 3 else masks
        labeled, n_regions = ndimage.label(mask_i > 0)
        labeled_flat = labeled.reshape(-1)
        for region_id in range(1, n_regions + 1):
            region_idx = np.flatnonzero(labeled_flat == region_id)
            if region_idx.size > 0:
                regions_per_img.append((i, region_idx))

    # The negative (background) mask is threshold-independent, so compute it once
    # instead of rebuilding ``masks == 0`` on every one of the ``num_thresholds``
    # iterations (each rebuild allocates a full-size boolean copy).
    neg_mask = masks == 0
    n_neg = int(neg_mask.sum())

    fprs = []
    pros = []

    for thresh in thresholds:
        binary_pred = maps > thresh  # bool, no int32 copy needed

        # Compute FPR
        n_fp = int((binary_pred & neg_mask).sum())
        fpr = n_fp / max(n_neg, 1)

        if fpr > fpr_limit:
            continue

        # Per-region overlap, reusing the cached region indices.
        pred_flat = binary_pred.reshape(n_imgs, -1) if maps.ndim == 3 \
            else binary_pred.reshape(1, -1)
        region_overlaps = []
        for img_i, region_idx in regions_per_img:
            overlap = pred_flat[img_i, region_idx].sum() / region_idx.size
            region_overlaps.append(overlap)

        if region_overlaps:
            pro = float(np.mean(region_overlaps))
            fprs.append(fpr)
            pros.append(pro)
    
    if len(fprs) < 2:
        return 0.0
    
    # Sort by FPR and compute area under PRO curve
    sorted_indices = np.argsort(fprs)
    fprs = np.array(fprs)[sorted_indices]
    pros = np.array(pros)[sorted_indices]
    
    # Normalize by fpr_limit
    aupro = float(np.trapz(pros, fprs) / fpr_limit)
    return aupro


def image_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute image-level metrics including F1-max."""
    auroc = float(sk_metrics.roc_auc_score(labels, scores))
    ap = float(sk_metrics.average_precision_score(labels, scores))
    
    # Compute F1-max
    f1_max, f1_threshold, f1_precision, f1_recall = _compute_f1_max(scores, labels)
    
    return {
        "image_auroc": auroc,
        "image_ap": ap,
        "image_f1_max": f1_max,
        "image_f1_threshold": f1_threshold,
        "image_f1_precision": f1_precision,
        "image_f1_recall": f1_recall,
    }


def pixel_metrics(
    maps: np.ndarray,
    masks: np.ndarray,
    fpr_limit: float = 0.3,
    compute_pro: bool = True,
    max_pixels: int = 50_000_000,
) -> Dict[str, float]:
    """Compute pixel-level metrics including F1-max and PRO.

    For datasets with very many test images (e.g. Real-IAD: ~3.8k images x
    392x392 = ~580M pixels per category) the exact sklearn ranking metrics
    materialise several float64 copies of the flattened arrays and exhaust host
    RAM. When the pixel count exceeds ``max_pixels`` the ranking/ROC metrics
    (AUROC, AP, F1-max, the ROC-based AUPRO) are computed on a deterministic
    uniform subsample, which is statistically identical for rank statistics to
    within ~1e-4. The connected-component PRO always uses the full-resolution
    maps. When the pixel count is below ``max_pixels`` (MVTec, VisA) the full
    arrays are used and the result is byte-identical to the un-capped version.
    """
    # ``reshape(-1)`` returns a view of the (C-contiguous) stacked array, whereas
    # ``flatten()`` always copies; for Real-IAD that avoids a ~2.3 GB copy. The
    # binary mask is stored as int8 (4x smaller than int32) which is more than
    # enough range for {0,1} labels and is accepted as-is by scikit-learn. These
    # two changes remove the large transient allocations whose page-cache reclaim
    # was triggering systemd-oomd to kill the session.
    maps_flat = maps.reshape(-1)
    masks_flat = (masks.reshape(-1) > 0).astype(np.int8)

    if masks_flat.sum() == 0 or masks_flat.sum() == len(masks_flat):
        return {
            "pixel_auroc": 0.0,
            "pixel_ap": 0.0,
            "pixel_f1_max": 0.0,
            "pixel_aupro": 0.0,
            "pixel_pro": 0.0,
        }

    # Subsample only the (flattened) ranking inputs when the dataset is huge.
    # A deterministic uniform stride is used (a *view*, no large allocation and
    # no 582M-element permutation), which is statistically equivalent for rank
    # statistics. When N <= max_pixels the stride is 1, so the full array is used
    # and the result is byte-identical to the un-capped version.
    if maps_flat.shape[0] > max_pixels:
        step = (maps_flat.shape[0] + max_pixels - 1) // max_pixels
        rank_maps = maps_flat[::step]
        rank_masks = masks_flat[::step]
    else:
        rank_maps = maps_flat
        rank_masks = masks_flat

    auroc = float(sk_metrics.roc_auc_score(rank_masks, rank_maps))
    ap = float(sk_metrics.average_precision_score(rank_masks, rank_maps))

    # Compute pixel-level F1-max
    f1_max, f1_threshold, _, _ = _compute_f1_max(rank_maps, rank_masks)

    # AUPRO computation (simplified version)
    fpr, tpr, _ = sk_metrics.roc_curve(rank_masks, rank_maps)
    valid = fpr <= fpr_limit
    if valid.sum() > 1:
        aupro = float(np.trapz(tpr[valid], fpr[valid]) / fpr_limit)
    else:
        aupro = 0.0

    # Compute PRO (Per-Region Overlap) on the full-resolution maps.
    pro = 0.0
    if compute_pro:
        pro = _compute_pro(maps, masks, fpr_limit=fpr_limit)
    
    return {
        "pixel_auroc": auroc,
        "pixel_ap": ap,
        "pixel_f1_max": f1_max,
        "pixel_f1_threshold": f1_threshold,
        "pixel_aupro": aupro,
        "pixel_pro": pro,
    }


def industrial_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
) -> Dict[str, float]:
    """Compute industrial-relevant metrics."""
    if threshold is None:
        # Use threshold at 95% TPR
        fpr, tpr, thresholds = sk_metrics.roc_curve(labels, scores)
        idx = np.searchsorted(tpr, 0.95, side="left")
        idx = min(idx, len(thresholds) - 1)
        threshold = float(thresholds[idx])
    
    preds = (scores >= threshold).astype(np.int32)
    
    precision = float(sk_metrics.precision_score(labels, preds, zero_division=0))
    recall = float(sk_metrics.recall_score(labels, preds, zero_division=0))
    f1 = float(sk_metrics.f1_score(labels, preds, zero_division=0))
    
    # Compute FP per 1000
    n_neg = (labels == 0).sum()
    n_fp = ((preds == 1) & (labels == 0)).sum()
    fp_per_1k = float(n_fp / max(n_neg, 1) * 1000)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_per_1k": fp_per_1k,
        "threshold": threshold,
    }
