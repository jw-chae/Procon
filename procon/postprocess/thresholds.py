"""Threshold selection utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def threshold_from_fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, tpr: float) -> Tuple[float, float]:
    """Find threshold that achieves target TPR."""
    order = np.argsort(scores)[::-1]
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    pos = labels_sorted == 1
    neg = labels_sorted == 0
    tp = np.cumsum(pos)
    fp = np.cumsum(neg)
    tpr_curve = tp / max(tp[-1], 1)
    fpr_curve = fp / max(fp[-1], 1)
    idx = np.searchsorted(tpr_curve, tpr, side="left")
    idx = min(idx, len(scores_sorted) - 1)
    return float(scores_sorted[idx]), float(fpr_curve[idx])


def threshold_from_fpr_limit(scores: np.ndarray, labels: np.ndarray, fpr_limit: float) -> Tuple[float, float]:
    """Find threshold that stays under FPR limit."""
    order = np.argsort(scores)[::-1]
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    pos = labels_sorted == 1
    neg = labels_sorted == 0
    tp = np.cumsum(pos)
    fp = np.cumsum(neg)
    tpr_curve = tp / max(tp[-1], 1)
    fpr_curve = fp / max(fp[-1], 1)
    valid = np.where(fpr_curve <= fpr_limit)[0]
    if len(valid) == 0:
        idx = len(scores_sorted) - 1
    else:
        idx = int(valid[-1])
    return float(scores_sorted[idx]), float(tpr_curve[idx])


def choose_threshold(scores: np.ndarray, labels: np.ndarray, cfg: Dict) -> Dict[str, float]:
    """Choose threshold based on configuration."""
    if cfg.get("type") == "fpr_at_tpr":
        threshold, fpr = threshold_from_fpr_at_tpr(scores, labels, float(cfg["tpr"]))
        return {"threshold": threshold, "fpr_at_tpr": fpr}
    if cfg.get("type") == "fpr_limit":
        threshold, tpr = threshold_from_fpr_limit(scores, labels, float(cfg["fpr"]))
        return {"threshold": threshold, "tpr_at_fpr": tpr}
    if cfg.get("type") == "fixed":
        return {"threshold": float(cfg["value"])}
    raise KeyError(f"Unsupported threshold type: {cfg.get('type')}")
