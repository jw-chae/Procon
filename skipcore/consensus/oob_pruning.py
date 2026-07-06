"""OOB-residual coreset pruning: remove *isolated* anchors, keep *rare* ones.

PatchCore's k-center coreset optimizes COVERAGE, which is mismatched with V9's
soft-projection scoring (anchors act as a local reconstruction basis, not as
nearest references). A coverage coreset keeps ISOLATED anchors (off the normal
manifold) that let V9 reconstruct anomalies as normal -> false negatives.

The probe that separates *rare* (keep) from *isolated* (remove) is the
out-of-bag (OOB) soft-projection residual: reconstruct anchor ``m`` from banks
that do NOT contain ``m`` (the seed-perturbed banks are independent k-center
selections, so they serve as OOB folds). Low OOB residual => ``m`` lies on the
normal manifold (rare, KEEP). High OOB residual => ``m`` is off-manifold
(isolated, REMOVE).

This differs from the failed V22, which pruned by *density* (softmax weight an
anchor receives from train normals). Density cannot tell rare from isolated --
both are low-density -- so V22 also removed rare anchors and collapsed
high-variation categories (transistor). OOB residual asks "is this rebuildable
from other normals?", which keeps rare and removes only isolated.

Training-free: only frozen features + existing banks are used.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from skipcore.memory.bank import MemoryBank


def _softproj_residual(x: torch.Tensor, anchors: torch.Tensor,
                       k: int, tau: Optional[float]) -> torch.Tensor:
    """Per-sample V9 soft-projection residual of ``x`` vs ``anchors`` -> [N]."""
    if anchors.shape[0] == 0:
        return torch.full((x.shape[0],), float("inf"), device=x.device)
    kk = min(k, anchors.shape[0])
    d2 = torch.cdist(x, anchors) ** 2
    knn_d2, knn_idx = torch.topk(d2, kk, dim=1, largest=False)
    t = tau if tau is not None else (float(torch.median(knn_d2).item()) or 1e-8)
    w = torch.softmax(-knn_d2 / t, dim=1)
    nb = anchors[knn_idx]
    z_hat = (w.unsqueeze(-1) * nb).sum(dim=1)
    return torch.linalg.vector_norm(x - z_hat, dim=1)


def _density(anchors: torch.Tensor, sample: torch.Tensor,
             k: int, tau: Optional[float]) -> torch.Tensor:
    """V22-style density: total softmax weight each anchor receives -> [M]."""
    if anchors.shape[0] == 0:
        return torch.zeros(0, device=anchors.device)
    kk = min(k, anchors.shape[0])
    d2 = torch.cdist(sample, anchors) ** 2          # [S, M]
    knn_d2, knn_idx = torch.topk(d2, kk, dim=1, largest=False)
    t = tau if tau is not None else (float(torch.median(knn_d2).item()) or 1e-8)
    w = torch.softmax(-knn_d2 / t, dim=1)
    contrib = torch.zeros(anchors.shape[0], device=anchors.device)
    contrib.scatter_add_(0, knn_idx.reshape(-1), w.reshape(-1))
    return contrib


def compute_oob_residual(target: torch.Tensor, others: List[torch.Tensor],
                         k: int, tau: Optional[float],
                         agg: str = "median") -> torch.Tensor:
    """OOB residual for each anchor of ``target`` reconstructed from ``others``.

    ``others`` are the banks that do NOT contain the target bank's anchors
    (every other seed-perturbed bank). The per-fold residuals are aggregated by
    ``agg`` (median / mean / max). High value => isolated (off-manifold).
    """
    if not others:
        # No OOB fold available -> rebuild from the target bank itself excluding
        # each anchor's own kNN-self (k+1 neighbours, drop the nearest = self).
        kk = min(k + 1, target.shape[0])
        d2 = torch.cdist(target, target) ** 2
        knn_d2, knn_idx = torch.topk(d2, kk, dim=1, largest=False)
        knn_d2, knn_idx = knn_d2[:, 1:], knn_idx[:, 1:]      # drop self
        t = tau if tau is not None else (float(torch.median(knn_d2).item()) or 1e-8)
        w = torch.softmax(-knn_d2 / t, dim=1)
        z_hat = (w.unsqueeze(-1) * target[knn_idx]).sum(dim=1)
        return torch.linalg.vector_norm(target - z_hat, dim=1)
    per_fold = torch.stack(
        [_softproj_residual(target, o, k, tau) for o in others], dim=0
    )  # [F, M]
    if agg == "mean":
        return per_fold.mean(dim=0)
    if agg == "max":
        return per_fold.max(dim=0).values
    return per_fold.median(dim=0).values  # default robust


def _train_recon_residual(sample: torch.Tensor, bank_embs: List[torch.Tensor],
                          k: int, tau: Optional[float]) -> float:
    """Mean V9 residual of train-normal sample under the given banks (median fuse)."""
    if not bank_embs:
        return float("nan")
    per_bank = torch.stack(
        [_softproj_residual(sample, e, k, tau) for e in bank_embs], dim=0
    )  # [B, S]
    fused = per_bank.median(dim=0).values
    return float(fused.mean().item())


def prune_banks_oob(
    banks: List[MemoryBank],
    all_patches: torch.Tensor,
    device: torch.device,
    *,
    k: int = 5,
    tau: Optional[float] = None,
    frac: float = 0.0,
    agg: str = "median",
    threshold: str = "quantile",
    refill: str = "none",
    abs_c: float = 2.0,
    sample_max: int = 4000,
    seed: int = 0,
) -> Tuple[List[MemoryBank], Dict[str, float]]:
    """Prune isolated anchors from each bank by OOB residual; return banks + audit.

    threshold:
      - ``quantile``: remove the top ``frac`` of anchors by OOB residual.
      - ``absolute``: remove anchors with OOB residual > median_train + c*IQR,
        tying the cut to the normal-residual scale (``abs_c`` = c).
    refill:
      - ``none``: keep the surviving anchors only.
      - ``refill``: farthest-point top-up from ``all_patches`` (excluding removed)
        back to the original bank size, restoring coverage.
    """
    embs = [b.embeddings.to(device=device, dtype=torch.float32) for b in banks]
    B = len(embs)

    # Train-normal sample for residual / density audits.
    n = all_patches.shape[0]
    rng = np.random.default_rng(seed)
    s = min(sample_max, n)
    sub_idx = torch.from_numpy(rng.choice(n, size=s, replace=False))
    sample = all_patches[sub_idx].to(device=device, dtype=torch.float32)

    train_resid_before = _train_recon_residual(sample, embs, k, tau)

    # Absolute-threshold reference scale from train-normal OOB residuals.
    abs_cut = None
    if threshold == "absolute":
        # Use the pooled train-sample residual distribution as the manifold scale.
        ref = torch.stack(
            [_softproj_residual(sample, e, k, tau) for e in embs], dim=0
        ).median(dim=0).values
        med = float(ref.median().item())
        q1 = float(torch.quantile(ref, 0.25).item())
        q3 = float(torch.quantile(ref, 0.75).item())
        abs_cut = med + abs_c * (q3 - q1)

    pruned_embs: List[torch.Tensor] = []
    removed_roob: List[float] = []
    removed_density: List[float] = []
    total_removed = 0
    total_anchors = 0
    # Rare bookkeeping (aggregated over banks).
    kept_rare = 0
    total_rare = 0

    for b in range(B):
        target = embs[b]
        others = [embs[j] for j in range(B) if j != b]
        roob = compute_oob_residual(target, others, k, tau, agg)   # [M]
        dens = _density(target, sample, k, tau)                    # [M]
        M = target.shape[0]
        total_anchors += M

        # rare = low-density (bottom 10%) AND low OOB residual (bottom 50%).
        if M >= 10:
            dens_thr = torch.quantile(dens, 0.10)
            roob_med = torch.quantile(roob, 0.50)
            rare_mask = (dens <= dens_thr) & (roob <= roob_med)
        else:
            rare_mask = torch.zeros(M, dtype=torch.bool, device=device)

        # Selection of anchors to REMOVE (isolated = high OOB residual).
        if frac <= 0.0 and threshold != "absolute":
            remove_mask = torch.zeros(M, dtype=torch.bool, device=device)
        elif threshold == "absolute":
            remove_mask = roob > abs_cut
        else:  # quantile
            n_rm = int(round(M * frac))
            remove_mask = torch.zeros(M, dtype=torch.bool, device=device)
            if n_rm > 0:
                rm_idx = torch.topk(roob, n_rm, largest=True).indices
                remove_mask[rm_idx] = True

        keep_mask = ~remove_mask
        kept = target[keep_mask]

        # Audit accumulation.
        if remove_mask.any():
            removed_roob.extend(roob[remove_mask].tolist())
            removed_density.extend(dens[remove_mask].tolist())
            total_removed += int(remove_mask.sum().item())
        total_rare += int(rare_mask.sum().item())
        kept_rare += int((rare_mask & keep_mask).sum().item())

        # Optional coverage refill via farthest-point top-up from full pool.
        if refill == "refill" and kept.shape[0] < M:
            kept = _farthest_point_refill(
                kept, all_patches.to(device), target_size=M,
                exclude=target[remove_mask], device=device,
            )
        pruned_embs.append(kept)

    train_resid_after = _train_recon_residual(sample, pruned_embs, k, tau)

    new_banks = [
        MemoryBank(embeddings=pruned_embs[b].cpu(), positions=None,
                   metadata={"method": "oob_pruned",
                             "count": str(pruned_embs[b].shape[0])})
        for b in range(B)
    ]
    audit = {
        "train_recon_residual_before": train_resid_before,
        "train_recon_residual_after": train_resid_after,
        "kept_rare_ratio": (kept_rare / total_rare) if total_rare else 1.0,
        "removed_count": int(total_removed),
        "removed_mean_density": float(np.mean(removed_density))
            if removed_density else 0.0,
        "removed_mean_Roob": float(np.mean(removed_roob))
            if removed_roob else 0.0,
        "total_anchors": int(total_anchors),
        "total_rare": int(total_rare),
        "memory_size": int(sum(e.shape[0] for e in pruned_embs)),
    }
    return new_banks, audit


def _farthest_point_refill(kept: torch.Tensor, pool: torch.Tensor,
                           target_size: int, exclude: torch.Tensor,
                           device: torch.device,
                           pool_cap: int = 20000) -> torch.Tensor:
    """Greedy farthest-point top-up of ``kept`` from ``pool`` to ``target_size``.

    Never re-adds anchors close to the removed (isolated) ``exclude`` set.
    """
    if kept.shape[0] >= target_size:
        return kept
    # Subsample the pool for tractable farthest-point search.
    n = pool.shape[0]
    if n > pool_cap:
        idx = torch.randperm(n, device=pool.device)[:pool_cap]
        pool = pool[idx]
    pool = pool.to(device=device, dtype=torch.float32)
    # Min distance from each pool point to the current kept set.
    min_d = torch.cdist(pool, kept).min(dim=1).values
    # Suppress points near the removed isolated anchors.
    if exclude.numel() > 0:
        excl_d = torch.cdist(pool, exclude).min(dim=1).values
        # If a pool point is closer to a removed anchor than to any kept anchor,
        # it is likely the same isolated region -> deprioritize.
        min_d = torch.where(excl_d < min_d, torch.zeros_like(min_d), min_d)
    added = []
    cur = kept
    need = target_size - kept.shape[0]
    for _ in range(need):
        j = int(torch.argmax(min_d).item())
        if min_d[j] <= 0:
            break
        added.append(j)
        new_pt = pool[j:j + 1]
        d_new = torch.cdist(pool, new_pt).squeeze(1)
        min_d = torch.minimum(min_d, d_new)
        min_d[j] = -1.0
    if added:
        cur = torch.cat([kept, pool[torch.tensor(added, device=device)]], dim=0)
    return cur
