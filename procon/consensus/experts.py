"""Heterogeneous expert ensembles for ConsensusCore (spec V11-V15).

Two raw evidence tensors are computed once per image from the consensus banks:

    nn_stack   [B, P]  -- per-bank nearest-anchor distances d_b(z)
    soft_stack [B, P]  -- per-bank soft-projection residuals e_b(z)

From these we derive several *experts* (quantile views) and combine them either
by quantile-curve interpolation (V11/V12/V15) or by per-image rank fusion
(V13/V14). Rank fusion deliberately discards score scale and keeps only ordering,
which avoids the normal-floor inflation that sank the additive prototype/full
recipes.

国밥: Q50 is the stable head chef, Q75 the off-flavor inspector, soft projection
the ingredient-mixing chef. We let them vote on rank, not pour raw broth together.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from procon.consensus.rank_fusion import rank_normalize_map


def _q(stack: torch.Tensor, q: float) -> torch.Tensor:
    """Quantile over the bank dimension ``[B, P] -> [P]``."""
    if stack.shape[0] == 1:
        return stack[0]
    return torch.quantile(stack, float(q), dim=0)


def _qcurve(stack: torch.Tensor, lambda1: float, lambda2: float) -> torch.Tensor:
    """Quantile-curve expert: Q50 + l1(Q75-Q50) + l2(Q90-Q75)."""
    q50 = _q(stack, 0.50)
    q75 = _q(stack, 0.75)
    out = q50 + lambda1 * (q75 - q50)
    if lambda2:
        q90 = _q(stack, 0.90)
        out = out + lambda2 * (q90 - q75)
    return out


def _qinterp(stack: torch.Tensor, eta: float) -> torch.Tensor:
    """Simple quantile interpolation: (1-eta)Q50 + eta Q75."""
    return (1.0 - eta) * _q(stack, 0.50) + eta * _q(stack, 0.75)


def needs_soft(recipe: str) -> bool:
    """Whether an expert recipe requires the soft-projection stack."""
    r = recipe
    if r in ("v12_soft_qcurve", "v13_rank_ensemble", "v13_rank_soft_only",
             "v13_rank_nn_soft50", "v14_expert_agreement"):
        return True
    if r.startswith("v15_soft"):
        return True
    return False


def needs_nn(recipe: str) -> bool:
    """Whether an expert recipe requires the NN-distance stack."""
    r = recipe
    if r in ("v11_nn_qcurve", "v13_rank_ensemble", "v13_rank_nn_only",
             "v13_rank_nn_soft50", "v14_expert_agreement"):
        return True
    if r.startswith("v15_nn"):
        return True
    return False


def combine_expert_score(
    recipe: str,
    args: Any,
    nn_stack: Optional[torch.Tensor],
    soft_stack: Optional[torch.Tensor],
) -> torch.Tensor:
    """Combine experts into a final patch-score vector ``[P]`` for one image.

    Args:
        recipe: one of the V11-V15 recipe names.
        args: parsed CLI namespace (for qcurve lambdas, eta, agreement params).
        nn_stack: ``[B, P]`` per-bank NN distances, or ``None``.
        soft_stack: ``[B, P]`` per-bank soft-projection residuals, or ``None``.
    """
    l1 = float(getattr(args, "qcurve_lambda1", 0.5))
    l2 = float(getattr(args, "qcurve_lambda2", 0.0))

    if recipe == "v11_nn_qcurve":
        return _qcurve(nn_stack, l1, l2)
    if recipe == "v12_soft_qcurve":
        return _qcurve(soft_stack, l1, l2)

    # --- V15: simple quantile interpolation (eta encoded in recipe name) ---
    if recipe.startswith("v15_"):
        eta = {"025": 0.25, "050": 0.50, "075": 0.75}.get(recipe[-3:], 0.5)
        stack = nn_stack if recipe.startswith("v15_nn") else soft_stack
        return _qinterp(stack, eta)

    # --- V13: per-image rank ensembles ---
    if recipe.startswith("v13_"):
        parts = []
        if recipe == "v13_rank_ensemble":
            parts = [_q(nn_stack, 0.50), _q(nn_stack, 0.75),
                     _q(soft_stack, 0.50), _q(soft_stack, 0.75)]
        elif recipe == "v13_rank_nn_only":
            parts = [_q(nn_stack, 0.50), _q(nn_stack, 0.75)]
        elif recipe == "v13_rank_soft_only":
            parts = [_q(soft_stack, 0.50), _q(soft_stack, 0.75)]
        elif recipe == "v13_rank_nn_soft50":
            parts = [_q(nn_stack, 0.50), _q(soft_stack, 0.50)]
        else:
            raise KeyError(f"Unknown V13 recipe: {recipe}")
        ranks = [rank_normalize_map(p) for p in parts]
        return torch.stack(ranks, dim=0).mean(dim=0)

    # --- V14: expert-agreement boost (heuristic, diagnostic) ---
    if recipe == "v14_expert_agreement":
        theta = float(getattr(args, "expert_agreement_theta", 0.90))
        min_votes = int(getattr(args, "expert_agreement_min_votes", 2))
        alpha = float(getattr(args, "expert_agreement_alpha", 0.1))
        base = rank_normalize_map(_q(nn_stack, 0.50))
        others = [
            rank_normalize_map(_q(nn_stack, 0.75)),
            rank_normalize_map(_q(soft_stack, 0.50)),
            rank_normalize_map(_q(soft_stack, 0.75)),
        ]
        votes = torch.zeros_like(base)
        for r in others:
            votes = votes + (r > theta).float()
        boost = (votes >= min_votes).float()
        return base + alpha * boost

    raise KeyError(f"Unknown expert recipe: {recipe}")


def is_multiscale(recipe: str) -> bool:
    """Whether a recipe is a multi-scale soft-projection family (V16-V19)."""
    return recipe.startswith(("v16_", "v17_", "v18_", "v19_"))


def multiscale_config(recipe: str, args: Any) -> dict:
    """Resolve (k_list, tau_scales, q_list, expert_fusion) for V16-V19.

    Each family builds per-(k,tau) bank-fused experts ``S_{k,tau}`` and then
    fuses those experts again. V19 instead varies the bank-fusion quantile.
    """
    k_list = list(getattr(args, "softproj_k_list", None) or [3, 5, 7])
    tau_scales = list(getattr(args, "softproj_tau_scales", None)
                      or [0.5, 1.0, 2.0])
    cfg = {
        "k_list": [5],
        "tau_scales": [1.0],
        "q_list": [0.5],          # bank-fusion quantiles per expert
        "expert_fusion": "median",
        "expert_q": float(getattr(args, "softproj_expert_quantile", 0.75)),
    }
    if recipe == "v16_multik_softproj_median":
        cfg.update(k_list=k_list, expert_fusion="median")
    elif recipe == "v16_multik_softproj_q75":
        cfg.update(k_list=k_list, expert_fusion="quantile")
    elif recipe == "v17_multitau_softproj_median":
        cfg.update(tau_scales=tau_scales, expert_fusion="median")
    elif recipe == "v17_multitau_softproj_q75":
        cfg.update(tau_scales=tau_scales, expert_fusion="quantile")
    elif recipe == "v18_multiscale_softproj_median":
        cfg.update(k_list=k_list, tau_scales=tau_scales, expert_fusion="median")
    elif recipe == "v18_multiscale_softproj_q75":
        cfg.update(k_list=k_list, tau_scales=tau_scales,
                   expert_fusion="quantile")
    elif recipe == "v19_softproj_qfamily_median":
        cfg.update(q_list=[0.5, 0.65, 0.75], expert_fusion="median")
    elif recipe == "v19_softproj_q65":
        cfg.update(q_list=[0.65], expert_fusion="median")
    else:
        raise KeyError(f"Unknown multiscale recipe: {recipe}")
    return cfg


def combine_multiscale(
    expert_scores: list,
    fusion: str,
    quantile: float,
) -> torch.Tensor:
    """Fuse multiple soft-projection expert maps ``[P]`` into one ``[P]``."""
    stack = torch.stack(expert_scores, dim=0)  # [E, P]
    if stack.shape[0] == 1:
        return stack[0]
    if fusion == "quantile":
        return torch.quantile(stack, float(quantile), dim=0)
    return stack.median(dim=0).values
