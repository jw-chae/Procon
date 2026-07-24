"""Anchor reliability estimation for soft ConsensusCore (spec V3 / V4).

A coreset anchor can be a *bad anchor*: close to a test patch yet not a reliable
representative of the normal manifold, which is exactly what causes false-normal
projections. We down-weight such anchors with a soft distance:

    d_b^soft(z) = min_{m in M_b} [ ||z - m|| + lambda * (1 - r(m)) ]

where ``r(m) in [0, 1]`` is an anchor reliability score. Two training-free
estimators are provided:

* ``stability`` (V3): an anchor is reliable if a similar anchor reappears across
  the other independently perturbed banks.
* ``oob`` (V4): MeDS-inspired out-of-bag estimate. For each anchor, look at its
  distance to the banks that do *not* contain it; a reliable anchor sits in a
  dense, low-variance region of the other banks.

Everything is computed once at fit time from the memory banks only -- no model
is trained.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from procon.memory.bank import MemoryBank


def _emb_f32(bank: MemoryBank, device: torch.device) -> torch.Tensor:
    return bank.embeddings.to(device=device, dtype=torch.float32)


def _auto_delta(banks: List[MemoryBank], device: torch.device) -> float:
    """Median nearest-neighbor distance among anchors of the first bank."""
    emb = _emb_f32(banks[0], device)
    n = emb.shape[0]
    if n < 2:
        return 1.0
    d = torch.cdist(emb, emb)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    return float(torch.median(nn).item())


def compute_stability_reliability(
    banks: List[MemoryBank],
    device: torch.device,
    delta: Optional[float] = None,
) -> List[torch.Tensor]:
    """Per-anchor stability reliability (V3).

    ``r_stab(m) = mean_b 1[ exists m_b in M_b with ||m - m_b|| < delta ]`` over
    the *other* banks. Returns one ``[M_b]`` reliability tensor per bank.
    """
    if delta is None:
        delta = _auto_delta(banks, device)
    embs = [_emb_f32(b, device) for b in banks]
    B = len(embs)
    out: List[torch.Tensor] = []
    for b in range(B):
        emb_b = embs[b]
        if B == 1:
            out.append(torch.ones(emb_b.shape[0], device=device))
            continue
        hits = torch.zeros(emb_b.shape[0], device=device)
        for bb in range(B):
            if bb == b:
                continue
            min_d = torch.cdist(emb_b, embs[bb]).min(dim=1).values
            hits += (min_d < delta).float()
        out.append(hits / float(B - 1))
    return out


def compute_oob_reliability(
    banks: List[MemoryBank],
    device: torch.device,
    tau_mu: Optional[float] = None,
    tau_sigma: Optional[float] = None,
) -> List[torch.Tensor]:
    """MeDS-inspired out-of-bag reliability (V4).

    For anchor ``m`` of bank ``b``, gather its min distances to every *other*
    bank, ``d_{b'}^OOB(m)``. With ``mu = median_b' d`` and ``sigma = IQR_b' d``:

        r_OOB(m) = exp(-mu / tau_mu) * exp(-sigma / tau_sigma)

    ``tau_mu`` / ``tau_sigma`` default (``auto``) to the median of ``mu`` /
    ``sigma`` across all anchors. Returns one ``[M_b]`` tensor per bank.
    """
    embs = [_emb_f32(b, device) for b in banks]
    B = len(embs)
    if B == 1:
        return [torch.ones(embs[0].shape[0], device=device)]

    mus: List[torch.Tensor] = []
    sigmas: List[torch.Tensor] = []
    for b in range(B):
        emb_b = embs[b]
        # [M_b, B-1] min distance to each other bank.
        cols = []
        for bb in range(B):
            if bb == b:
                continue
            cols.append(torch.cdist(emb_b, embs[bb]).min(dim=1).values)
        d_oob = torch.stack(cols, dim=1)  # [M_b, B-1]
        mu = torch.median(d_oob, dim=1).values
        q = torch.quantile(d_oob, torch.tensor([0.25, 0.75], device=device), dim=1)
        iqr = q[1] - q[0]
        mus.append(mu)
        sigmas.append(iqr)

    all_mu = torch.cat(mus)
    all_sigma = torch.cat(sigmas)
    tm = float(torch.median(all_mu).item()) if tau_mu is None else float(tau_mu)
    ts = float(torch.median(all_sigma).item()) if tau_sigma is None else float(tau_sigma)
    tm = max(tm, 1e-6)
    ts = max(ts, 1e-6)

    out: List[torch.Tensor] = []
    for mu, sigma in zip(mus, sigmas):
        r = torch.exp(-mu / tm) * torch.exp(-sigma / ts)
        out.append(r.clamp(0.0, 1.0))
    return out


def reliability_to_penalty(
    reliabilities: List[torch.Tensor],
    lambda_reliability: float,
) -> List[torch.Tensor]:
    """Convert reliability ``r`` into additive distance penalty ``lambda(1-r)``."""
    return [lambda_reliability * (1.0 - r) for r in reliabilities]
