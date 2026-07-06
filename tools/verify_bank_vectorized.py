"""Verify bank-vectorized soft projection == per-bank legacy loop, bit-for-bit.

Champion path: per image, per group, loop over B=5 banks calling
``soft_projection_bank``. The (alpha) optimisation stacks the 5 banks into a
``[B, M, D]`` tensor and does ONE batched cdist/topk, but must:
  1. produce per-bank residuals bit-identical to the loop, and
  2. consume the ``auto_tau`` RNG in the *same per-bank order* (tau uses
     ``torch.randperm`` on the global generator).

This script measures (1) the residual diff and (2) confirms the tau values
match when computed in legacy order. If WORST diff == 0 the vectorised path is
safe for the legacy/champion reproduction.
"""
import torch

from skipcore.consensus.soft_projection import soft_projection_bank, auto_tau

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P, D, M, B, K = 784, 512, 120, 5, 5


def legacy(queries, banks):
    """Per-bank loop, exactly as the champion runs it (tau=auto each call)."""
    torch.manual_seed(123)
    res = []
    taus = []
    for b in range(B):
        # replicate soft_projection_bank with tau=None to capture tau too
        bank = banks[b]
        tau = auto_tau(queries, bank, K)
        taus.append(tau)
        q32 = queries.float()
        b32 = bank.float()
        d2 = torch.cdist(q32, b32) ** 2
        knn_d2, knn_idx = torch.topk(d2, K, dim=1, largest=False)
        w = torch.softmax(-knn_d2 / tau, dim=1)
        nb = b32[knn_idx]
        z_hat = (w.unsqueeze(-1) * nb).sum(dim=1)
        res.append(torch.linalg.vector_norm(q32 - z_hat, dim=1))
    return torch.stack(res, 0), taus  # [B, P]


def vectorized(queries, banks):
    """Bank-vectorized: tau per-bank in legacy order, then batched residual."""
    torch.manual_seed(123)
    # (1) tau in the SAME order as legacy (consumes RNG identically)
    taus = [auto_tau(queries, banks[b], K) for b in range(B)]
    tau_t = torch.tensor(taus, device=DEV).view(B, 1, 1)
    # (2) batched cdist over the bank dimension
    q32 = queries.float().unsqueeze(0).expand(B, P, D)   # [B, P, D]
    bstack = torch.stack([banks[b].float() for b in range(B)], 0)  # [B, M, D]
    d2 = torch.cdist(q32, bstack) ** 2                   # [B, P, M]
    knn_d2, knn_idx = torch.topk(d2, K, dim=2, largest=False)  # [B, P, K]
    w = torch.softmax(-knn_d2 / tau_t, dim=2)            # [B, P, K]
    # gather neighbors per bank
    nb = torch.gather(
        bstack.unsqueeze(1).expand(B, P, M, D), 2,
        knn_idx.unsqueeze(-1).expand(B, P, K, D))        # [B, P, K, D]
    z_hat = (w.unsqueeze(-1) * nb).sum(dim=2)            # [B, P, D]
    res = torch.linalg.vector_norm(q32 - z_hat, dim=2)  # [B, P]
    return res, taus


def main():
    torch.manual_seed(0)
    queries = torch.randn(P, D, device=DEV)
    banks = [torch.randn(M, D, device=DEV) for _ in range(B)]

    r_old, t_old = legacy(queries, banks)
    r_new, t_new = vectorized(queries, banks)

    tau_diff = max(abs(a - b) for a, b in zip(t_old, t_new))
    res_diff = (r_old - r_new).abs().max().item()
    # median over banks (the actual per-layer score), which is what feeds scoring
    med_old = r_old.median(dim=0).values
    med_new = r_new.median(dim=0).values
    med_diff = (med_old - med_new).abs().max().item()

    print(f"tau max diff      = {tau_diff:.3e}")
    print(f"residual max diff = {res_diff:.3e}")
    print(f"median  max diff  = {med_diff:.3e}")
    ok = res_diff < 1e-6 and tau_diff < 1e-9
    print("=> BIT-IDENTICAL" if ok else "=> DIFFERS (investigate)")


if __name__ == "__main__":
    main()
