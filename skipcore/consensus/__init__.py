"""ConsensusCore: training-free retrieval UAD via consensus over discrete normal projectors.

This package treats a PatchCore-style coreset memory bank as a *discrete normal
projector* and stabilizes it by building several independently perturbed banks and
fusing their nearest-neighbor distances with a robust quantile operator.

Modules:
    banks      -- construct B perturbed coreset memory banks
    fusion     -- robust distance-fusion operators (median / quantile / trimmed-mean)
    prototypes -- INP-inspired intrinsic per-image prototype refinement
    runner     -- per-category build + evaluate pipeline
"""
from skipcore.consensus.fusion import fuse_distances
from skipcore.consensus.banks import build_consensus_banks
from skipcore.consensus.prototypes import intrinsic_prototype_refine

__all__ = [
    "fuse_distances",
    "build_consensus_banks",
    "intrinsic_prototype_refine",
]
