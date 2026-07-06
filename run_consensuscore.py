#!/usr/bin/env python
"""ConsensusCore: training-free retrieval UAD via consensus over discrete projectors.

We reinterpret a PatchCore-style coreset memory bank as a *discrete normal
projector* and stabilize it by building several independently perturbed banks and
fusing their nearest-neighbor distances with a robust quantile operator. No
encoder/decoder is trained.

Examples
--------
ConsensusCore (median fusion, 5 banks):
    python run_consensuscore.py \
        --dataset mvtec --feature dinov2_vitb14 \
        --memory_ratio 0.01 --num_banks 5 --bank_diversity seed \
        --fusion median --image_score topmean --topmean_ratio 0.01

ConsensusCore-Q75:
    python run_consensuscore.py \
        --dataset mvtec --memory_ratio 0.01 --num_banks 5 \
        --bank_diversity seed --fusion quantile --quantile 0.75 \
        --image_score topmean --topmean_ratio 0.01

ConsensusCore + intrinsic prototypes:
    python run_consensuscore.py \
        --dataset mvtec --memory_ratio 0.01 --num_banks 5 --fusion median \
        --use_intrinsic_proto --proto_select_quantile 0.7 \
        --num_prototypes 8 --proto_fusion add --proto_alpha 0.5

Ablation baselines (sections 6.A-6.C):
    python run_consensuscore.py --dataset mvtec --method single
    python run_consensuscore.py --dataset mvtec --method full
    python run_consensuscore.py --dataset mvtec --method random
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

# Populate component registries (datasets, backbones, extractors, backends).
import skipcore.data.datasets  # noqa: F401
from skipcore.models.feature_extractors import vit_patchcore  # noqa: F401
from skipcore.models.backbones import dinov2_multilayer  # noqa: F401
from skipcore.inference import torch_knn, faiss_gpu  # noqa: F401

from skipcore.consensus.recipes import get_recipe, list_recipes
from skipcore.consensus.runner import (
    evaluate_category,
    evaluate_category_layergroup,
)
from skipcore.engine import build_feature_extractor
from skipcore.utils import get_dataset_categories, load_yaml, save_json


DEFAULT_CONFIGS = {
    "mvtec": "configs/mvtec_default.yaml",
    "visa": "configs/visa_default.yaml",
    "realiad": "configs/realiad_default.yaml",
    "uni_medical": "configs/uni_medical_default.yaml",
    "btad": "configs/btad_default.yaml",
    "mpdd": "configs/mpdd_default.yaml",
}
VISA_DEFAULT_ROOT = "/media/jjack/Extreme SSD/dataset/visa/visa_mvtec"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ConsensusCore training-free UAD")
    p.add_argument("--dataset", default="mvtec", choices=["mvtec", "visa", "realiad", "uni_medical", "btad", "mpdd"])
    p.add_argument("--config", default=None, help="Base YAML (backbone/features/dataset)")
    p.add_argument("--feature", default="dinov2_vitb14", help="Feature extractor tag (informational)")
    p.add_argument("--category", default=None, help="Single category (default: all)")
    p.add_argument("--seed", type=int, default=0)

    # Recipe shortcut (V0-V7). Expands to option overrides; manual flags win.
    p.add_argument("--recipe", default=None, choices=list_recipes(),
                   help="Versioned recipe; expands into the options below")

    # Method selection: consensus (proposed) or single/full/random ablations.
    p.add_argument("--method", default="consensus", choices=["consensus", "single", "full", "random"])

    # Memory / consensus construction.
    p.add_argument("--memory_ratio", type=float, default=0.01)
    p.add_argument("--num_banks", type=int, default=5)
    p.add_argument("--bank_diversity", default="seed", choices=["seed", "ratio", "bootstrap"])
    p.add_argument("--ratios", "--memory_ratios", dest="ratios",
                   type=_parse_ratios, default=None,
                   help="Ratio list for --bank_diversity ratio, e.g. 0.005,0.01,0.02")

    # Distance fusion.
    p.add_argument("--fusion", default="median",
                   choices=["mean", "median", "quantile", "trimmed_mean", "max", "min"])
    p.add_argument("--quantile", type=float, default=0.75)
    p.add_argument("--trim_ratio", type=float, default=0.1)

    # Image-level aggregation.
    p.add_argument("--image_score", default="topmean", choices=["topmean"])
    p.add_argument("--topmean_ratio", type=float, default=0.01)

    # Anchor reliability (soft ConsensusCore: V3 / V4).
    p.add_argument("--use_anchor_reliability", action="store_true")
    p.add_argument("--reliability_type", default="stability",
                   choices=["stability", "oob"])
    p.add_argument("--reliability_lambda", type=float, default=0.03)
    p.add_argument("--stability_delta", default=None,
                   help="Euclidean delta for stability; 'auto' or a float")
    p.add_argument("--tau_mu", default="auto", help="OOB mu temperature ('auto' or float)")
    p.add_argument("--tau_sigma", default="auto", help="OOB sigma temperature ('auto' or float)")

    # Intrinsic prototype refinement (INP-inspired).
    p.add_argument("--use_intrinsic_proto", action="store_true")
    p.add_argument("--proto_select_quantile", type=float, default=0.7)
    p.add_argument("--num_prototypes", type=int, default=8)
    p.add_argument("--proto_fusion", default="add", choices=["add", "max"])
    p.add_argument("--proto_alpha", type=float, default=0.5)

    # Decoder-free soft projection (V8 / V9 / V10).
    p.add_argument("--use_soft_projection", action="store_true")
    p.add_argument("--softproj_k", type=int, default=5)
    p.add_argument("--softproj_tau", default="auto",
                   help="Softmax temperature ('auto' or float)")
    # (alpha) bank-vectorized soft projection (V28 layergroup speed-up).
    p.add_argument("--bank_vectorized", action="store_true",
                   help="Batch the B banks into one cdist (numerically "
                        "equivalent ~1e-6, much faster).")
    p.add_argument("--sp_query_chunk", type=int, default=0,
                   help="Chunk queries in bank-vectorized soft projection "
                        "to bound the [B,chunk,M] tensor (0 = no chunking).")
    p.add_argument("--softproj_alpha", type=float, default=0.5)
    p.add_argument("--softproj_fusion", default="direct",
                   choices=["direct", "gate_residual", "max"])
    p.add_argument("--entropy_lambda", type=float, default=0.0)
    # Multi-scale soft projection (V16-V19).
    p.add_argument("--softproj_k_list", type=_parse_int_list, default=None,
                   help="k neighborhood sizes, e.g. 3,5,7")
    p.add_argument("--softproj_tau_scales", type=_parse_ratios, default=None,
                   help="tau scale factors, e.g. 0.5,1.0,2.0")
    p.add_argument("--softproj_expert_fusion", default="median",
                   choices=["median", "quantile"])
    p.add_argument("--softproj_expert_quantile", type=float, default=0.75)

    # V9-aware memory selection (V20-V22).
    p.add_argument("--mem_select", default=None,
                   choices=["recgreedy", "recgreedy_diverse", "rec_pruned"])
    p.add_argument("--candidate_ratio", type=float, default=0.05)
    p.add_argument("--large_memory_ratio", type=float, default=0.05)
    p.add_argument("--recgreedy_sample_patches", type=int, default=20000)
    p.add_argument("--div_lambda", type=float, default=0.01)

    # Local geometry residual (V26 / V27).
    p.add_argument("--score_mode", default="softproj",
                   choices=["softproj", "local_pca", "local_maha"])
    p.add_argument("--pca_neighbors", type=int, default=10)
    p.add_argument("--pca_rank", type=int, default=2)
    p.add_argument("--maha_eps", type=float, default=1e-4)

    # V28 two-group layer-wise V9.
    p.add_argument("--v28_pool", default="quantile",
                   choices=["quantile", "mean", "max"])
    p.add_argument("--v28_q", type=float, default=0.5)
    p.add_argument("--v28_groupnorm", default="zscore",
                   choices=["zscore", "robust", "none", "perimage_robust"])
    p.add_argument("--v28_readout", default="topmean",
                   choices=["topmean", "max"],
                   help="Axis-3 image readout from the final patch map")
    p.add_argument("--v28", action="store_true",
                   help="Enable V28 two-group layer-wise path")
    p.add_argument("--v28_groups_json", default=None,
                   help='Group->layers JSON, e.g. '
                        "'{\"low\":[-9,-12],\"high\":[-1,-3,-6]}'")
    p.add_argument("--layer_combine", default="none",
                   choices=["none", "mean", "median", "q75", "trimmed_mean",
                            "max", "product", "rank"],
                   help="Axis-1 two-level combine: bank-consensus per layer, "
                        "then combine per-layer scores (overrides v28_pool)")

    # OOB-residual coreset pruning (rare vs isolated).
    p.add_argument("--oob_prune", action="store_true",
                   help="Prune isolated anchors by OOB soft-projection residual")
    p.add_argument("--oob_frac", type=float, default=0.0,
                   help="Quantile prune fraction (top-frac by OOB residual)")
    p.add_argument("--oob_agg", default="median",
                   choices=["median", "mean", "max"],
                   help="Aggregation of per-fold OOB residuals")
    p.add_argument("--oob_threshold", default="quantile",
                   choices=["quantile", "absolute"])
    p.add_argument("--oob_refill", default="none", choices=["none", "refill"],
                   help="Farthest-point coverage refill after pruning")
    p.add_argument("--oob_abs_c", type=float, default=2.0,
                   help="Absolute-threshold IQR multiplier (median + c*IQR)")

    # Heterogeneous expert ensembles (V11-V15).
    p.add_argument("--expert_recipe", default=None,
                   help="Expert combine rule (set by V11-V15 recipes)")
    p.add_argument("--qcurve_lambda1", type=float, default=0.5)
    p.add_argument("--qcurve_lambda2", type=float, default=0.0)
    p.add_argument("--expert_agreement_theta", type=float, default=0.90)
    p.add_argument("--expert_agreement_min_votes", type=int, default=2)
    p.add_argument("--expert_agreement_alpha", type=float, default=0.1)

    # Diagnostics.
    p.add_argument("--debug_save_components", action="store_true",
                   help="Save s_ext/s_int/s_final/mask maps per image")

    p.add_argument("--output", default=None, help="Output directory for results")
    return p.parse_args()


def _parse_ratios(text: str) -> List[float]:
    """Parse a comma- or space-separated ratio list into floats."""
    parts = str(text).replace(",", " ").split()
    return [float(x) for x in parts if x]


def _parse_int_list(text: str) -> List[int]:
    """Parse a comma- or space-separated list into ints."""
    parts = str(text).replace(",", " ").split()
    return [int(float(x)) for x in parts if x]


def _explicit_flags(argv: List[str]) -> set:
    """Names of options the user passed explicitly (so recipes never override)."""
    alias = {"--memory_ratios": "ratios"}
    flags = set()
    for tok in argv:
        if tok.startswith("--"):
            name = tok.split("=", 1)[0]
            flags.add(alias.get(name, name.lstrip("-")))
    return flags


def apply_recipe(args: argparse.Namespace, argv: List[str]) -> argparse.Namespace:
    """Expand ``--recipe`` into option overrides, preserving explicit CLI flags."""
    if not args.recipe:
        return args
    overrides = get_recipe(args.recipe)
    explicit = _explicit_flags(argv)
    for key, value in overrides.items():
        if key in explicit:
            continue  # user override wins
        setattr(args, key, value)
    return args


def _mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def main() -> None:
    import sys
    args = parse_args()
    args = apply_recipe(args, sys.argv[1:])
    config_path = args.config or DEFAULT_CONFIGS[args.dataset]
    cfg = load_yaml(config_path)

    cfg["dataset"]["name"] = args.dataset
    if args.dataset == "visa":
        cfg["dataset"]["root"] = cfg["dataset"].get("visa_root", VISA_DEFAULT_ROOT)

    # Recipe-driven backbone feature-space override (V23 / V25).
    backbone_override = getattr(args, "backbone_override", None)
    if backbone_override:
        cfg.setdefault("backbone", {})
        cfg["backbone"].update(backbone_override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # V28: build one frozen extractor per layer group (low / high).
    v28 = bool(getattr(args, "v28", False))
    group_extractors = {}
    if v28:
        groups = getattr(args, "v28_groups", None) or {"low": [-9, -12],
                                                        "high": [-1, -3, -6]}
        # CLI JSON override (Phase 2/3 arbitrary group configs without code edit).
        gj = getattr(args, "v28_groups_json", None)
        if gj:
            import json as _json
            groups = {k: [int(x) for x in v]
                      for k, v in _json.loads(gj).items()}
        for gname, layers in groups.items():
            gcfg = dict(cfg.get("backbone", {}))
            gcfg["layers"] = layers
            gcfg["layer_fusion"] = "concat"
            gx = build_feature_extractor(cfg["features"], backbone_cfg=gcfg)
            if hasattr(gx, "backbone"):
                gx.backbone.to(device)
            group_extractors[gname] = gx
        extractor = None
    else:
        # Build the frozen feature extractor once and reuse across categories.
        extractor = build_feature_extractor(cfg["features"],
                                            backbone_cfg=cfg.get("backbone", {}))
        if hasattr(extractor, "backbone"):
            extractor.backbone.to(device)

    categories = [args.category] if args.category else get_dataset_categories(args.dataset)

    # Recipe name (if given) defines the independent output folder; otherwise
    # fall back to a descriptive tag.
    if args.recipe:
        tag = args.recipe
    else:
        tag = args.method if args.method != "consensus" else f"consensus_{args.fusion}"
        if args.method == "consensus" and args.fusion == "quantile":
            tag = f"consensus_q{int(round(args.quantile * 100))}"
        if args.use_anchor_reliability:
            tag += f"_{args.reliability_type}"
        if args.use_intrinsic_proto:
            tag += "_proto"

    print(f"\n{'='*64}")
    print(f"  ConsensusCore | dataset={args.dataset} | recipe={args.recipe} | tag={tag}")
    print(f"  method={args.method} banks={args.num_banks} diversity={args.bank_diversity} "
          f"fusion={args.fusion} ratio={args.memory_ratio}")
    print(f"  reliability={args.use_anchor_reliability}({args.reliability_type}) "
          f"proto={args.use_intrinsic_proto}({args.proto_fusion})")
    print(f"{'='*64}\n")

    results: List[Dict[str, Any]] = []
    for cat in categories:
        if v28:
            res = evaluate_category_layergroup(cfg, cat, args, device,
                                               group_extractors)
        else:
            res = evaluate_category(cfg, cat, args, device, extractor)
        results.append(res)
        print(
            f"[{cat:>12}] "
            f"I-AUROC={res.get('image_auroc', 0):.4f} "
            f"I-AP={res.get('image_ap', 0):.4f} "
            f"I-F1={res.get('image_f1_max', 0):.4f} | "
            f"P-AUROC={res.get('pixel_auroc', 0):.4f} "
            f"P-AP={res.get('pixel_ap', 0):.4f} "
            f"P-F1={res.get('pixel_f1_max', 0):.4f} "
            f"AUPRO={res.get('pixel_aupro', 0):.4f} "
            f"bank={res.get('bank_size', 0)}"
        )

    summary = {
        "image_auroc": _mean([r.get("image_auroc") for r in results]),
        "image_ap": _mean([r.get("image_ap") for r in results]),
        "image_f1_max": _mean([r.get("image_f1_max") for r in results]),
        "pixel_auroc": _mean([r.get("pixel_auroc") for r in results]),
        "pixel_ap": _mean([r.get("pixel_ap") for r in results]),
        "pixel_f1_max": _mean([r.get("pixel_f1_max") for r in results]),
        "pixel_aupro": _mean([r.get("pixel_aupro") for r in results]),
        "pixel_pro": _mean([r.get("pixel_pro") for r in results]),
    }

    print(f"\n{'-'*72}")
    print(
        f"  MEAN | I-AUROC={summary['image_auroc']:.4f} "
        f"I-AP={summary['image_ap']:.4f} "
        f"I-F1={summary['image_f1_max']:.4f} | "
        f"P-AUROC={summary['pixel_auroc']:.4f} "
        f"P-AP={summary['pixel_ap']:.4f} "
        f"P-F1={summary['pixel_f1_max']:.4f} "
        f"AUPRO={summary['pixel_aupro']:.4f}"
    )
    print(f"{'-'*72}\n")

    out_dir = Path(args.output) if args.output else Path("runs_consensuscore") / args.dataset / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config_path,
        "args": vars(args),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "per_category": results,
    }
    save_json(out_dir / f"results_seed{args.seed}.json", payload)

    # Per-category CSV for downstream win/loss analysis.
    csv_fields = [
        "category", "bank_size",
        "image_auroc", "image_ap", "image_f1_max",
        "pixel_auroc", "pixel_ap", "pixel_f1_max",
        "pixel_aupro", "pixel_pro",
    ]
    csv_path = out_dir / f"per_category_seed{args.seed}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        mean_row = {"category": "MEAN", "bank_size": ""}
        mean_row.update({k: summary.get(k) for k in csv_fields if k in summary})
        writer.writerow(mean_row)

    print(f"Saved results to {out_dir / f'results_seed{args.seed}.json'}")
    print(f"Saved per-category CSV to {csv_path}")


if __name__ == "__main__":
    main()
