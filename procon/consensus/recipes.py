"""Versioned ConsensusCore recipes (spec V0-V7).

Each recipe is a named bundle of CLI option overrides so that a single
``--recipe`` flag expands into the right configuration. Manual CLI flags always
win over the recipe (handled in ``run_procon.py`` by only applying recipe
values to options the user did not pass explicitly).

国밥 philosophy:
    V0 one pot · V1 multi-pot consensus · V2 small+large pots ·
    V3 ingredient reliability · V4 reliability distilled from partial pots ·
    V5 in-bowl taste calibration · V6 reliable pots + bowl calibration ·
    V7 full special 国밥.
"""
from __future__ import annotations

from typing import Any, Dict

# Shared defaults that most consensus recipes share.
_BANKS = {"method": "consensus", "num_banks": 5, "memory_ratio": 0.01,
          "bank_diversity": "seed"}
_RATIO_ENS = {"method": "consensus", "num_banks": 3, "bank_diversity": "ratio",
              "ratios": [0.005, 0.01, 0.02]}
_PROTO = {"use_intrinsic_proto": True, "proto_select_quantile": 0.7,
          "num_prototypes": 8, "proto_alpha": 0.5}


RECIPES: Dict[str, Dict[str, Any]] = {
    # V0: single PatchCore baseline.
    "v0_single": {"method": "single", "memory_ratio": 0.01, "num_banks": 1},

    # V1: ConsensusCore (median / Q75 fusion).
    "v1_median": {**_BANKS, "fusion": "median"},
    "v1_q75": {**_BANKS, "fusion": "quantile", "quantile": 0.75},

    # V2: memory-ratio ensemble.
    "v2_ratio_median": {**_RATIO_ENS, "fusion": "median"},
    "v2_ratio_q75": {**_RATIO_ENS, "fusion": "quantile", "quantile": 0.75},

    # V3: soft consensus with stability reliability.
    "v3_soft_median": {**_BANKS, "fusion": "median",
                       "use_anchor_reliability": True,
                       "reliability_type": "stability",
                       "reliability_lambda": 0.03},
    "v3_soft_q75": {**_BANKS, "fusion": "quantile", "quantile": 0.75,
                    "use_anchor_reliability": True,
                    "reliability_type": "stability",
                    "reliability_lambda": 0.03},

    # V4: soft consensus with MeDS-inspired OOB reliability.
    "v4_oob_median": {**_BANKS, "fusion": "median",
                      "use_anchor_reliability": True,
                      "reliability_type": "oob",
                      "reliability_lambda": 0.03,
                      "tau_mu": "auto", "tau_sigma": "auto"},
    "v4_oob_q75": {**_BANKS, "fusion": "quantile", "quantile": 0.75,
                   "use_anchor_reliability": True,
                   "reliability_type": "oob",
                   "reliability_lambda": 0.03,
                   "tau_mu": "auto", "tau_sigma": "auto"},

    # V5: INP-style internal prototype refinement on top of V1 median.
    "v5_proto_add": {**_BANKS, "fusion": "median", **_PROTO,
                     "proto_fusion": "add"},
    "v5_proto_max": {**_BANKS, "fusion": "median", **_PROTO,
                     "proto_fusion": "max"},

    # V6: soft consensus + internal prototype.
    "v6_soft_proto_add": {**_BANKS, "fusion": "median",
                          "use_anchor_reliability": True,
                          "reliability_type": "stability",
                          "reliability_lambda": 0.03,
                          **_PROTO, "proto_fusion": "add"},
    "v6_oob_proto_add": {**_BANKS, "fusion": "median",
                         "use_anchor_reliability": True,
                         "reliability_type": "oob",
                         "reliability_lambda": 0.03,
                         "tau_mu": "auto", "tau_sigma": "auto",
                         **_PROTO, "proto_fusion": "add"},

    # V7: full special 国밥 (OOB soft consensus + prototype).
    "v7_full": {**_BANKS, "fusion": "quantile", "quantile": 0.75,
                "use_anchor_reliability": True,
                "reliability_type": "oob",
                "reliability_lambda": 0.03,
                "tau_mu": "auto", "tau_sigma": "auto",
                **_PROTO, "proto_fusion": "add"},

    # V8: single-bank decoder-free soft projection (local kNN reconstruction).
    "v8_softproj_single": {"method": "single", "memory_ratio": 0.01,
                           "num_banks": 1, "fusion": "median",
                           "use_soft_projection": True, "softproj_k": 5,
                           "softproj_tau": "auto", "softproj_fusion": "direct"},

    # V9: consensus soft projection.
    "v9_softproj_median": {**_BANKS, "fusion": "median",
                           "use_soft_projection": True, "softproj_k": 5,
                           "softproj_tau": "auto", "softproj_fusion": "direct"},
    "v9_softproj_q75": {**_BANKS, "fusion": "quantile", "quantile": 0.75,
                        "use_soft_projection": True, "softproj_k": 5,
                        "softproj_tau": "auto", "softproj_fusion": "direct"},
    # V9 + entropy penalty.
    "v9_softproj_entropy_median": {**_BANKS, "fusion": "median",
                                   "use_soft_projection": True, "softproj_k": 5,
                                   "softproj_tau": "auto",
                                   "softproj_fusion": "direct",
                                   "entropy_lambda": 0.01},

    # V10: gated soft projection -- keep V1 NN base, residual-only boost.
    "v10_softproj_gate_median": {**_BANKS, "fusion": "median",
                                 "use_soft_projection": True, "softproj_k": 5,
                                 "softproj_tau": "auto", "softproj_alpha": 0.5,
                                 "softproj_fusion": "gate_residual"},
    "v10_softproj_gate_q75": {**_BANKS, "fusion": "quantile", "quantile": 0.75,
                              "use_soft_projection": True, "softproj_k": 5,
                              "softproj_tau": "auto", "softproj_alpha": 0.5,
                              "softproj_fusion": "gate_residual"},
    "v10_softproj_max_median": {**_BANKS, "fusion": "median",
                                "use_soft_projection": True, "softproj_k": 5,
                                "softproj_tau": "auto", "softproj_alpha": 0.5,
                                "softproj_fusion": "max"},
}

# ---- Heterogeneous expert ensembles (V11-V15) ----
# All set ``expert_recipe`` so the runner takes the expert path. They reuse the
# consensus banks (B=5) and the soft-projection settings (k=5, tau=auto).
_EXPERT = {**_BANKS, "softproj_k": 5, "softproj_tau": "auto"}


def _expert(name: str, **extra: Any) -> Dict[str, Any]:
    return {**_EXPERT, "expert_recipe": name, **extra}


RECIPES.update({
    # V11 / V12: quantile-curve experts (NN / soft).
    "v11_nn_qcurve": _expert("v11_nn_qcurve", qcurve_lambda1=0.5,
                             qcurve_lambda2=0.0),
    "v12_soft_qcurve": _expert("v12_soft_qcurve", qcurve_lambda1=0.5,
                               qcurve_lambda2=0.0),

    # V13: per-image rank ensembles + ablations.
    "v13_rank_ensemble": _expert("v13_rank_ensemble"),
    "v13_rank_nn_only": _expert("v13_rank_nn_only"),
    "v13_rank_soft_only": _expert("v13_rank_soft_only"),
    "v13_rank_nn_soft50": _expert("v13_rank_nn_soft50"),

    # V14: expert-agreement boost (heuristic / diagnostic).
    "v14_expert_agreement": _expert("v14_expert_agreement",
                                    expert_agreement_theta=0.90,
                                    expert_agreement_min_votes=2,
                                    expert_agreement_alpha=0.1),

    # V15: simple quantile interpolation (cleanest alternative).
    "v15_nn_qinterp_025": _expert("v15_nn_qinterp_025"),
    "v15_nn_qinterp_050": _expert("v15_nn_qinterp_050"),
    "v15_nn_qinterp_075": _expert("v15_nn_qinterp_075"),
    "v15_soft_qinterp_025": _expert("v15_soft_qinterp_025"),
    "v15_soft_qinterp_050": _expert("v15_soft_qinterp_050"),
    "v15_soft_qinterp_075": _expert("v15_soft_qinterp_075"),

    # ---- Multi-scale soft-projection family (V16-V19) ----
    # V16: multi-k soft projection (k in {3,5,7}).
    "v16_multik_softproj_median": _expert("v16_multik_softproj_median",
                                          softproj_k_list=[3, 5, 7]),
    "v16_multik_softproj_q75": _expert("v16_multik_softproj_q75",
                                       softproj_k_list=[3, 5, 7],
                                       softproj_expert_quantile=0.75),
    # V17: multi-temperature soft projection (tau scales {0.5,1,2}).
    "v17_multitau_softproj_median": _expert("v17_multitau_softproj_median",
                                            softproj_tau_scales=[0.5, 1.0, 2.0]),
    "v17_multitau_softproj_q75": _expert("v17_multitau_softproj_q75",
                                         softproj_tau_scales=[0.5, 1.0, 2.0],
                                         softproj_expert_quantile=0.75),
    # V18: multi-k multi-temperature (9 experts).
    "v18_multiscale_softproj_median": _expert("v18_multiscale_softproj_median",
                                              softproj_k_list=[3, 5, 7],
                                              softproj_tau_scales=[0.5, 1.0, 2.0]),
    "v18_multiscale_softproj_q75": _expert("v18_multiscale_softproj_q75",
                                           softproj_k_list=[3, 5, 7],
                                           softproj_tau_scales=[0.5, 1.0, 2.0],
                                           softproj_expert_quantile=0.75),
    # V19: soft-projection quantile family / direct q65.
    "v19_softproj_qfamily_median": _expert("v19_softproj_qfamily_median"),
    "v19_softproj_q65": _expert("v19_softproj_q65"),
})

# ---- V9-aware memory selection (V20-V22) ----
# Reuse V9 scoring (soft projection median), only the memory builder changes.
_V9SCORE = {**_BANKS, "fusion": "median", "use_soft_projection": True,
            "softproj_k": 5, "softproj_tau": "auto", "softproj_fusion": "direct"}

RECIPES.update({
    "v20_recgreedy_softproj": {**_V9SCORE, "mem_select": "recgreedy",
                               "candidate_ratio": 0.05,
                               "recgreedy_sample_patches": 20000},
    "v21_recgreedy_diverse_softproj": {**_V9SCORE,
                                       "mem_select": "recgreedy_diverse",
                                       "candidate_ratio": 0.05,
                                       "div_lambda": 0.01},
    "v22_rec_pruned_kcenter": {**_V9SCORE, "mem_select": "rec_pruned",
                               "large_memory_ratio": 0.05},

    # ---- Local geometry residual (V26 / V27) ----
    "v26_local_pca_softproj": {**_BANKS, "fusion": "median",
                               "score_mode": "local_pca",
                               "pca_neighbors": 10, "pca_rank": 2},
    "v27_local_mahalanobis": {**_BANKS, "fusion": "median",
                              "score_mode": "local_maha",
                              "pca_neighbors": 10, "pca_rank": 4,
                              "maha_eps": 1e-4},
})

# ---- Layer-wise V9 (V23 / V25): backbone feature-space overrides ----
# These set ``backbone_override`` consumed by run_procon.py to rebuild the
# frozen extractor with different layers / per-layer normalization. Scoring stays
# V9 (soft projection median).
_DINO_LAYERS = {-3: "l3", -6: "l6", -9: "l9", -12: "l12"}

RECIPES.update({
    "v23_layer_l3": {**_V9SCORE,
                     "backbone_override": {"layers": [-3],
                                           "layer_fusion": "concat"}},
    "v23_layer_l6": {**_V9SCORE,
                     "backbone_override": {"layers": [-6],
                                           "layer_fusion": "concat"}},
    "v23_layer_l9": {**_V9SCORE,
                     "backbone_override": {"layers": [-9],
                                           "layer_fusion": "concat"}},
    "v23_layer_l12": {**_V9SCORE,
                      "backbone_override": {"layers": [-12],
                                            "layer_fusion": "concat"}},
    "v25_layer_l2norm_concat": {**_V9SCORE,
                                "backbone_override": {"layer_norm_mode": "l2"}},
    "v25_layer_standardized_concat": {
        **_V9SCORE,
        "backbone_override": {"layer_norm_mode": "layernorm"}},
})

# ---- V28: two-group layer-wise V9 (low/high semantic consensus) ----
# low group = shallow blocks (texture/edge), high group = deep blocks (structure).
# Each group has its own extractor + banks; group-normalized residuals are pooled
# with Q_q consensus (mean/max are ablations).
_V28_GROUPS = {"low": [-9, -12], "high": [-1, -3, -6]}
_V28 = {"method": "consensus", "num_banks": 5, "memory_ratio": 0.01,
        "v28": True, "v28_groups": _V28_GROUPS, "softproj_k": 5,
        "softproj_tau": "auto"}

RECIPES.update({
    "v28_layergroup_consensus": {**_V28, "v28_pool": "quantile", "v28_q": 0.5,
                                 "v28_groupnorm": "zscore"},
    # Ablation A: pooling rule.
    "v28_group_mean": {**_V28, "v28_pool": "mean", "v28_groupnorm": "zscore"},
    "v28_group_max": {**_V28, "v28_pool": "max", "v28_groupnorm": "zscore"},
    # Ablation B: normalization.
    "v28_no_groupnorm": {**_V28, "v28_pool": "quantile", "v28_q": 0.5,
                         "v28_groupnorm": "none"},
    # Ablation C: grouping itself (one group = all layers).
    "v28_singlegroup_allsum": {**_V28, "v28_pool": "quantile", "v28_q": 0.5,
                               "v28_groupnorm": "zscore",
                               "v28_groups": {"all": [-1, -3, -6, -9, -12]}},
})


# ---- Layer study (Phase 1-3): single-layer profiling + group configs ----
# Phase 1: expanded single-layer pool. Each uses V9 scoring on one DINOv2 layer.
_LAYERSTUDY_SINGLE = [-1, -2, -3, -4, -6, -8, -9, -10, -12]
for _idx in _LAYERSTUDY_SINGLE:
    RECIPES[f"ls_single_l{abs(_idx)}"] = {
        **_V9SCORE,
        "backbone_override": {"layers": [_idx], "layer_fusion": "concat"},
    }


def _ls_group(name, groups, pool="mean", groupnorm="zscore", q=0.5):
    return {"method": "consensus", "num_banks": 5, "memory_ratio": 0.01,
            "v28": True, "v28_groups": groups, "softproj_k": 5,
            "softproj_tau": "auto", "v28_pool": pool, "v28_groupnorm": groupnorm,
            "v28_q": q}


RECIPES.update({
    # Phase 3 boundary sweep (group_mean consensus, train-normal zscore norm).
    # A = current V28 mechanical-half baseline.
    "ls_A_half": _ls_group("A", {"low": [-9, -12], "high": [-1, -3, -6]}),
    # D-series: slide the low/high boundary across the depth axis.
    "ls_D1": _ls_group("D1", {"low": [-12], "high": [-1, -3, -6, -9]}),
    "ls_D3": _ls_group("D3", {"low": [-6, -9, -12], "high": [-1, -3]}),
    "ls_D4": _ls_group("D4", {"low": [-3, -6, -9, -12], "high": [-1]}),
    # E: three-group shallow/mid/deep split.
    "ls_E_three": _ls_group("E", {"deep": [-1, -3], "mid": [-6],
                                  "shallow": [-9, -12]}),
})

# ---- Phase 2: consensus-operation 3-axis sweep (axis-1/2/3) ----
# L-12 confirmed toxic -> removed. Healthy layer pool is FIXED here:
_P2_POOL = [-3, -4, -6, -8, -9]
_P2_GROUPS = {f"l{abs(_i)}": [_i] for _i in _P2_POOL}
# baseline_A: standard V9 (concat the same 5-layer pool -> single coreset).
_P2_BASELINE_A = {**_V9SCORE,
                  "backbone_override": {"layers": list(_P2_POOL),
                                        "layer_fusion": "concat"}}

# Champion-pool concat baseline (layer-independence ablation): the SAME champion
# pool {-3,-6,-8,-9} concatenated into ONE feature space -> ONE coreset family
# (B=5 seed banks, soft-projection median), identical to the champion in every
# way EXCEPT that layers are fused before the coreset instead of kept as
# independent per-layer banks. Run on full MVTec at 1% with readout topmean
# 0.005 (CLI --topmean_ratio) to slot directly into the per-layer table.
_P3_POOL_CONCAT = [-3, -6, -8, -9]
_P3_CONCAT_3689 = {**_V9SCORE,
                   "backbone_override": {"layers": list(_P3_POOL_CONCAT),
                                         "layer_fusion": "concat"}}


def _p2(combine="mean", groupnorm="none", readout="topmean", topmean=0.01):
    """One axis combination over the fixed healthy pool (independent banks)."""
    return {"method": "consensus", "num_banks": 5, "memory_ratio": 0.01,
            "v28": True, "v28_groups": dict(_P2_GROUPS), "softproj_k": 5,
            "softproj_tau": "auto", "v28_groupnorm": groupnorm,
            "layer_combine": combine, "v28_readout": readout,
            "topmean_ratio": topmean}


RECIPES.update({
    # Fair baselines.
    "ls_baseline_A": _P2_BASELINE_A,             # concat 5-layer -> single bank
    "ls_baseline_B": _p2("mean", "none", "topmean", 0.01),  # indep banks, mean
    # Layer-independence ablation: champion pool {-3,-6,-8,-9} concatenated.
    "concat4_3689": _P3_CONCAT_3689,    # --- Axis 1: aggregation operator (norm=none, readout=topmean_0.01) ---    "ls_agg_median": _p2("median", "none", "topmean", 0.01),
    "ls_agg_q75": _p2("q75", "none", "topmean", 0.01),
    "ls_agg_trimmed": _p2("trimmed_mean", "none", "topmean", 0.01),
    "ls_agg_max": _p2("max", "none", "topmean", 0.01),
    # --- Axis 2: normalization (agg=mean, readout=topmean_0.01) ---
    "ls_norm_none": _p2("mean", "none", "topmean", 0.01),
    "ls_norm_perimg": _p2("mean", "perimage_robust", "topmean", 0.01),
    "ls_norm_trainstat": _p2("mean", "zscore", "topmean", 0.01),
    # --- Axis 3: readout (agg=mean, norm=none) ---
    "ls_read_tm005": _p2("mean", "none", "topmean", 0.005),
    "ls_read_tm01": _p2("mean", "none", "topmean", 0.01),
    "ls_read_tm02": _p2("mean", "none", "topmean", 0.02),
    "ls_read_max": _p2("mean", "none", "max", 0.01),
})

# Full 3-axis cross grid lc_{AGG}__{NORM}__{RO} (5 x 3 x 4 = 60 recipes).
# Naming maps directly to the run plan in the Phase 2 instruction.
_LC_AGG = {"mean": "mean", "median": "median", "q75": "q75",
           "trimmed_mean": "trimmed_mean", "max": "max"}
_LC_NORM = {"none": "none", "robust_perimg": "perimage_robust",
            "train_stat": "zscore"}
_LC_RO = {"topmean_0.005": ("topmean", 0.005),
          "topmean_0.01": ("topmean", 0.01),
          "topmean_0.02": ("topmean", 0.02),
          "max": ("max", 0.01)}
for _agg, _comb in _LC_AGG.items():
    for _norm, _gn in _LC_NORM.items():
        for _ro, (_rdout, _tm) in _LC_RO.items():
            RECIPES[f"lc_{_agg}__{_norm}__{_ro}"] = _p2(_comb, _gn, _rdout, _tm)

# ---- Phase 3: layer-pool pruning (cooking method FIXED) ----
# Phase 2 winner: aggregation=mean, normalization=none, readout=topmean_0.005.
# Keep that frozen; vary ONLY the layer pool. Each layer keeps its own bank.
_P3_FIXED = dict(combine="mean", groupnorm="none", readout="topmean",
                 topmean=0.005)


def _p3_pool(pool):
    groups = {f"l{abs(_i)}": [_i] for _i in pool}
    r = _p2(_P3_FIXED["combine"], _P3_FIXED["groupnorm"],
            _P3_FIXED["readout"], _P3_FIXED["topmean"])
    r["v28_groups"] = groups
    return r


# Unique pools: full {-3,-4,-6,-8,-9} + all five one-drop variants
# (the spec's 4-layer pools coincide with the one-drop set).
_P3_POOLS = {
    "p3_full_34689": [-3, -4, -6, -8, -9],     # full healthy pool
    "p3_drop3_4689": [-4, -6, -8, -9],         # P \ {-3}
    "p3_drop4_3689": [-3, -6, -8, -9],         # P \ {-4}
    "p3_drop6_3489": [-3, -4, -8, -9],         # P \ {-6}
    "p3_drop8_3469": [-3, -4, -6, -9],         # P \ {-8}  (mandatory check)
    "p3_drop9_3468": [-3, -4, -6, -8],         # P \ {-9}
}
for _name, _pool in _P3_POOLS.items():
    RECIPES[_name] = _p3_pool(_pool)

# Phase 3b: deep layers -3,-4 individually hurt P-AP, so test dropping BOTH and
# pushing further toward the mid-depth core {-8,-9}.
_P3B_POOLS = {
    "p3_core_689": [-6, -8, -9],       # drop both deep semantic layers
    "p3_core_89": [-8, -9],            # mid-depth core only
    "p3_core_3689_ref": [-3, -6, -8, -9],  # current best (re-check, exists too)
    "p3_core_4689_ref": [-4, -6, -8, -9],  # second best (re-check)
    "p3_core_369": [-3, -6, -9],       # alt: skip -8
    "p3_core_69": [-6, -9],            # broader-spaced pair
}
for _name, _pool in _P3B_POOLS.items():
    RECIPES[_name] = _p3_pool(_pool)


# ---- Coreset-budget sweep on the champion pool {-3,-6,-8,-9} ----
# Same cooking (mean+none+topmean_0.005); vary only the per-layer memory ratio.
# Paired V9 references at the same ratio for a fair comparison.
for _r, _tag in ((0.05, "r05"), (0.10, "r10")):
    _champ = _p3_pool([-3, -6, -8, -9])
    _champ["memory_ratio"] = _r
    RECIPES[f"p3_3689_{_tag}"] = _champ
    RECIPES[f"v9_softproj_median_{_tag}"] = {**_V9SCORE, "memory_ratio": _r}


# ---- OOB-residual coreset pruning (rare vs isolated) ----
# Two frameworks: FW_A = V9 default (concat -1,-3,-6,-9,-12, single coreset);
# FW_B = layer-consensus survivor (pool {-3,-6,-8,-9}, mean+none+topmean_0.005).
# OOB pruning removes ISOLATED anchors (high OOB soft-projection residual) while
# keeping RARE ones (on-manifold). Contrasted against v22 density pruning.
_FWA_BASE = {**_V9SCORE}                              # concat, B=5, topmean ref
_FWB_BASE = _p3_pool([-3, -6, -8, -9])               # layer-consensus survivor


def _oob(base, frac=0.0, agg="median", threshold="quantile", refill="none",
         abs_c=2.0):
    r = dict(base)
    r.update({"oob_prune": True, "oob_frac": frac, "oob_agg": agg,
              "oob_threshold": threshold, "oob_refill": refill,
              "oob_abs_c": abs_c})
    return r


# References (no pruning).
RECIPES["fwa_ref"] = dict(_FWA_BASE)
RECIPES["fwb_ref"] = dict(_FWB_BASE)
# v22 density-prune baseline (FW_A; expected to collapse transistor).
RECIPES["fwa_v22_density"] = {**_FWA_BASE, "mem_select": "rec_pruned",
                              "large_memory_ratio": 0.05}

# Screening grid (bottom-3 first): quantile fracs x agg, plus absolute + refill.
for _fw, _base in (("fwa", _FWA_BASE), ("fwb", _FWB_BASE)):
    # Axis P1 x P2 (quantile, refill=none).
    for _frac in (5, 10, 20, 30):
        for _agg in ("median", "max"):
            RECIPES[f"{_fw}_oob_p{_frac}_{_agg}_quant_none"] = _oob(
                _base, frac=_frac / 100.0, agg=_agg, threshold="quantile")
    # Axis P3 absolute thresholds (median agg).
    RECIPES[f"{_fw}_oob_abs2_median_none"] = _oob(
        _base, agg="median", threshold="absolute", abs_c=2.0)
    RECIPES[f"{_fw}_oob_abs3_median_none"] = _oob(
        _base, agg="median", threshold="absolute", abs_c=3.0)
    # Axis P4 refill probe (drop isolated, then re-cover).
    RECIPES[f"{_fw}_oob_p10_median_quant_refill"] = _oob(
        _base, frac=0.10, agg="median", threshold="quantile", refill="refill")
    RECIPES[f"{_fw}_oob_p20_median_quant_refill"] = _oob(
        _base, frac=0.20, agg="median", threshold="quantile", refill="refill")




def list_recipes() -> list:
    return list(RECIPES.keys())


def get_recipe(name: str) -> Dict[str, Any]:
    if name not in RECIPES:
        raise KeyError(
            f"Unknown recipe '{name}'. Available: {', '.join(RECIPES.keys())}"
        )
    return dict(RECIPES[name])
