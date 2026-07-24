"""Benchmark utilities for running full dataset evaluations."""
from __future__ import annotations

import csv
import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from procon.utils.io import save_json, save_yaml


# ==================== DATASET CATEGORIES ====================
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]

REALIAD_CATEGORIES = [
    "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser",
    "fire_hood", "mint", "mounts", "pcb", "phone_battery", "plastic_nut",
    "plastic_plug", "porcelain_doll", "regulator", "rolled_strip_base",
    "sim_card_set", "switch", "tape", "terminalblock", "toothbrush", "toy",
    "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill",
    "wooden_beads", "woodstick", "zipper",
]

# Uni-Medical (BMAD): 3 subsets carry pixel masks (brain, liver, retina_resc);
# the image-only subsets can be appended when only image-level metrics matter.
UNI_MEDICAL_CATEGORIES = [
    "brain", "liver", "retina_resc",
]

# BTAD: three product categories.
BTAD_CATEGORIES = ["01", "02", "03"]

# MPDD: six metal-part categories (MVTec-format).
MPDD_CATEGORIES = [
    "bracket_black", "bracket_brown", "bracket_white",
    "connector", "metal_plate", "tubes",
]


def get_dataset_categories(dataset_name: str) -> List[str]:
    """Get categories for a dataset."""
    name = dataset_name.lower()
    if "realiad" in name or "real_iad" in name or "real-iad" in name:
        return REALIAD_CATEGORIES
    elif "uni_medical" in name or "uni-medical" in name or "bmad" in name:
        return UNI_MEDICAL_CATEGORIES
    elif "btad" in name:
        return BTAD_CATEGORIES
    elif "mpdd" in name:
        return MPDD_CATEGORIES
    elif "visa" in name:
        return VISA_CATEGORIES
    elif "mvtec" in name:
        return MVTEC_CATEGORIES
    return []


# ==================== RUN DIRECTORY ====================
def build_run_dir(
    dataset: str,
    category: str,
    exp_name: str,
    seed: int,
    base_dir: str = "runs",
) -> Path:
    """Build standard run directory path."""
    return Path(base_dir) / dataset / category / exp_name / f"seed{seed}"


def build_exp_name(
    layer_name: str = "multilayer",
    coreset_pct: float = 0.01,
    bank_size: int = 16,
) -> str:
    """Build experiment name from parameters."""
    return f"{layer_name}_cs{int(coreset_pct * 100)}_bank{bank_size}"


# ==================== RESULT STREAMING ====================
# Default CSV fields for benchmark results
DEFAULT_CSV_FIELDS = [
    "exp_name", "routing_agg", "dataset", "category", "seed",
    "coreset_pct", "layers", "bank_size",
    "image_auroc", "image_ap", "image_f1_max",
    "i_auroc_base", "i_auroc_struct_only", "i_auroc_hybrid",
    "pixel_auroc", "pixel_ap", "pixel_f1_max", "pixel_aupro", "pixel_pro",
    "routing_acc",
    "extract_ms", "knn_ms", "postprocess_ms", "total_ms", "fps",
    "bank_memory_mb", "gpu_peak_mb",
]


@dataclass
class ResultStreamer:
    """Stream results to CSV and JSONL files as they are computed."""
    
    output_dir: Path
    csv_fields: List[str] = field(default_factory=lambda: DEFAULT_CSV_FIELDS.copy())
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.output_dir / "all_results.csv"
        self.jsonl_path = self.output_dir / "all_results.jsonl"
        
        self._csv_file = self.csv_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, 
            fieldnames=self.csv_fields, 
            extrasaction="ignore"
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()
        
        self._jsonl_file = self.jsonl_path.open("w")
        self._results: List[Dict[str, Any]] = []
    
    def write(self, row: Dict[str, Any]) -> None:
        """Write a single result row."""
        import json
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        self._jsonl_file.write(json.dumps(row) + "\n")
        self._jsonl_file.flush()
        self._results.append(row)
    
    def write_mean_row(self, row: Dict[str, Any]) -> None:
        """Write a mean row (usually for summary)."""
        self.write(row)
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all collected results."""
        return self._results
    
    def close(self) -> None:
        """Close all file handles."""
        self._csv_file.close()
        self._jsonl_file.close()


# ==================== SUMMARY GENERATION ====================
def compute_dataset_summary(
    results: List[Dict[str, Any]],
    dataset: str,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute mean and std for dataset results."""
    import numpy as np
    
    if metrics is None:
        metrics = [
            "image_auroc", "image_ap", "image_f1_max",
            "pixel_auroc", "pixel_ap", "pixel_f1_max", "pixel_aupro", "pixel_pro",
            "routing_acc", "fps",
        ]
    
    dataset_results = [r for r in results if r.get("dataset") == dataset]
    if not dataset_results:
        return {}
    
    summary = {}
    for key in metrics:
        values = [r.get(key) for r in dataset_results if r.get(key) is not None]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    
    return summary


def save_category_metrics_table(
    output_dir: Path,
    results: List[Dict[str, Any]],
    dataset: str,
) -> None:
    """Save per-category metrics table with a mean row."""
    output_dir = Path(output_dir)
    rows = [r for r in results if r.get("dataset") == dataset]
    if not rows:
        return

    fields = [
        "dataset", "method", "category",
        "image_auroc", "image_ap", "image_f1_max",
        "pixel_auroc", "pixel_ap", "pixel_f1_max", "pixel_aupro",
    ]
    method = rows[0].get("exp_name", "")
    out_path = output_dir / f"category_results_{dataset}.csv"

    def _mean(key: str) -> Optional[float]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        import numpy as np
        return float(np.mean(vals))

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "dataset": dataset,
                "method": method,
                "category": r.get("category"),
                "image_auroc": r.get("image_auroc"),
                "image_ap": r.get("image_ap"),
                "image_f1_max": r.get("image_f1_max"),
                "pixel_auroc": r.get("pixel_auroc"),
                "pixel_ap": r.get("pixel_ap"),
                "pixel_f1_max": r.get("pixel_f1_max"),
                "pixel_aupro": r.get("pixel_aupro"),
            })
        writer.writerow({
            "dataset": dataset,
            "method": method,
            "category": "mean",
            "image_auroc": _mean("image_auroc"),
            "image_ap": _mean("image_ap"),
            "image_f1_max": _mean("image_f1_max"),
            "pixel_auroc": _mean("pixel_auroc"),
            "pixel_ap": _mean("pixel_ap"),
            "pixel_f1_max": _mean("pixel_f1_max"),
            "pixel_aupro": _mean("pixel_aupro"),
        })


def save_benchmark_results(
    output_dir: Path,
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    datasets: List[str],
) -> None:
    """Save all benchmark results and summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw JSON
    save_json(output_dir / "results_raw.json", results)
    
    # Save config
    save_yaml(output_dir / "config.yaml", config)
    
    # Save per-dataset summaries
    for dataset in datasets:
        summary = compute_dataset_summary(results, dataset)
        if summary:
            save_json(output_dir / f"summary_{dataset}.json", summary)
            save_category_metrics_table(output_dir, results, dataset)
            print(f"\n📊 {dataset.upper()} Summary:")
            print(f"   Image AUROC: {summary.get('image_auroc_mean', 0)*100:.2f}%")
            print(f"   Image AP: {summary.get('image_ap_mean', 0)*100:.2f}%")
            print(f"   Image F1-max: {summary.get('image_f1_max_mean', 0)*100:.2f}%")
            print(f"   Pixel AUROC: {summary.get('pixel_auroc_mean', 0)*100:.2f}%")
            print(f"   Pixel AP: {summary.get('pixel_ap_mean', 0)*100:.2f}%")
            print(f"   Pixel F1-max: {summary.get('pixel_f1_max_mean', 0)*100:.2f}%")
            print(f"   Pixel AUPRO: {summary.get('pixel_aupro_mean', 0)*100:.2f}%")
            if "fps_mean" in summary:
                print(f"   FPS: {summary.get('fps_mean', 0):.1f}")


# ==================== CONFIG HELPERS ====================
def update_config_for_experiment(
    base_cfg: Dict[str, Any],
    dataset: str,
    category: str,
    exp_name: str,
    seed: int,
    coreset_pct: Optional[float] = None,
    layers: Optional[List[int]] = None,
    bank_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Create updated config for a specific experiment."""
    cfg = copy.deepcopy(base_cfg)
    
    cfg["dataset"]["name"] = dataset
    cfg["dataset"]["category"] = category
    cfg["experiment"]["name"] = exp_name
    cfg["experiment"]["seed"] = seed
    
    if coreset_pct is not None:
        cfg["memory"]["percentage"] = coreset_pct
    if layers is not None:
        cfg["backbone"]["layers"] = layers
    if bank_size is not None:
        cfg["routing"]["bank_size"] = bank_size
    
    # Update dataset root for VisA
    if dataset == "visa":
        cfg["dataset"]["root"] = cfg["dataset"].get(
            "visa_root", 
            "datasets/visa"
        )
    
    return cfg


# ==================== AUROC VARIANT COMPUTATION ====================
def compute_image_auroc_variants(
    base_scores,
    struct_scores,
    labels,
    struct_lambda: float = 0.05,
) -> Dict[str, float]:
    """Compute I-AUROC for base, struct-only, and hybrid variants."""
    import numpy as np
    from sklearn import metrics as sk_metrics
    from procon.eval.metrics import image_metrics
    
    base_scores = np.asarray(base_scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    
    results = {}
    
    # Base (max pooling) metrics
    auroc_base = float(sk_metrics.roc_auc_score(labels, base_scores))
    ap_base = float(sk_metrics.average_precision_score(labels, base_scores))
    results["i_auroc_base"] = auroc_base
    results["i_ap_base"] = ap_base
    
    try:
        base_metrics = image_metrics(base_scores, labels)
        results["i_f1_base"] = float(base_metrics.get("image_f1_max", 0.0))
    except Exception:
        results["i_f1_base"] = 0.0
    
    # Struct-only and hybrid (if struct_scores available)
    if struct_scores is not None and len(struct_scores) > 0:
        struct_scores = np.asarray(struct_scores, dtype=np.float32)
        
        auroc_struct = float(sk_metrics.roc_auc_score(labels, struct_scores))
        ap_struct = float(sk_metrics.average_precision_score(labels, struct_scores))
        results["i_auroc_struct_only"] = auroc_struct
        results["i_ap_struct_only"] = ap_struct
        
        try:
            struct_metrics = image_metrics(struct_scores, labels)
            results["i_f1_struct_only"] = float(struct_metrics.get("image_f1_max", 0.0))
        except Exception:
            results["i_f1_struct_only"] = 0.0
        
        # Hybrid
        hybrid_scores = base_scores + struct_lambda * struct_scores
        auroc_hybrid = float(sk_metrics.roc_auc_score(labels, hybrid_scores))
        ap_hybrid = float(sk_metrics.average_precision_score(labels, hybrid_scores))
        results["i_auroc_hybrid"] = auroc_hybrid
        results["i_ap_hybrid"] = ap_hybrid
        
        try:
            hybrid_metrics = image_metrics(hybrid_scores, labels)
            results["i_f1_hybrid"] = float(hybrid_metrics.get("image_f1_max", 0.0))
        except Exception:
            results["i_f1_hybrid"] = 0.0
    
    return results
