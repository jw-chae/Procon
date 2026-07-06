from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_TRAIN_GOOD_SCORES_NAME = "train_good_scores.csv"
DEFAULT_TRAIN_GOOD_THRESHOLDS_NAME = "thresholds_train_good.json"


@dataclass(frozen=True)
class TrainGoodThresholds:
    quantile: float
    base: float
    hybrid: float
    struct: Optional[float]
    n: int


def train_good_scores_path(run_dir: Path) -> Path:
    return run_dir / DEFAULT_TRAIN_GOOD_SCORES_NAME


def train_good_thresholds_path(run_dir: Path) -> Path:
    return run_dir / DEFAULT_TRAIN_GOOD_THRESHOLDS_NAME


def compute_train_good_thresholds_from_scores_csv(
    scores_csv: Path,
    quantile: float = 0.995,
) -> TrainGoodThresholds:
    df = pd.read_csv(scores_csv)

    if "base_score" not in df.columns or "final_score" not in df.columns:
        raise KeyError(f"Missing required columns in {scores_csv}: base_score/final_score")

    base_thr = float(np.quantile(df["base_score"].to_numpy(dtype=np.float64), quantile))
    hybrid_thr = float(np.quantile(df["final_score"].to_numpy(dtype=np.float64), quantile))

    struct_thr: Optional[float]
    if "struct_score" in df.columns:
        struct_thr = float(np.quantile(df["struct_score"].to_numpy(dtype=np.float64), quantile))
    else:
        struct_thr = None

    return TrainGoodThresholds(
        quantile=float(quantile),
        base=base_thr,
        hybrid=hybrid_thr,
        struct=struct_thr,
        n=int(len(df)),
    )


def save_train_good_thresholds(run_dir: Path, thresholds: TrainGoodThresholds) -> Path:
    out_path = train_good_thresholds_path(run_dir)
    payload = {
        "quantile": thresholds.quantile,
        "n": thresholds.n,
        "base": thresholds.base,
        "hybrid": thresholds.hybrid,
        "struct": thresholds.struct,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_train_good_thresholds(
    run_dir: Path,
    *,
    quantile: float = 0.995,
    strict: bool = True,
) -> TrainGoodThresholds:
    """Load thresholds computed from train-good scores.

    Priority:
    1) run_dir/thresholds_train_good.json
    2) run_dir/train_good_scores.csv -> compute & save thresholds

    If strict=True and neither exists, raises FileNotFoundError.
    """

    thr_path = train_good_thresholds_path(run_dir)
    if thr_path.exists():
        data = json.loads(thr_path.read_text(encoding="utf-8"))
        loaded = TrainGoodThresholds(
            quantile=float(data.get("quantile", quantile)),
            base=float(data["base"]),
            hybrid=float(data["hybrid"]),
            struct=(None if data.get("struct", None) is None else float(data["struct"])),
            n=int(data.get("n", 0) or 0),
        )
        return loaded

    scores_path = train_good_scores_path(run_dir)
    if scores_path.exists():
        thresholds = compute_train_good_thresholds_from_scores_csv(scores_path, quantile=quantile)
        save_train_good_thresholds(run_dir, thresholds)
        return thresholds

    if strict:
        raise FileNotFoundError(
            "Train-good thresholds not found. Expected one of:\n"
            f"- {thr_path}\n"
            f"- {scores_path}\n\n"
            "Generate them with:\n"
            "  python3 appendix/scripts/cache_train_good_scores.py --dataset <mvtec|visa> --category <cat> --exp-name <exp> --seed 0"
        )

    return TrainGoodThresholds(quantile=float(quantile), base=float("nan"), hybrid=float("nan"), struct=None, n=0)


def load_train_good_scores_df(run_dir: Path) -> pd.DataFrame:
    scores_path = train_good_scores_path(run_dir)
    if not scores_path.exists():
        raise FileNotFoundError(f"Train-good scores not found: {scores_path}")
    return pd.read_csv(scores_path)
