"""Runtime optimization settings."""
from __future__ import annotations

from typing import Any, Dict

import torch


def apply_runtime_settings(runtime: Dict[str, Any]) -> bool:
    """Apply performance-related torch settings. Returns inference_mode flag."""
    if not isinstance(runtime, dict):
        return False

    if torch.cuda.is_available():
        if "tf32" in runtime:
            use_tf32 = bool(runtime.get("tf32", False))
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
            torch.backends.cudnn.allow_tf32 = use_tf32
        if "cudnn_benchmark" in runtime:
            torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", False))

    matmul_precision = runtime.get("matmul_precision")
    if matmul_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(str(matmul_precision))
        except Exception:
            pass

    return bool(runtime.get("inference_mode", False))
