"""Data type utilities."""
import torch


def resolve_dtype(dtype: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    key = str(dtype).lower()
    if key in ("float16", "fp16", "half"):
        return torch.float16
    if key in ("float32", "fp32", "single"):
        return torch.float32
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32
