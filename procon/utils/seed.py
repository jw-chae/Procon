"""Random seed utilities for reproducibility."""
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    """Set random seed for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
