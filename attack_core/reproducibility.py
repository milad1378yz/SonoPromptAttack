import os
import random

import numpy as np
import torch


def seed_everything(seed: int | None) -> int | None:
    """Seed supported RNGs and request deterministic Torch behavior."""
    if seed is None:
        return None

    seed = int(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed
