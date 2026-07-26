"""Shared TextAttack setup for MedGemma VLM datasets."""

import os
import random
import sys
import types
from pathlib import Path

from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)

VLM_INPUT_COLUMNS = ["premise", "hypothesis", "context"]
VLM_FROZEN_COLUMNS = {"premise", "context"}
IMAGE_TSV_ROW_PREFIX = "__IMAGE_TSV_ROW_"


def decode_textattack_field(text: str) -> str:
    """Decode one TextAttack CSV field the same way transferability does."""
    return str(text or "").replace("\\n", "\n").strip()


def compose_transfer_prompt(editable_prompt: str, frozen_suffix: str) -> str:
    """Build the full question string that transferability reconstructs from CSV."""
    return f"{decode_textattack_field(editable_prompt)}{decode_textattack_field(frozen_suffix)}".strip()


def maybe_set_seed() -> int | None:
    seed_text = os.getenv("MEDGEMMA_TEXTATTACK_SEED", "").strip()
    if not seed_text:
        return None
    try:
        seed = int(seed_text)
    except ValueError:
        return None

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    return seed


def image_placeholder(tsv_row_index: int) -> str:
    return f"{IMAGE_TSV_ROW_PREFIX}{int(tsv_row_index)}__"


def char_level_constraints(max_edit_distance: int = 30):
    return [
        RepeatModification(),
        StopwordModification(),
        InputColumnModification(VLM_INPUT_COLUMNS, VLM_FROZEN_COLUMNS),
        LevenshteinEditDistance(max_edit_distance),
    ]


def vlm_column_constraints():
    return [
        RepeatModification(),
        StopwordModification(),
        InputColumnModification(VLM_INPUT_COLUMNS, VLM_FROZEN_COLUMNS),
    ]


def register_for_parallel_pickling(module_name: str, file_path: Path, exports: dict[str, object]) -> None:
    """Register custom classes under a stable module name for TextAttack --parallel."""
    resolved = str(Path(file_path).resolve())
    if module_name not in sys.modules:
        mod = types.ModuleType(module_name)
        mod.__file__ = resolved
        sys.modules[module_name] = mod
    mod = sys.modules[module_name]
    for qualname, obj in exports.items():
        obj.__module__ = module_name
        obj.__qualname__ = qualname
        setattr(mod, qualname, obj)
