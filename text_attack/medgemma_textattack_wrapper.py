import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import textattack
import torch

from attack_core.model_loader import load_vlm
from attack_core.u2bench import decode_base64_image, normalize_u2bench_row
from attack_core.vlm_scoring import score_candidate
from text_attack.medgemma_attack_common import (
    IMAGE_TSV_ROW_PREFIX,
    compose_transfer_prompt,
    maybe_set_seed,
    register_for_parallel_pickling,
)


@dataclass(frozen=True)
class TextAttackSettings:
    tsv_path: Path
    model_id: str
    system_prompt: str

    @classmethod
    def from_environment(
        cls,
        *,
        tsv_path: str | Path | None = None,
        model_id: str | None = None,
        system_prompt: str | None = None,
    ) -> "TextAttackSettings":
        raw_tsv_path = (
            str(tsv_path)
            if tsv_path is not None
            else os.getenv("MEDGEMMA_TEXTATTACK_TSV", "")
        ).strip()
        if not raw_tsv_path:
            raise ValueError("MEDGEMMA_TEXTATTACK_TSV is required.")

        resolved_model_id = (
            model_id
            if model_id is not None
            else os.getenv("MEDGEMMA_TEXTATTACK_MODEL_ID", "google/medgemma-4b-it")
        ).strip()
        resolved_system_prompt = (
            system_prompt
            if system_prompt is not None
            else os.getenv("MEDGEMMA_TEXTATTACK_SYSTEM_PROMPT", "")
        )

        return cls(
            tsv_path=Path(raw_tsv_path).expanduser(),
            model_id=resolved_model_id,
            system_prompt=resolved_system_prompt,
        )


def _textattack_parallel_enabled() -> bool:
    raw = os.getenv("TEXTATTACK_PARALLEL", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_main_process() -> bool:
    return mp.current_process().name == "MainProcess"


def _ensure_vlm_device_map() -> None:
    if "VLM_DEVICE_MAP" in os.environ:
        return
    if _textattack_parallel_enabled() and _is_main_process():
        # Parent only pickles the attack for workers; avoid a GPU-resident copy.
        os.environ["VLM_DEVICE_MAP"] = "cpu"
    else:
        os.environ["VLM_DEVICE_MAP"] = (
            os.getenv("MEDGEMMA_TEXTATTACK_DEVICE_MAP", "cuda:0").strip() or "cuda:0"
        )


def _load_tsv_metadata(tsv_path: Path) -> tuple[list[str], dict[int, str]]:
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t")
    if df.empty:
        raise ValueError(f"TextAttack TSV is empty: {tsv_path}")

    _, label_names, _ = normalize_u2bench_row(df.iloc[0])
    label_names = [str(option).strip() for option in label_names if str(option).strip()]
    if not label_names:
        raise ValueError(f"First TSV row has no usable options: {tsv_path}")

    label_map = {name: idx for idx, name in enumerate(label_names)}
    images: dict[int, str] = {}
    for tsv_row_index, row in df.iterrows():
        _, options, truth = normalize_u2bench_row(row)
        normalized_options = [
            str(option).strip() for option in options if str(option).strip()
        ]
        if normalized_options != label_names or truth not in label_map:
            continue
        images[int(tsv_row_index)] = str(row["img_data"])
    return label_names, images


@lru_cache(maxsize=256)
def _decode_image_cached(img_b64: str):
    return decode_base64_image(img_b64)


class MedGemmaTextAttackWrapper(textattack.models.wrappers.ModelWrapper):
    def __init__(self, settings: TextAttackSettings | None = None):
        settings = settings or TextAttackSettings.from_environment()
        self.tsv_path = settings.tsv_path
        self.system_prompt = settings.system_prompt
        self.options, self._image_b64_by_tsv_row = _load_tsv_metadata(self.tsv_path)
        _ensure_vlm_device_map()
        self.vlm, self.processor = load_vlm(settings.model_id)
        self.model = self.vlm
        self.tokenizer = getattr(self.processor, "tokenizer", None)

    def _resolve_image_b64(self, premise: str) -> str:
        if premise.startswith(IMAGE_TSV_ROW_PREFIX) and premise.endswith("__"):
            tsv_row_index = int(premise[len(IMAGE_TSV_ROW_PREFIX) : -2])
            try:
                return self._image_b64_by_tsv_row[tsv_row_index]
            except KeyError as exc:
                raise KeyError(
                    f"No image for TSV row index {tsv_row_index} in {self.tsv_path}"
                ) from exc
        if premise.startswith("__IMAGE_ROW_") and premise.endswith("__"):
            raise ValueError(
                "Legacy __IMAGE_ROW_N__ placeholders are unsupported; "
                "regenerate the dataset with __IMAGE_TSV_ROW_<index>__."
            )
        return premise

    def _normalize_input(self, item):
        if isinstance(item, (tuple, list)):
            if len(item) < 2:
                raise ValueError(
                    "Expected (img_data, editable_prompt[, frozen_suffix]) tuple for MedGemma TextAttack wrapper."
                )
            premise = str(item[0])
            image_b64 = self._resolve_image_b64(premise)
            editable_prompt = str(item[1])
            frozen_suffix = str(item[2]) if len(item) >= 3 else ""
            # Score the same prompt transferability rebuilds from TextAttack CSV logs.
            full_prompt = compose_transfer_prompt(editable_prompt, frozen_suffix)
            return image_b64, full_prompt
        raise TypeError(
            f"Expected tuple/list input from TextAttack dataset, got {type(item)}."
        )

    def _score_options(self, image_b64: str, question_text: str) -> np.ndarray:
        image = _decode_image_cached(image_b64).copy()

        scores = [
            float(
                score_candidate(
                    self.vlm,
                    self.processor,
                    image,
                    self.system_prompt,
                    question_text,
                    opt,
                )
            )
            for opt in self.options
        ]

        tensor_scores = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(tensor_scores, dim=-1).cpu().numpy()
        return probs

    def __call__(self, text_input_list: Sequence[tuple[str, str]], **kwargs):
        outputs = []
        for item in text_input_list:
            image_b64, full_prompt = self._normalize_input(item)
            outputs.append(self._score_options(image_b64, full_prompt))
        return np.stack(outputs, axis=0)

    def get_grad(self, text_input):
        raise NotImplementedError(
            "Gradient access is not implemented for MedGemma wrapper."
        )

    def _tokenize(self, inputs: Iterable):
        tokenized = []
        for item in inputs:
            if isinstance(item, (tuple, list)):
                text = str(item[1]) if len(item) >= 2 else str(item[-1])
            else:
                text = str(item)
            if self.tokenizer is not None and hasattr(self.tokenizer, "tokenize"):
                tokenized.append(self.tokenizer.tokenize(text))
            else:
                tokenized.append(text.split())
        return tokenized


# TextAttack loads --model-from-file under a temp_* module name, which breaks
# multiprocessing pickling when --parallel is enabled. Re-register this file
# under a stable import path so worker processes can unpickle the wrapper.
_STABLE_MODULE = "text_attack.medgemma_textattack_wrapper"
register_for_parallel_pickling(
    _STABLE_MODULE,
    Path(__file__),
    {
        "TextAttackSettings": TextAttackSettings,
        "MedGemmaTextAttackWrapper": MedGemmaTextAttackWrapper,
    },
)

# Workers unpickle the attack object; loading another VLM at import time duplicates GPU memory.
if _is_main_process():
    maybe_set_seed()
    model = MedGemmaTextAttackWrapper()
else:
    model = None
sys.modules[_STABLE_MODULE].model = model
