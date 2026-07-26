from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import os
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import textattack

THIS_DIR = Path(__file__).resolve().parent
ATTACK_VLM_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ATTACK_VLM_DIR) not in sys.path:
    sys.path.insert(0, str(ATTACK_VLM_DIR))

from medgemma_attack_common import (  # noqa: E402
    IMAGE_TSV_ROW_PREFIX,
    compose_transfer_prompt,
    maybe_set_seed,
)

from attack_core.model_loader import load_vlm  # noqa: E402
from attack_core.u2bench import decode_base64_image  # noqa: E402
from attack_core.vlm_scoring import score_candidate  # noqa: E402

try:
    from attack_core.vlm_scoring import score_options_from_generation_step  # noqa: E402
except ImportError:
    score_options_from_generation_step = None


TSV_PATH = Path(os.getenv("MEDGEMMA_TEXTATTACK_TSV", "/home/yasamin/Documents/VLM/medgemma/23.tsv"))
MODEL_ID = os.getenv("MEDGEMMA_TEXTATTACK_MODEL_ID", "google/medgemma-4b-it")
SYSTEM_PROMPT = os.getenv("MEDGEMMA_TEXTATTACK_SYSTEM_PROMPT", "")
SCORE_MODE = os.getenv("MEDGEMMA_TEXTATTACK_SCORE_MODE", "teacher_forced").strip().lower()


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


def _load_image_b64_by_tsv_row(tsv_path: Path, label_names: list[str]) -> dict[int, str]:
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t")
    label_map = {name: idx for idx, name in enumerate(label_names)}
    images: dict[int, str] = {}
    for tsv_row_index, row in df.iterrows():
        if str(row["class_label"]) in label_map:
            images[int(tsv_row_index)] = str(row["img_data"])
    return images


def _load_options(tsv_path: Path) -> list[str]:
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t")
    options_field = str(df.iloc[0]["options"]).strip()
    try:
        parsed = json.loads(options_field)
    except Exception:
        parsed = ast.literal_eval(options_field)
    if isinstance(parsed, dict):
        values = next(iter(parsed.values()))
    else:
        values = parsed
    return [str(x).strip() for x in values if str(x).strip()]


@lru_cache(maxsize=256)
def _decode_image_cached(img_b64: str):
    return decode_base64_image(img_b64)


class MedGemmaTextAttackWrapper(textattack.models.wrappers.ModelWrapper):
    def __init__(self, model_id: str = MODEL_ID, tsv_path: Path = TSV_PATH):
        self.model_id = model_id
        self.tsv_path = Path(tsv_path)
        self.options = _load_options(self.tsv_path)
        self._image_b64_by_tsv_row = _load_image_b64_by_tsv_row(self.tsv_path, self.options)
        _ensure_vlm_device_map()
        self.vlm, self.processor = load_vlm(model_id)
        self.model = self.vlm
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.score_mode = SCORE_MODE

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
            return image_b64, editable_prompt, frozen_suffix, full_prompt
        raise TypeError(
            f"Expected tuple/list input from TextAttack dataset, got {type(item)}."
        )

    def _score_options(self, image_b64: str, question_text: str) -> np.ndarray:
        image = _decode_image_cached(image_b64).copy()

        if self.score_mode == "generation_step":
            if score_options_from_generation_step is None:
                raise RuntimeError(
                    "MEDGEMMA_TEXTATTACK_SCORE_MODE=generation_step requires "
                    "attack_core.vlm_scoring.score_options_from_generation_step, but it is not available."
                )
            summary = score_options_from_generation_step(
                self.vlm,
                self.processor,
                image,
                SYSTEM_PROMPT,
                question_text,
                self.options,
            )
            score_map = summary.get("scores", {}) if summary else {}
            scores = [float(score_map.get(opt, float("-inf"))) for opt in self.options]
        else:
            scores = [
                float(
                    score_candidate(
                        self.vlm,
                        self.processor,
                        image,
                        SYSTEM_PROMPT,
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
            image_b64, _editable_prompt, _frozen_suffix, full_prompt = self._normalize_input(item)
            outputs.append(self._score_options(image_b64, full_prompt))
        return np.stack(outputs, axis=0)

    def get_grad(self, text_input):
        raise NotImplementedError("Gradient access is not implemented for MedGemma wrapper.")

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
WRAPPER_MODULE = "medgemma_textattack_wrapper"
MedGemmaTextAttackWrapper.__module__ = WRAPPER_MODULE
MedGemmaTextAttackWrapper.__qualname__ = "MedGemmaTextAttackWrapper"
if WRAPPER_MODULE not in sys.modules:
    _stable_module = types.ModuleType(WRAPPER_MODULE)
    _stable_module.__file__ = str(Path(__file__).resolve())
    sys.modules[WRAPPER_MODULE] = _stable_module
_stable_module = sys.modules[WRAPPER_MODULE]
_stable_module.MedGemmaTextAttackWrapper = MedGemmaTextAttackWrapper

# Workers unpickle the attack object; loading another VLM at import time duplicates GPU memory.
if _is_main_process():
    maybe_set_seed()
    model = MedGemmaTextAttackWrapper()
else:
    model = None
_stable_module.model = model


def _parse_args():
    parser = argparse.ArgumentParser(description="MedGemma TextAttack wrapper CLI")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--tsv", type=str, default=None, help="Path to TSV dataset.")
    parser.add_argument("--model-id", type=str, default=None, help="Hugging Face model id.")
    parser.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt.")
    parser.add_argument(
        "--score-mode",
        type=str,
        choices=["teacher_forced", "generation_step"],
        default=None,
        help="Scoring mode.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.seed is not None:
        os.environ["MEDGEMMA_TEXTATTACK_SEED"] = str(args.seed)
    if args.tsv:
        os.environ["MEDGEMMA_TEXTATTACK_TSV"] = args.tsv
    if args.model_id:
        os.environ["MEDGEMMA_TEXTATTACK_MODEL_ID"] = args.model_id
    if args.system_prompt is not None:
        os.environ["MEDGEMMA_TEXTATTACK_SYSTEM_PROMPT"] = args.system_prompt
    if args.score_mode:
        os.environ["MEDGEMMA_TEXTATTACK_SCORE_MODE"] = args.score_mode

    maybe_set_seed()
    wrapper = MedGemmaTextAttackWrapper()
    print(
        "MedGemma TextAttack wrapper initialized. "
        f"model_id={wrapper.model_id} tsv_path={wrapper.tsv_path} score_mode={wrapper.score_mode}"
    )


if __name__ == "__main__":
    main()
