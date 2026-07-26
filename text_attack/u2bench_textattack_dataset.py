import os
import sys
from pathlib import Path

import ast
import json
import re
import pandas as pd
import textattack

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text_attack.medgemma_attack_common import image_placeholder


def _textattack_tsv_path() -> Path:
    raw_path = os.getenv("MEDGEMMA_TEXTATTACK_TSV", "").strip()
    if not raw_path:
        raise ValueError("MEDGEMMA_TEXTATTACK_TSV is required.")
    return Path(raw_path).expanduser()


def split_prompt(prompt: str):
    text = str(prompt or "")
    patterns = [
        r"(\\n\\noptions:.*)$",
        r"(\n\noptions:.*)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return text[: match.start()].strip(), match.group(1)
    return text.strip(), ""


def _label_names_from_options(options_field: str) -> list[str]:
    try:
        parsed = json.loads(options_field)
    except Exception:
        parsed = ast.literal_eval(options_field)
    if isinstance(parsed, dict):
        return [str(x) for x in next(iter(parsed.values()))]
    return [str(x) for x in parsed]


def _load_dataset(tsv_path: Path):
    df = pd.read_csv(tsv_path, sep="\t")
    label_names = _label_names_from_options(str(df.iloc[0]["options"]).strip())
    label_map = {name: idx for idx, name in enumerate(label_names)}
    rows = []

    # premise = image placeholder keyed by TSV row index (resolved in medgemma_textattack_wrapper)
    for tsv_row_index, row in df.iterrows():
        if str(row["class_label"]) not in label_map:
            continue
        editable, suffix = split_prompt(str(row["prompt"]))
        rows.append(
            (
                (image_placeholder(tsv_row_index), editable, suffix),
                label_map[str(row["class_label"])],
            )
        )

    return textattack.datasets.Dataset(
        rows,
        input_columns=("premise", "hypothesis", "context"),
        label_names=label_names,
        shuffle=False,
    )


dataset = _load_dataset(_textattack_tsv_path())
