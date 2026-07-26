import os
from pathlib import Path

import pandas as pd
import textattack

from attack_core.u2bench import parse_options, split_prompt_options
from text_attack.medgemma_attack_common import image_placeholder


def _textattack_tsv_path() -> Path:
    raw_path = os.getenv("MEDGEMMA_TEXTATTACK_TSV", "").strip()
    if not raw_path:
        raise ValueError("MEDGEMMA_TEXTATTACK_TSV is required.")
    return Path(raw_path).expanduser()


def _load_dataset(tsv_path: Path):
    df = pd.read_csv(tsv_path, sep="\t")
    label_names = parse_options(df.iloc[0]["options"])
    label_map = {name: idx for idx, name in enumerate(label_names)}
    rows = []

    # premise = image placeholder keyed by TSV row index (resolved in medgemma_textattack_wrapper)
    for tsv_row_index, row in df.iterrows():
        if str(row["class_label"]) not in label_map:
            continue
        editable, suffix = split_prompt_options(row["prompt"])
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
