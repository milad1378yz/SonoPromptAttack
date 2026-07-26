import os
from pathlib import Path

import pandas as pd
import textattack

from attack_core.u2bench import normalize_u2bench_row, split_prompt_options
from text_attack.medgemma_attack_common import image_placeholder


def _textattack_tsv_path() -> Path:
    raw_path = os.getenv("MEDGEMMA_TEXTATTACK_TSV", "").strip()
    if not raw_path:
        raise ValueError("MEDGEMMA_TEXTATTACK_TSV is required.")
    return Path(raw_path).expanduser()


def _load_dataset(tsv_path: Path):
    df = pd.read_csv(tsv_path, sep="\t")
    if df.empty:
        raise ValueError(f"TextAttack TSV is empty: {tsv_path}")

    _, label_names, _ = normalize_u2bench_row(df.iloc[0])
    label_names = [str(option).strip() for option in label_names if str(option).strip()]
    if not label_names:
        raise ValueError(f"First TSV row has no usable options: {tsv_path}")

    label_map = {name: idx for idx, name in enumerate(label_names)}
    rows = []

    # premise = image placeholder keyed by TSV row index (resolved in medgemma_textattack_wrapper)
    for tsv_row_index, row in df.iterrows():
        prompt, options, truth = normalize_u2bench_row(row)
        normalized_options = [
            str(option).strip() for option in options if str(option).strip()
        ]
        if normalized_options != label_names or truth not in label_map:
            continue
        editable, suffix = split_prompt_options(prompt)
        rows.append(
            (
                (image_placeholder(tsv_row_index), editable, suffix),
                label_map[truth],
            )
        )

    return textattack.datasets.Dataset(
        rows,
        input_columns=("premise", "hypothesis", "context"),
        label_names=label_names,
        shuffle=False,
    )


dataset = _load_dataset(_textattack_tsv_path())
