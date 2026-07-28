#!/usr/bin/env python3
"""Build the balanced result-explorer dataset from local run summaries."""

import argparse
import base64
import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

csv.field_size_limit(sys.maxsize)

PROPOSERS = {
    "gemma-4-E4B-it": "Gemma-4-E4B-it",
    "gemma-4-12B-it": "Gemma-4-12B-it",
    "qwen-2.5-7b-instruct": "Qwen2.5-7B-Instruct",
    "qwen-2.5-14b-instruct": "Qwen2.5-14B-Instruct",
    "qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
}
TARGETS = {
    "medgemma-4b": "MedGemma-4B",
    "medgemma": "MedGemma-4B",
    "medgemma-27b": "MedGemma-27B",
    "qoq-med-7b": "QoQ-Med-7B",
    "qoq_med-7b": "QoQ-Med-7B",
    "qoq-med-32b": "QoQ-Med-32B",
    "qoq_med-32b": "QoQ-Med-32B",
    "llava-med-7b": "LLaVA-Med-7B",
    "llava_med_7b": "LLaVA-Med-7B",
    "llava-med": "LLaVA-Med-7B",
}
TASKS = {
    "DD": ("Disease diagnosis", "disease_diagnosis"),
    "LL": ("Lesion localization", "lesion_localisation"),
    "VR": ("View recognition", "view_recognition"),
}
WEIRD = re.compile(r"[{}\[\]<>$`]|\\u[0-9a-fA-F]{4}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-pair", type=int, default=10)
    return parser.parse_args()


def canonical_pair(path: Path, runs_root: Path):
    parts = path.relative_to(runs_root).parts
    if len(parts) < 3:
        return None
    proposer = PROPOSERS.get(parts[0])
    target = TARGETS.get(parts[1].lower())
    task = parts[2]
    if not proposer or not target or task not in TASKS:
        return None
    return proposer, target, task


def clean_score(record):
    changes = record.get("transitions") or []
    change_text = " ".join(
        f"{item.get('prev', '')} {item.get('new', '')}" for item in changes
    )
    return (
        bool(WEIRD.search(change_text)),
        abs(len(str(record.get("final_question", ""))) - len(str(record.get("original_question", "")))),
        len(changes),
        str(record.get("key", "")),
    )


def collect_candidates(runs_root: Path):
    grouped = defaultdict(dict)
    for path in sorted(runs_root.rglob("*.attack_summary.json")):
        pair = canonical_pair(path, runs_root)
        if not pair:
            continue
        proposer, target, task = pair
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for record in payload:
            if not record.get("attack_success") or not record.get("transitions"):
                continue
            if str(record.get("base_pred")) != str(record.get("real_label")):
                continue
            key = str(record.get("key") or "")
            dedupe_key = (task, key)
            candidate = {
                "record": record,
                "task": task,
            }
            current = grouped[(proposer, target)].get(dedupe_key)
            if current is None or clean_score(record) < clean_score(current["record"]):
                grouped[(proposer, target)][dedupe_key] = candidate
    return grouped


def read_tsv_row(path: Path, row_index: int):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return next(row for index, row in enumerate(reader) if index == row_index)


def save_image(encoded: str, output_path: Path):
    encoded = encoded.strip()
    if encoded.lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    image.thumbnail((900, 900), Image.Resampling.LANCZOS)
    image.save(output_path, quality=88, optimize=True)


def normalized_text(value):
    return str(value or "").replace("\\n", "\n").strip()


def build():
    args = parse_args()
    grouped = collect_candidates(args.runs_root)
    output = args.output_root
    image_dir = output / "images"
    data_dir = output / "data"
    image_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    expected = {
        (proposer, target)
        for proposer in sorted(set(PROPOSERS.values()))
        for target in sorted(set(TARGETS.values()))
    }
    examples = []
    shortages = []
    for proposer, target in sorted(expected):
        candidates = sorted(
            grouped.get((proposer, target), {}).values(),
            key=lambda item: clean_score(item["record"]),
        )
        usable = []
        for candidate in candidates:
            record = candidate["record"]
            task = candidate["task"]
            key = str(record["key"])
            tsv_name, index_text = key.rsplit(":", 1)
            row_index = int(index_text)
            dataset_path = args.dataset_root / TASKS[task][1] / tsv_name
            if not dataset_path.exists():
                continue
            try:
                row = read_tsv_row(dataset_path, row_index)
            except (StopIteration, OSError, csv.Error):
                continue
            if normalized_text(record.get("original_question")) != normalized_text(
                row.get("prompt")
            ):
                continue
            if str(record.get("real_label")) != str(row.get("class_label")):
                continue
            usable.append(
                (candidate, key, tsv_name, dataset_path, row_index, row)
            )
            if len(usable) == args.per_pair:
                break
        if len(usable) < args.per_pair:
            shortages.append((proposer, target, len(usable)))
        for rank, (
            candidate,
            key,
            tsv_name,
            dataset_path,
            row_index,
            row,
        ) in enumerate(usable, 1):
            record = candidate["record"]
            task = candidate["task"]
            image_name = f"{task.lower()}-{dataset_path.stem}-{row_index}.jpg"
            image_path = image_dir / image_name
            if not image_path.exists():
                save_image(row["img_data"], image_path)
            example_id = (
                f"{proposer}-{target}-{task}-{rank}"
                .lower()
                .replace("_", "-")
                .replace(" ", "-")
            )
            examples.append(
                {
                    "id": example_id,
                    "title": f"{TASKS[task][0]} · {tsv_name} row {row_index}",
                    "proposer_model": proposer,
                    "target_vlm": target,
                    "task": TASKS[task][0],
                    "image": f"images/{image_name}",
                    "key": key,
                    "dataset_source": str(dataset_path.relative_to(args.dataset_root.parent.parent)),
                    "dataset_row_index": row_index,
                    "ground_truth": record.get("real_label"),
                    "prediction_before": record.get("base_pred"),
                    "prediction_after": record.get("final_pred"),
                    "original_prompt": str(record.get("original_question", "")).replace("\\n", "\n"),
                    "attacked_prompt": str(record.get("final_question", "")).replace("\\n", "\n"),
                    "changes": [
                        {
                            "step": change.get("step"),
                            "previous": change.get("prev"),
                            "replacement": change.get("new"),
                        }
                        for change in record.get("transitions", [])
                    ],
                }
            )

    (data_dir / "examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False)
    )
    print(f"Wrote {len(examples)} examples and {len(list(image_dir.glob('*.jpg')))} images.")
    if shortages:
        for proposer, target, count in shortages:
            print(f"SHORTAGE: {proposer} × {target}: {count}/{args.per_pair}")
        raise SystemExit(2)


if __name__ == "__main__":
    build()
