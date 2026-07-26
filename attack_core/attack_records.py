import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from attack_core.u2bench import parse_options, resolve_label


def split_prompt(prompt: str) -> Tuple[str, str]:
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


def _find_resume_files(attack_input: Path) -> List[Path]:
    if attack_input.is_file():
        return [attack_input]

    summary_files = []
    for pattern in (
        "*.attack_summary.json",
        "*.pair_summary.json",
        "*.summary.json",
    ):
        summary_files.extend(sorted(attack_input.rglob(pattern)))
    textattack_csv_files = sorted(attack_input.rglob("*.csv"))
    attack_log_files = sorted(attack_input.rglob("attack_log_*.txt"))
    if summary_files or textattack_csv_files or attack_log_files:
        return summary_files + textattack_csv_files + attack_log_files

    return sorted(attack_input.rglob("_resume_progress.jsonl"))


def _decode_textattack_text(text: str) -> str:
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", str(text or ""), flags=re.DOTALL)
    return text.replace("\\n", "\n").strip()


def _parse_textattack_input(text: str) -> Tuple[str, str, str]:
    raw_fields: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for part in str(text or "").split("<SPLIT>"):
        match = re.match(
            r"\s*\[\[\[\[(Premise|Hypothesis|Context)\]\]\]\]\s*:\s?(.*)$",
            part,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            current = match.group(1).lower()
            raw_fields.setdefault(current, []).append(match.group(2))
        elif current is not None:
            raw_fields[current].append(part)
    fields = {
        name: _decode_textattack_text("\n".join(chunks)) for name, chunks in raw_fields.items()
    }
    return fields.get("premise", ""), fields.get("hypothesis", ""), fields.get("context", "")


def _textattack_prompt_from_input(text: str) -> str:
    _premise, hypothesis, context = _parse_textattack_input(text)
    return f"{hypothesis}{context}".strip()


def _textattack_row_index(text: str) -> Optional[int]:
    premise, _hypothesis, _context = _parse_textattack_input(text)
    match = re.search(r"__IMAGE_TSV_ROW_(\d+)__", premise)
    if not match:
        match = re.search(r"__IMAGE_TSV_ROW_(\d+)__", str(text or ""))
    if not match:
        return None
    return int(match.group(1))


def _is_textattack_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return {"original_text", "perturbed_text", "result_type"}.issubset(columns)


def _load_textattack_csv_records(path: Path) -> List[dict]:
    df = pd.read_csv(path)
    records: List[dict] = []
    tsv_stem = re.sub(r"^[A-Za-z]+(?:_[A-Za-z]+)*_", "", path.stem)
    tsv_file = f"{tsv_stem}.tsv"
    for csv_index, row in df.iterrows():
        row_index = _textattack_row_index(row.get("original_text", ""))
        if row_index is None:
            row_index = _textattack_row_index(row.get("perturbed_text", ""))
        if row_index is None:
            continue

        result_type = str(row.get("result_type", "")).strip()
        original_question = _textattack_prompt_from_input(row.get("original_text", ""))
        attacked_question = _textattack_prompt_from_input(row.get("perturbed_text", ""))
        if not original_question:
            continue
        if not attacked_question:
            attacked_question = original_question

        records.append(
            {
                "key": f"{tsv_file}:{row_index}",
                "tsv_file": tsv_file,
                "index": int(row_index),
                "csv_index": int(csv_index),
                "attack_success": result_type.lower() == "successful",
                "original_question": original_question,
                "attacked_question": attacked_question,
                "final_question": attacked_question,
                "label_after_attack": str(row.get("perturbed_output", "")),
                "prediction_source": "textattack",
                "reward_source": "textattack",
                "source_attack_format": "textattack_csv",
                "source_attack_file": str(path),
                "result_type": result_type,
            }
        )
    return records


def _extract_log_block_field(block: str, start_label: str, stop_labels: Tuple[str, ...]) -> str:
    start = block.find(start_label)
    if start < 0:
        return ""
    start += len(start_label)
    stops = [block.find(label, start) for label in stop_labels]
    stops = [pos for pos in stops if pos >= 0]
    end = min(stops) if stops else len(block)
    return block[start:end].strip()


def _parse_log_transitions(block: str) -> Tuple[List[dict], str]:
    transitions: List[dict] = []
    transitions_text_parts: List[str] = []
    section = _extract_log_block_field(
        block,
        "Transitions (path):",
        ("Final attack question:", "Final question:", "real_label=", "Result:", "--- End Run ---"),
    )
    for line in section.splitlines():
        match = re.match(
            r"\s*(\d+)\)\s*(.*?)\s*->\s*(.*?)\s*\(\u0394truth\s*([+-]?[0-9.]+)\)", line
        )
        if not match:
            continue
        prev = match.group(2).strip()
        new = match.group(3).strip()
        delta_truth = float(match.group(4))
        transitions.append(
            {
                "step": int(match.group(1)),
                "prev": prev,
                "new": new,
                "delta_truth": delta_truth,
            }
        )
        transitions_text_parts.append(f"{prev} -> {new}")
    return transitions, " | ".join(transitions_text_parts)


def _load_attack_log_records(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?=^==== Attacking sample )", text, flags=re.MULTILINE)
    records: List[dict] = []
    for block in blocks:
        if not block.startswith("==== Attacking sample "):
            continue
        header = block.splitlines()[0]
        key_match = re.search(r"\|\s*key=([^|]+?)\s*\|", header)
        if not key_match:
            continue
        key = key_match.group(1).strip()
        if ":" not in key:
            continue
        tsv_file, row_index_text = key.rsplit(":", 1)
        try:
            row_index = int(row_index_text)
        except ValueError:
            continue

        real_label_match = re.search(r"^Real label:\s*(.+)$", block, flags=re.MULTILINE)
        final_line_match = re.search(
            r"^real_label=(.*?)\s+.*?\bpred=(.*?)\s+", block, flags=re.MULTILINE
        )
        chosen_match = re.search(r"\bchosen_pred=([^\n]+?)\s+prediction_source=", block)
        prediction_source_match = re.search(r"\bprediction_source=([^\s]+)", block)
        reward_source_match = re.search(r"\breward_source=([^\s]+)", block)

        original_question = _extract_log_block_field(
            block,
            "Base question:",
            ("Initial gap", "--- Steps ---", "Base:"),
        )
        final_question = _extract_log_block_field(
            block,
            "Final attack question:",
            ("real_label=", "Result:", "--- End Run ---"),
        )
        if not final_question:
            final_question = _extract_log_block_field(
                block,
                "Final question:",
                ("real_label=", "Result:", "--- End Run ---"),
            )
        if not final_question:
            final_question = original_question

        transitions, transitions_text = _parse_log_transitions(block)
        attack_success = (
            "Success: VLM misclassified under this path." in block
            or "Result: misclassified" in block
        )
        label_after_attack = ""
        if chosen_match:
            label_after_attack = chosen_match.group(1).strip()
        elif final_line_match:
            label_after_attack = final_line_match.group(2).strip()

        records.append(
            {
                "key": key,
                "tsv_file": Path(tsv_file).name,
                "index": row_index,
                "search_mode": "mcts",
                "success": attack_success,
                "attack_success": attack_success,
                "prediction_source": (
                    prediction_source_match.group(1) if prediction_source_match else "pred"
                ),
                "reward_source": reward_source_match.group(1) if reward_source_match else "margin",
                "real_label": real_label_match.group(1).strip() if real_label_match else "",
                "actual_label": real_label_match.group(1).strip() if real_label_match else "",
                "original_question": original_question,
                "final_question": final_question,
                "transition_count": len(transitions),
                "transitions": transitions,
                "transitions_text": transitions_text,
                "label_after_attack": label_after_attack,
                "source_attack_format": "attack_log",
                "source_attack_file": str(path),
            }
        )
    return records


def _dedupe_attack_records(records: List[dict]) -> List[dict]:
    by_key: Dict[str, dict] = {}
    unkeyed: List[dict] = []
    for rec in records:
        key = str(rec.get("key") or "")
        if not key:
            unkeyed.append(rec)
            continue
        old = by_key.get(key)
        if old is None:
            by_key[key] = rec
            continue
        old_score = (
            1 if bool(old.get("attack_success")) else 0,
            1 if old.get("final_question") else 0,
            len(old.get("transitions") or []),
        )
        new_score = (
            1 if bool(rec.get("attack_success")) else 0,
            1 if rec.get("final_question") else 0,
            len(rec.get("transitions") or []),
        )
        if new_score > old_score:
            by_key[key] = rec
    return list(by_key.values()) + unkeyed


def load_attack_records(attack_input: Path) -> List[dict]:
    records: List[dict] = []
    for path in _find_resume_files(attack_input):
        if _is_textattack_csv(path):
            records.extend(_load_textattack_csv_records(path))
            continue
        if path.name.startswith("attack_log_") and path.suffix == ".txt":
            records.extend(_load_attack_log_records(path))
            continue
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
    return _dedupe_attack_records(records)


def resolve_tsv_path(
    tsv_file: str,
    key: str,
    dataset_root: Path,
    resolution_cache: Dict[Tuple[str, str], Path],
) -> Path:
    cache_key = (str(tsv_file), str(key))
    if cache_key in resolution_cache:
        return resolution_cache[cache_key]

    candidates: List[Path] = []
    tsv_file = str(tsv_file or "").strip()
    if tsv_file:
        tsv_path = Path(tsv_file)
        if tsv_path.is_absolute() and tsv_path.exists():
            resolution_cache[cache_key] = tsv_path
            return tsv_path
        joined = dataset_root / tsv_path
        if joined.exists():
            resolution_cache[cache_key] = joined
            return joined
        candidates.extend(dataset_root.rglob(tsv_path.name))

    if key and ":" in key:
        key_path = key.split(":", 1)[0]
        rel = Path(key_path)
        direct = dataset_root / rel
        if direct.exists():
            resolution_cache[cache_key] = direct
            return direct
        candidates.extend(dataset_root.rglob(rel.name))

    if not candidates:
        raise FileNotFoundError(f"Could not resolve TSV for tsv_file={tsv_file!r}, key={key!r}")

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)

    if len(unique) > 1 and tsv_file:
        for candidate in unique:
            if candidate.as_posix().endswith(tsv_file.replace("\\", "/")):
                resolution_cache[cache_key] = candidate
                return candidate

    resolution_cache[cache_key] = unique[0]
    return unique[0]


def load_attacked_samples(
    attack_input: Path,
    dataset_root: Path,
    successful_only: bool,
) -> List[dict]:
    records = load_attack_records(attack_input)
    if not records:
        raise ValueError(f"No attack summary records found under: {attack_input}")

    resolution_cache: Dict[Tuple[str, str], Path] = {}
    df_cache: Dict[Path, pd.DataFrame] = {}
    samples: List[dict] = []

    for rec in records:
        if successful_only and not bool(rec.get("attack_success")):
            continue

        key = str(rec.get("key", ""))
        tsv_path = resolve_tsv_path(
            rec.get("tsv_file", ""),
            key,
            dataset_root,
            resolution_cache,
        )
        if tsv_path not in df_cache:
            df_cache[tsv_path] = pd.read_csv(tsv_path, sep="\t")
        df = df_cache[tsv_path]

        row_index = rec.get("index")
        if row_index is None and key and ":" in key:
            try:
                row_index = int(key.rsplit(":", 1)[1])
            except Exception:
                row_index = None
        if row_index is None:
            continue

        row_index = int(row_index)
        if row_index < 0 or row_index >= len(df):
            continue

        row = df.iloc[row_index]
        options = parse_options(row.get("options", ""))
        truth = resolve_label(row.get("class_label"), options)
        if not options or not truth:
            continue

        original_question = str(
            rec.get("original_question") or rec.get("base_question") or row.get("prompt", "")
        )
        attacked_question = str(
            rec.get("final_question") or rec.get("attacked_question") or row.get("prompt", "")
        )
        label_after_attack = str(
            rec.get("label_after_attack")
            or rec.get("final_chosen_pred")
            or rec.get("final_pred")
            or ""
        )
        if label_after_attack.isdigit():
            label_idx = int(label_after_attack)
            if 0 <= label_idx < len(options):
                label_after_attack = str(options[label_idx])
        editable_prompt, frozen_suffix = split_prompt(attacked_question)
        samples.append(
            {
                "record": rec,
                "key": key or f"{tsv_path}:{row_index}",
                "tsv_path": str(tsv_path),
                "row_index": row_index,
                "img_data": str(row.get("img_data", "")),
                "original_question": original_question,
                "attacked_question": attacked_question,
                "editable_prompt": editable_prompt,
                "frozen_suffix": frozen_suffix,
                "options": options,
                "truth": truth,
                "attack_success": bool(rec.get("attack_success")),
                "label_after_attack": label_after_attack,
                "prediction_source": str(rec.get("prediction_source") or ""),
                "reward_source": str(rec.get("reward_source") or ""),
            }
        )
    return samples
