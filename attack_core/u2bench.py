import ast
import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

from attack_core.run_outputs import append_jsonl_records


def decode_base64_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    img = Image.open(BytesIO(data)).convert("RGB")
    return img


def parse_options(options_field):
    s = str(options_field).strip()
    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            obj = None

    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            if isinstance(v, (list, tuple)):
                out.extend([str(x) for x in v])
            else:
                out.append(str(v))
        return out
    if isinstance(obj, (list, tuple)):
        return [str(x) for x in obj]

    parts = re.split(r"[\|,;\n]+", s)
    return [p.strip().strip("'\"") for p in parts if p.strip()]


def split_prompt_options(prompt: str) -> tuple[str, str]:
    """Split a U2-Bench prompt into editable text and its options suffix."""
    text = str(prompt or "")
    for pattern in (r"(\\n\\noptions:.*)$", r"(\n\noptions:.*)$"):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return text[: match.start()].strip(), match.group(1)
    return text.strip(), ""


def _placeholder_value_to_text(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return text

    if isinstance(parsed, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in parsed if str(item).strip())
    if isinstance(parsed, dict):
        return ", ".join(str(key).strip() for key in parsed.keys() if str(key).strip())
    return str(parsed).strip()


def fill_prompt_placeholders(prompt: str, row) -> str:
    text = str(prompt or "")
    if "{" not in text or "}" not in text:
        return text.strip()

    try:
        fields = list(row.index)
    except Exception:
        fields = []

    for field in fields:
        field_name = str(field)
        value_text = _placeholder_value_to_text(row.get(field, ""))
        if not value_text:
            continue

        escaped_field = field_name.replace("_", r"\_")
        for placeholder in (f"{{{field_name}}}", f"{{{escaped_field}}}"):
            text = text.replace(placeholder, value_text)

    return text.strip()


LOCATION_OPTIONS = [
    "upper left",
    "upper center",
    "upper right",
    "middle left",
    "center",
    "middle right",
    "lower left",
    "lower center",
    "lower right",
    "not visible",
]


def _is_location_prompt(prompt: str) -> bool:
    text = str(prompt or "").lower()
    return (
        "primary location" in text
        and "upper left" in text
        and "lower right" in text
        and "not visible" in text
    )


def _location_from_xy(x_value, y_value) -> str:
    x = float(x_value)
    y = float(y_value)

    horizontal = "left" if x < (1 / 3) else "center" if x < (2 / 3) else "right"
    vertical = "upper" if y < (1 / 3) else "middle" if y < (2 / 3) else "lower"

    if vertical == "middle" and horizontal == "center":
        return "center"
    return f"{vertical} {horizontal}"


def _location_label_from_bbox(gt_bbox_field) -> Optional[str]:
    text = str(gt_bbox_field or "").strip()
    if not text or text.lower() == "nan":
        return None

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None

    boxes = parsed if isinstance(parsed, list) else [parsed]
    for item in boxes:
        if not isinstance(item, dict) or not item:
            continue
        for bbox in item.values():
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
                try:
                    return _location_from_xy(bbox[0], bbox[1])
                except Exception:
                    continue
    return "not visible"


def resolve_row_options_and_truth(row):
    raw_prompt = str(row.get("prompt", ""))
    if _is_location_prompt(raw_prompt):
        truth = _location_label_from_bbox(row.get("gt_bbox", ""))
        return list(LOCATION_OPTIONS), truth

    options = [o for o in parse_options(row.get("options", "")) if str(o).strip()]
    truth = resolve_label(row.get("class_label", ""), options)
    return options, truth


def resolve_label(class_label, options):
    if class_label is None:
        return None

    opts = [str(o).strip() for o in options if str(o).strip()]
    if not opts:
        return None

    lower_map = {o.lower(): o for o in opts}
    cl = str(class_label).strip()
    if not cl or cl.lower() == "nan":
        return None

    key = cl.lower()
    if key in lower_map:
        return lower_map[key]

    if cl.isdigit():
        idx = int(cl)
        if 0 <= idx < len(opts):
            return opts[idx]
        if 1 <= idx <= len(opts):
            return opts[idx - 1]
    return None


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _find_cached_candidate_files(cache_source: Path):
    if cache_source.is_file():
        return [cache_source]

    files = []
    for pattern in ("*.attack_summary.json", "_resume_progress.jsonl"):
        files.extend(sorted(cache_source.rglob(pattern)))
    return files


def _resolve_cached_tsv_path(
    tsv_file: str,
    key: str,
    dataset_path: Path,
    resolution_cache: dict,
):
    cache_key = (str(tsv_file or ""), str(key or ""))
    if cache_key in resolution_cache:
        return resolution_cache[cache_key]

    if dataset_path.is_file():
        target_name = dataset_path.name
        candidates = set()
        tsv_file = str(tsv_file or "").strip()
        if tsv_file:
            candidates.add(Path(tsv_file).name)
        if key and ":" in key:
            candidates.add(Path(key.split(":", 1)[0]).name)
        if target_name in candidates:
            resolution_cache[cache_key] = dataset_path
            return dataset_path
        raise FileNotFoundError(
            f"Cached record belongs to a different TSV ({tsv_file!r}, key={key!r}) than requested dataset file {dataset_path}"
        )

    tsv_file = str(tsv_file or "").strip()
    if tsv_file:
        direct = dataset_path / tsv_file
        if direct.exists():
            resolution_cache[cache_key] = direct
            return direct
        matches = list(dataset_path.rglob(Path(tsv_file).name))
        if len(matches) == 1:
            resolution_cache[cache_key] = matches[0]
            return matches[0]
        if len(matches) > 1:
            for match in matches:
                if match.as_posix().endswith(tsv_file.replace("\\", "/")):
                    resolution_cache[cache_key] = match
                    return match

    if key and ":" in key:
        rel = key.split(":", 1)[0]
        direct = dataset_path / rel
        if direct.exists():
            resolution_cache[cache_key] = direct
            return direct
        matches = list(dataset_path.rglob(Path(rel).name))
        if matches:
            resolution_cache[cache_key] = matches[0]
            return matches[0]

    raise FileNotFoundError(
        f"Could not resolve cached TSV path for tsv_file={tsv_file!r}, key={key!r} under {dataset_path}"
    )


def load_candidate_samples_from_results_cache(
    dataset_path: str,
    candidate_cache_source: str,
    prediction_source: str = "pred",
):
    dataset_root = Path(dataset_path)
    cache_source = Path(candidate_cache_source)
    cache_files = _find_cached_candidate_files(cache_source)
    if not cache_files:
        raise ValueError(f"No cached candidate summary files found under: {cache_source}")

    resolution_cache = {}
    df_cache = {}
    seen_keys = set()
    candidates = []

    for path in cache_files:
        try:
            if path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    records = [json.loads(line) for line in f if line.strip()]
            else:
                with open(path, "r", encoding="utf-8") as f:
                    records = json.load(f)
        except Exception:
            continue

        if not isinstance(records, list):
            continue

        for rec in records:
            key = str(rec.get("key") or "").strip()
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)

            try:
                tsv_path = _resolve_cached_tsv_path(
                    rec.get("tsv_file", ""),
                    key,
                    dataset_root,
                    resolution_cache,
                )
            except Exception:
                continue

            if tsv_path not in df_cache:
                try:
                    df_cache[tsv_path] = pd.read_csv(tsv_path, sep="\t")
                except Exception:
                    continue
            df = df_cache[tsv_path]

            row_index = rec.get("index")
            if row_index is None and ":" in key:
                try:
                    row_index = int(key.rsplit(":", 1)[1])
                except Exception:
                    row_index = None
            if row_index is None:
                continue
            try:
                row_index = int(row_index)
            except Exception:
                continue
            if row_index < 0 or row_index >= len(df):
                continue

            row = df.iloc[row_index]
            options, truth = resolve_row_options_and_truth(row)
            if len(options) < 2:
                continue
            if not truth:
                continue

            img_b64 = str(row.get("img_data", "")).strip()
            question_text = fill_prompt_placeholders(row.get("prompt", ""), row)
            if not img_b64 or not question_text:
                continue
            try:
                img = decode_base64_image(img_b64)
            except Exception:
                continue

            cached_base_pred = str(rec.get("base_pred") or rec.get("root_pred") or "").strip()
            if cached_base_pred and cached_base_pred.lower() != str(truth).lower():
                continue

            source_file = str(rec.get("tsv_file") or tsv_path.name)
            gap_val = _safe_float(rec.get("base_gap"))
            if gap_val is None:
                gap_val = _safe_float(rec.get("gap"))
            if gap_val is None:
                gap_val = float("inf")

            candidates.append(
                {
                    "image": img,
                    "ground_truth": truth,
                    "prompt": question_text,
                    "options": options,
                    "gap": gap_val,
                    "key": key,
                    "source_file": source_file,
                    "row_index": int(row_index),
                    "scores_summary": None,
                }
            )

    candidates.sort(
        key=lambda x: (
            x["gap"] if isinstance(x.get("gap"), (int, float)) else float("inf"),
            str(x.get("key", "")),
        )
    )
    return candidates


def _extract_scores_from_cache(rec, options):
    if not rec:
        return None

    score_map = rec.get("scores")
    if isinstance(score_map, dict):
        if all(opt in score_map for opt in options):
            try:
                return {opt: float(score_map[opt]) for opt in options}
            except Exception:
                return None

    prefix_scores = {}
    for k, v in rec.items():
        if isinstance(k, str) and k.startswith("s_"):
            prefix_scores[k[2:]] = v
    if not prefix_scores:
        return None

    option_tokens = {opt: re.sub(r"[^a-z0-9]+", "", str(opt).strip().lower()) for opt in options}
    tokens = {token: opt for opt, token in option_tokens.items()}
    if len(tokens) == len(set(tokens)) and all(tok in prefix_scores for tok in tokens):
        try:
            return {opt: float(prefix_scores[option_tokens[opt]]) for opt in options}
        except Exception:
            return None

    first_letters = {}
    for opt in options:
        key = option_tokens[opt][:1]
        first_letters.setdefault(key, []).append(opt)

    if len(first_letters) != len(options):
        return None

    if not all(k in prefix_scores and len(v) == 1 for k, v in first_letters.items()):
        return None

    try:
        return {v[0]: float(prefix_scores[k]) for k, v in first_letters.items()}
    except Exception:
        return None


def _load_initial_score_cache(cache_path: Optional[str]):
    cache = {}
    if not cache_path or not Path(cache_path).exists():
        return cache
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec.get("key")
                    if key:
                        cache[key] = rec
                except Exception:
                    continue
    except Exception:
        return {}
    return cache


def _cache_record(key, rel_path, row_index, truth, scores_summary, gap_val):
    return {
        "key": key,
        "file": str(rel_path),
        "row": int(row_index),
        "pred": scores_summary.get("pred"),
        "truth": truth,
        "runner_up": scores_summary.get("runner_up"),
        "pred_score": float(scores_summary.get("pred_score", 0.0)),
        "runner_up_score": float(scores_summary.get("runner_up_score", 0.0)),
        "pred_gap": gap_val,
        "gap": gap_val,
        "truth_score": float(scores_summary.get("truth_score", 0.0)),
        "best_other_score": float(scores_summary.get("best_other_score", 0.0)),
        "truth_gap": float(scores_summary.get("truth_gap", 0.0)),
        "margin": float(scores_summary.get("margin", 0.0)),
        "options": scores_summary.get("options", []),
        "scores": scores_summary.get("scores", {}),
    }


def _scores_for_row(
    cached_rec,
    options,
    truth,
    prediction_source,
    reward_source,
    reasoning_off,
    vlm,
    proc,
    image,
    system_prompt,
    question_text,
):
    from attack_core.vlm_scoring import (
        attach_reward_fields,
        compute_scores,
        summarize_option_scores,
    )

    cached_scores = _extract_scores_from_cache(cached_rec, options) if cached_rec else None
    cache_complete = bool(
        cached_rec
        and cached_scores
        and isinstance(cached_rec.get("scores"), dict)
        and bool(cached_rec.get("options"))
    )
    scores_summary = None
    if cached_scores:
        try:
            scores_summary = summarize_option_scores(cached_scores, options, truth)
            scores_summary = attach_reward_fields(
                scores_summary, truth, prediction_source, reward_source
            )
        except Exception:
            scores_summary = None

    if scores_summary is None:
        scores_summary = compute_scores(
            vlm,
            proc,
            image,
            system_prompt,
            question_text,
            options,
            truth,
            prediction_source=prediction_source,
            reward_source=reward_source,
            reasoning_off=reasoning_off,
        )

    return scores_summary, cache_complete


def find_lowconf_correct_samples_dataset_path(
    u2_path: str,
    vlm,
    proc,
    system_prompt: str = "",
    diff_threshold: Optional[float] = 5.0,
    cache_path: Optional[str] = None,
    prediction_source: str = "pred",
    reward_source: str = "margin",
    reasoning_off: bool = False,
    completed_keys=None,
):
    task_path = Path(u2_path)
    if task_path.is_file():
        dataset_root = task_path.parent
        tsv_files = [task_path]
    else:
        dataset_root = task_path
        tsv_files = sorted(dataset_root.rglob("*.tsv"))

    cache = _load_initial_score_cache(cache_path)
    new_cache_records = []
    candidates = []
    completed_keys = set(completed_keys or ())
    if not tsv_files:
        return []

    for tsv in tqdm(tsv_files, desc="Scanning dataset"):
        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception:
            continue
        for ridx, row in df.iterrows():
            options, truth = resolve_row_options_and_truth(row)
            if len(options) < 2:
                continue
            rel_path = tsv.relative_to(dataset_root)
            key = f"{rel_path}:{int(ridx)}"
            if key in completed_keys:
                continue
            if not truth:
                continue

            img_b64 = str(row.get("img_data", "")).strip()
            question_text = fill_prompt_placeholders(row.get("prompt", ""), row)
            if not img_b64 or not question_text:
                continue

            try:
                img = decode_base64_image(img_b64)
            except Exception:
                continue

            cached_rec = cache.get(key)
            try:
                scores_summary, cache_complete = _scores_for_row(
                    cached_rec,
                    options,
                    truth,
                    prediction_source,
                    reward_source,
                    reasoning_off,
                    vlm,
                    proc,
                    img,
                    system_prompt,
                    question_text,
                )
            except Exception:
                continue

            gap_val = float(scores_summary.get("pred_gap", scores_summary.get("gap", 0.0)))
            if cache_path and (cached_rec is None or not cache_complete):
                new_cache_records.append(
                    _cache_record(key, rel_path, ridx, truth, scores_summary, gap_val)
                )

            pred = str(scores_summary.get("pred") or "")
            if not pred or truth is None or pred.lower() != truth.lower():
                continue

            if diff_threshold is None or gap_val < float(diff_threshold):
                candidates.append(
                    {
                        "image": img,
                        "ground_truth": truth,
                        "prompt": question_text,
                        "options": options,
                        "gap": gap_val,
                        "key": key,
                        "source_file": str(rel_path),
                        "row_index": int(ridx),
                        "scores_summary": scores_summary,
                    }
                )

    candidates.sort(key=lambda x: x["gap"])
    if cache_path and new_cache_records:
        try:
            append_jsonl_records(Path(cache_path), new_cache_records)
        except Exception:
            pass
    return candidates


def find_lowconf_correct_samples_dataset_id(
    u2_path: str,
    vlm,
    proc,
    system_prompt: str = "",
    diff_threshold: Optional[float] = 5.0,
    cache_path: Optional[str] = None,
    prediction_source: str = "pred",
    reward_source: str = "margin",
    reasoning_off: bool = False,
):
    task_dir = Path(u2_path)
    cache = _load_initial_score_cache(cache_path)
    new_cache_records = []
    candidates = []
    disease_dir = task_dir / "disease_diagnosis"
    tsv_files = sorted(disease_dir.rglob("*.tsv")) if disease_dir.exists() else []
    print("tsv_files:", tsv_files)

    for tsv in tqdm(tsv_files, desc="Scanning dataset"):
        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception:
            continue
        for ridx, row in df.iterrows():
            options, truth = resolve_row_options_and_truth(row)
            if len(options) < 2 or not truth:
                continue

            img_b64 = str(row.get("img_data", "")).strip()
            question_text = fill_prompt_placeholders(row.get("prompt", ""), row)
            if not img_b64 or not question_text:
                continue

            try:
                img = decode_base64_image(img_b64)
            except Exception:
                continue

            key = f"{tsv.relative_to(task_dir)}:{int(ridx)}"
            cached_rec = cache.get(key)
            try:
                scores_summary, cache_complete = _scores_for_row(
                    cached_rec,
                    options,
                    truth,
                    prediction_source,
                    reward_source,
                    reasoning_off,
                    vlm,
                    proc,
                    img,
                    system_prompt,
                    question_text,
                )
            except Exception:
                continue

            gap_val = float(scores_summary.get("pred_gap", scores_summary.get("gap", 0.0)))
            if cache_path and (cached_rec is None or not cache_complete):
                new_cache_records.append(
                    _cache_record(
                        key, tsv.relative_to(task_dir), ridx, truth, scores_summary, gap_val
                    )
                )

            pred = str(scores_summary.get("pred") or "")
            if not pred or truth is None or pred.lower() != truth.lower():
                continue

            if diff_threshold is None or gap_val < float(diff_threshold):
                candidates.append(
                    {
                        "image": img,
                        "ground_truth": truth,
                        "prompt": question_text,
                        "options": options,
                        "gap": gap_val,
                        "key": key,
                        "source_file": str(tsv.relative_to(task_dir)),
                        "row_index": int(ridx),
                        "scores_summary": scores_summary,
                    }
                )

    if cache_path and new_cache_records:
        try:
            append_jsonl_records(Path(cache_path), new_cache_records)
        except Exception:
            pass

    candidates.sort(key=lambda x: x["gap"])
    return candidates
