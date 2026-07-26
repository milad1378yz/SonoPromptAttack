import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from attack_core.attack_records import load_attacked_samples
from attack_core.model_loader import load_vlm
from attack_core.run_outputs import append_jsonl, append_text, write_csv, write_json
from attack_core.u2bench import decode_base64_image
from attack_core.vlm_scoring import compute_scores, format_option_scores


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _safe_ratio(num: int, den: int) -> Optional[float]:
    if not den:
        return None
    return float(num) / float(den)


def _sanitize_path_component(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return text or fallback


def _infer_source_attack_layout(attack_input: Path):
    parts = list(attack_input.resolve().parts)
    if "Trillium_results" in parts:
        idx = parts.index("Trillium_results")
        trailing = parts[idx + 1 :]
        if len(trailing) >= 3:
            source_llm = trailing[0]
            task_name = trailing[2]
            return source_llm, task_name

    if attack_input.is_dir():
        if attack_input.name.endswith("_sample_summaries") and attack_input.parent.name:
            return attack_input.parent.parent.name or attack_input.parent.name, attack_input.parent.name
        return attack_input.parent.name or attack_input.name, attack_input.name

    parent = attack_input.parent
    if parent.name.endswith("_sample_summaries"):
        grandparent = parent.parent
        source_llm = grandparent.parent.name if grandparent.parent.name else grandparent.name
        task_name = grandparent.name
        return source_llm, task_name
    return parent.name or attack_input.stem, attack_input.stem


def _default_output_dir(attack_input: Path, target_vlm_id: str, repo_root: Path) -> Path:
    safe_model = _sanitize_path_component(Path(target_vlm_id).name or str(target_vlm_id), "target_vlm")
    source_llm, task_name = _infer_source_attack_layout(attack_input)
    safe_source_llm = _sanitize_path_component(source_llm, "source_llm")
    safe_task_name = _sanitize_path_component(task_name, "task")
    return repo_root / "transferability" / "results" / safe_model / safe_source_llm / safe_task_name


def evaluate_transfer_sample(sample: dict, vlm, proc, prediction_source: str, reward_source: str):
    image = decode_base64_image(sample["img_data"])
    options = [str(o).strip() for o in sample["options"] if str(o).strip()]
    truth = str(sample["truth"])
    original_question = str(sample["original_question"])
    attacked_question = str(sample["attacked_question"])

    base_scores = compute_scores(
        vlm,
        proc,
        image,
        "",
        original_question,
        options,
        truth,
        prediction_source=prediction_source,
        reward_source=reward_source,
    )
    attacked_scores = compute_scores(
        vlm,
        proc,
        image,
        "",
        attacked_question,
        options,
        truth,
        prediction_source=prediction_source,
        reward_source=reward_source,
    )

    base_label = str(base_scores.get("pred") or "")
    attacked_label = str(attacked_scores.get("pred") or "")

    target_base_correct = bool(base_label) and base_label.lower() == truth.lower()
    target_attacked_fooled = bool(attacked_label) and attacked_label.lower() != truth.lower()
    strict_transfer_success = target_base_correct and target_attacked_fooled

    return {
        "base_scores": base_scores,
        "attacked_scores": attacked_scores,
        "base_label": base_label,
        "attacked_label": attacked_label,
        "target_base_correct": target_base_correct,
        "target_attacked_fooled": target_attacked_fooled,
        "strict_transfer_success": strict_transfer_success,
    }


def export_transfer_summaries(records: List[dict], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    aggregate_json = output_dir / "transferability_summary.json"
    aggregate_csv = output_dir / "transferability_summary.csv"

    written.append(write_json(aggregate_json, records))

    fieldnames = [
        "key",
        "tsv_file",
        "index",
        "actual_label",
        "source_label_after_attack",
        "source_prediction_source",
        "target_prediction_source",
        "target_vlm_id",
        "target_base_label",
        "target_attacked_label",
        "target_base_correct",
        "target_attacked_fooled",
        "strict_transfer_success",
        "original_question",
        "attacked_question",
        "target_base_scores_text",
        "target_attacked_scores_text",
    ]
    written.append(write_csv(aggregate_csv, records, fieldnames))

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["tsv_file"])].append(record)

    for tsv_file, rows in grouped.items():
        rel = Path(tsv_file)
        out_dir = output_dir / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{rel.stem}.transfer_summary.json"
        csv_path = out_dir / f"{rel.stem}.transfer_summary.csv"

        written.append(write_json(json_path, rows))

        written.append(write_csv(csv_path, rows, fieldnames))

    return written


def build_overview(
    records: List[dict],
    *,
    attack_input: Path,
    dataset_root: Path,
    target_vlm_id: str,
    prediction_source: str,
    successful_only: bool,
    skipped: int,
    seed: Optional[int] = None,
):
    total = len(records)
    target_base_correct = sum(1 for r in records if bool(r.get("target_base_correct")))
    any_transfer = sum(1 for r in records if bool(r.get("target_attacked_fooled")))
    strict_transfer = sum(1 for r in records if bool(r.get("strict_transfer_success")))
    overview = {
        "attack_input": str(attack_input),
        "dataset_root": str(dataset_root),
        "target_vlm_id": target_vlm_id,
        "prediction_source": prediction_source,
        "source_successful_only": bool(successful_only),
        "evaluated_samples": total,
        "skipped_samples": int(skipped),
        "target_base_correct_count": int(target_base_correct),
        "any_transfer_count": int(any_transfer),
        "strict_transfer_count": int(strict_transfer),
        "any_transfer_rate_over_evaluated": _safe_ratio(any_transfer, total),
        "strict_transfer_rate_over_evaluated": _safe_ratio(strict_transfer, total),
        "strict_transfer_rate_over_target_base_correct": _safe_ratio(strict_transfer, target_base_correct),
    }
    if seed is not None:
        overview["seed"] = int(seed)
    return overview


def parse_args(repo_root: Path):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack-input",
        type=str,
        required=True,
        help="Source attack run directory, attack_summary/pair_summary file, _resume_progress.jsonl file, or TextAttack CSV file/directory.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(repo_root / "dataset" / "u2-bench"),
        help="Root directory containing local TSV files.",
    )
    parser.add_argument(
        "--target-vlm-id",
        type=str,
        required=True,
        help="Second VLM to test transferability against.",
    )
    parser.add_argument(
        "--prediction-source",
        type=str,
        default="pred",
        choices=["pred", "real_pred"],
    )
    parser.add_argument("--reward-source", type=str, default="margin")
    parser.add_argument("--seed", type=int, default=765, help="Random seed for reproducibility.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--include-unsuccessful", action="store_true")
    return parser.parse_args()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args(repo_root)
    if args.seed is not None:
        _set_seed(args.seed)

    attack_input = Path(args.attack_input).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(attack_input, args.target_vlm_id, repo_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        Path(args.log_path).expanduser().resolve()
        if args.log_path
        else output_dir / "transferability_log.txt"
    )
    detail_log_path = log_path.parent / f"{log_path.stem}_details.jsonl"
    if log_path.exists():
        log_path.unlink()
    if detail_log_path.exists():
        detail_log_path.unlink()

    samples = load_attacked_samples(
        attack_input=attack_input,
        dataset_root=dataset_root,
        successful_only=not args.include_unsuccessful,
    )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    append_text(
        log_path,
        [
            f"Attack input: {attack_input}",
            f"Dataset root: {dataset_root}",
            f"Target VLM: {args.target_vlm_id}",
            f"Prediction source: {args.prediction_source}",
            f"Reward source: {args.reward_source}",
            f"Seed: {args.seed if args.seed is not None else '<unset>'}",
            f"Samples loaded: {len(samples)}",
            f"Source successful only: {not args.include_unsuccessful}",
        ]
    )

    vlm, proc = load_vlm(args.target_vlm_id)

    summary_records: List[dict] = []
    skipped = 0
    any_transfer = 0
    strict_transfer = 0
    target_base_correct_count = 0

    pbar = tqdm(samples, desc="Transferability")
    for sample in pbar:
        try:
            result = evaluate_transfer_sample(
                sample,
                vlm,
                proc,
                args.prediction_source,
                args.reward_source,
            )
        except Exception as exc:
            skipped += 1
            append_text(log_path, f"Skipping {sample['key']} due to runtime failure: {exc}")
            append_jsonl(detail_log_path, {"key": sample["key"], "status": "skipped", "error": str(exc)})
            pbar.set_description(
                f"[Base Correct / Strict Xfer / Any Fooled / Skipped / Total] "
                f"{target_base_correct_count} / {strict_transfer} / {any_transfer} / {skipped} / "
                f"{strict_transfer + (len(summary_records) - strict_transfer) + skipped}"
            )
            continue

        if result["target_base_correct"]:
            target_base_correct_count += 1
        if result["target_attacked_fooled"]:
            any_transfer += 1
        if result["strict_transfer_success"]:
            strict_transfer += 1

        record = {
            "key": sample["key"],
            "tsv_file": Path(sample["tsv_path"]).name,
            "index": sample["row_index"],
            "actual_label": sample["truth"],
            "source_label_after_attack": sample["label_after_attack"],
            "source_prediction_source": sample["prediction_source"],
            "target_prediction_source": args.prediction_source,
            "target_vlm_id": args.target_vlm_id,
            "target_base_label": result["base_label"],
            "target_attacked_label": result["attacked_label"],
            "target_base_correct": result["target_base_correct"],
            "target_attacked_fooled": result["target_attacked_fooled"],
            "strict_transfer_success": result["strict_transfer_success"],
            "original_question": sample["original_question"],
            "attacked_question": sample["attacked_question"],
            "target_base_scores_text": format_option_scores(result["base_scores"].get("scores")),
            "target_attacked_scores_text": format_option_scores(result["attacked_scores"].get("scores")),
        }
        summary_records.append(record)

        append_text(
            log_path,
            [
                f"Sample {sample['key']}: truth={sample['truth']} "
                f"target_base={result['base_label']} target_attack={result['attacked_label']} "
                f"base_correct={result['target_base_correct']} any_transfer={result['target_attacked_fooled']} strict_transfer={result['strict_transfer_success']}",
            ]
        )
        append_jsonl(
            detail_log_path,
            {
                "key": sample["key"],
                "status": "ok",
                "record": record,
                "target_base_scores": result["base_scores"],
                "target_attacked_scores": result["attacked_scores"],
            }
        )
        total_done = len(summary_records) + skipped
        pbar.set_description(
            f"[Base Correct / Strict Xfer / Any Fooled / Skipped / Total] "
            f"{target_base_correct_count} / {strict_transfer} / {any_transfer} / {skipped} / {total_done}"
        )

    written = export_transfer_summaries(summary_records, output_dir)
    overview = build_overview(
        summary_records,
        attack_input=attack_input,
        dataset_root=dataset_root,
        target_vlm_id=args.target_vlm_id,
        prediction_source=args.prediction_source,
        successful_only=not args.include_unsuccessful,
        skipped=skipped,
        seed=args.seed,
    )
    overview_path = output_dir / "transferability_overview.json"
    write_json(overview_path, overview)

    append_text(
        log_path,
        [
            "",
            "Transferability overview:",
            json.dumps(overview, ensure_ascii=False, indent=2),
            "Written:",
            *[str(path) for path in written],
            str(overview_path),
            f"Details JSONL: {detail_log_path}",
        ]
    )
    print(
        f"Transferability complete. base_correct={overview['target_base_correct_count']} "
        f"strict_transfer={overview['strict_transfer_count']} "
        f"any_transfer={overview['any_transfer_count']} skipped={overview['skipped_samples']} "
        f"overview={overview_path}"
    )


if __name__ == "__main__":
    main()
