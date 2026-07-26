"""
SmoothVLM defense adapted from SmoothLLM.

This file reuses the core smoothing idea from:
https://github.com/arobey1/smooth-llm
SmoothLLM is MIT-licensed. This adaptation targets VLM classification
workflows in this repository rather than text-only jailbreak detection.
"""

import argparse
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from attack_core.attack_records import load_attacked_samples
from attack_core.model_loader import load_vlm
from attack_core.reproducibility import seed_everything
from attack_core.run_outputs import append_jsonl, append_text, write_csv, write_json
from attack_core.u2bench import decode_base64_image
from attack_core.vlm_scoring import compute_scores, format_option_scores


class Perturbation:
    """Base class for random perturbations."""

    def __init__(self, q: int):
        self.q = q
        self.alphabet = string.ascii_letters + string.digits + string.punctuation + " "

    def _num_edits(self, s: str) -> int:
        if not s:
            return 0
        return max(1, int(len(s) * self.q / 100))


class RandomSwapPerturbation(Perturbation):
    """Random character replacement perturbation."""

    def __call__(self, s: str) -> str:
        if not s:
            return s
        list_s = list(s)
        k = min(len(s), self._num_edits(s))
        for i in random.sample(range(len(s)), k):
            list_s[i] = random.choice(self.alphabet)
        return "".join(list_s)


class RandomPatchPerturbation(Perturbation):
    """Random contiguous patch replacement perturbation."""

    def __call__(self, s: str) -> str:
        if not s:
            return s
        substring_width = min(len(s), self._num_edits(s))
        max_start = max(0, len(s) - substring_width)
        start_index = random.randint(0, max_start)
        sampled_chars = "".join(random.choice(self.alphabet) for _ in range(substring_width))
        list_s = list(s)
        list_s[start_index : start_index + substring_width] = sampled_chars
        return "".join(list_s)


class RandomInsertPerturbation(Perturbation):
    """Random character insertion perturbation."""

    def __call__(self, s: str) -> str:
        if not s:
            return s
        list_s = list(s)
        k = min(len(s), self._num_edits(s))
        for i in sorted(random.sample(range(len(s)), k), reverse=True):
            list_s.insert(i, random.choice(self.alphabet))
        return "".join(list_s)


PERTURBATION_TYPES = {
    "RandomSwapPerturbation": RandomSwapPerturbation,
    "RandomPatchPerturbation": RandomPatchPerturbation,
    "RandomInsertPerturbation": RandomInsertPerturbation,
}


def _prediction_support_score(summary: dict, label: str) -> float:
    if not label:
        return float("-inf")
    scores = summary.get("scores") or {}
    if label in scores:
        try:
            return float(scores[label])
        except (TypeError, ValueError):
            return float("-inf")
    return float("-inf")


def choose_majority_prediction(results: List[dict]) -> Tuple[str, List[dict]]:
    if not results:
        return "", []
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for rec in results:
        label = str(rec["scores"].get("pred") or "")
        grouped[label].append(rec)

    if not grouped:
        return "", []

    best_label = ""
    best_group: List[dict] = []
    best_count = -1
    best_mean_score = float("-inf")
    for label, group in grouped.items():
        count = len(group)
        mean_score = sum(_prediction_support_score(g["scores"], label) for g in group) / max(1, count)
        if count > best_count or (count == best_count and mean_score > best_mean_score):
            best_label = label
            best_group = group
            best_count = count
            best_mean_score = mean_score
    return best_label, best_group


class SmoothVLMDefense:
    def __init__(
        self,
        vlm,
        proc,
        pert_type: str,
        pert_pct: int,
        num_copies: int,
        system_prompt: str = "",
    ):
        self.vlm = vlm
        self.proc = proc
        self.num_copies = num_copies
        self.system_prompt = system_prompt
        self.perturbation_fn = PERTURBATION_TYPES[pert_type](q=pert_pct)

    def __call__(self, sample: dict) -> dict:
        image = decode_base64_image(sample["img_data"])
        base_scores = compute_scores(
            self.vlm,
            self.proc,
            image,
            self.system_prompt,
            sample["attacked_question"],
            sample["options"],
            sample["truth"],
        )

        copy_results = []
        for copy_index in range(self.num_copies):
            question = (
                self.perturbation_fn(sample["editable_prompt"])
                + sample["frozen_suffix"]
            )
            scores = compute_scores(
                self.vlm,
                self.proc,
                image,
                self.system_prompt,
                question,
                sample["options"],
                sample["truth"],
            )
            copy_results.append(
                {
                    "copy_index": copy_index,
                    "question": question,
                    "scores": scores,
                }
            )

        defended_label, majority_group = choose_majority_prediction(copy_results)
        defended_scores = None
        representative_question = sample["attacked_question"]
        if majority_group:
            defended_scores = max(
                majority_group,
                key=lambda rec: _prediction_support_score(rec["scores"], defended_label),
            )["scores"]
            representative_question = max(
                majority_group,
                key=lambda rec: _prediction_support_score(rec["scores"], defended_label),
            )["question"]

        truth = sample["truth"]
        base_label = str(base_scores.get("pred") or "")
        base_attack_success = bool(base_label) and base_label.lower() != str(truth).lower()
        defended_attack_success = bool(defended_label) and defended_label.lower() != str(truth).lower()
        majority_count = len(majority_group)

        return {
            "base_scores": base_scores,
            "base_label": base_label,
            "base_attack_success": base_attack_success,
            "defended_label": defended_label,
            "defended_scores": defended_scores,
            "defended_attack_success": defended_attack_success,
            "majority_count": majority_count,
            "majority_fraction": (majority_count / self.num_copies) if self.num_copies else 0.0,
            "representative_question": representative_question,
            "copies": copy_results,
        }


def _default_output_dir(attack_input: Path, vlm_id: str, repo_root: Path) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(vlm_id).name or str(vlm_id))
    safe_attack = re.sub(r"[^A-Za-z0-9._-]+", "_", attack_input.name)
    return repo_root / "defence" / "results" / safe_model / safe_attack


def export_results(records: List[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "smooth_vlm_summary.json"
    csv_path = output_dir / "smooth_vlm_summary.csv"
    write_json(json_path, records)

    fieldnames = [
        "key",
        "tsv_file",
        "index",
        "actual_label",
        "prediction_source",
        "base_label",
        "label_after_attack",
        "defended_label",
        "attack_success_before_defense",
        "attack_success_after_defense",
        "majority_count",
        "majority_fraction",
        "num_copies",
        "perturbation_type",
        "perturbation_pct",
        "original_question",
        "attacked_question",
        "representative_question",
        "base_scores_text",
        "defended_scores_text",
    ]
    write_csv(csv_path, records, fieldnames)

    return [json_path, csv_path]


def parse_args(repo_root: Path):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack-input",
        type=str,
        required=True,
        help="Attack run directory, _resume_progress.jsonl file, or attack_summary.json file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(repo_root / "dataset" / "u2-bench"),
        help="Root directory containing the local u2-bench TSV files.",
    )
    parser.add_argument("--vlm-id", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--num-copies", type=int, default=10)
    parser.add_argument(
        "--perturbation-type",
        type=str,
        default="RandomSwapPerturbation",
        choices=["RandomSwapPerturbation", "RandomPatchPerturbation", "RandomInsertPerturbation"],
    )
    parser.add_argument("--perturbation-pct", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-unsuccessful", action="store_true")
    return parser.parse_args()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args(repo_root)
    seed_everything(args.seed)

    attack_input = Path(args.attack_input).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir(attack_input, args.vlm_id, repo_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        Path(args.log_path).expanduser().resolve()
        if args.log_path
        else output_dir / "smooth_vlm_log.txt"
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
            f"VLM: {args.vlm_id}",
            f"Smooth copies: {args.num_copies}",
            f"Perturbation: {args.perturbation_type} ({args.perturbation_pct}%)",
            f"Samples loaded: {len(samples)}",
        ]
    )

    vlm, proc = load_vlm(args.vlm_id)
    defense = SmoothVLMDefense(
        vlm=vlm,
        proc=proc,
        pert_type=args.perturbation_type,
        pert_pct=args.perturbation_pct,
        num_copies=args.num_copies,
    )

    summary_records = []
    recovered = 0
    still_fooled = 0
    skipped = 0

    pbar = tqdm(samples, desc="SmoothVLM")
    for sample in pbar:
        try:
            result = defense(sample)
        except Exception as exc:
            skipped += 1
            append_text(log_path, f"Skipping {sample['key']} due to defense/runtime failure: {exc}")
            append_jsonl(detail_log_path, {"key": sample["key"], "status": "skipped", "error": str(exc)})
            pbar.set_description(
                f"[Recovered / Still Fooled / Skipped / Total] {recovered} / {still_fooled} / {skipped} / {recovered + still_fooled + skipped}"
            )
            continue

        if result["base_attack_success"] and not result["defended_attack_success"]:
            recovered += 1
        else:
            still_fooled += 1

        defended_scores = result["defended_scores"] or {}
        record = {
            "key": sample["key"],
            "tsv_file": Path(sample["tsv_path"]).name,
            "index": sample["row_index"],
            "actual_label": sample["truth"],
            "prediction_source": "pred",
            "base_label": result["base_label"],
            "label_after_attack": sample["label_after_attack"] or result["base_label"],
            "defended_label": result["defended_label"],
            "attack_success_before_defense": result["base_attack_success"],
            "attack_success_after_defense": result["defended_attack_success"],
            "majority_count": result["majority_count"],
            "majority_fraction": result["majority_fraction"],
            "num_copies": args.num_copies,
            "perturbation_type": args.perturbation_type,
            "perturbation_pct": args.perturbation_pct,
            "original_question": sample["original_question"],
            "attacked_question": sample["attacked_question"],
            "representative_question": result["representative_question"],
            "base_scores_text": format_option_scores(result["base_scores"].get("scores")),
            "defended_scores_text": format_option_scores(defended_scores.get("scores")),
        }
        summary_records.append(record)
        append_text(
            log_path,
            [
                f"Sample {sample['key']}: truth={sample['truth']} base={result['base_label']} defended={result['defended_label']} "
                f"base_attack={result['base_attack_success']} defended_attack={result['defended_attack_success']} "
                f"majority={result['majority_count']}/{args.num_copies}",
            ]
        )
        append_jsonl(
            detail_log_path,
            {
                "key": sample["key"],
                "status": "ok",
                "record": record,
                "copies": [
                    {
                        "copy_index": c["copy_index"],
                        "prediction": str(c["scores"].get("pred") or ""),
                        "question": c["question"],
                    }
                    for c in result["copies"]
                ],
            }
        )
        pbar.set_description(
            f"[Recovered / Still Fooled / Skipped / Total] {recovered} / {still_fooled} / {skipped} / {recovered + still_fooled + skipped}"
        )

    written = export_results(summary_records, output_dir)
    append_text(
        log_path,
        [
            "",
            f"SmoothVLM summary: recovered={recovered} still_fooled={still_fooled} skipped={skipped} total={recovered + still_fooled + skipped}",
            "Written:",
            *[str(path) for path in written],
            f"Details JSONL: {detail_log_path}",
        ]
    )
    print(
        f"SmoothVLM complete. recovered={recovered} still_fooled={still_fooled} skipped={skipped} "
        f"summary={written[0]}"
    )


if __name__ == "__main__":
    main()
