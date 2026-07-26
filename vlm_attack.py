import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm

from attack_core.mcts_search import MCTS
from attack_core.model_loader import load_llm, load_vlm
from attack_core.proposer import apply_minimal_replacement, llm_suggest_pairs
from attack_core.reproducibility import seed_everything
from attack_core.run_outputs import (
    append_jsonl,
    append_text,
    write_csv,
    write_json,
)
from attack_core.search_baselines import (
    BeamSearch,
    GeneticSearch,
    GreedySearch,
    RandomSearch,
)
from attack_core.u2bench import (
    find_lowconf_correct_samples_dataset_id,
    find_lowconf_correct_samples_dataset_path,
    load_candidate_samples_from_results_cache,
)
from attack_core.vlm_scoring import attach_reward_fields, compute_scores, format_option_scores


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _sample_source_info(sample):
    source_file = str(sample.get("source_file") or "").strip()
    row_index = sample.get("row_index")
    if source_file and row_index is not None:
        try:
            return source_file, int(row_index)
        except Exception:
            return source_file, row_index

    key = str(sample.get("key") or "")
    if ":" in key:
        source_file, row_text = key.rsplit(":", 1)
        try:
            row_index = int(row_text)
        except Exception:
            row_index = row_text
        return source_file, row_index
    return source_file or "unknown.tsv", row_index


def _transition_text(transitions):
    parts = []
    for item in transitions:
        prev = str(item.get("prev", ""))
        new = str(item.get("new", ""))
        parts.append(f"{prev} -> {new}")
    return " | ".join(parts)


def _load_resume_records(progress_path: Path):
    records = []
    seen_keys = set()
    if not progress_path.exists():
        return records, seen_keys

    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = str(rec.get("key") or "")
                if not key or key in seen_keys:
                    continue
                records.append(rec)
                seen_keys.add(key)
    except Exception:
        return [], set()
    return records, seen_keys


def _load_completed_keys_from_tree_log(tree_log_path: Path):
    completed_keys = set()
    if not tree_log_path or not tree_log_path.exists():
        return completed_keys

    try:
        with open(tree_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = str(rec.get("key") or "")
                if key:
                    completed_keys.add(key)
    except Exception:
        return set()
    return completed_keys


def _build_summary_record(
    sample,
    *,
    truth,
    base_question,
    final_question,
    transitions,
    base_scores,
    final_scores,
    success,
    search_mode,
    evaluations,
):
    source_file, row_index = _sample_source_info(sample)
    final_scores = final_scores or {}
    base_scores = base_scores or {}
    final_pred = str(final_scores.get("pred") or "")
    return {
        "key": str(sample.get("key", "")),
        "tsv_file": source_file,
        "index": row_index,
        "search_mode": search_mode,
        "evaluations": int(evaluations),
        "success": bool(success),
        "attack_success": bool(success),
        "prediction_source": "pred",
        "reward_source": "margin",
        "real_label": truth,
        "actual_label": truth,
        "original_question": base_question,
        "final_question": final_question,
        "transition_count": len(transitions),
        "transitions": transitions,
        "transitions_text": _transition_text(transitions),
        "base_pred": base_scores.get("pred"),
        "base_runner_up": base_scores.get("runner_up"),
        "base_gap": _safe_float(base_scores.get("pred_gap", base_scores.get("gap"))),
        "base_reward": _safe_float(base_scores.get("reward", base_scores.get("margin"))),
        "final_pred": final_scores.get("pred"),
        "final_runner_up": final_scores.get("runner_up"),
        "final_reward": _safe_float(final_scores.get("reward", final_scores.get("margin"))),
        "chosen_pred": final_pred,
        "label_after_attack": final_pred,
    }


def _export_attack_summaries(records, summary_dir: Path, summary_format: str):
    if not records or summary_format == "none":
        return []

    summary_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for rec in records:
        tsv_file = str(rec.get("tsv_file") or "unknown.tsv")
        grouped.setdefault(tsv_file, []).append(rec)

    written_paths = []
    for tsv_file, rows in grouped.items():
        rel_path = Path(tsv_file)
        out_dir = summary_dir / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{rel_path.stem}.attack_summary.json"
        csv_path = out_dir / f"{rel_path.stem}.attack_summary.csv"

        if summary_format in {"json", "both"}:
            written_paths.append(write_json(json_path, rows))

        if summary_format in {"csv", "both"}:
            csv_rows = []
            for rec in rows:
                csv_rec = dict(rec)
                csv_rec["transitions"] = json.dumps(rec.get("transitions", []), ensure_ascii=False)
                csv_rows.append(csv_rec)
            fieldnames = [
                "key",
                "tsv_file",
                "index",
                "search_mode",
                "evaluations",
                "success",
                "attack_success",
                "prediction_source",
                "reward_source",
                "real_label",
                "actual_label",
                "original_question",
                "final_question",
                "transition_count",
                "transitions_text",
                "transitions",
                "base_pred",
                "base_runner_up",
                "base_gap",
                "base_reward",
                "final_pred",
                "final_runner_up",
                "final_reward",
                "elapsed_seconds",
                "chosen_pred",
                "label_after_attack",
            ]
            written_paths.append(write_csv(csv_path, csv_rows, fieldnames))

    return written_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Run an MCTS-based prompt attack against a VLM.")
    parser.add_argument(
        "--vlm-id",
        default="google/medgemma-4b-it",
        help="Vision-language model to attack (default: google/medgemma-4b-it).",
    )
    parser.add_argument(
        "--llm-id",
        default="qwen/qwen3-30b-a3b-instruct-2507",
        help="LLM to try for proposing edits.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="api key for the LLM model.",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Whether to use the API for the LLM model.",
    )
    parser.add_argument(
        "--llm-api-provider",
        choices=["auto", "openrouter", "openai", "gemini", "openai_compatible"],
        default="auto",
        help="API backend for the proposer LLM. Use openai_compatible for a local llama.cpp/vLLM-style server.",
    )
    parser.add_argument(
        "--llm-api-base-url",
        default=None,
        help="Base URL for an OpenAI-compatible local server, e.g. http://127.0.0.1:8080/v1",
    )
    parser.add_argument(
        "--llm-quantization",
        choices=["auto", "fp16", "4bit"],
        default="auto",
        help="Quantization mode for the LLM (auto enables 4-bit for very large models).",
    )
    parser.add_argument(
        "--log-path",
        default="attack_log.txt",
        help="Path to append run logs (default: attack_log.txt).",
    )
    parser.add_argument(
        "--dataset-id",
        default="DolphinAI/u2-bench",
        help="Dataset repo id to download from HuggingFace (default: DolphinAI/u2-bench).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Local dataset path. Pass either a single TSV file or a directory that will be scanned recursively for *.tsv files.",
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Directory for per-TSV attack summaries. Default: <log-dir>/<log-stem>_sample_summaries.",
    )
    parser.add_argument(
        "--summary-format",
        choices=["none", "json", "csv", "both"],
        default="both",
        help="Per-TSV summary export format for attacked samples.",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional path for the initial score cache (default: cache/initial_scores_u2bench_<VLM>.jsonl).",
    )
    parser.add_argument(
        "--candidate-cache-source",
        default=None,
        help="Optional previous run directory or summary file to rebuild the candidate list from cached attack summaries instead of rescanning the dataset.",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=None,
        help="Maximum pred-runner gap to keep a sample (None keeps all correct samples).",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt to prepend when querying the VLM.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["mcts", "ga", "random", "greedy", "beam"],
        default="mcts",
        help="Search strategy: tree search (mcts), genetic search (ga), random, greedy, or beam.",
    )
    parser.add_argument(
        "--ga-max-steps",
        type=int,
        default=50,
        help="Maximum accepted edits during genetic search.",
    )
    parser.add_argument(
        "--ga-generations-per-step",
        type=int,
        default=3,
        help="LLM proposal batches per genetic-search step.",
    )
    parser.add_argument(
        "--ga-attempt-multiplier",
        type=int,
        default=8,
        help="Safety multiplier for retries when no improving edits are found.",
    )
    parser.add_argument(
        "--mcts-max-depth",
        type=int,
        default=8,
        help="Max depth for the MCTS search tree.",
    )
    parser.add_argument(
        "--mcts-exploration",
        type=float,
        default=1.4,
        help="Exploration constant for UCT.",
    )
    parser.add_argument(
        "--max-vlm-evaluations",
        "--mcts-max-iterations",
        dest="max_vlm_evaluations",
        type=int,
        default=80,
        help="Maximum scorer evaluations during search (shared budget for all search modes).",
    )
    parser.add_argument(
        "--mcts-max-children",
        type=int,
        default=3,
        help="Maximum children per expansion.",
    )
    parser.add_argument(
        "--proposal-attempts",
        type=int,
        default=2,
        help="Number of LLM generations when proposing edit pairs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic runs (applied to random/np/torch; default: 0).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on how many candidate samples to attack (e.g. 10 for a quick flow check).",
    )
    parser.add_argument(
        "--reasoning-off",
        action="store_true",
        help="Whether to disable reasoning features in Gemma-4-26B",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    vlm_id = args.vlm_id
    llm_id = args.llm_id
    log_path = Path(args.log_path)
    tree_log_path = log_path.parent / f"{log_path.stem}_trees.jsonl"

    print(f"Loading VLM: {vlm_id}")
    vlm, vlm_proc = load_vlm(vlm_id)
    if not args.use_api:
        print(f"Loading LLM: {llm_id}")
        llm, llm_tok = load_llm(llm_id, quantization=args.llm_quantization)

    run_header = [
        f"=== Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===",
        f"VLM: {vlm_id}",
        f"LLM: {llm_id}",
        "Prediction source: pred",
        "Reward source: margin",
        f"Seed: {args.seed}",
    ]
    append_text(log_path, run_header)

    if args.dataset_path:
        u2_path = args.dataset_path
        if not Path(u2_path).exists():
            raise FileNotFoundError(f"Dataset path not found: {u2_path}")
        print(f"Using local dataset path: {u2_path}")
    else:
        print(f"Downloading dataset: {args.dataset_id}")
        u2_path = snapshot_download(repo_id=args.dataset_id, repo_type="dataset")
    # Add a model- and dataset-specific cache for initial scores
    if args.cache_path:
        cache_path = args.cache_path
    else:
        safe_vlm = re.sub(r"[^a-zA-Z0-9_.-]+", "_", vlm_id)
        cache_path = str(Path("cache") / f"initial_scores_u2bench_{safe_vlm}.jsonl")

    summary_dir = (
        Path(args.summary_dir)
        if args.summary_dir
        else log_path.parent / f"{log_path.stem}_sample_summaries"
    )
    progress_path = summary_dir / "_resume_progress.jsonl"
    complete_marker = summary_dir / "_COMPLETE"
    if complete_marker.exists():
        print(f"Complete marker already exists; nothing to resume: {complete_marker}")
        append_text(
            log_path, f"Complete marker already exists; nothing to resume: {complete_marker}"
        )
        return

    prior_records, completed_keys = _load_resume_records(progress_path)
    if not completed_keys:
        completed_keys = _load_completed_keys_from_tree_log(tree_log_path)
        if completed_keys:
            tree_resume_msg = (
                f"Resume fallback: found {len(completed_keys)} completed sample key(s) "
                f"in tree log {tree_log_path}."
            )
            print(tree_resume_msg)
            append_text(log_path, tree_resume_msg)

    gap_threshold = args.gap_threshold
    print(
        f"Selecting correctly-predicted, low-confidence samples "
        f"(pred-runner gap < {gap_threshold})..."
    )
    if args.candidate_cache_source:
        print(f"Loading candidate list from cached summaries: {args.candidate_cache_source}")
        lowconf = load_candidate_samples_from_results_cache(
            dataset_path=u2_path,
            candidate_cache_source=args.candidate_cache_source,
        )
    elif args.dataset_path:
        lowconf = find_lowconf_correct_samples_dataset_path(
            u2_path,
            vlm=vlm,
            proc=vlm_proc,
            system_prompt=args.system_prompt,
            diff_threshold=gap_threshold,
            cache_path=cache_path,
            reasoning_off=args.reasoning_off,
            completed_keys=completed_keys,
        )
    else:
        lowconf = find_lowconf_correct_samples_dataset_id(
            u2_path,
            vlm=vlm,
            proc=vlm_proc,
            system_prompt=args.system_prompt,
            diff_threshold=gap_threshold,
            cache_path=cache_path,
            reasoning_off=args.reasoning_off,
            completed_keys=completed_keys,
        )

    if completed_keys:
        lowconf = [sample for sample in lowconf if str(sample.get("key", "")) not in completed_keys]
        resume_msg = (
            f"Resume enabled: found {len(completed_keys)} completed sample(s) in "
            f"{progress_path}; {len(lowconf)} sample(s) remain."
        )
        print(resume_msg)
        append_text(log_path, resume_msg)

    if not lowconf:
        if completed_keys:
            msg = (
                "No remaining low-confidence correct samples after resume filtering; "
                f"{len(completed_keys)} completed key(s) were already recorded."
            )
            print(msg)
            append_text(log_path, msg)
            _export_attack_summaries(prior_records, summary_dir, args.summary_format)
            write_json(
                complete_marker,
                {
                    "completed_at": datetime.now().isoformat(),
                    "total_candidates": len(prior_records),
                    "total_records": len(prior_records),
                },
            )
            print(f"Complete marker written: {complete_marker}")
            append_text(log_path, f"Complete marker written: {complete_marker}")
            return
        print("No low-confidence correct samples found. Falling back to first valid sample.")
        raise RuntimeError("No valid samples found for attack.")

    remaining_candidates = len(lowconf)
    if args.max_samples is not None:
        max_samples = max(0, int(args.max_samples))
        lowconf = lowconf[:max_samples]
        print(
            f"Found {remaining_candidates} remaining candidates. "
            f"Limiting run to {len(lowconf)} sample(s) due to --max-samples={args.max_samples}."
        )
        append_text(
            log_path,
            f"Found {remaining_candidates} remaining candidates. "
            f"Limiting run to {len(lowconf)} sample(s) due to --max-samples={args.max_samples}.",
        )
    else:
        print(f"Found {remaining_candidates} candidates. Will attack all (sorted by gap).")

    selected_candidate_count = len(lowconf)

    success_count = 0
    failed_count = 0
    skipped_count = 0
    attack_records = list(prior_records)

    def attack_one(sample):
        nonlocal success_count
        image = sample["image"]
        truth = sample["ground_truth"]
        base_question = sample["prompt"].strip()
        system_prompt = args.system_prompt

        options = [str(o).strip() for o in sample.get("options", []) if str(o).strip()]

        def _score_question(question_text):
            return compute_scores(
                vlm,
                vlm_proc,
                image,
                system_prompt,
                question_text,
                options,
                truth,
                reasoning_off=args.reasoning_off,
            )

        def _label_summary(scores):
            chosen_pred = str(scores.get("pred") or "") or "-"
            return (
                f"real_label={truth} pred={scores.get('pred', '')} "
                f"runner_up={scores.get('runner_up') or '-'} "
                f"chosen_pred={chosen_pred} prediction_source=pred"
            )

        def _transition_records_from_list(items):
            records = []
            for idx, item in enumerate(items or [], start=1):
                records.append(
                    {
                        "step": idx,
                        "prev": str(item.get("prev", "")),
                        "new": str(item.get("new", "")),
                        "delta_truth": _safe_float(item.get("delta")),
                    }
                )
            return records

        # Baseline metrics for the initial question (reuse cached summary when available)
        base_scores = sample.get("scores_summary")
        if not base_scores:
            base_scores = _score_question(base_question)
        else:
            base_scores = dict(base_scores)
        base_scores = attach_reward_fields(base_scores)
        initial_gap = sample.get("gap")
        if initial_gap is None:
            initial_gap = base_scores.get("pred_gap", base_scores.get("gap", float("nan")))

        append_text(
            log_path,
            [
                f"Real label: {truth}",
                "Base question:",
                base_question.strip(),
                f"Initial gap (pred-runner): {initial_gap:.3f}",
                "--- Steps ---",
            ],
        )

        score_text = format_option_scores(base_scores.get("scores", {}))
        base_line = (
            f"Base: real_label={truth} pred={base_scores['pred']} "
            f"runner_up={base_scores.get('runner_up', '-')}"
            f" gap={base_scores.get('pred_gap', base_scores.get('gap', 0.0)):.3f} "
            f"truth_score={base_scores.get('truth_score', float('nan')):.3f} "
            f"margin={base_scores.get('margin', float('nan')):.3f} "
            f"reward={base_scores.get('reward', float('nan')):.3f} "
            f"reward_source={base_scores.get('reward_source', 'margin')}"
        )
        if score_text:
            base_line += f" scores=[{score_text}]"
        print(base_line)
        append_text(log_path, base_line)

        proposal_model = args.llm_id if args.use_api else llm
        proposal_tokenizer = None if args.use_api else llm_tok
        pair_proposer = lambda question_text, trans_history: llm_suggest_pairs(
            proposal_model,
            trans_history,
            question_text,
            blocked_tokens=options,
            attempts=args.proposal_attempts,
            api_key=args.api_key,
            api_provider=args.llm_api_provider,
            api_base_url=args.llm_api_base_url or "",
            use_api=args.use_api,
            tok=proposal_tokenizer,
            log_path=log_path,
        )

        if args.search_mode in {"random", "greedy", "beam"}:
            search_classes = {
                "random": RandomSearch,
                "greedy": GreedySearch,
                "beam": BeamSearch,
            }
            search_cls = search_classes[args.search_mode]
            searcher = search_cls(
                scorer=_score_question,
                proposer=pair_proposer,
                apply_edit=apply_minimal_replacement,
                truth_label=truth,
                max_iterations=args.max_vlm_evaluations,
                max_depth=args.mcts_max_depth,
                max_children_per_expand=args.mcts_max_children,
            )
            desc = args.search_mode.upper()
            with tqdm(total=args.max_vlm_evaluations, desc=desc, leave=False) as pbar:
                search_result = searcher.search(base_question, base_scores, progress=pbar)

            final_scores = search_result.get("scores", {})
            final_question = search_result.get("question", base_question)
            transitions = _transition_records_from_list(search_result.get("transitions", []))
            label_line = _label_summary(final_scores)
            method_name = args.search_mode.capitalize()
            evaluation_line = (
                f"{method_name} scorer evaluations: "
                f"{search_result['evaluations']}/{args.max_vlm_evaluations}."
            )
            print(evaluation_line)
            append_text(log_path, evaluation_line)
            if search_result.get("success"):
                trans_lines = [f"Transitions ({args.search_mode}):"]
                for item in transitions:
                    d = float(item.get("delta_truth") or 0.0)
                    sign = "+" if d >= 0 else ""
                    trans_lines.append(
                        f"  {item['step']}) {item.get('prev','')} -> {item.get('new','')} "
                        f"(Δtruth {sign}{d:.3f})"
                    )
                print(f"\nSuccess: VLM misclassified with {method_name} search.")
                print("\n".join(trans_lines))
                print("\nFinal question:")
                print(final_question)
                print(label_line)
                append_text(
                    log_path,
                    [
                        f"Result: misclassified via {method_name}.",
                        *trans_lines,
                        "Final question:",
                        final_question,
                        label_line,
                        "--- End Run ---",
                        "",
                    ],
                )
                success_count += 1
                return _build_summary_record(
                    sample,
                    truth=truth,
                    base_question=base_question,
                    final_question=final_question,
                    transitions=transitions,
                    base_scores=base_scores,
                    final_scores=final_scores,
                    success=True,
                    search_mode=args.search_mode,
                    evaluations=search_result["evaluations"],
                )

            final_margin = float(final_scores.get("margin", float("nan")))
            final_pred = final_scores.get("pred", "")
            done_line = (
                f"{method_name} search finished without misclassification. "
                f"real_label={truth} final_pred={final_pred} "
                f"margin={final_margin:.3f} "
                f"reward={final_scores.get('reward', float('nan')):.3f} "
                f"reward_source={final_scores.get('reward_source', 'margin')} "
                f"evaluations={search_result['evaluations']}"
            )
            print(done_line)
            print("\nFinal question:")
            print(final_question)
            append_text(
                log_path,
                [
                    done_line,
                    "Final question:",
                    final_question,
                    f"Result: done ({method_name}).",
                    "--- End Run ---",
                    "",
                ],
            )
            return _build_summary_record(
                sample,
                truth=truth,
                base_question=base_question,
                final_question=final_question,
                transitions=transitions,
                base_scores=base_scores,
                final_scores=final_scores,
                success=False,
                search_mode=args.search_mode,
                evaluations=search_result["evaluations"],
            )

        if args.search_mode == "ga":
            ga_searcher = GeneticSearch(
                scorer=_score_question,
                proposer=pair_proposer,
                apply_edit=apply_minimal_replacement,
                truth_label=truth,
                max_steps=args.ga_max_steps,
                generations_per_step=args.ga_generations_per_step,
                attempt_multiplier=args.ga_attempt_multiplier,
                max_evaluations=args.max_vlm_evaluations,
            )
            with tqdm(total=args.ga_max_steps, desc="Genetic search", leave=False) as pbar:
                ga_result = ga_searcher.search(base_question, base_scores, progress=pbar)
            evaluation_line = (
                "Genetic-search scorer evaluations: "
                f"{ga_result['evaluations']}/{args.max_vlm_evaluations}."
            )
            print(evaluation_line)
            append_text(log_path, evaluation_line)
            if ga_result["success"]:
                trans_lines = ["Transitions (genetic search):"]
                for i, t in enumerate(ga_result["transitions"], start=1):
                    d = float(t.get("delta", 0.0))
                    sign = "+" if d >= 0 else ""
                    trans_lines.append(
                        f"  {i}) {t.get('prev','')} -> {t.get('new','')} (Δtruth {sign}{d:.3f})"
                    )
                final_scores = ga_result.get("scores", {})
                final_question = ga_result.get("question", base_question)
                label_line = _label_summary(final_scores)
                print("\nSuccess: VLM misclassified with genetic search.")
                print("\n".join(trans_lines))
                print("\nFinal question:")
                print(final_question)
                print(label_line)
                append_text(
                    log_path,
                    [
                        (
                            "Result: misclassified via genetic search "
                            f"after {ga_result['evaluations']} scorer evaluations."
                        ),
                        *trans_lines,
                        "Final question:",
                        final_question,
                        label_line,
                        "--- End Run ---",
                        "",
                    ],
                )
                success_count += 1
                return _build_summary_record(
                    sample,
                    truth=truth,
                    base_question=base_question,
                    final_question=final_question,
                    transitions=_transition_records_from_list(
                        ga_result.get("transitions", [])
                    ),
                    base_scores=base_scores,
                    final_scores=final_scores,
                    success=True,
                    search_mode="ga",
                    evaluations=ga_result["evaluations"],
                )
            else:
                final_scores = ga_result.get("scores", {})
                final_margin = float(final_scores.get("margin", float("nan")))
                final_pred = final_scores.get("pred", "")
                final_question = ga_result.get("question", base_question)
                done_line = (
                    f"Genetic search finished without misclassification. "
                    f"real_label={truth} final_pred={final_pred} "
                    f"margin={final_margin:.3f} "
                    f"reward={final_scores.get('reward', float('nan')):.3f} "
                    f"reward_source={final_scores.get('reward_source', 'margin')} "
                    f"steps={len(ga_result.get('history', []))} "
                    f"evaluations={ga_result['evaluations']}"
                )
                print(done_line)
                print("\nFinal question:")
                print(final_question)
                append_text(
                    log_path,
                    [
                        done_line,
                        "Final question:",
                        final_question,
                        "Result: done (genetic search).",
                        "--- End Run ---",
                        "",
                    ],
                )
                return _build_summary_record(
                    sample,
                    truth=truth,
                    base_question=base_question,
                    final_question=final_question,
                    transitions=_transition_records_from_list(
                        ga_result.get("transitions", [])
                    ),
                    base_scores=base_scores,
                    final_scores=final_scores,
                    success=False,
                    search_mode="ga",
                    evaluations=ga_result["evaluations"],
                )

        searcher = MCTS(
            scorer=_score_question,
            proposer=pair_proposer,
            apply_edit=apply_minimal_replacement,
            truth_label=truth,
            max_depth=args.mcts_max_depth,
            exploration=args.mcts_exploration,
            max_iterations=args.max_vlm_evaluations,
            max_children_per_expand=args.mcts_max_children,
        )

        print(f"Starting MCTS attack search (max depth {args.mcts_max_depth})...\n")
        with tqdm(total=args.max_vlm_evaluations, desc="MCTS", leave=False) as pbar:
            result = searcher.search(base_question, base_scores, progress=pbar)

        evaluation_line = (
            f"MCTS scorer evaluations: {result['evaluations']}/{args.max_vlm_evaluations}."
        )
        print(evaluation_line)
        append_text(log_path, evaluation_line)

        trace = result.get("trace", [])
        for idx, entry in enumerate(trace, start=1):
            trans = entry.get("transition")
            if isinstance(trans, tuple):
                trans_text = f"{trans[0]} -> {trans[1]}"
            else:
                trans_text = "-"
            score_text = format_option_scores(entry.get("scores", {}))
            step_line = (
                f"Node {idx} depth={entry['depth']}: real_label={truth} pred={entry.get('pred','')} "
                f"runner_up={entry.get('runner_up') or '-'} "
                f"chosen_pred={entry.get('chosen_pred') or '-'} "
                f"prediction_source={entry.get('prediction_source') or 'pred'} "
                f"gap={entry.get('pred_gap', entry.get('gap', float('nan'))):.3f} "
                f"truth_score={entry['truth_score']:.3f} other={entry['best_other_score']:.3f} "
                f"margin={entry['margin']:.3f} "
                f"reward={entry.get('reward', float('nan')):.3f} "
                f"reward_source={entry.get('reward_source') or 'margin'} "
                f"Δtruth={entry['delta_truth_score']:.3f} "
                f"Δmargin={entry['delta_margin']:.3f} "
                f"Δreward={entry.get('delta_reward', float('nan')):.3f} transition={trans_text}"
            )
            if score_text:
                step_line += f" scores=[{score_text}]"
            print(step_line)
            append_text(log_path, step_line)

        attack_node = result.get("attack_node")
        best_node = result.get("best_node") or result.get("root")
        final_node = attack_node or best_node

        score_tree = result.get("score_tree")
        if score_tree:

            def _node_json_summary(node):
                if not node:
                    return None
                scores = node.scores or {}
                return {
                    "depth": int(node.depth),
                    "question": node.question,
                    "transition": list(node.transition) if node.transition else None,
                    "truth": truth,
                    "real_label": truth,
                    "actual_label": truth,
                    "pred": scores.get("pred"),
                    "runner_up": scores.get("runner_up"),
                    "pred_score": scores.get("pred_score"),
                    "runner_up_score": scores.get("runner_up_score"),
                    "chosen_pred": str(scores.get("pred") or ""),
                    "label_after_attack": str(scores.get("pred") or ""),
                    "prediction_source": "pred",
                    "margin": float(scores.get("margin", float("nan"))),
                    "reward": float(scores.get("reward", scores.get("margin", float("nan")))),
                    "reward_source": scores.get("reward_source"),
                    "pred_gap": float(scores.get("pred_gap", scores.get("gap", float("nan")))),
                    "truth_score": float(scores.get("truth_score", float("nan"))),
                }

            tree_record = {
                "key": sample.get("key", ""),
                "truth": truth,
                "real_label": truth,
                "actual_label": truth,
                "prediction_source": "pred",
                "evaluations": result["evaluations"],
                "root_pred": base_scores.get("pred"),
                "root_runner_up": base_scores.get("runner_up"),
                "root_chosen_pred": str(base_scores.get("pred") or ""),
                "root_margin": float(base_scores.get("margin", float("nan"))),
                "root_reward": float(
                    base_scores.get("reward", base_scores.get("margin", float("nan")))
                ),
                "attack_found": bool(attack_node),
                "attack_success": bool(attack_node),
                "final_question": final_node.question if final_node else None,
                "final_pred": final_node.scores.get("pred") if final_node else None,
                "final_runner_up": (final_node.scores.get("runner_up") if final_node else None),
                "final_chosen_pred": (
                    str(final_node.scores.get("pred") or "") if final_node else None
                ),
                "label_after_attack": (
                    str(final_node.scores.get("pred") or "") if final_node else None
                ),
                "final_reward": (
                    float(
                        final_node.scores.get(
                            "reward", final_node.scores.get("margin", float("nan"))
                        )
                    )
                    if final_node
                    else None
                ),
                "trace": trace,
                "final_node": _node_json_summary(final_node),
                "best_node": _node_json_summary(best_node),
                "attack_node": _node_json_summary(attack_node),
                "tree": score_tree,
            }
            append_jsonl(tree_log_path, tree_record)

        def _path(node):
            path_nodes = []
            cur = node
            while cur:
                path_nodes.append(cur)
                cur = cur.parent
            path_nodes.reverse()
            return path_nodes

        def _path_transition_lines(node):
            if not node:
                return []
            lines = ["Transitions (path):"]
            for item in _path_transition_records(node):
                prev = item["prev"]
                new = item["new"]
                d = float(item["delta_truth"] or 0.0)
                i = item["step"]
                sign = "+" if d >= 0 else ""
                lines.append(f"  {i}) {prev} -> {new} (Δtruth {sign}{d:.3f})")
            return lines

        def _path_transition_records(node):
            if not node:
                return []
            path_nodes = _path(node)
            records = []
            for i, path_node in enumerate(path_nodes[1:], start=1):  # skip root
                prev, new = path_node.transition
                records.append(
                    {
                        "step": i,
                        "prev": str(prev),
                        "new": str(new),
                        "delta_truth": _safe_float(path_node.delta_truth),
                    }
                )
            return records

        if attack_node:
            trans_lines = _path_transition_lines(attack_node)
            trans_records = _path_transition_records(attack_node)
            label_line = _label_summary(attack_node.scores)

            print("\nSuccess: VLM misclassified under this path.")
            print("\n".join(trans_lines))
            print("\nFinal question:")
            print(attack_node.question)
            print(label_line)
            append_text(
                log_path,
                [
                    "Success: VLM misclassified under this path.",
                    *trans_lines,
                    "Final attack question:",
                    attack_node.question,
                    label_line,
                    "Result: misclassified via MCTS.",
                    "--- End Run ---",
                    "",
                ],
            )
            success_count += 1
            return _build_summary_record(
                sample,
                truth=truth,
                base_question=base_question,
                final_question=attack_node.question,
                transitions=trans_records,
                base_scores=base_scores,
                final_scores=attack_node.scores,
                success=True,
                search_mode="mcts",
                evaluations=result["evaluations"],
            )
        if best_node:
            trans_lines = _path_transition_lines(best_node)
            trans_records = _path_transition_records(best_node)
            best_line = (
                f"Best node margin={best_node.margin:.3f} reward={best_node.scores.get('reward', best_node.margin):.3f} "
                f"reward_source={best_node.scores.get('reward_source', 'margin')} "
                f"real_label={truth} pred={best_node.pred} "
                f"runner_up={best_node.scores.get('runner_up') or '-'} depth={best_node.depth}"
            )
            label_line = _label_summary(best_node.scores)
            print(best_line)
            if trans_lines:
                print("\n".join(trans_lines))
            print("\nFinal question:")
            print(best_node.question)
            print(label_line)
            append_text(
                log_path,
                [
                    best_line,
                    *trans_lines,
                    "Final question:",
                    best_node.question,
                    label_line,
                    "Result: done.",
                    "--- End Run ---",
                    "",
                ],
            )
            return _build_summary_record(
                sample,
                truth=truth,
                base_question=base_question,
                final_question=best_node.question,
                transitions=trans_records,
                base_scores=base_scores,
                final_scores=best_node.scores,
                success=False,
                search_mode="mcts",
                evaluations=result["evaluations"],
            )
        return _build_summary_record(
            sample,
            truth=truth,
            base_question=base_question,
            final_question=base_question,
            transitions=[],
            base_scores=base_scores,
            final_scores=base_scores,
            success=False,
            search_mode="mcts",
            evaluations=result["evaluations"],
        )

    # Attack all candidates, from easiest to hardest
    with tqdm(total=len(lowconf), desc="Attack", dynamic_ncols=True) as attack_pbar:
        for idx, sample in enumerate(lowconf, start=1):
            key = sample.get("key", str(idx))
            print(
                f"\n==== Attacking sample {idx}/{len(lowconf)} | key={key} | gap={sample.get('gap', float('nan')):.3f} ===="
            )
            append_text(
                log_path,
                f"==== Attacking sample {idx}/{len(lowconf)} | key={key} | gap={sample.get('gap', float('nan')):.3f} ====",
            )
            prev_success_count = success_count
            sample_started = time.monotonic()
            record = attack_one(sample)
            if record is None:
                skipped_count += 1
            else:
                record["elapsed_seconds"] = round(time.monotonic() - sample_started, 3)
                attack_records.append(record)
                append_jsonl(progress_path, record)
                if success_count == prev_success_count:
                    failed_count += 1

            processed = success_count + failed_count + skipped_count
            attack_pbar.set_description(
                f"[Succeeded / Failed / Skipped / Total] "
                f"{success_count} / {failed_count} / {skipped_count} / {processed}"
            )
            attack_pbar.update(1)

    print(
        f"\nAttack summary: succeeded={success_count}, failed={failed_count}, "
        f"skipped={skipped_count}, total={len(lowconf)}."
    )
    append_text(
        log_path,
        [
            f"Attack summary: succeeded={success_count}, failed={failed_count}, skipped={skipped_count}, total={len(lowconf)}.",
        ],
    )
    written_paths = _export_attack_summaries(attack_records, summary_dir, args.summary_format)
    if written_paths:
        print(f"Per-TSV attack summaries written under: {summary_dir}")
        append_text(log_path, f"Per-TSV attack summaries written under: {summary_dir}")
    write_json(
        complete_marker,
        {
            "completed_at": datetime.now().isoformat(),
            "total_candidates": len(prior_records) + selected_candidate_count,
            "total_records": len(attack_records),
        },
    )
    print(f"Complete marker written: {complete_marker}")
    append_text(log_path, f"Complete marker written: {complete_marker}")


if __name__ == "__main__":
    main()
