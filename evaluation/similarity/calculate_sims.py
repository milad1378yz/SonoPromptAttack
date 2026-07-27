import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    from sacrebleu.metrics import BLEU, CHRF
except ImportError:
    BLEU = None
    CHRF = None


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calculate_rouge(base, final):
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(base, final)
        return {
            "rouge1_fmeasure": scores["rouge1"].fmeasure,
            "rougeL_fmeasure": scores["rougeL"].fmeasure,
        }

    base_tokens = base.split()
    final_tokens = final.split()
    rouge1 = f1_from_overlap(base_tokens, final_tokens)
    rouge_l = rouge_l_f1(base_tokens, final_tokens)
    return {
        "rouge1_fmeasure": rouge1,
        "rougeL_fmeasure": rouge_l,
    }


def calculate_bleu(base, final):
    if BLEU is not None:
        bleu = BLEU(effective_order=True)
        return float(bleu.sentence_score(final, [base]).score)
    return simple_bleu(base, final)


def calculate_chrf(base, final):
    if CHRF is not None:
        chrf = CHRF()
        return float(chrf.sentence_score(final, [base]).score)
    return simple_chrf(base, final)


def f1_from_overlap(reference_tokens, candidate_tokens):
    if not reference_tokens and not candidate_tokens:
        return 1.0
    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    overlap = sum((reference_counts & candidate_counts).values())
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(a, b):
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(reference_tokens, candidate_tokens):
    if not reference_tokens and not candidate_tokens:
        return 1.0
    if not reference_tokens or not candidate_tokens:
        return 0.0

    lcs = lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def simple_bleu(base, final, max_order=4):
    reference_tokens = base.split()
    candidate_tokens = final.split()
    if not candidate_tokens:
        return 0.0
    if not reference_tokens:
        return 100.0 if not candidate_tokens else 0.0

    precisions = []
    for order in range(1, max_order + 1):
        cand_ngrams = ngrams(candidate_tokens, order)
        ref_ngrams = Counter(ngrams(reference_tokens, order))
        if not cand_ngrams:
            precisions.append(1.0)
            continue

        cand_counts = Counter(cand_ngrams)
        clipped = sum(min(count, ref_ngrams[gram]) for gram, count in cand_counts.items())
        # Add-one smoothing to avoid zeroing the sentence score.
        precisions.append((clipped + 1) / (len(cand_ngrams) + 1))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    ref_len = len(reference_tokens)
    cand_len = len(candidate_tokens)
    brevity_penalty = 1.0 if cand_len > ref_len else math.exp(1 - (ref_len / cand_len))
    return 100.0 * brevity_penalty * geo_mean


def char_ngrams(text, n):
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def simple_chrf(base, final, max_order=6, beta=2.0):
    if not base and not final:
        return 100.0
    if not base or not final:
        return 0.0

    scores = []
    beta_sq = beta * beta
    for order in range(1, max_order + 1):
        ref_ngrams = Counter(char_ngrams(base, order))
        cand_ngrams = Counter(char_ngrams(final, order))
        if not ref_ngrams or not cand_ngrams:
            scores.append(0.0)
            continue

        overlap = sum((ref_ngrams & cand_ngrams).values())
        precision = overlap / sum(cand_ngrams.values())
        recall = overlap / sum(ref_ngrams.values())
        if precision + recall == 0:
            scores.append(0.0)
            continue
        scores.append((1 + beta_sq) * precision * recall / (beta_sq * precision + recall))

    return 100.0 * (sum(scores) / len(scores))


def get_default_device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_semantic_model(model_name: str, device: str):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    try:
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ValueError as exc:
        if "Unrecognized model" not in str(exc):
            raise

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if getattr(config, "model_type", None) == "xlm-roberta":
            from transformers import XLMRobertaModel

            model = XLMRobertaModel.from_pretrained(model_name, config=config).to(device)
        else:
            # Some embedding models require remote code because their config/model
            # type is not registered in the local transformers build.
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    model.eval()
    return model, tokenizer


def calculate_semantic_similarity(base, final, model, tokenizer, max_tokens=2048):
    import torch

    model_max_positions = getattr(getattr(model, "config", None), "max_position_embeddings", max_tokens)
    tokenizer_max = getattr(tokenizer, "model_max_length", max_tokens)
    effective_max_tokens = min(
        max_tokens,
        tokenizer_max if isinstance(tokenizer_max, int) and tokenizer_max > 0 else max_tokens,
        model_max_positions if isinstance(model_max_positions, int) and model_max_positions > 0 else max_tokens,
    )

    base_ids = tokenizer.encode(base)
    final_ids = tokenizer.encode(final)
    base_tokens = len(base_ids)
    final_tokens = len(final_ids)

    if base_tokens > effective_max_tokens:
        print(f"WARNING: Base question exceeds {effective_max_tokens} tokens ({base_tokens} tokens)")
    if final_tokens > effective_max_tokens:
        print(f"WARNING: Final question exceeds {effective_max_tokens} tokens ({final_tokens} tokens)")

    def embed(text: str):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=effective_max_tokens,
            padding=True,
        )
        encoded = {key: value.to(next(model.parameters()).device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        attention = encoded["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden * attention
        pooled = masked_hidden.sum(dim=1) / attention.sum(dim=1).clamp(min=1)
        return pooled[0]

    base_embedding = embed(base)
    final_embedding = embed(final)
    similarity = torch.nn.functional.cosine_similarity(
        base_embedding.unsqueeze(0), final_embedding.unsqueeze(0)
    )
    return float(similarity.item())


def load_lm(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {}
    if device.startswith("cuda"):
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def calculate_perplexity(text, model, tokenizer):
    import torch

    device = next(model.parameters()).device
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=min(tokenizer.model_max_length, 2048),
    )
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return float(math.exp(loss.item()))


def resolve_input_csvs(csv_file: str | None, input_dir: str | None):
    if csv_file:
        path = Path(csv_file)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return [path]

    if input_dir:
        path = Path(input_dir)
        if not path.exists():
            raise FileNotFoundError(f"Input directory not found: {path}")
        return sorted(p for p in path.glob("*.csv") if p.name != "scenario_counts.csv")

    raise ValueError("Provide either --csv_file or --input_dir")


def scenario_from_df(df: pd.DataFrame, csv_path: Path):
    if {"attacker_model", "target_model", "search_mode"}.issubset(df.columns):
        first = df.iloc[0]
        return first["attacker_model"], first["target_model"], first["search_mode"]

    parts = csv_path.stem.split("__")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return "", "", csv_path.stem


def keep_successful_attacks(df: pd.DataFrame):
    success_column = next(
        (name for name in ("attack_success", "success") if name in df.columns),
        None,
    )
    if success_column is None:
        return df

    def is_success(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not pd.isna(value):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes", "successful"}

    return df[df[success_column].map(is_success)].copy()


def compute_metrics_for_df(
    df: pd.DataFrame,
    include_semantic: bool,
    emb_model,
    emb_tokenizer,
    include_perplexity: bool,
    lm_model,
    lm_tokenizer,
):
    df = df.copy()
    df["base_question"] = df["base_question"].fillna("").astype(str)
    df["final_question"] = df["final_question"].fillna("").astype(str)

    df["levenshtein_distance"] = df.apply(
        lambda row: levenshtein_distance(row["base_question"], row["final_question"]),
        axis=1,
    )

    print("Calculating BLEU scores...")
    df["bleu_score"] = [
        calculate_bleu(row.base_question, row.final_question)
        for row in tqdm(df.itertuples(index=False), total=len(df))
    ]

    print("Calculating chrF scores...")
    df["chrf_score"] = [
        calculate_chrf(row.base_question, row.final_question)
        for row in tqdm(df.itertuples(index=False), total=len(df))
    ]

    print("Calculating ROUGE scores...")
    rouge_scores = [
        calculate_rouge(row.base_question, row.final_question)
        for row in tqdm(df.itertuples(index=False), total=len(df))
    ]
    df = pd.concat([df, pd.DataFrame(rouge_scores)], axis=1)

    if include_semantic:
        print("Calculating semantic similarity...")
        df["semantic_similarity"] = [
            calculate_semantic_similarity(
                row.base_question,
                row.final_question,
                emb_model,
                emb_tokenizer,
            )
            for row in tqdm(df.itertuples(index=False), total=len(df))
        ]

    if include_perplexity:
        print("Calculating base perplexity...")
        df["base_perplexity"] = [
            calculate_perplexity(row.base_question, lm_model, lm_tokenizer)
            for row in tqdm(df.itertuples(index=False), total=len(df))
        ]

        print("Calculating final perplexity...")
        df["final_perplexity"] = [
            calculate_perplexity(row.final_question, lm_model, lm_tokenizer)
            for row in tqdm(df.itertuples(index=False), total=len(df))
        ]

    return df


def summarise_metrics(
    df: pd.DataFrame,
    csv_path: Path,
    embedding_model_name: str | None = None,
    perplexity_model_name: str | None = None,
):
    attacker_model, target_model, search_mode = scenario_from_df(df, csv_path)
    stats = {
        "attacker_model": attacker_model,
        "target_model": target_model,
        "search_mode": search_mode,
        "input_csv": str(csv_path),
        "successful_attack_count": len(df),
    }
    if "semantic_similarity" in df.columns:
        stats["semantic_similarity_model"] = embedding_model_name
    if "base_perplexity" in df.columns or "final_perplexity" in df.columns:
        stats["perplexity_model"] = perplexity_model_name

    metric_columns = [
        "levenshtein_distance",
        "bleu_score",
        "chrf_score",
        "rouge1_fmeasure",
        "rougeL_fmeasure",
        "semantic_similarity",
        "base_perplexity",
        "final_perplexity",
    ]

    for column in metric_columns:
        if column in df.columns:
            stats[f"{column}_mean"] = df[column].mean()
            stats[f"{column}_median"] = df[column].median()
            stats[f"{column}_min"] = df[column].min()
            stats[f"{column}_max"] = df[column].max()

    return stats


def process_csv(
    csv_path: Path,
    output_dir: Path,
    include_semantic: bool,
    emb_model,
    emb_tokenizer,
    include_perplexity: bool,
    lm_model,
    lm_tokenizer,
    embedding_model_name: str | None,
    perplexity_model_name: str | None,
    skip_existing: bool,
):
    sims_path = output_dir / f"{csv_path.stem}.sims.csv"
    stats_path = output_dir / f"{csv_path.stem}.statistics.csv"

    if skip_existing and sims_path.exists() and stats_path.exists():
        print(f"\nSkipping {csv_path} because outputs already exist.")
        return pd.read_csv(stats_path).iloc[0].to_dict()

    print(f"\nProcessing {csv_path}...")
    try:
        df = pd.read_csv(
            csv_path,
            dtype={"example": "str", "base_question": "str", "final_question": "str"},
        )
    except EmptyDataError:
        print("Input CSV has no rows or header. Writing empty outputs.")
        empty_df = pd.DataFrame(columns=["base_question", "final_question"])
        empty_df.to_csv(sims_path, index=False)
        return {
            "attacker_model": "",
            "target_model": "",
            "search_mode": csv_path.stem,
            "input_csv": str(csv_path),
            "successful_attack_count": 0,
        }

    if df.empty:
        print("Input CSV is empty. Writing empty outputs.")
        df.to_csv(sims_path, index=False)
        return {
            "attacker_model": "",
            "target_model": "",
            "search_mode": csv_path.stem,
            "input_csv": str(csv_path),
            "successful_attack_count": 0,
        }

    df = keep_successful_attacks(df)

    if "base_question" not in df.columns and "original_question" in df.columns:
        df["base_question"] = df["original_question"]

    required_columns = {"base_question", "final_question"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    df = compute_metrics_for_df(
        df=df,
        include_semantic=include_semantic,
        emb_model=emb_model,
        emb_tokenizer=emb_tokenizer,
        include_perplexity=include_perplexity,
        lm_model=lm_model,
        lm_tokenizer=lm_tokenizer,
    )

    metrics_order = [
        "example",
        "attacker_model",
        "target_model",
        "search_mode",
        "tsv_file",
        "index",
        "sample_id",
        "success",
        "attack_success",
        "transition_count",
        "levenshtein_distance",
        "bleu_score",
        "chrf_score",
        "rouge1_fmeasure",
        "rougeL_fmeasure",
        "semantic_similarity",
        "base_perplexity",
        "final_perplexity",
        "base_question",
        "final_question",
        "source_file",
    ]
    output_columns = [column for column in metrics_order if column in df.columns]

    df[output_columns].to_csv(sims_path, index=False)
    print(f"Wrote metrics to {sims_path}")

    stats = summarise_metrics(
        df,
        csv_path,
        embedding_model_name=embedding_model_name,
        perplexity_model_name=perplexity_model_name,
    )
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    print(f"Wrote statistics to {stats_path}")

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate similarity metrics between original and final questions"
    )
    parser.add_argument("--csv_file", help="Path to a single input CSV")
    parser.add_argument("--input_dir", help="Directory containing scenario CSVs")
    parser.add_argument("--output_dir", help="Directory for output CSVs")
    parser.add_argument(
        "--include_semantic",
        action="store_true",
        help="Compute embedding-based semantic similarity",
    )
    parser.add_argument(
        "--include_perplexity",
        action="store_true",
        help="Compute question perplexities with a causal LM",
    )
    parser.add_argument(
        "--embedding_model",
        default="google/embeddinggemma-300m",
        help="Sentence embedding model for semantic similarity",
    )
    parser.add_argument(
        "--lm_model",
        default="google/gemma-3-4b-pt",
        help="Language model used for perplexity",
    )
    parser.add_argument(
        "--device",
        default=get_default_device(),
        help="Execution device, for example cpu or cuda",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse existing per-scenario outputs in the output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_paths = resolve_input_csvs(args.csv_file, args.input_dir)
    default_output_dir = Path(args.output_dir) if args.output_dir else csv_paths[0].parent
    default_output_dir.mkdir(parents=True, exist_ok=True)

    emb_model = None
    emb_tokenizer = None
    if args.include_semantic:
        print(f"Loading embedding model on {args.device}...")
        emb_model, emb_tokenizer = load_semantic_model(args.embedding_model, args.device)

    lm_model = None
    lm_tokenizer = None
    if args.include_perplexity:
        print(f"Loading language model on {args.device}...")
        lm_model, lm_tokenizer = load_lm(args.lm_model, args.device)

    scenario_stats = []
    for csv_path in csv_paths:
        stats = process_csv(
            csv_path=csv_path,
            output_dir=default_output_dir,
            include_semantic=args.include_semantic,
            emb_model=emb_model,
            emb_tokenizer=emb_tokenizer,
            include_perplexity=args.include_perplexity,
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            embedding_model_name=args.embedding_model if args.include_semantic else None,
            perplexity_model_name=args.lm_model if args.include_perplexity else None,
            skip_existing=args.skip_existing,
        )
        scenario_stats.append(stats)

    summary_path = default_output_dir / "scenario_similarity_summary.csv"
    pd.DataFrame(scenario_stats).to_csv(summary_path, index=False)
    print(f"\nWrote scenario summary to {summary_path}")


if __name__ == "__main__":
    main()
