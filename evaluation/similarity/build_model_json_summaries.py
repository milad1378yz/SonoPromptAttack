import argparse
import json
from pathlib import Path

import pandas as pd


def weighted_mean(frame: pd.DataFrame, value_column: str, weight_column: str):
    valid = frame[[value_column, weight_column]].dropna()
    if valid.empty:
        return None
    weights = valid[weight_column]
    weight_sum = weights.sum()
    if weight_sum == 0:
        return None
    return float((valid[value_column] * weights).sum() / weight_sum)


def collect_metric_means(frame: pd.DataFrame):
    metrics = {}
    for column in frame.columns:
        if not column.endswith("_mean"):
            continue
        if column in {"successful_attack_count_mean"}:
            continue
        metric_name = column[: -len("_mean")]
        metrics[metric_name] = weighted_mean(frame, column, "successful_attack_count")
    return metrics


def build_role_summary(frame: pd.DataFrame, role: str, paired_column: str):
    if frame.empty:
        return {
            "role": role,
            "scenario_count": 0,
            "successful_attack_count": 0,
            "total_attack_count": 0,
            "metric_weighted_means": {},
            "scenarios": [],
        }

    scenarios = []
    for row in frame.sort_values(["successful_attack_count", "search_mode"], ascending=[False, True]).to_dict("records"):
        successful_attack_count = int(row["successful_attack_count"]) if pd.notna(row["successful_attack_count"]) else 0
        total_attack_count = int(row["total_attacks"]) if pd.notna(row.get("total_attacks")) else 0
        scenario_entry = {
            "search_mode": row["search_mode"],
            "paired_model": row[paired_column],
            "successful_attack_count": successful_attack_count,
            "total_attack_count": total_attack_count,
        }
        for key, value in row.items():
            if key.endswith("_mean") or key in {"semantic_similarity_model", "perplexity_model"}:
                if pd.notna(value):
                    scenario_entry[key] = value
        scenarios.append(scenario_entry)

    return {
        "role": role,
        "scenario_count": int(len(frame)),
        "successful_attack_count": int(frame["successful_attack_count"].fillna(0).sum()),
        "total_attack_count": int(frame["total_attacks"].fillna(0).sum()) if "total_attacks" in frame.columns else None,
        "metric_weighted_means": collect_metric_means(frame),
        "semantic_similarity_model": (
            frame["semantic_similarity_model"].dropna().iloc[0]
            if "semantic_similarity_model" in frame.columns and not frame["semantic_similarity_model"].dropna().empty
            else None
        ),
        "perplexity_model": (
            frame["perplexity_model"].dropna().iloc[0]
            if "perplexity_model" in frame.columns and not frame["perplexity_model"].dropna().empty
            else None
        ),
        "scenarios": scenarios,
    }


def sanitize_filename(name: str):
    return name.replace("/", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Build JSON summaries per model from scenario similarity outputs")
    parser.add_argument("--summary_csv", required=True, help="Path to scenario_similarity_summary.csv")
    parser.add_argument("--counts_csv", required=True, help="Path to scenario_counts.csv")
    parser.add_argument("--output_dir", required=True, help="Directory for JSON summaries")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    counts = pd.read_csv(args.counts_csv)

    if "successful_attack_count" in summary.columns:
        summary = summary.rename(columns={"successful_attack_count": "summary_successful_attack_count"})

    merged = counts[["attacker_model", "target_model", "search_mode", "total_attacks", "successful_attacks"]].merge(
        summary,
        on=["attacker_model", "target_model", "search_mode"],
        how="outer",
    )

    if "successful_attacks" in merged.columns:
        merged["successful_attack_count"] = merged["successful_attacks"]
    elif "summary_successful_attack_count" in merged.columns:
        merged["successful_attack_count"] = merged["summary_successful_attack_count"]
    else:
        merged["successful_attack_count"] = 0

    if "summary_successful_attack_count" in merged.columns:
        merged["successful_attack_count"] = merged["successful_attack_count"].fillna(
            merged["summary_successful_attack_count"]
        )
    merged["successful_attack_count"] = merged["successful_attack_count"].fillna(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = sorted(set(merged["attacker_model"].dropna()).union(set(merged["target_model"].dropna())))
    index = []

    for model_name in model_names:
        attacker_frame = merged[merged["attacker_model"] == model_name].copy()
        target_frame = merged[merged["target_model"] == model_name].copy()

        payload = {
            "model_name": model_name,
            "summary_source": str(Path(args.summary_csv)),
            "successful_attacks_source": str(Path(args.counts_csv)),
            "as_attacker": build_role_summary(attacker_frame, "attacker", "target_model"),
            "as_target": build_role_summary(target_frame, "target", "attacker_model"),
        }

        payload["total_successful_attacks_involving_model"] = (
            payload["as_attacker"]["successful_attack_count"] + payload["as_target"]["successful_attack_count"]
        )

        out_path = output_dir / f"{sanitize_filename(model_name)}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        index.append(
            {
                "model_name": model_name,
                "json_file": str(out_path),
                "attacker_successful_attacks": payload["as_attacker"]["successful_attack_count"],
                "target_successful_attacks": payload["as_target"]["successful_attack_count"],
                "total_successful_attacks_involving_model": payload["total_successful_attacks_involving_model"],
            }
        )

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Wrote {len(index)} model summary JSON files to {output_dir}")
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()
