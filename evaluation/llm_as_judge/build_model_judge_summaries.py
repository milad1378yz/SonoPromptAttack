import argparse
import json
from pathlib import Path

import pandas as pd


def mean_or_none(series: pd.Series):
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def collect_rate_means(frame: pd.DataFrame):
    metrics = {}
    for column in frame.columns:
        if column.endswith("_rate"):
            metrics[column] = mean_or_none(frame[column])
    return metrics


def collect_count_sums(frame: pd.DataFrame):
    metrics = {}
    for column in frame.columns:
        if "__" in column and column.endswith("_count"):
            metrics[column] = int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
    return metrics


def build_role_summary(frame: pd.DataFrame, role: str, paired_column: str):
    if frame.empty:
        return {
            "role": role,
            "scenario_count": 0,
            "successful_attack_count": 0,
            "mean_rates": {},
            "category_count_sums": {},
            "scenarios": [],
        }

    scenarios = []
    sort_columns = ["successful_attack_count", "search_mode"]
    ascending = [False, True]
    for row in frame.sort_values(sort_columns, ascending=ascending).to_dict("records"):
        entry = {
            "search_mode": row["search_mode"],
            "paired_model": row[paired_column],
            "successful_attack_count": int(row["successful_attack_count"]),
        }
        for key, value in row.items():
            if key.endswith("_rate") or key.endswith("_count"):
                if pd.notna(value):
                    entry[key] = float(value) if key.endswith("_rate") else int(value)
        scenarios.append(entry)

    return {
        "role": role,
        "scenario_count": int(len(frame)),
        "successful_attack_count": int(frame["successful_attack_count"].sum()),
        "judge_model": (
            frame["judge_model"].dropna().iloc[0]
            if "judge_model" in frame.columns and not frame["judge_model"].dropna().empty
            else None
        ),
        "rubric_name": (
            frame["rubric_name"].dropna().iloc[0]
            if "rubric_name" in frame.columns and not frame["rubric_name"].dropna().empty
            else None
        ),
        "mean_rates": collect_rate_means(frame),
        "category_count_sums": collect_count_sums(frame),
        "scenarios": scenarios,
    }


def sanitize_filename(name: str):
    return name.replace("/", "_")


def load_summary(summary_file: Path):
    if summary_file.suffix.lower() == ".json":
        payload = json.loads(summary_file.read_text())
        frame = pd.DataFrame(payload)
    else:
        frame = pd.read_csv(summary_file)

    numeric_columns = [
        column
        for column in frame.columns
        if column.endswith("_rate")
        or column.endswith("_count")
        or column in {"total_attacks", "successful_attacks"}
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Build JSON summaries per model from LLM judge outputs")
    parser.add_argument("--summary_file", required=True, help="Path to scenario_judge_summary.json or .csv")
    parser.add_argument("--counts_csv", required=True, help="Path to scenario_counts.csv")
    parser.add_argument("--output_dir", required=True, help="Directory for JSON summaries")
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.summary_file)
    summary = load_summary(summary_path)
    counts = pd.read_csv(args.counts_csv)

    merged = summary.merge(
        counts[["attacker_model", "target_model", "search_mode", "total_attacks", "successful_attacks"]],
        on=["attacker_model", "target_model", "search_mode"],
        how="left",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = sorted(set(merged["attacker_model"]).union(set(merged["target_model"])))
    index = []

    for model_name in model_names:
        attacker_frame = merged[merged["attacker_model"] == model_name].copy()
        target_frame = merged[merged["target_model"] == model_name].copy()

        payload = {
            "model_name": model_name,
            "summary_source": str(summary_path),
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
