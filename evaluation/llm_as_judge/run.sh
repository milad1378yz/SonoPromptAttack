#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_PATH="${1:-}"
OUTPUT_DIR="${2:-$REPO_ROOT/evaluation/results/llm_as_judge}"
JUDGE_MODEL="${3:-${OPENROUTER_MODEL:-}}"

if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: bash evaluation/llm_as_judge/run.sh INPUT_CSV_OR_DIR [OUTPUT_DIR] [JUDGE_MODEL]" >&2
  exit 2
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set." >&2
  exit 2
fi
if [[ -z "$JUDGE_MODEL" ]]; then
  echo "Provide JUDGE_MODEL as argument 3 or set OPENROUTER_MODEL." >&2
  exit 2
fi

INPUT_FLAG="--csv_file"
if [[ -d "$INPUT_PATH" ]]; then
  INPUT_FLAG="--input_dir"
elif [[ ! -f "$INPUT_PATH" ]]; then
  echo "Input path does not exist: $INPUT_PATH" >&2
  exit 2
fi

"$PYTHON_BIN" "$SCRIPT_DIR/judge_generated_questions.py" \
  "$INPUT_FLAG" "$INPUT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --judge_model "$JUDGE_MODEL" \
  --skip_existing

COUNTS_CSV="${JUDGE_COUNTS_CSV:-}"
if [[ -z "$COUNTS_CSV" && -d "$INPUT_PATH" && -f "$INPUT_PATH/scenario_counts.csv" ]]; then
  COUNTS_CSV="$INPUT_PATH/scenario_counts.csv"
fi

if [[ -n "$COUNTS_CSV" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/build_model_judge_summaries.py" \
    --summary_file "$OUTPUT_DIR/scenario_judge_summary.json" \
    --counts_csv "$COUNTS_CSV" \
    --output_dir "$OUTPUT_DIR/model_summaries"
fi
