#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_PATH="${1:-}"

if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: bash evaluation/similarity/run.sh INPUT_CSV_OR_DIR [OUTPUT_DIR] [METRIC_OPTIONS]" >&2
  exit 2
fi
shift

OUTPUT_DIR="$REPO_ROOT/evaluation/results/similarity"
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUTPUT_DIR="$1"
  shift
fi

INPUT_FLAG="--csv_file"
if [[ -d "$INPUT_PATH" ]]; then
  INPUT_FLAG="--input_dir"
elif [[ ! -f "$INPUT_PATH" ]]; then
  echo "Input path does not exist: $INPUT_PATH" >&2
  exit 2
fi

"$PYTHON_BIN" "$SCRIPT_DIR/calculate_sims.py" \
  "$INPUT_FLAG" "$INPUT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --skip_existing \
  "$@"

COUNTS_CSV="${SIMILARITY_COUNTS_CSV:-}"
if [[ -z "$COUNTS_CSV" && -d "$INPUT_PATH" && -f "$INPUT_PATH/scenario_counts.csv" ]]; then
  COUNTS_CSV="$INPUT_PATH/scenario_counts.csv"
fi

if [[ -n "$COUNTS_CSV" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/build_model_json_summaries.py" \
    --summary_csv "$OUTPUT_DIR/scenario_similarity_summary.csv" \
    --counts_csv "$COUNTS_CSV" \
    --output_dir "$OUTPUT_DIR/model_summaries"
fi
