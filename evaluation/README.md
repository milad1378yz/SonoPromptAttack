# Evaluation

This folder contains the paper's evaluation utilities. By default, LLM-judge
outputs and all aggregate metrics are written to
`evaluation/results/llm_as_judge/`. The runner creates that directory
automatically, and generated results are ignored by Git.

## LLM as judge

The judge compares an original question with its attacked version. It reports:

- naturalness and grammaticality rates;
- weird/corrupted-artifact rate;
- task-intent and output-constraint preservation rates;
- JSON parse-success rate and mean judge confidence;
- counts and rates for grammar, weirdness, context-fit, intent, and primary
  quality categories.

Input is either one CSV or a directory of CSVs. Every CSV must contain
`final_question` and either `base_question` or `original_question` (the latter
is emitted by this repository's attack runner). The optional columns
`attacker_model`, `target_model`, and `search_mode` make scenario and per-model
summaries more descriptive.

Install the repository dependencies, set `OPENROUTER_API_KEY` in your shell or
secret manager, and run from the repository root:

```bash
export OPENROUTER_MODEL="openai/gpt-4o-mini"
bash evaluation/llm_as_judge/run.sh path/to/successful_attacks.csv
```

For a directory:

```bash
bash evaluation/llm_as_judge/run.sh \
  path/to/successful_attack_csvs \
  evaluation/results/llm_as_judge
```

The runner never reads a key from a repository file or accepts one as a command
line argument. `OPENROUTER_SITE_URL` and `OPENROUTER_APP_NAME` are optional.

Main outputs:

- `*.judge.csv`: row-level normalized judgments and the raw model response;
- `*.judge.statistics.json`: all aggregate metrics for one input CSV;
- `scenario_judge_summary.json`: metrics for every processed scenario;
- `model_summaries/*.json`: optional attacker/target summaries.

Per-model summaries are produced automatically when a directory contains
`scenario_counts.csv`. For a counts file elsewhere, set `JUDGE_COUNTS_CSV`.
It must contain `attacker_model`, `target_model`, `search_mode`,
`total_attacks`, and `successful_attacks`.

For a quick validation, call the Python entry point with `--limit 1`. To change
the categories or wording of the rubric, edit
`evaluation/llm_as_judge/categories.json`.
