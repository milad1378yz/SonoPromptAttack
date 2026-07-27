# Evaluation

This folder contains the paper's evaluation utilities. By default, LLM-judge
outputs and all aggregate metrics are written to
`evaluation/results/`. Each runner creates its output directory automatically,
and generated results are ignored by Git.

## Similarity metrics

The similarity evaluator compares the original and attacked questions using:

- character-level Levenshtein distance;
- sentence BLEU and chrF;
- ROUGE-1 and ROUGE-L F1;
- optional embedding cosine similarity;
- optional original and attacked-question perplexity.

The base metrics run without downloading another model:

```bash
bash evaluation/similarity/run.sh path/to/attack_summary.csv
```

Add embedding similarity, perplexity, or both with:

```bash
bash evaluation/similarity/run.sh path/to/attack_summary.csv \
  --include_semantic \
  --include_perplexity
```

The default embedding and perplexity models can be changed with
`--embedding_model` and `--lm_model`; use `--device cpu` or `--device cuda` to
select the device. Model-backed metrics download their selected Hugging Face
models on first use.

Inputs must contain `final_question` and either `base_question` or
`original_question`. Outputs include row-level `*.sims.csv`, per-input
`*.statistics.csv`, and `scenario_similarity_summary.csv`. Per-model JSON
summaries are generated when the input directory contains
`scenario_counts.csv`, or when `SIMILARITY_COUNTS_CSV` points to one.
When an input has `attack_success` or `success`, only successful attacks are
evaluated.

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
When an input has `attack_success` or `success`, only successful attacks are
sent to the judge.

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
