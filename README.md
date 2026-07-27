<div align="center">
  <h1>When Minor Edits Matter</h1>
  <p><strong>LLM-Driven Prompt Attack for Medical VLM Robustness in Ultrasound</strong></p>
  <p>
    <a href="https://arxiv.org/abs/2603.21047"><img src="https://img.shields.io/badge/arXiv-2603.21047-b31b1b.svg" alt="arXiv"></a>
    <img src="https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10">
    <img src="https://img.shields.io/badge/PyTorch-2.7-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch 2.7">
  </p>
</div>

Official implementation of
[When Minor Edits Matter: LLM-Driven Prompt Attack for Medical VLM Robustness in Ultrasound](https://arxiv.org/abs/2603.21047).
The repository includes the MCTS attack, conventional search strategies and TextAttack baselines.

Evaluation utilities include lexical and semantic similarity metrics,
perplexity, and an LLM-as-judge quality rubric. See
[`evaluation/README.md`](evaluation/README.md) for the full metric definitions.

## Setup

Python 3.10 and an NVIDIA GPU are recommended. Create an environment and install
the dependencies:

```bash
conda create -n vlm-attack python=3.10 -y
conda activate vlm-attack
pip install -r requirements.txt
```

Authenticate with Hugging Face and make sure your account can access the selected
models:

```bash
huggingface-cli login
```

The primary attack downloads `DolphinAI/u2-bench` automatically. For all
experiments, download a reusable local copy:

```bash
huggingface-cli download DolphinAI/u2-bench \
  --repo-type dataset \
  --local-dir dataset/u2-bench
```

## Running the code

Run all commands from the repository root.

### Main attack

The following example uses MCTS, MedGemma, and a remote proposer LLM:

```bash
export LLM_API_KEY="your-api-key"

python vlm_attack.py \
  --dataset-path dataset/u2-bench/disease_diagnosis \
  --vlm-id google/medgemma-4b-it \
  --llm-id qwen/qwen3-30b-a3b-instruct-2507 \
  --use-api \
  --llm-api-provider openrouter \
  --api-key "$LLM_API_KEY" \
  --search-mode mcts \
  --max-samples 10 \
  --log-path runs/mcts/attack_log.txt \
  --summary-dir runs/mcts/summaries
```

Remove `--max-samples` for a full run and use a new summary directory for each
completed run. Available search modes are `mcts`, `ga`, `random`, `greedy`, and
`beam`. To use a local proposer, omit `--use-api` and `--api-key`, then provide a
local or Hugging Face model with `--llm-id`.

Results are written beside `--log-path` and under `--summary-dir`. Interrupted
runs resume from the summary directory.

## Evaluation

The attack runner's `*.attack_summary.csv` files can be passed directly to both
evaluators. Run commands from the repository root.

Calculate Levenshtein distance, BLEU, chrF, ROUGE-1, and ROUGE-L:

```bash
bash evaluation/similarity/run.sh \
  runs/mcts/summaries/path/to/task.attack_summary.csv
```

To also calculate embedding cosine similarity and perplexity:

```bash
bash evaluation/similarity/run.sh \
  runs/mcts/summaries/path/to/task.attack_summary.csv \
  evaluation/results/similarity \
  --include_semantic \
  --include_perplexity
```

Run the LLM judge after setting its model and providing the OpenRouter
credential through the environment:

```bash
export OPENROUTER_MODEL="openai/gpt-4o-mini"
bash evaluation/llm_as_judge/run.sh \
  runs/mcts/summaries/path/to/task.attack_summary.csv
```

Both tools also accept a directory of CSV files. Results are created
automatically under `evaluation/results/`, which is excluded from Git. No
credential is stored in the repository or accepted on the command line.


### TextAttack baselines

TextAttack processes one TSV file at a time:

```bash
export MEDGEMMA_TEXTATTACK_TSV="dataset/u2-bench/disease_diagnosis/path/to/task.tsv"
export MEDGEMMA_TEXTATTACK_MODEL_ID="google/medgemma-4b-it"
export MEDGEMMA_TEXTATTACK_SEED=765
mkdir -p runs/textattack

python -m textattack attack \
  --model-from-file text_attack/medgemma_textattack_wrapper.py \
  --dataset-from-file text_attack/u2bench_textattack_dataset.py \
  --attack-from-file text_attack/medgemma_textfooler_attack.py \
  --num-examples 10 \
  --query-budget 100 \
  --model-batch-size 1 \
  --random-seed 765 \
  --log-to-csv runs/textattack/textfooler.csv
```

Replace the attack file with any of:

- `medgemma_checklist_attack.py`
- `medgemma_deepwordbug_attack.py`
- `medgemma_greedy_char_substitution_attack.py`
- `medgemma_random_char_search_attack.py`
- `medgemma_textbugger_attack.py`
- `medgemma_textfooler_attack.py`
