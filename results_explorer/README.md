# SonoPromptAttack result explorer

This static explorer filters recorded examples by proposer LLM, target VLM,
task, and example. It shows the exact dataset image, full original and attacked
prompts, every recorded edit, predictions, ground truth, and source provenance.

Install the small web wrapper and launch the explorer:

```bash
cd results_explorer
npm install
npm run dev
```

Then open the local URL printed by vinext (normally `http://localhost:3001`).

## Included result matrix

The generated dataset contains 200 successful attacks: ten examples for every
combination of four proposer LLMs and five target Med-VLMs.

- Proposers: Gemma-4-E4B-it, Gemma-4-12B-it, Qwen2.5-7B-Instruct, and
  Qwen2.5-14B-Instruct.
- Targets: MedGemma-4B, MedGemma-27B, QoQ-Med-7B, QoQ-Med-32B, and
  LLaVA-Med-7B.

The interface intentionally shows only the record key and dataset image
location. Internal run-summary paths are not displayed.

Regenerate the balanced dataset with `scripts/build_examples.py`.
