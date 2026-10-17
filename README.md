# Large-Language-Model-Practices

A sandbox repository for learning, experimenting, and prototyping with large language models (LLMs). This README includes visual placeholders, sample results, and condensed explanations of key LLM concepts to help you run experiments and present outcomes.

---

## Table of contents
- Overview
- Repository structure
- Quick start
- Key concepts (concise)
- Visuals: where to put them & how to create them
- Results: example metrics, tables, and interpretation
- Example workflows and commands
- Reproducibility & experiment tracking
- Notebook & demo pointers
- Contributing & license

---

## Overview
This repository is for exploring LLM workflows: dataset prep, tokenization, embeddings, fine-tuning, evaluation, prompt engineering, and lightweight deployment. Use small toy experiments to iterate quickly, then scale.

Goals:
- Learn core LLM concepts and tools
- Record reproducible experiments with visuals and metrics
- Build a catalog of prompts, models, and evaluation scripts

---

## Repository structure
- `data/`
  - `raw/`
  - `processed/`
- `notebooks/`
  - EDA, prompt experiments, evaluation notebooks
- `src/`
  - `tokenization.py`
  - `train.py`
  - `infer.py`
  - `eval.py`
- `configs/`
  - `train_toy.yaml`
  - `train_full.yaml`
- `scripts/`
  - `prepare_data.py`
  - `plot_results.py`
- `models/`
  - `checkpoint-<timestamp>/`
- `assets/`
  - `figures/`
    - `training_curve.png`
    - `embeddings_tsne.png`
    - `confusion_matrix.png`
- `results/`
  - `run-YYYYMMDD-<tag>.jsonl`
- `README.md`
- `requirements.txt`

---

## Quick start
1. Clone
   ```bash
   git clone https://github.com/KrishnaKumar2002/Large-Language-Model-Practices.git
   ```
2. Python env
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```
3. Prepare data
   ```bash
   python scripts/prepare_data.py --input data/raw --output data/processed
   ```
4. Toy training
   ```bash
   python src/train.py --config configs/train_toy.yaml --device cpu
   ```
5. Evaluate
   ```bash
   python src/eval.py --model models/checkpoint-toy --data data/processed/test.jsonl --out results/run-toy.jsonl
   ```
6. Generate plots
   ```bash
   python scripts/plot_results.py --results results/run-toy.jsonl --out assets/figures/training_curve.png
   ```

---

## Key concepts (short, practical)

- Tokenization
  - Turns text into token ids.
  - Subword tokenizers (Byte-Pair Encoding, SentencePiece) are common.
  - Keep vocabulary and tokenizer configs with experiment metadata.

- Embeddings
  - Vector representations of tokens or sentences.
  - Useful for retrieval, clustering, semantic search, kNN baselines.
  - Visualize with PCA or t-SNE for sanity checks.

- Transformer & Attention
  - Self-attention lets each token attend to others.
  - Transformer blocks = multi-head attention + MLP + residuals + layer-norm.
  - Key hyperparams: number of layers, hidden size, heads, sequence length.

- Fine-tuning vs. Instruction Tuning vs. RLHF
  - Fine-tuning: supervised training on task-specific data.
  - Instruction tuning: train on (instruction, response) pairs to improve instruction-following.
  - RLHF: refine model outputs using human preference signals (more complex).

- Parameter-efficient methods (LoRA / Adapters)
  - Fine-tune a small set of parameters (low-rank adapters) instead of full model weights.
  - Saves memory and speeds up iteration.

- Evaluation metrics
  - Perplexity (language modeling baseline)
  - BLEU / ROUGE (for summarization or sequence tasks)
  - Exact match / F1 (QA)
  - Task-specific accuracy / precision / recall
  - Human evaluation for instruction-following or quality

---

## Visuals — where & how
Store plots in `assets/figures/`. Recommended visuals:
- Training curve: loss / validation loss vs. steps
- Learning rate schedule
- Embedding projection (PCA / t-SNE)
- Confusion matrix for classification tasks
- Examples table: prompt -> model output -> reference -> human score

Markdown examples to embed visuals:

```markdown
![Training curve](assets/figures/training_curve.png)
![Embeddings t-SNE](assets/figures/embeddings_tsne.png)
```

Short snippet: plotting training curve (matplotlib)

```python
# scripts/plot_results.py
import json, matplotlib.pyplot as plt
from pathlib import Path

results = [json.loads(line) for line in open('results/run-toy.jsonl')]
steps = [r['step'] for r in results]
train_loss = [r['train_loss'] for r in results]
val_loss = [r.get('val_loss') for r in results]

plt.plot(steps, train_loss, label='train')
plt.plot(steps, val_loss, label='val')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend()
plt.savefig('assets/figures/training_curve.png')
```

Embedding projection example (sklearn + TSNE)

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
X = np.load('models/embeddings.npy')  # (N, D)
labels = np.load('models/labels.npy')  # (N,)
proj = TSNE(n_components=2, perplexity=30).fit_transform(X)
plt.scatter(proj[:,0], proj[:,1], c=labels, cmap='tab10', s=5)
plt.savefig('assets/figures/embeddings_tsne.png')
```

---

## Results — example layout and sample numbers
Keep a `results/` folder containing a JSONL or CSV per run with metadata and metrics.

Example run summary (store as JSON):

```json
{
  "run_id": "run-20261017-toy",
  "model": "gpt-small-v0",
  "dataset": "toy-dataset-v1",
  "params": 60000000,
  "steps": 2000,
  "batch_size": 32,
  "train_loss_final": 1.82,
  "val_loss_final": 2.03,
  "accuracy": 0.78,
  "notes": "Toy run, CPU. Short schedule for iteration."
}
```

Example result table for README (replace with your real numbers):

| Run | Model | Steps | Val Loss | Accuracy | Notes |
|-----|-------|------:|--------:|--------:|-------|
| run-toy | gpt-small-v0 | 2k | 2.03 | 78% | quick validation |
| run-pt1 | gpt-small-v0 + LoRA | 10k | 1.45 | 85% | LoRA improved sample efficiency |
| run-instr | gpt-small-v0 instr-tuned | 8k | 1.30 | 87% | better instruction following |

Interpretation tips:
- Compare validation loss and task metrics (accuracy) together — lower loss does not always mean better downstream performance.
- Use human assessments for instruction quality or safety checks.

---

## Example experiment log template
Save per-run metadata (YAML/JSON) and outputs:
- `run.yaml`
  - run_id, date, model_base, model_version, seed, config_path, git_commit, notes
- `metrics.json`
  - arrays of (step, train_loss, val_loss, lr)
- `predictions.jsonl`
  - one line per example: {id, prompt, output, reference, metrics}

---

## Example configs
`configs/train_toy.yaml`:

```yaml
model: gpt-small-v0
seed: 42
batch_size: 32
lr: 5e-5
warmup_steps: 100
total_steps: 2000
optimizer: adamw
weight_decay: 0.01
device: cpu
```

---

## Reproducibility & tracking
- Track code with git; include `git_commit` in run metadata.
- Use experiment trackers: Weights & Biases, MLFlow, or simple JSON logs.
- Save the exact tokenizer, config, and model checkpoint for each run.
- Fix random seeds and record hardware (GPU type, CPU, memory).

---

## Notebooks & demos
- `notebooks/01-eda.ipynb` — dataset inspection and token length histograms
- `notebooks/02-probes.ipynb` — small prompt engineering tests
- `notebooks/03-eval.ipynb` — produce evaluation tables and charts

---

## Suggested README visuals checklist (for PRs)
- [ ] Training curve (loss vs. steps)
- [ ] Validation metric curves (accuracy / BLEU / ROUGE)
- [ ] Embedding projection image
- [ ] Confusion matrix (if classification)
- [ ] Short examples table: prompt -> output -> score

---

## How to add your own results
1. Run an experiment and save run metadata and metrics to `results/run-<date>-<tag>.jsonl`.
2. Use `scripts/plot_results.py` to generate PNGs into `assets/figures/`.
3. Commit `results/` and `assets/figures/` (or publish externally and link). For large model files, use external storage or Git LFS.

---

## Example small workflow to produce visuals & results
1. `python scripts/prepare_data.py --input data/raw --output data/processed`
2. `python src/train.py --config configs/train_toy.yaml --outdir models/run-toy`
3. `python src/eval.py --model models/run-toy/checkpoint.pt --data data/processed/test.jsonl --out results/run-toy.jsonl`
4. `python scripts/plot_results.py --results results/run-toy.jsonl --out assets/figures/training_curve.png`
5. Add a short summary to README under "Results" with the key figures and link images.

---

## Development / contributing
- Use feature branches: `feature/short-desc`
- Add tests to `tests/` for new modules
- Keep `requirements.txt` updated with library versions
- Open issues for tasks or experiments to avoid duplicate work

---

## License
Choose and add a LICENSE file (MIT recommended for learning projects).

---

## Contact
Created by Krishna Kumar (https://github.com/KrishnaKumar2002). For questions open an issue or message me at https://github.com/KrishnaKumar2002
