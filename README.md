# Text Mining Project

Small notebook-first text-mining workspace for exploring a train/test CSV dataset with reusable preprocessing, feature-engineering, classification, and agentic workflow helpers.

## Status

This repository centers on three project notebooks:

- `notebooks/tm_tests_10.ipynb`: experimentation notebook for exploration, preprocessing, feature engineering, model benchmarking, and extra-credit experiments.
- `notebooks/tm_final_10.ipynb`: final ready-to-run solution notebook.
- `notebooks/agent_10.ipynb`: conversational agent workflow that routes classification requests through project model tools.

Reusable logic is factored into `src/preprocessing.py`, `src/plotting.py`, `src/features.py`, and `src/models.py`. Generated FinBERT artifacts are cached locally under `data/processed/` when produced and are not committed.

The current strongest classification model verified from the notebook outputs is `finbert_full_finetune`, which reached validation macro F1 `0.8221`, weighted F1 `0.8645`, and accuracy `0.8617`. The strongest traditional ML fallback is `tfidf_logreg_balanced`, using `raw_lower_regex_clean_stemmed` features.

## Setup

This project uses `uv` and Python 3.13.

```bash
uv sync
uv run python -m ipykernel install --user --name text-mining-project --display-name "Python (text-mining-project)"
```

Open the current notebooks in a notebook client such as VS Code or JupyterLab and select the `Python (text-mining-project)` kernel. Use `notebooks/tm_tests_10.ipynb` for the full experiment history, `notebooks/tm_final_10.ipynb` for the final pipeline, and `notebooks/agent_10.ipynb` for the agentic workflow.

FinBERT training and feature-generation artifacts are written under `data/processed/`, which remains ignored by git. A fresh clone will recreate the required generated artifacts when the relevant notebook sections are run.

For the optional decoder-only extra-credit and agent sections, create a project-root `.env` from `.env.example` and set these variables:

- `OPENROUTER_API_KEY`
- `OPENAI_API_BASE`
- `OPENAI_MODEL`
- `OPENAI_TEMPERATURE`
- `OPENAI_MAX_TOKENS`

The notebooks load `.env` automatically from the project root. `.env.example` defaults `OPENAI_API_BASE` to OpenRouter and uses `deepseek/deepseek-v4-flash` as a sample model configuration. The decoder-only extra-credit section runs across the full validation split, so rerunning that section can generate substantially more API calls than earlier capped experiments.

## File Tree

```text
.
|-- .env.example
|-- .gitignore
|-- README.md
|-- data/
|   `-- raw/
|       |-- test.csv
|       `-- train.csv
|-- notebooks/
|   |-- agent_10.ipynb
|   |-- tm_final_10.ipynb
|   `-- tm_tests_10.ipynb
|-- pyproject.toml
|-- src/
|   |-- __init__.py
|   |-- features.py
|   |-- models.py
|   |-- plotting.py
|   `-- preprocessing.py
`-- uv.lock
```

## File Purpose

- `.env.example`: Template environment variables for optional OpenAI-compatible API sections.
- `.gitignore`: Repo-local ignore rules for generated files, feature caches, local environment state, and agent-support files.
- `README.md`: Project overview, current status, setup, and maintained repository map.
- `data/raw/train.csv`: Raw labelled training dataset for text-mining experiments.
- `data/raw/test.csv`: Raw unlabelled test dataset paired with the training data.
- `notebooks/agent_10.ipynb`: Agentic AI workflow notebook with callable sentiment-classification tools.
- `notebooks/tm_final_10.ipynb`: Final ready-to-run solution notebook for the selected pipeline.
- `notebooks/tm_tests_10.ipynb`: Main experimentation notebook for exploration, preprocessing, features, models, and evaluation.
- `pyproject.toml`: Project metadata and Python dependency declarations managed with `uv`.
- `src/__init__.py`: Marks `src` as a Python package.
- `src/features.py`: Reusable feature-engineering helpers for Word2Vec embeddings, transformer encoders, and feature caches.
- `src/models.py`: Reusable classification helpers for sklearn benchmarks and FinBERT sequence-classifier training.
- `src/plotting.py`: Reusable EDA plotting helpers for notebook histogram and relative-frequency visualizations.
- `src/preprocessing.py`: Reusable text preprocessing helpers such as regex cleaning, stopword removal, lemmatization, and stemming.
- `uv.lock`: Locked dependency resolution for reproducible environments.
