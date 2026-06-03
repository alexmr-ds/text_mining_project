# Text Mining Project

Small notebook-first text-mining workspace for exploring a train/test CSV dataset with reusable preprocessing, feature-engineering, classification, and reporting helpers.

## Status

This repository centers on `notebooks/workflow.ipynb`, with reusable logic factored into `src/preprocessing.py`, `src/plotting.py`, `src/features.py`, and `src/models.py`. The notebook covers data exploration, preprocessing assessment, feature engineering, classical baselines, FinBERT benchmarks, and optional extra-credit transformer and decoder-only experiments. Standalone reports in `reports/` summarize the notebook findings and reuse exported figures from `reports/figures/`.

The current strongest classification model is `finbert_full_finetune`, which reached validation macro F1 `0.8277`, weighted F1 `0.8699`, and accuracy `0.8685`. The strongest traditional ML fallback is `tfidf_logreg_balanced`, using `raw_lower_regex_clean_stemmed` features.

## Setup

This project uses `uv` and Python 3.13.

```bash
uv sync
uv run python -m ipykernel install --user --name text-mining-project --display-name "Python (text-mining-project)"
```

Open `notebooks/workflow.ipynb` in a notebook client such as VS Code or JupyterLab and select the `Python (text-mining-project)` kernel.

The first FinBERT run downloads `ProsusAI/finbert` through Hugging Face and caches generated feature arrays under `data/processed/`, which is ignored by git.

For the optional decoder-only extra-credit section, create a project-root `.env` from `.env.example` and set these variables:

- `OPENROUTER_API_KEY`
- `OPENAI_API_BASE`
- `OPENAI_MODEL`
- `OPENAI_TEMPERATURE`
- `OPENAI_MAX_TOKENS`

The notebook loads `.env` automatically from the project root. `.env.example` defaults `OPENAI_API_BASE` to OpenRouter and uses `deepseek/deepseek-v4-flash` as a sample model configuration. The decoder-only extra-credit section now runs across the full validation split, so rerunning that section is slower and can generate substantially more API calls than the earlier capped setup.

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
|   `-- workflow.ipynb
|-- pyproject.toml
|-- reports/
|   |-- classification_models.md
|   |-- data_exploration.md
|   |-- feature_engineering.md
|   |-- figures/
|   |   |-- ticker_symbols_by_sentiment.png
|   |   |-- tweet_length_by_sentiment.png
|   |   |-- tweet_length_distributions.png
|   |   `-- url_mention_presence_by_sentiment.png
|   `-- preprocessing_quality_assessment.md
|-- src/
|   |-- __init__.py
|   |-- features.py
|   |-- models.py
|   |-- plotting.py
|   `-- preprocessing.py
|-- tests/
|   |-- __init__.py
|   `-- test_models.py
`-- uv.lock
```

## File Purpose

- `.env.example`: Template environment variables for the optional decoder-only notebook section that uses an OpenAI-compatible API.
- `.gitignore`: Repo-local ignore rules for generated files, feature caches, local environment state, and agent-support files.
- `README.md`: Project overview, current status, setup, and maintained repository map.
- `data/raw/train.csv`: Raw training dataset for text-mining experiments.
- `data/raw/test.csv`: Raw test dataset paired with the training data.
- `notebooks/workflow.ipynb`: Primary notebook workflow for exploration, modeling, and optional extra-credit experiments.
- `pyproject.toml`: Project metadata and Python dependency declarations managed with `uv`.
- `reports/classification_models.md`: Standalone report for the traditional ML and transformer classification benchmark.
- `reports/data_exploration.md`: Standalone report for the data exploration findings.
- `reports/feature_engineering.md`: Standalone report for the completed feature-engineering benchmark and selected feature representation.
- `reports/figures/`: Exported notebook figures used by the Markdown reports.
- `reports/figures/ticker_symbols_by_sentiment.png`: Ticker-count distribution figure by sentiment label.
- `reports/figures/tweet_length_by_sentiment.png`: Tweet length distribution figure split by sentiment label.
- `reports/figures/tweet_length_distributions.png`: Overall tweet character and word length distribution figure.
- `reports/figures/url_mention_presence_by_sentiment.png`: URL and mention presence figure by sentiment label.
- `reports/preprocessing_quality_assessment.md`: Standalone report for preprocessing and quality assessment findings.
- `src/__init__.py`: Marks `src` as a Python package.
- `src/features.py`: Reusable feature-engineering helpers for Word2Vec embeddings, transformer encoders, and feature caches.
- `src/models.py`: Reusable classification helpers for sklearn benchmarks and FinBERT sequence-classifier training.
- `src/plotting.py`: Reusable EDA plotting helpers for notebook histogram and relative-frequency visualizations.
- `src/preprocessing.py`: Reusable text preprocessing helpers such as regex cleaning, stopword removal, lemmatization, and stemming.
- `tests/__init__.py`: Marks the local unittest suite as an importable test package.
- `tests/test_models.py`: Unit tests for reusable classification metric and sklearn benchmark helpers.
- `uv.lock`: Locked dependency resolution for reproducible environments.
