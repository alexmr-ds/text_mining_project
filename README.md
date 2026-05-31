# Text Mining Project

Small text-mining workspace for exploring a train/test CSV dataset with a notebook-driven workflow and reusable preprocessing, feature-engineering, and classification-model helpers.

## Status

This repository is in an exploratory notebook-driven stage with preprocessing, feature-engineering, and classification-model benchmarks now implemented in `notebooks/workflow.ipynb`. The current workflow centers on `src/preprocessing.py`, `src/plotting.py`, `src/features.py`, and `src/models.py` for shared preprocessing, EDA plotting, feature-engineering utilities, and classification benchmarks. Standalone reports in `reports/` document data exploration, preprocessing quality assessment, feature engineering, and classification models, with notebook-backed figures exported to `reports/figures/`.

The current strongest classification model is `finbert_full_finetune`, which reached validation macro F1 `0.8212`, weighted F1 `0.8653`, and accuracy `0.8633`. The strongest traditional ML fallback is `tfidf_logreg_balanced`, using `raw_lower_regex_clean_stemmed` features.

## Setup

This project uses `uv` and Python 3.13.

```bash
uv sync
uv run jupyter notebook
```

Open `notebooks/workflow.ipynb` and run the notebook in a clean kernel.

The first FinBERT run downloads `ProsusAI/finbert` through Hugging Face and caches generated feature arrays under `data/processed/`, which is ignored by git.

## File Tree

```text
.
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

- `.gitignore`: Repo-local ignore rules for generated files, feature caches, and local environment state.
- `README.md`: Project overview, current status, setup, and maintained repository map.
- `data/raw/train.csv`: Raw training dataset for text-mining experiments.
- `data/raw/test.csv`: Raw test dataset paired with the training data.
- `notebooks/workflow.ipynb`: Exploratory notebook for the current text-mining workflow.
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
- `src/features.py`: Reusable feature-engineering helpers for Word2Vec embeddings, FinBERT embeddings, and feature caches.
- `src/models.py`: Reusable classification helpers for sklearn benchmarks and FinBERT sequence-classifier training.
- `src/plotting.py`: Reusable EDA plotting helpers for notebook histogram and relative-frequency visualizations.
- `src/preprocessing.py`: Reusable text preprocessing helpers such as regex cleaning, stopword removal, lemmatization, and stemming.
- `tests/__init__.py`: Marks the local unittest suite as an importable test package.
- `tests/test_models.py`: Unit tests for reusable classification metric and sklearn benchmark helpers.
- `uv.lock`: Locked dependency resolution for reproducible environments.
