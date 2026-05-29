# Text Mining Project

Small text-mining workspace for exploring a train/test CSV dataset with a notebook-driven workflow and reusable preprocessing helpers.

## Status

This repository is in an early exploratory stage. The current workflow centers on `notebooks/workflow.ipynb`, with `src/preprocessing.py` providing shared cleaning and normalization utilities for downstream experiments.

## Setup

This project uses `uv` and Python 3.13.

```bash
uv sync
uv run jupyter notebook
```

Open `notebooks/workflow.ipynb` and run the notebook in a clean kernel.

## File Tree

```text
.
|-- .gitignore
|-- .python-version
|-- README.md
|-- data/
|   `-- raw/
|       |-- test.csv
|       `-- train.csv
|-- main.py
|-- notebooks/
|   `-- workflow.ipynb
|-- pyproject.toml
|-- src/
|   |-- __init__.py
|   `-- preprocessing.py
`-- uv.lock
```

## File Purpose

- `.gitignore`: Repo-local ignore rules for generated files and local environment state.
- `.python-version`: Local Python version used for this project environment.
- `README.md`: Project overview, current status, setup, and maintained repository map.
- `data/raw/train.csv`: Raw training dataset for text-mining experiments.
- `data/raw/test.csv`: Raw test dataset paired with the training data.
- `main.py`: Minimal Python entrypoint for the project package.
- `notebooks/workflow.ipynb`: Exploratory notebook for the current text-mining workflow.
- `pyproject.toml`: Project metadata and Python dependency declarations managed with `uv`.
- `src/__init__.py`: Marks `src` as a Python package.
- `src/preprocessing.py`: Reusable text preprocessing helpers such as regex cleaning, stopword removal, lemmatization, and stemming.
- `uv.lock`: Locked dependency resolution for reproducible environments.
