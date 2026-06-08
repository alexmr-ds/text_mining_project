"""Classification model helpers for sentiment experiments."""

from dataclasses import dataclass
from typing import Any, Iterable
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class SklearnModelSpec:
    """Configuration for one sklearn classification experiment."""

    variant: str
    representation: str
    classifier_family: str
    model: Any


class TextClassificationDataset(Dataset):
    """Tokenized text dataset for transformer classification."""

    def __init__(self, texts: Iterable[str], labels: Iterable[int], tokenizer, max_length: int):
        self.texts = [str(text) for text in texts]
        self.labels = [int(label) for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def set_random_seed(seed: int = 73) -> None:
    """Seed Python, NumPy, and torch for reproducible model runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_row(y_true, y_pred) -> dict[str, float]:
    """Calculate shared classification metrics."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def benchmark_sklearn_model(
    spec: SklearnModelSpec,
    train_features: Any,
    val_features: Any,
    y_train: Any,
    y_val: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit and evaluate one sklearn-style classifier."""
    spec.model.fit(train_features, y_train)
    val_pred = spec.model.predict(val_features)
    row = {
        "variant": spec.variant,
        "representation": spec.representation,
        "classifier_family": spec.classifier_family,
        **metric_row(y_val, val_pred),
    }
    report = classification_report(y_val, val_pred, output_dict=True, zero_division=0)
    return row, report


def benchmark_sklearn_specs(
    specs: list[SklearnModelSpec],
    feature_sets: dict[str, tuple[Any, Any]],
    y_train: Any,
    y_val: Any,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Benchmark multiple sklearn model specifications."""
    rows = []
    reports = {}

    for spec in specs:
        train_features, val_features = feature_sets[spec.representation]
        row, report = benchmark_sklearn_model(
            spec,
            train_features,
            val_features,
            y_train,
            y_val,
        )
        rows.append(row)
        reports[spec.variant] = report

    return pd.DataFrame(rows), reports


def freeze_transformer_encoder(model) -> None:
    """Freeze all transformer encoder parameters while leaving the classifier trainable."""
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        return

    for parameter in base_model.parameters():
        parameter.requires_grad = False


def train_transformer_classifier(
    variant: str,
    model_name: str,
    train_texts: Iterable[str],
    val_texts: Iterable[str],
    y_train: Iterable[int],
    y_val: Iterable[int],
    num_labels: int,
    device: str,
    max_length: int = 128,
    batch_size: int = 16,
    epochs: int = 2,
    learning_rate: float = 2e-5,
    freeze_encoder: bool = False,
    seed: int = 73,
    save_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fine-tune and evaluate a transformer sequence classifier.

    If *save_path* is provided the trained model and tokenizer are saved there
    with ``save_pretrained`` so they can be reloaded later for inference.
    """
    set_random_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    ).to(device)

    if freeze_encoder:
        freeze_transformer_encoder(model)

    train_dataset = TextClassificationDataset(train_texts, y_train, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, y_val, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()

    val_pred = []
    val_true = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].cpu().numpy().tolist()
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
            val_true.extend(labels)
            val_pred.extend(predictions)

    if save_path is not None:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    row = {
        "variant": variant,
        "representation": "finbert_sequence",
        "classifier_family": "Fine-tuned FinBERT",
        **metric_row(val_true, val_pred),
    }
    report = classification_report(val_true, val_pred, output_dict=True, zero_division=0)
    return row, report


def predict_transformer_classifier(
    model_path: str,
    texts: Iterable[str],
    device: str,
    max_length: int = 128,
    batch_size: int = 16,
) -> list[int]:
    """Load a saved transformer classifier and predict labels for *texts*.

    Parameters
    ----------
    model_path:
        Directory previously written by ``save_pretrained`` (from
        ``train_transformer_classifier`` with *save_path* set).
    texts:
        Raw input strings to classify.
    device:
        Torch device string, e.g. ``"cpu"``, ``"cuda"``, or ``"mps"``.
    max_length:
        Tokeniser truncation length — must match the value used during training.
    batch_size:
        Number of samples per inference batch.

    Returns
    -------
    list[int]
        Predicted integer class labels in the same order as *texts*.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    texts_list = [str(t) for t in texts]
    # Dummy labels — not used during inference but required by the Dataset class.
    dummy_labels = [0] * len(texts_list)
    dataset = TextClassificationDataset(texts_list, dummy_labels, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
            predictions.extend(preds)

    return predictions
