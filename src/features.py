"""Feature engineering helpers for tweet sentiment experiments."""

from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


def tokenize_for_word2vec(texts: Iterable[str]) -> list[list[str]]:
    """Tokenize whitespace-normalized text for Word2Vec training and pooling."""
    return [str(text).split() for text in texts]


def train_word2vec_model(
    tokenized_texts: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    seed: int = 73,
    epochs: int = 20,
) -> Word2Vec:
    """Train a deterministic Word2Vec model for notebook feature experiments."""
    return Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=1,
        seed=seed,
        epochs=epochs,
    )


def average_word2vec_embeddings(
    tokenized_texts: list[list[str]],
    model: Word2Vec,
) -> np.ndarray:
    """Represent each document by the mean of its known Word2Vec token vectors."""
    vector_size = model.wv.vector_size
    embeddings = np.zeros((len(tokenized_texts), vector_size), dtype=np.float32)

    for row_idx, tokens in enumerate(tokenized_texts):
        token_vectors = [model.wv[token] for token in tokens if token in model.wv]
        if token_vectors:
            embeddings[row_idx] = np.mean(token_vectors, axis=0)

    return embeddings


def get_torch_device() -> str:
    """Return the best available torch device for local embedding generation."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_pool_last_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token embeddings while ignoring padded tokens."""
    token_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    pooled = (last_hidden_state * token_mask).sum(dim=1)
    token_counts = token_mask.sum(dim=1).clamp(min=1e-9)
    return pooled / token_counts


def encode_transformer_mean_pool(
    texts: Iterable[str],
    model_name: str,
    batch_size: int = 16,
    max_length: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """Encode texts with a transformer encoder and attention-mask mean pooling."""
    device = device or get_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    text_values = [str(text) for text in texts]
    pooled_batches = []

    for start in range(0, len(text_values), batch_size):
        batch_texts = text_values[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            pooled = mean_pool_last_hidden_state(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

        pooled_batches.append(pooled.cpu().numpy())

    return np.vstack(pooled_batches).astype(np.float32)


def load_or_create_feature_cache(
    cache_path: Path,
    create_features: Callable[[], np.ndarray],
) -> np.ndarray:
    """Load a cached feature array or create and persist it if missing."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return np.load(cache_path)

    features = create_features()
    np.save(cache_path, features)
    return features


def benchmark_feature_matrix(
    variant: str,
    feature_family: str,
    train_features: Any,
    val_features: Any,
    y_train: Any,
    y_val: Any,
    scale_dense: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Benchmark one feature representation on the validation split."""
    if scale_dense:
        diagnostic_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=73)),
            ]
        )
    else:
        diagnostic_model = LogisticRegression(max_iter=1000, random_state=73)

    diagnostic_model.fit(train_features, y_train)
    val_pred = diagnostic_model.predict(val_features)

    benchmark_row = {
        "variant": variant,
        "feature_family": feature_family,
        "macro_f1": f1_score(y_val, val_pred, average="macro"),
        "weighted_f1": f1_score(y_val, val_pred, average="weighted"),
        "accuracy": diagnostic_model.score(val_features, y_val),
    }
    shape_row = {
        "variant": variant,
        "feature_family": feature_family,
        "train_shape": train_features.shape,
        "val_shape": val_features.shape,
        "feature_dim": train_features.shape[1],
    }
    report = classification_report(y_val, val_pred, output_dict=True, zero_division=0)

    return benchmark_row, shape_row, report
