"""Reusable text preprocessing helpers for the notebook workflow."""

import re

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def ensure_nltk_resource(resource_path, download_name):
    """Ensure an NLTK resource is available before it is used."""
    candidates = [resource_path]
    if not resource_path.endswith(".zip"):
        candidates.append(f"{resource_path}.zip")

    for candidate in candidates:
        try:
            nltk.data.find(candidate)
            return
        except LookupError:
            continue

    nltk.download(download_name)


def regex_clean(text):
    """Remove URLs, mentions, and redundant whitespace from text."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lowercase_text(text):
    """Lowercase text before later preprocessing steps are applied."""
    return text.lower()


def remove_stopwords(text, stop_words):
    """Remove configured stopwords from a whitespace-tokenized string."""
    tokens = text.split()

    filtered_tokens = [token for token in tokens if token not in stop_words]

    return " ".join(filtered_tokens)


def build_revised_stop_words(english_stop_words, custom_stopwords, preserve_terms):
    """Build a stopword set that preserves explicitly sentiment-bearing terms."""
    return (set(english_stop_words) | set(custom_stopwords)) - set(preserve_terms)


def lemmatize_text(text):
    """Lemmatize all tokens in a whitespace-tokenized string."""
    tokens = text.split()

    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(lemmas)


def stem_text(text):
    """Stem all tokens in a whitespace-tokenized string."""
    tokens = text.split()

    stems = [stemmer.stem(token) for token in tokens]

    return " ".join(stems)


def preprocess_text(text, stop_words):
    """Apply the notebook's default lowercase-clean-filter preprocessing flow."""
    text = lowercase_text(text)
    text = regex_clean(text)
    return remove_stopwords(text, stop_words)


def preprocess_text_revised(text, stop_words):
    """Apply a revised cleaner that preserves selected sentiment-bearing tokens."""
    text = lowercase_text(text)
    text = regex_clean(text)
    return remove_stopwords(text, stop_words)


def benchmark_variants(variants, y_train, y_val):
    """Benchmark preprocessing text variants with a shared TF-IDF classifier."""
    rows = []

    for variant, (train_text, val_text) in variants.items():
        diagnostic_model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(min_df=2)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=73)),
            ]
        )
        diagnostic_model.fit(train_text, y_train)
        val_pred = diagnostic_model.predict(val_text)
        rows.append(
            {
                "variant": variant,
                "macro_f1": f1_score(y_val, val_pred, average="macro"),
                "weighted_f1": f1_score(y_val, val_pred, average="weighted"),
                "accuracy": diagnostic_model.score(val_text, y_val),
            }
        )

    return pd.DataFrame(rows).set_index("variant")
