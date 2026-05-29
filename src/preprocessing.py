"""Reusable text preprocessing helpers for the notebook workflow."""

import re

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def ensure_nltk_resource(resource_path, download_name):
    """Ensure an NLTK resource is available before it is used."""
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name)


def regex_clean(text):
    """Remove URLs, mentions, and redundant whitespace from text."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_stopwords(text, stop_words):
    """Remove configured stopwords from a whitespace-tokenized string."""
    tokens = text.split()

    filtered_tokens = [token for token in tokens if token not in stop_words]

    return " ".join(filtered_tokens)


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
    text = text.lower()
    text = regex_clean(text)
    return remove_stopwords(text, stop_words)
