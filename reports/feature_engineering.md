# Feature Engineering Report

This report documents the completed results from section `4. Feature Engineering` in `notebooks/workflow.ipynb`. The notebook benchmarked four required feature families: Bag of Words, TF-IDF, Word2Vec, and a Transformer Encoder representation using FinBERT.

## Feature Variants

| Variant | Feature family | Text input | Implementation |
| --- | --- | --- | --- |
| `bow_count_raw_lower` | BoW | `raw_lower` | `CountVectorizer(min_df=2)` |
| `tfidf_raw_lower` | TF-IDF | `raw_lower` | `TfidfVectorizer(min_df=2)` |
| `word2vec_skipgram_mean_raw_lower` | Word2Vec | `raw_lower` | skip-gram Word2Vec with mean-pooled token vectors |
| `finbert_mean_pool_original` | Transformer Encoder | original raw text | `ProsusAI/finbert` last-hidden-state mean pooling |

The preprocessing assessment still drives the text choice. Lowercase-only raw text remained the best classical baseline, so BoW, TF-IDF, and Word2Vec use `raw_lower`. FinBERT uses the original tweet text because the pretrained tokenizer expects the raw sequence format.

## Implemented Representations

The sparse representations are built directly in the notebook:

- BoW uses token counts with `min_df=2`.
- TF-IDF uses the same vocabulary threshold and weighting over token counts.

The dense representations use `src/features.py` helpers:

- Word2Vec is trained on the training tweets with `sg=1`, `vector_size=100`, `window=5`, `min_count=2`, `epochs=20`, and `seed=73`.
- Tweet-level Word2Vec features are produced by averaging all in-vocabulary token vectors in each tweet.
- FinBERT embeddings are generated with `ProsusAI/finbert` using attention-mask mean pooling over the final hidden states, with `batch_size=16` and `max_length=128`.
- Dense feature arrays are cached under `data/processed/features/` so repeated notebook runs can reuse the generated embeddings.

During the recorded local run, FinBERT embeddings were generated on `mps`. That device choice is environment-specific and does not change the representation definition.

## Benchmark Setup

All feature variants are compared with the same diagnostic classifier:

| Feature type | Diagnostic model |
| --- | --- |
| Sparse BoW and TF-IDF | `LogisticRegression(max_iter=1000, random_state=73)` |
| Dense Word2Vec and FinBERT | `StandardScaler()` followed by `LogisticRegression(max_iter=1000, random_state=73)` |

The benchmark reports validation `macro_f1`, `weighted_f1`, and `accuracy`. Macro F1 is the primary comparison metric because the dataset is class-imbalanced and the Bearish and Bullish classes matter beyond majority-class accuracy.

## Observed Results

The notebook produced the following feature matrix shapes:

| Variant | Train shape | Validation shape | Feature dimension |
| --- | --- | --- | --- |
| `bow_count_raw_lower` | `(7634, 6292)` | `(1909, 6292)` | `6292` |
| `tfidf_raw_lower` | `(7634, 6292)` | `(1909, 6292)` | `6292` |
| `word2vec_skipgram_mean_raw_lower` | `(7634, 100)` | `(1909, 100)` | `100` |
| `finbert_mean_pool_original` | `(7634, 768)` | `(1909, 768)` | `768` |

The validation benchmark ranked the feature families as follows:

| Rank | Variant | Feature family | Macro F1 | Weighted F1 | Accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | `finbert_mean_pool_original` | Transformer Encoder | `0.7412` | `0.8063` | `0.8078` |
| 2 | `bow_count_raw_lower` | BoW | `0.7076` | `0.7914` | `0.7999` |
| 3 | `tfidf_raw_lower` | TF-IDF | `0.6833` | `0.7768` | `0.7968` |
| 4 | `word2vec_skipgram_mean_raw_lower` | Word2Vec | `0.5970` | `0.7235` | `0.7470` |

## Conclusion

The strongest representation from the completed run is `finbert_mean_pool_original`. It leads the comparison on all tracked metrics and improves the primary macro F1 score from the best sparse baseline (`bow_count_raw_lower`, `0.7076`) to `0.7412`.

For the next modeling stage, `finbert_mean_pool_original` should be treated as the main feature candidate, while the other three variants remain documented comparison baselines.
