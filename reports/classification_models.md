# Classification Models

This report documents the classification-model benchmark implemented in Section 5 and evaluated in Section 6 of `notebooks/workflow.ipynb`. The setup keeps the selected preprocessing choice from the component-ablation work: sparse and Word2Vec features use `raw_lower_regex_clean_stemmed`, while FinBERT-based methods use the original text because the transformer tokenizer expects natural-language input.

## Model Coverage

| Feature engineering | Classifier variation 1 | Classifier variation 2 | Purpose |
| --- | --- | --- | --- |
| TF-IDF on `raw_lower_regex_clean_stemmed` | Logistic Regression | Linear SVM | Strong sparse-text baselines with linear decision boundaries. |
| Word2Vec mean embeddings on `raw_lower_regex_clean_stemmed` | Logistic Regression | MLP | Tests whether dense static embeddings help linear and nonlinear classifiers. |
| FinBERT transformer encoder | Logistic Regression on mean-pooled embeddings | Fine-tuned FinBERT sequence classifier | Compares transformer feature extraction against end-to-end transformer classification. |

Each classifier family also includes two variations where useful: default versus class-balanced linear models, shallow versus deeper MLP, and classifier-head-only versus full FinBERT fine-tuning.

## Final Evaluation Results

| Rank | Variant | Representation | Classifier family | Macro F1 | Weighted F1 | Accuracy |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `finbert_full_finetune` | `finbert_sequence` | Fine-tuned FinBERT | 0.8212 | 0.8653 | 0.8633 |
| 2 | `finbert_embedding_logreg_baseline` | `finbert_mean_pool_original` | FinBERT Embeddings + Logistic Regression | 0.7412 | 0.8063 | 0.8078 |
| 3 | `tfidf_logreg_balanced` | `tfidf_raw_lower_regex_clean_stemmed` | Logistic Regression | 0.7409 | 0.8045 | 0.7994 |
| 4 | `tfidf_linear_svm_balanced` | `tfidf_raw_lower_regex_clean_stemmed` | Linear SVM | 0.7339 | 0.8039 | 0.8030 |
| 5 | `tfidf_linear_svm_baseline` | `tfidf_raw_lower_regex_clean_stemmed` | Linear SVM | 0.7337 | 0.8037 | 0.8088 |
| 6 | `finbert_embedding_logreg_balanced` | `finbert_mean_pool_original` | FinBERT Embeddings + Logistic Regression | 0.7260 | 0.7822 | 0.7748 |
| 7 | `tfidf_logreg_baseline` | `tfidf_raw_lower_regex_clean_stemmed` | Logistic Regression | 0.7045 | 0.7895 | 0.8051 |
| 8 | `word2vec_mlp_deep` | `word2vec_skipgram_mean_raw_lower_regex_clean_stemmed` | MLP | 0.6895 | 0.7769 | 0.7816 |
| 9 | `finbert_classifier_head_only` | `finbert_sequence` | Fine-tuned FinBERT | 0.6813 | 0.7452 | 0.7433 |
| 10 | `word2vec_mlp_shallow` | `word2vec_skipgram_mean_raw_lower_regex_clean_stemmed` | MLP | 0.6804 | 0.7753 | 0.7863 |
| 11 | `word2vec_logreg_baseline` | `word2vec_skipgram_mean_raw_lower_regex_clean_stemmed` | Logistic Regression | 0.6088 | 0.7266 | 0.7465 |
| 12 | `word2vec_logreg_balanced` | `word2vec_skipgram_mean_raw_lower_regex_clean_stemmed` | Logistic Regression | 0.5994 | 0.6826 | 0.6616 |

Section 6 uses the stored classification reports to add the explicit precision, recall, F1, and accuracy comparison required by the rubric. Accuracy measures overall correctness, but the dataset is imbalanced toward Neutral tweets, so accuracy alone can hide weak Bearish or Bullish detection. Precision measures how reliable a predicted sentiment label is, recall measures how many true examples of that label are recovered, and F1 balances those two views. Macro F1 is the primary final metric because it weights Bearish, Bullish, and Neutral performance equally.

## Conclusion

The best-performing model is `finbert_full_finetune`, with validation macro F1 `0.8212`. It is the preferred classification model because it improves clearly over both the best traditional sparse-text setup and the FinBERT embedding-extraction baseline.

For a faster or more explainable fallback, `tfidf_logreg_balanced` is the strongest traditional ML option, reaching macro F1 `0.7409` while using the selected `raw_lower_regex_clean_stemmed` preprocessing pipeline.
