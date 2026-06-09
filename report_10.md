

**Text Mining**   
Master’s in Data Science and Advanced Analytics   
 

Project Report 

**Group 10**  

Alexander Batista, 20250419

Mehmet Karaca, 20250344

Luis Mendes, 20221949

Veronica Mendes, 20221945

Spring Semester 2025-2026 

# **Table of Contents**

[1\. Introduction (All)	1](#1.-introduction-\(all\))

[2\. Data Exploration (Vero)	1](#2.-data-exploration-\(vero\))

[3\. Data Preprocessing (Luis)	1](#3.-data-preprocessing-\(luis\))

[4\. Classification Models (Mehmet)	1](#4.-classification-models-\(mehmet\))

[4.1 Experimental Design and Benchmark Framework	1](#4.1-experimental-design-and-benchmark-framework)

[4.2 Traditional Machine Learning Classifiers	2](#4.2-traditional-machine-learning-classifiers)

[4.2.1 Logistic Regression	2](#4.2.1-logistic-regression)

[4.2.2 Linear Support Vector Machine	2](#4.2.2-linear-support-vector-machine)

[4.2.3 Multilayer Perceptron	3](#4.2.3-multilayer-perceptron)

[4.3 Transformer Encoder Classifiers	3](#4.3-transformer-encoder-classifiers)

[4.3.1 Frozen FinBERT Embeddings with Logistic Regression	3](#4.3.1-frozen-finbert-embeddings-with-logistic-regression)

[4.3.2 Fine-tuned FinBERT for Sequence Classification	4](#4.3.2-fine-tuned-finbert-for-sequence-classification)

[4.4 Additional Transformer Encoders (Extended Experiments)	5](#4.4-additional-transformer-encoders-\(extended-experiments\))

[4.4.1 Sentence-BERT	5](#4.4.1-sentence-bert)

[4.4.2 RoBERTa	5](#4.4.2-roberta)

[4.5 Decoder-Only Language Model (Extended Experiments)	6](#4.5-decoder-only-language-model-\(extended-experiments\))

[4.6 Agentic AI Workflow (Extra Challenge 2)	6](#4.6-agentic-ai-workflow)

[4.7 Summary	7](#4.7-summary)

[5\. Evaluation and Results (Alexandre)	8](#5.-evaluation-and-results-\(alexandre\))

# 1\. Introduction (All) {#1.-introduction-(all)}

Market sentiment on social media has become a reliable signal for short-term price movement, making automated classification of financial tweets a practically relevant NLP task. This project develops a pipeline to classify tweets as Bearish (0), Bullish (1), or Neutral (2), working from a corpus of 9,543 labelled training examples with a pronounced class imbalance toward Neutral (64.7%). Our approach spans classical sparse-text methods, static word embeddings, frozen transformer encoders, and full fine-tuning of a domain-specific language model, allowing a systematic comparison across representation families rather than a search for a single best technique. The report covers exploratory data analysis (Section 2), preprocessing (Section 3), classification models (Section 4), and final evaluation (Section 5).

# 2\. Data Exploration (Vero) {#2.-data-exploration-(vero)}

Here you should analyze the corpora and provide some conclusions and visual information (bar charts, word clouds, etc.) that contextualize the data.

# 3\. Data Preprocessing (Luis) {#3.-data-preprocessing-(luis)}

Explanation of the different preprocessing methods developed.

# 4\. Classification Models (Mehmet) {#4.-classification-models-(mehmet)}

## 4.1 Experimental Design and Benchmark Framework {#4.1-experimental-design-and-benchmark-framework}

The classification stage was designed around a consistent evaluation framework: every model variant was trained on the same 7,634-sample training split and evaluated on the same held-out 1,909-sample validation set (an 80/20 stratified split). Macro-averaged F1 was the primary ranking metric throughout, a deliberate choice driven by the dataset's pronounced class imbalance — Neutral tweets constitute 64.7% of the corpus, with Bullish and Bearish accounting for only 20.2% and 15.1% respectively. Optimising for accuracy in this setting would trivially reward a model that collapses predictions toward the majority class, so macro F1 anchors all comparisons to balanced per-class performance.

Fourteen distinct classification configurations were evaluated across three broad families: traditional machine learning classifiers operating on sparse and dense feature representations, fine-tuned transformer encoder classifiers, and a decoder-only large language model used in a zero-shot prompting regime. Each family was given at least two variants to isolate the effect of specific design choices — such as class weighting, encoder freezing, and model architecture — from the choice of representation itself.

## 4.2 Traditional Machine Learning Classifiers {#4.2-traditional-machine-learning-classifiers}

Traditional classifiers were evaluated on two representations: TF-IDF vectors built from the `raw_lower_regex_clean_stemmed` preprocessing variant (5,482 dimensions, `min_df=2`) and 100-dimensional Word2Vec skip-gram embeddings averaged at the tweet level. The TF-IDF representation scored higher at the feature-benchmark stage (macro F1 0.705 with a diagnostic Logistic Regression) than Word2Vec (0.609), which informed the emphasis placed on sparse-text experiments. Dense Word2Vec embeddings were standardised with `StandardScaler` before classification, since gradient-based and distance-sensitive methods are sensitive to feature scale.

### 4.2.1 Logistic Regression {#4.2.1-logistic-regression}

Logistic Regression served as the primary linear baseline. Two variants were evaluated on TF-IDF features: a default-weighted model and a class-balanced variant that sets inverse class frequencies as sample weights during training. Both used L2 regularisation with the `lbfgs` solver and a maximum of 1,000 iterations to ensure convergence on the high-dimensional TF-IDF input.

The baseline achieved a macro F1 of 0.705 and accuracy of 0.805, while the balanced variant improved macro F1 substantially to 0.741 — the strongest result among all traditional models — at a modest cost to overall accuracy (0.799). The gap is instructive: the default model concentrates on the Neutral majority class, while the balanced variant recovers meaningful Bearish and Bullish recall. The same pairing was repeated on Word2Vec embeddings, but performance was noticeably lower (macro F1 0.609 and 0.599), suggesting that the 100-dimensional average-pooled representation loses too much discriminative lexical information for a linear classifier to recover.

### 4.2.2 Linear Support Vector Machine {#4.2.2-linear-support-vector-machine}

Linear SVMs (`LinearSVC`) were paired with TF-IDF features as a second family of linear boundary classifiers. Two variants were tested: a default model with `C=1.0` and a balanced-weight variant. Both achieved macro F1 values of approximately 0.733–0.734, sitting just below the balanced Logistic Regression. The SVM's margin-maximising objective provided no measurable advantage over the probabilistic LogReg on this high-dimensional sparse representation, a well-documented tendency for both methods to converge to similar solutions on TF-IDF input.

### 4.2.3 Multilayer Perceptron {#4.2.3-multilayer-perceptron}

Two MLP configurations were evaluated on Word2Vec embeddings: a shallow network with a single hidden layer of 100 units and a deeper configuration with layers of 128 and 64 units. Both used early stopping (max 300 epochs, `adam` solver) and upstream feature scaling.

The deeper MLP achieved macro F1 0.690 and the shallow variant 0.680 — a consistent improvement over Logistic Regression on the same Word2Vec features (0.609), confirming that the non-linear capacity of the MLP extracts more from the dense embedding space than a linear classifier can. However, both MLP results fall below the TF-IDF linear classifiers, which highlights that the 100-dimensional skip-gram representation simply does not encode enough sentiment signal to outperform well-tuned sparse methods on this corpus, regardless of the classifier's expressive power.

## 4.3 Transformer Encoder Classifiers {#4.3-transformer-encoder-classifiers}

Transformer-based classifiers were approached in two modes: frozen encoder embeddings fed to a lightweight downstream classifier, and end-to-end fine-tuning of the transformer itself on the classification task. Both modes used ProsusAI/finbert, a BERT-base variant pre-trained on financial news and earnings call transcripts, as the backbone. This choice was motivated by domain alignment: generic language model pre-training does not necessarily capture the idiosyncratic vocabulary of financial Twitter, where cashtickers, abbreviated jargon, and market-specific phrasing are common.

### 4.3.1 Frozen FinBERT Embeddings with Logistic Regression {#4.3.1-frozen-finbert-embeddings-with-logistic-regression}

In the embedding-only approach, FinBERT was run in inference mode to produce 768-dimensional mean-pooled sentence representations for every tweet, averaging final hidden-layer token vectors across the sequence (excluding padding). Encoding used a batch size of 16 and truncated at 128 tokens; the resulting embeddings were standardised before being passed to Logistic Regression.

This setup achieved a macro F1 of 0.741 (baseline) — identical to the best TF-IDF result — with overall accuracy of 0.808. The balanced variant scored 0.725, slightly lower, suggesting the class-weighting mechanism interacts poorly with the high-dimensional dense space. The result is noteworthy: FinBERT's frozen representations alone are competitive with carefully tuned sparse lexical models without any task-specific fine-tuning.

### 4.3.2 Fine-tuned FinBERT for Sequence Classification {#4.3.2-fine-tuned-finbert-for-sequence-classification}

Full fine-tuning of FinBERT with a classification head added on top of the [CLS] token yielded the strongest result across all experiments. Two fine-tuning strategies were compared: freezing all transformer encoder layers and training only the newly added classification head, and updating all model parameters jointly.

Frozen-head fine-tuning reached macro F1 of only 0.681, performing worse than the frozen-embedding Logistic Regression approach. This is a common but counterintuitive finding: a classification head trained over a frozen encoder has strictly less capacity than the same head over a fully trained encoder, and in practice the head cannot overcome the mismatch between the frozen representations and the classification boundary needed for the task.

Full fine-tuning broke decisively from this pattern. Trained with the AdamW optimiser at a learning rate of 2×10⁻⁵, a batch size of 16, for 2 epochs over 7,634 samples (padded and truncated to 128 tokens), the fully fine-tuned model achieved a macro F1 of 0.822, weighted F1 of 0.865, and overall accuracy of 0.862 on the validation set. The learning rate and epoch count follow the conventions established by the original BERT fine-tuning recommendations (Devlin et al., 2019\) and avoided overfitting on this relatively small corpus.

| Class | Precision | Recall | F1-Score | Support |
| :---- | :---- | :---- | :---- | :---- |
| Bearish | 0.705 | 0.819 | 0.758 | 288 |
| Bullish | 0.769 | 0.831 | 0.799 | 385 |
| Neutral | 0.940 | 0.881 | 0.910 | 1,236 |
| Macro avg | 0.805 | 0.844 | 0.822 | 1,909 |

Neutral performance is expectedly strong given its prevalence, but the model also generalises well to the minority classes — Bullish F1 of 0.799 and Bearish F1 of 0.758 indicate that fine-tuning on the downstream task successfully reorients the model's representations toward the sentiment distinctions that matter. This model was selected as the final submission model, with its weights saved to disk for inference on the test set.

## 4.4 Additional Transformer Encoders (Extended Experiments) {#4.4-additional-transformer-encoders-(extended-experiments)}

Beyond the required FinBERT experiments, two further encoder models were evaluated in the embedding-plus-classifier paradigm to assess whether different pre-training objectives or architectures change the picture.

### 4.4.1 Sentence-BERT {#4.4.1-sentence-bert}

Sentence-BERT (all-mpnet-base-v2) is specifically trained via contrastive learning on sentence-pair tasks, producing embeddings that are geometrically meaningful at the sentence level rather than being pooled from token-level representations. Tweets were encoded in batches of 32, yielding 768-dimensional vectors that were then standardised and passed to Logistic Regression. The resulting macro F1 of 0.758 (accuracy 0.811) places this model just above the frozen FinBERT embedding result, despite the fact that the model was not pre-trained on financial text. This suggests that Sentence-BERT's contrastive objective produces more separable embeddings for short texts like tweets, partially compensating for its lack of domain specificity.

### 4.4.2 RoBERTa {#4.4.2-roberta}

roberta-base was evaluated using the same mean-pooling strategy as frozen FinBERT: attention-mask-weighted average of the final hidden layer, encoded in batches of 16 with a maximum sequence length of 128 tokens, yielding 768-dimensional representations. RoBERTa achieved macro F1 of 0.757 (accuracy 0.817), nearly identical to Sentence-BERT despite the different pooling methodology and training objective. Both models cluster around the 0.757–0.758 macro F1 range, forming a clear performance tier between the frozen FinBERT baseline (0.741) and the fully fine-tuned FinBERT (0.822). The close parity of RoBERTa and Sentence-BERT also suggests that, at the frozen-embedding level, the choice of pooling strategy and objective matters less than the overall scale and quality of pre-training.

## 4.5 Decoder-Only Language Model (Extended Experiments) {#4.5-decoder-only-language-model-(extended-experiments)}

As an additional experiment, a decoder-only large language model was used for classification in a zero-shot prompting setup, without any gradient updates or fine-tuning. The model deepseek-chat was accessed through a DeepSeek-compatible API endpoint and prompted with a structured instruction template that specified the three sentiment categories and requested a single numeric label in response:

```
You are a financial sentiment classifier.
Classify the following tweet into exactly one category:
  0 = Bearish  |  1 = Bullish  |  2 = Neutral
Return only the numeric label.
Tweet: {tweet}
```

Responses were parsed with a regular expression to extract the first occurrence of 0, 1, or 2; the 1,909 validation tweets were classified sequentially with a generation temperature of 0.7. All predictions were successfully parsed (no rejected outputs).

The results were the weakest across all evaluated models: macro F1 of 0.602 and accuracy of 0.587. The per-class breakdown reveals the mechanism: the model strongly over-predicted Bullish (recall 0.958, precision 0.366) while severely under-recovering Neutral tweets (recall 0.458). This behaviour is consistent with a general-purpose instruction model that has not been calibrated to the corpus distribution — in a zero-shot regime it has no mechanism to infer that Neutral is the dominant class. The experiment nonetheless serves as a useful reference point: scale alone, without task adaptation, does not compete with moderately trained discriminative classifiers.

## 4.6 Agentic AI Workflow (Extra Challenge 2) {#4.6-agentic-ai-workflow}

To address Extra Challenge 2, a conversational agent was implemented in `agent_10.ipynb` that orchestrates the project's classification models as callable tools, using the DeepSeek API as the reasoning backend via the OpenAI function-calling protocol.

### Architecture and Tools

The agent exposes four tools to the underlying LLM:

| Tool | Purpose | Model invoked |
| :---- | :---- | :---- |
| `classify_fast` | Quick classification with class probabilities | TF-IDF + balanced LogReg |
| `classify_best` | High-accuracy classification with confidence scores | Fine-tuned FinBERT |
| `compare_models` | Run both classifiers and compare — core coordination tool | Both |
| `get_model_info` | Return validation metrics for both models | — |

Full message history is maintained across turns, so the agent can refer back to prior tweets or decisions without the user repeating context.

### Decision Protocol

The system prompt encodes a routing policy the LLM follows when selecting tools: quick or exploratory requests are routed to `classify_fast`; requests for maximum accuracy trigger `classify_best`; ambiguous tweets or verification requests invoke `compare_models`; questions about model performance call `get_model_info`. Critically, if the two classifiers disagree on a prediction, the agent is instructed to automatically escalate to `compare_models` and explain the discrepancy rather than silently picking one result.

### Demonstrated Capabilities

A four-turn scripted conversation in the notebook exercises every routing path:

1. **Turn 1** — *"Quickly classify…"* → `classify_fast` invoked; AAPL earnings beat classified as Bullish (89.5% confidence).
2. **Turn 2** — *"Double-check with your best model"* → `classify_best` invoked; fine-tuned FinBERT confirms Bullish with 99.0% confidence.
3. **Turn 3** — Ambiguous Fed rate-pause tweet → `compare_models` invoked automatically; TF-IDF predicts Bearish (41.7%, barely above its Bullish score of 41.3%), FinBERT predicts Neutral (73.4%); the agent defers to FinBERT and explains the conflicting market signals (rate pause = mildly bullish, persistent inflation = hawkish, divided analysts = uncertainty).
4. **Turn 4** — *"Which model should I trust?"* → `get_model_info` invoked; agent recommends FinBERT for important decisions, TF-IDF for bulk screening.

Turn 3 is the most illustrative: genuine disagreement between classifiers triggers automatic escalation, the agent surfaces both predictions with confidence scores, and resolves the conflict through reasoning about model quality — a coordination task that goes beyond direct classification. An interactive `chat()` function is also available for real-time sessions.

## 4.7 Summary {#4.7-summary}

Table 4.1 presents the full classification benchmark ranked by macro F1.

Three patterns emerge clearly from this benchmark. First, full fine-tuning of a domain-appropriate transformer is the single most effective strategy by a considerable margin — 0.822 macro F1 versus 0.741–0.758 for the next tier. Second, frozen contextual embeddings (FinBERT, Sentence-BERT, RoBERTa) and well-tuned sparse TF-IDF classifiers occupy the same performance band, with differences of less than 2 percentage points between them. This suggests that for short, noisy financial texts, the main bottleneck at the frozen-embedding stage is the inability to adapt the representation to the task, not the quality of the base embedding itself. Third, static Word2Vec embeddings consistently trail both sparse and contextual representations, and the zero-shot decoder is the weakest of all — underscoring that classification performance on this task requires either strong lexical overlap (TF-IDF) or task-adapted representations (fine-tuning).

| Rank | Model | Feature Type | Acc. | Precision | Recall | Macro F1 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | Fine-tuned FinBERT (full) | Fine-tuned sequence | 0.862 | 0.805 | 0.844 | 0.822 |
| 2 | Sentence-BERT + LogReg | Frozen encoder | 0.811 | 0.754 | 0.763 | 0.758 |
| 3 | RoBERTa + LogReg | Frozen encoder | 0.817 | 0.760 | 0.756 | 0.757 |
| 4 | FinBERT embed. + LogReg (baseline) | Frozen encoder | 0.808 | 0.747 | 0.736 | 0.741 |
| 5 | TF-IDF + LogReg (balanced) | Sparse lexical | 0.799 | 0.724 | 0.763 | 0.741 |
| 6 | TF-IDF + LinearSVM (balanced) | Sparse lexical | 0.803 | 0.731 | 0.738 | 0.734 |
| 7 | TF-IDF + LinearSVM (baseline) | Sparse lexical | 0.809 | 0.762 | 0.715 | 0.734 |
| 8 | FinBERT embed. + LogReg (balanced) | Frozen encoder | 0.774 | 0.703 | 0.762 | 0.725 |
| 9 | TF-IDF + LogReg (baseline) | Sparse lexical | 0.805 | 0.792 | 0.664 | 0.705 |
| 10 | Word2Vec + MLP (deep) | Dense static | 0.782 | 0.705 | 0.680 | 0.690 |
| 11 | Fine-tuned FinBERT (head only) | Fine-tuned head | 0.743 | 0.674 | 0.694 | 0.681 |
| 12 | Word2Vec + MLP (shallow) | Dense static | 0.786 | 0.719 | 0.657 | 0.680 |
| 13 | Word2Vec + LogReg (baseline) | Dense static | 0.747 | 0.662 | 0.583 | 0.609 |
| 14 | Decoder LLM (zero-shot) | Prompt-only | 0.587 | 0.653 | 0.686 | 0.602 |
| 15 | Word2Vec + LogReg (balanced) | Dense static | 0.662 | 0.589 | 0.645 | 0.599 |

Table 4.1 — Classification benchmark, ranked by macro F1 (validation set, n = 1,909)

# 5\. Evaluation and Results (Alexandre) {#5.-evaluation-and-results-(alexandre)}

Description of the performance of the models and main conclusions.

