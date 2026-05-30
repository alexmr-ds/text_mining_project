# Preprocessing and Quality Assessment Report

This report summarizes notebook sections `3. Text Preprocessing`, `3.4 Preprocessing Quality Assessment`, and `3.5 Corrected Preprocessing Benchmark` from `notebooks/workflow.ipynb`. The goal is to document how the current preprocessing pipeline behaves, measure whether it preserves useful sentiment signal, and identify which parts of the cleaner reduce predictive power.

## Preprocessing Setup

The preprocessing stage starts from the train, validation, and test text splits. It prepares NLTK resources for stopword removal, lemmatization, and stemming:

| Resource | Purpose |
| --- | --- |
| `stopwords` | Remove common English stopwords |
| `wordnet` | Support lemmatization |
| `omw-1.4` | Support WordNet metadata used by the lemmatizer |

The notebook now works with two stopword configurations:

| Stopword set | Description |
| --- | --- |
| `stop_words` | Original stopword set used by the first cleaned baseline |
| `revised_stop_words` | Corrected stopword set that preserves selected sentiment-bearing terms |

The shared custom stopwords are:

| Custom stopword | Rationale |
| --- | --- |
| `https` | URL artifact |
| `http` | URL artifact |
| `www` | URL artifact |
| `marketscreener` | Frequent source-specific term |
| `u` | Fragment from `U.S.` token handling |
| `s` | Fragment from `U.S.` token handling |

The revised stopword configuration explicitly preserves:

- negations: `no`, `nor`, `not`
- market direction terms: `up`, `down`
- high-value financial sentiment terms: `beats`, `misses`, `cuts`, `raised`, `gain`, `loss`

This means the notebook has moved beyond the earlier recommendation to revise stopword handling: that correction is now implemented and benchmarked.

## Text Variants

The notebook currently uses these text representations:

| Variant | Description |
| --- | --- |
| `original` | Raw tweet text |
| `raw_lower` | Raw text lowercased only |
| `cleaned` | Original lowercased and cleaned text after regex filtering and stopword removal |
| `cleaned_revised` | Corrected cleaned text using the revised stopword set |
| `lemmatized` | Original cleaned text after lemmatization |
| `stemmed` | Original cleaned text after stemming |

The original `cleaned` representation remains useful as a historical baseline. The new `cleaned_revised` representation is the corrected cleaner introduced to preserve terms that were previously being removed.

## Original 3.4 Findings

The original `3.4` quality assessment still provides the baseline diagnostic picture for the first cleaning pipeline.

### Sample Comparison Across Variants

The notebook samples three examples per label and displays `original`, `cleaned`, `lemmatized`, and `stemmed` side by side. The main pattern is unchanged:

- `cleaned` removes case variation, URL artifacts, and many stopwords
- `lemmatized` makes smaller form-level changes
- `stemmed` is the most aggressive and least readable variant

These examples show that the original cleaner preserves some financial meaning, but it also removes a noticeable amount of useful surface detail.

### Text Retention

| Variant | Empty docs | Near-empty docs | Avg chars | Avg words | Avg char reduction | Avg word reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 0 | 33 | 85.96 | 12.19 | 0.00% | 0.00% |
| `cleaned` | 6 | 73 | 62.33 | 8.89 | 27.49% | 27.06% |
| `lemmatized` | 6 | 73 | 61.22 | 8.89 | 28.78% | 27.06% |
| `stemmed` | 6 | 73 | 56.44 | 8.89 | 34.34% | 27.06% |

The original cleaner is aggressive. It increases near-empty documents from `33` to `73` and produces `6` empty documents. This remains an important warning sign because short financial tweets can still contain high-value sentiment cues.

### Largest Cleaning Reductions

The most heavily reduced tweets are mostly mentions, URLs, or short structural posts. Some collapse to an empty string; others retain only a single token such as `sell`, `hour`, or `open`.

This confirms that the cleaner removes obvious noise, but it can also strip away too much content from already short tweets.

### Original Stopword and Signal Audit

| Check | Tokens |
| --- | --- |
| Negations removed as stopwords | `no`, `nor`, `not` |
| Financial terms removed as stopwords | `down`, `up` |
| Financial terms preserved | `$`, `%`, `beat`, `beats`, `cut`, `cuts`, `fall`, `falls`, `gain`, `gains`, `loss`, `losses`, `miss`, `misses`, `raise`, `raises` |

This was the central problem in the original cleaner. It removed both negations and market direction terms, which are directly tied to financial sentiment.

### Removed and Remaining Tokens

The removed-token list is dominated by function words such as `to`, `the`, `of`, `in`, and `and`, which is expected. However, `up` also appears among the most frequently removed tokens, at `359` occurrences, which confirms that the original stopword list removed a strong Bullish signal.

The remaining-token inspection still shows useful vocabulary:

| Sentiment | Examples of retained tokens |
| --- | --- |
| Bearish | `misses`, `cut`, `coronavirus`, `oil`, `price` |
| Bullish | `beats`, `target`, `raised`, `eps`, `hedge` |
| Neutral | `results`, `earnings`, `dividend`, `reports`, `call`, `transcript` |

### Label-Concentrated Tokens

Many strongly concentrated terms are Neutral-specific, including `transcript`, `edited`, `gmt`, `preview`, `buy?`, `mutual`, `retirement`, `de`, `impeachment`, and `presentation.` These terms act more like headline or content-distribution markers than direct sentiment words, but they still help separate Neutral tweets from Bearish and Bullish ones.

### Original Diagnostic Benchmark

| Variant | Macro F1 | Weighted F1 | Accuracy |
| --- | ---: | ---: | ---: |
| `raw_lower` | 0.6833 | 0.7768 | 0.7968 |
| `stemmed` | 0.6721 | 0.7658 | 0.7852 |
| `lemmatized` | 0.6509 | 0.7543 | 0.7784 |
| `cleaned` | 0.6434 | 0.7498 | 0.7763 |

The original conclusion still holds: `raw_lower` is stronger than the original `cleaned`, `lemmatized`, and `stemmed` variants. The original cleaning pipeline removes more predictive information than it adds.

## Corrected Preprocessing Benchmark (`3.5`)

Section `3.5` implements the corrected stopword logic and compares the new cleaner against `raw_lower` and the original cleaned baseline.

### Preserved-Term Audit

| Check | Tokens |
| --- | --- |
| `preserved_terms_still_in_stopwords` | `[]` |
| `preserved_terms_available_for_model` | `beats`, `cuts`, `down`, `gain`, `loss`, `misses`, `no`, `nor`, `not`, `raised`, `up` |

This confirms that the corrected cleaner no longer removes the specific sentiment-bearing terms identified in the earlier report.

### Step Inspection

The notebook now shows a side-by-side dataframe with these stages:

- `original`
- `cleaned`
- `raw_lower`
- `raw_lower_regex_clean`
- `cleaned_revised`
- `revised_without_regex_clean`
- `revised_without_stopword_removal`

This makes the effect of each stage visible. For example, the tweet `Futures up https://...` becomes:

- `cleaned`: `futures`
- `cleaned_revised`: `futures up`

That is the intended correction: the revised cleaner still removes the URL, but it now preserves the directional term `up`.

### Revised Benchmark Results

| Variant | Macro F1 | Weighted F1 | Accuracy |
| --- | ---: | ---: | ---: |
| `raw_lower` | 0.6833 | 0.7768 | 0.7968 |
| `cleaned_revised` | 0.6558 | 0.7580 | 0.7826 |
| `cleaned` | 0.6434 | 0.7498 | 0.7763 |

Interpretation:

- `cleaned_revised` improves over the original `cleaned` baseline by `+0.0123` macro F1
- `cleaned_revised` still trails `raw_lower` by `-0.0275` macro F1
- the stopword correction helps, but it does not fully recover the predictive power lost by cleaning

The strongest variant in the corrected benchmark remains `raw_lower`.

### Per-Class Results for the Stronger Raw-vs-Corrected Option

Because `raw_lower` still wins the corrected benchmark, the notebook prints the detailed classification report for `raw_lower`:

| Sentiment | Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bearish | `0` | 0.8060 | 0.3750 | 0.5118 | 288 |
| Bullish | `1` | 0.7797 | 0.5792 | 0.6647 | 385 |
| Neutral | `2` | 0.7992 | 0.9628 | 0.8734 | 1236 |

The class-level interpretation is unchanged: Neutral is easiest, Bearish recall is weakest, and macro F1 remains more informative than accuracy for model comparison.

## Component Ablation

The notebook now isolates the revised cleaner's components in two complementary ways.

### Cumulative View

This view asks: what happens as stages are added one by one?

| Variant | Macro F1 | Weighted F1 | Accuracy |
| --- | ---: | ---: | ---: |
| `raw_lower` | 0.6833 | 0.7768 | 0.7968 |
| `raw_lower_regex_clean` | 0.6850 | 0.7780 | 0.7978 |
| `cleaned_revised` | 0.6558 | 0.7580 | 0.7826 |

Interpretation:

- adding regex cleaning slightly improves macro F1 by `+0.0017`
- adding stopword removal after regex cleaning reduces macro F1 by `-0.0292`

Regex cleaning appears neutral to slightly helpful. The large drop happens when stopword removal is applied.

### Leave-One-Out View

This view asks: if the full revised cleaner is used, what happens when one stage is removed?

| Variant | Macro F1 | Weighted F1 | Accuracy |
| --- | ---: | ---: | ---: |
| `cleaned_revised` | 0.6558 | 0.7580 | 0.7826 |
| `revised_without_regex_clean` | 0.6552 | 0.7576 | 0.7821 |
| `revised_without_stopword_removal` | 0.6850 | 0.7780 | 0.7978 |

Interpretation:

- removing regex cleaning slightly hurts performance by `-0.0006`
- removing stopword removal improves macro F1 by `+0.0292`

This confirms the cumulative result: regex cleaning is not the problem, while stopword removal is the dominant source of performance loss in the revised cleaner.

### Ablation Conclusion

The notebook's summary cell reports:

- largest cumulative drop: `add_stopword_removal`
- largest leave-one-out improvement: `remove_stopword_removal`

The practical conclusion is direct: stopword removal remains the main performance bottleneck, even after preserving the most obvious sentiment-bearing terms.

## Conclusion

The current notebook evidence supports three conclusions.

1. `raw_lower` remains the strongest baseline for feature engineering.
   It still has the best overall diagnostic performance, with macro F1 `0.6833`, weighted F1 `0.7768`, and accuracy `0.7968`.

2. `cleaned_revised` is meaningfully better than the original `cleaned` baseline.
   Preserving negations and key financial direction terms improves macro F1 from `0.6434` to `0.6558`.

3. The remaining weakness is mainly stopword removal, not regex cleaning.
   The ablation results show that removing URLs and mentions is slightly helpful, while stopword removal is still responsible for the largest drop in predictive power.

The best current recommendation is therefore:

- use `raw_lower` as the primary baseline for downstream feature engineering
- keep `cleaned_revised` as the better cleaned alternative for comparison experiments
- avoid broader stopword removal unless it becomes more selective or is redesigned around domain-specific evidence
