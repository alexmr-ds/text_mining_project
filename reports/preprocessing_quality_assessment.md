# Preprocessing and Quality Assessment Report

This report summarizes notebook sections `3. Text Preprocessing`, `3.4 Preprocessing Quality Assessment`, and `3.5 Preprocessing Combination Benchmark` from `notebooks/workflow.ipynb`. The goal is to document how the current preprocessing pipeline behaves, measure whether it preserves useful sentiment signal, and identify which preprocessing combination performs best.

## Preprocessing Setup

The preprocessing stage starts from the train, validation, and test text splits. It prepares NLTK resources for stopword removal, lemmatization, and stemming:

| Resource | Purpose |
| --- | --- |
| `stopwords` | Remove common English stopwords |
| `wordnet` | Support lemmatization |
| `omw-1.4` | Support WordNet metadata used by the lemmatizer |

The notebook uses one stopword configuration:

| Stopword set | Description |
| --- | --- |
| `stop_words` | English stopwords plus a small set of URL/source artifacts |

The shared custom stopwords are:

| Custom stopword | Rationale |
| --- | --- |
| `https` | URL artifact |
| `http` | URL artifact |
| `www` | URL artifact |
| `marketscreener` | Frequent source-specific term |
| `u` | Fragment from `U.S.` token handling |
| `s` | Fragment from `U.S.` token handling |

## Text Variants

The notebook first inspects these baseline text representations:

| Variant | Description |
| --- | --- |
| `original` | Raw tweet text |
| `raw_lower` | Raw text lowercased only |
| `cleaned` | Original lowercased and cleaned text after regex filtering and stopword removal |
| `lemmatized` | Original cleaned text after lemmatization |
| `stemmed` | Original cleaned text after stemming |

Section `3.5` then expands this into a full combination benchmark over regex cleaning, stopword removal, lemmatization, and stemming.

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

This diagnostic was limited to the original four variants. Section `3.5` broadens the comparison and changes the selected preprocessing choice.

## Preprocessing Combination Benchmark (`3.5`)

Section `3.5` compares all meaningful combinations of the preprocessing strategies used in the notebook. Each row is evaluated with the same TF-IDF Logistic Regression diagnostic model.

The benchmark dimensions are:

- `raw_lower`: original tweet text converted to lowercase
- `regex_clean`: URL, mention, and repeated-whitespace removal
- `stopwords`: removal using the original `stop_words` list
- `lemmatized`: token lemmatization
- `stemmed`: token stemming

Lemmatization and stemming are treated as alternatives, not stacked transformations.

### Combination Results

| Variant | Macro F1 | Weighted F1 | Accuracy |
| --- | ---: | ---: | ---: |
| `raw_lower_regex_clean_stemmed` | 0.7045 | 0.7895 | 0.8051 |
| `raw_lower_stemmed` | 0.7019 | 0.7881 | 0.8041 |
| `raw_lower_lemmatized` | 0.6888 | 0.7782 | 0.7962 |
| `raw_lower_regex_clean_lemmatized` | 0.6865 | 0.7768 | 0.7952 |
| `raw_lower_regex_clean` | 0.6850 | 0.7780 | 0.7978 |
| `original` | 0.6833 | 0.7768 | 0.7968 |
| `raw_lower` | 0.6833 | 0.7768 | 0.7968 |
| `raw_lower_regex_clean_stopwords_stemmed` | 0.6721 | 0.7658 | 0.7852 |
| `raw_lower_stopwords_stemmed` | 0.6717 | 0.7669 | 0.7868 |
| `raw_lower_regex_clean_stopwords_lemmatized` | 0.6509 | 0.7543 | 0.7784 |
| `raw_lower_stopwords_lemmatized` | 0.6473 | 0.7519 | 0.7763 |
| `raw_lower_regex_clean_stopwords` | 0.6434 | 0.7498 | 0.7763 |
| `raw_lower_stopwords` | 0.6394 | 0.7474 | 0.7748 |

Interpretation:

- the strongest variant is `raw_lower_regex_clean_stemmed`
- stemming improves over the lowercase-only baseline, especially when combined with regex cleaning
- stopword removal hurts performance across the tested combinations
- regex cleaning is mildly helpful, with the best score coming from regex cleaning plus stemming

## Conclusion

The current notebook evidence supports three conclusions.

1. `raw_lower_regex_clean_stemmed` is the strongest preprocessing representation in the full diagnostic grid.
   It reaches macro F1 `0.7045`, weighted F1 `0.7895`, and accuracy `0.8051`.

2. Stemming is useful in this setup.
   The two best variants both apply stemming, and both outperform the original lowercase-only baseline.

3. Stopword removal remains harmful.
   The stopword-removal combinations sit below their comparable no-stopword alternatives, which suggests that short financial tweets depend on words that generic stopword lists remove.

The best current recommendation is therefore:

- select `raw_lower_regex_clean_stemmed` as the best diagnostic preprocessing variant
- keep `raw_lower` as a simple baseline for comparison
- avoid stopword removal unless the stopword list is redesigned around domain-specific evidence
