# Text Mining

**Spring Semester 2025/2026**

*Project Assignment Handout {version 1.0}*

This handout details the rules for the mandatory practical project for Text Mining, to be developed and completed during the academic calendar of the Text Mining class.

---

## 1. Project Summary

> "Over time, major indexes go up and down based on internal and external factors. Performance like that excites investors, but typically in opposite ways. Constant gains lead some investors to expect more of the same. Others worry the good times are surely about to end. The former sentiment is sometimes called 'bullish,' while the latter is referred to as 'bearish.'"
> — [SmartAsset](https://smartasset.com/financial-advisor/bullish-vs-bearish)

The goal of this project is to develop an NLP model capable of predicting Market sentiment based on tweets. With the NLP techniques learned during class, you must implement a classification model that receives tweets as inputs and is able to predict, for each tweet, if it describes a **Bearish (0)**, **Bullish (1)**, or **Neutral (2)** attitude.

The project should be developed using **Python 3** and libraries such as *NLTK*, *Scikit Learn*, *Hugging Face*, and *LangChain*. The project can be solved in various ways — there is no single correct solution.

---

## 2. Group Rules

The project should be done in a group of between **one (1) to four (4)** students. Groups larger than four will **NOT** have their project graded.

### Important Delivery Deadline

| Deliverable | Deadline |
|---|---|
| Project Report + Notebooks + Predictions | **Midnight, 15th of June 2026** |

---

## 3. Project Starting Point – Corpora

The data is divided into a training file (`train.csv`) and a testing file (`test.csv`):

- **Train** (9,543 lines): Contains tweets (`text`) and the sentiment label (`label`). Each tweet has one of the following labels: Bearish (0), Bullish (1), or Neutral (2). You may divide this into Train/Validation sets.
- **Test** (299 lines): Same structure as the train set, but without the `label` column. You are expected to provide the predicted label (0, 1, or 2) for each tweet — **your predictions will be compared with the actual (true) labels**.

---

## 4. Solution Requirements

Your solution must cover the following points:

1. **Data Exploration** — `2.00 pts`
   Analyze the corpora and provide conclusions and visual information (bar charts, word clouds, etc.) that contextualize the data.

2. **Corpus Split** — `0.50 pts`
   Apply a method to split the training corpus into train/validation sets to evaluate model performance. K-Fold cross validation is also acceptable.

3. **Data Preprocessing** — `3.00 pts`
   Correctly implement at least **four (4)** preprocessing techniques shown in class (stop words, regular expressions, lemmatization, stemming, etc.).

4. **Feature Engineering** — `5.50 pts`
   Correctly implement and experiment with at least one variation of each of the following techniques: **BoW**, **word2vec**, **Transformer Encoder**.

5. **Classification Models** — `4.50 pts`
   Correctly implement and test at least two variations of each of the following methods: **Traditional ML** (KNN, MLP, Logistic Regression, Random Forest, XGBoost, etc.) and **Transformer Encoders**.

6. **Evaluation and Analysis** — `1.50 pts`
   Evaluate and compare your models using at least **Recall**, **Precision**, **Accuracy**, and **F1-Score**, and explain what the evaluation means in the context of the problem.

### Extra Work (up to 2.00 pts)

Development of techniques beyond those in point 5 is highly recommended:

1. **Feature Engineering** — `0.50 pts` for each extra Transformer Encoder method applied (maximum of 2 extra methods).
2. **Classification Models** — `1.00 pts` for correctly using a decoder model for classification.

---

## 5. Delivery Guide

You must deliver the following files (see the delivery template folder shared on Moodle):

1. **`tm_tests_xx.ipynb`** — Notebook following the structure in Section 4, containing the techniques experimented with and their evaluation. (`xx` = group number)
2. **`tm_final_xx.ipynb`** — Notebook with only your ready-to-run final solution, including a single pipeline with a single classification model.
3. **`pred_xx.csv`** — CSV file with only two columns: the test set ID and your predicted labels.

Additionally, you must submit a **PDF report** named `report_XX` with the following structure (other structures are also accepted):

1. **Data Exploration** — data presentation and explanation of main findings from exploratory analysis (accounts for 50% of criteria 4.1).
2. **Data Preprocessing** — explanation of the preprocessing methods developed (accounts for 25% of criteria 4.2 and 4.3).
3. **Feature Engineering** — description and explanation of feature engineering methods applied (accounts for 30% of criteria 4.4).
4. **Classification Models** — description and explanation of the models implemented (accounts for 30% of criteria 4.5).
5. **Evaluation and Results** — description of model performance and main conclusions (accounts for 50% of criteria 4.6).

### Extra Information

- The PDF report has a **maximum of 15 pages**. Exceeding this will incur a **0.5-point penalty per extra page**.
- Any extra work must be **clearly defined as such in the PDF report**, or it will not be considered for evaluation.
- All files must be saved in a folder named `group_xx` and submitted through Moodle's project submission section by **23:59 of the 23rd of January**.
- Late delivery incurs a **1.0-point penalty per half-day late**.
- Failure to comply with the delivery guide incurs up to a **1.0-point penalty**.
- To prevent plagiarism and misuse of Gen AI, students may be **randomly chosen for an oral defense** to assess their understanding.

---

## 6. Extra Challenges

### Extra Challenge 1

Predictions will be compared against the actual labels from `test.csv`. The three groups with the highest performance will receive:

- **1.00 pt** — group with the best model
- **0.50 pts** — group with the 2nd best model
- **0.25 pts** — group with the 3rd best model

### Extra Challenge 2

Up to **1.50 pts** for groups that correctly design and implement an **agentic AI-based workflow** that orchestrates the classification pipeline using tools or multiple models. The agent must have a **conversational interface** and perform at least one non-trivial decision or coordination task, such as:

- Choosing between alternative models
- Comparing outputs from multiple classifiers
- Routing tweets to different models
- Automating evaluation

Simple use of a single LLM prompt for direct classification, without tool use, orchestration, or decision-making, will **not** be considered. Extra work must be described in detail in the report.

> **Final note:** Grades can hypothetically exceed 20 points with extra challenges, but the final project grade is **capped at 20**.

---

## Questions and Clarifications

Further clarifications and answers to student questions will be provided on the Moodle page as appropriate.

*Good luck with your project!*
