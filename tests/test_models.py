"""Tests for classification model helper functions."""

from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.dummy import DummyClassifier

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import models


class ModelHelperTests(unittest.TestCase):
    def test_metric_row_returns_expected_scores(self):
        row = models.metric_row([0, 1, 1, 2], [0, 1, 0, 2])

        self.assertAlmostEqual(row["accuracy"], 0.75)
        self.assertGreater(row["macro_f1"], 0)
        self.assertGreater(row["weighted_f1"], 0)

    def test_benchmark_sklearn_specs_evaluates_all_specs(self):
        feature_sets = {
            "toy_features": (
                np.array([[0], [1], [2], [3], [4], [5]]),
                np.array([[0], [1], [2]]),
            )
        }
        y_train = np.array([0, 0, 1, 1, 2, 2])
        y_val = np.array([0, 0, 1])
        specs = [
            models.SklearnModelSpec(
                variant="dummy_most_frequent",
                representation="toy_features",
                classifier_family="Dummy",
                model=DummyClassifier(strategy="most_frequent"),
            ),
            models.SklearnModelSpec(
                variant="dummy_stratified",
                representation="toy_features",
                classifier_family="Dummy",
                model=DummyClassifier(strategy="stratified", random_state=73),
            ),
        ]

        results, reports = models.benchmark_sklearn_specs(
            specs,
            feature_sets,
            y_train,
            y_val,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(set(results["variant"]), {"dummy_most_frequent", "dummy_stratified"})
        self.assertEqual(set(reports), {"dummy_most_frequent", "dummy_stratified"})


if __name__ == "__main__":
    unittest.main()
