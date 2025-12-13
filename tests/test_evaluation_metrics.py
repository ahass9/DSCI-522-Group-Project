"""
Unit tests for the evaluation_metrics function.

The tests verify that evaluation_metrics:
- returns the expected keys and shapes for a fitted classifier
- produces sensible metric values on a simple binary classification problem
- raises clear errors for invalid inputs (None, empty arrays, mismatched lengths,
  or models without the required methods).
"""

import pytest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model_evaluation import evaluation_metrics


# expected
def test_evaluation_metrics_basic_correct_output():
    """
    Basic test: 1-NN classifier on a small binary dataset.

    Because evaluation_metrics fits the model on X_test and y_test and then
    evaluates on the same data, a 1-NN classifier should achieve perfect
    accuracy on this simple dataset.
    """
    X_test = np.array([[0], [1], [2], [3]])
    y_test = np.array([0, 0, 1, 1])

    model = KNeighborsClassifier(n_neighbors=1)

    metrics = evaluation_metrics(model, X_test, y_test)

    # Check if required keys exist
    for key in [
        "accuracy",
        "confusion_matrix",
        "fpr",
        "tpr",
        "auc",
        "y_pred",
        "y_proba",
    ]:
        assert key in metrics

    # Accuracy should be perfect with this setup
    assert metrics["accuracy"] == pytest.approx(1.0)

    # Confusion matrix should be 2x2 for binary classification
    cm = metrics["confusion_matrix"]
    assert cm.shape == (2, 2)

    # y_pred and y_proba should have the same length as y_test
    assert len(metrics["y_pred"]) == len(y_test)
    assert len(metrics["y_proba"]) == len(y_test)

    # AUC should be between 0 and 1
    assert 0.0 <= metrics["auc"] <= 1.0


# error cases
def test_evaluation_metrics_none_inputs():
    """
    Raises error when X_test or y_test are None.
    """
    model = KNeighborsClassifier(n_neighbors=1)

    with pytest.raises(ValueError, match="must not be None"):
        evaluation_metrics(model, None, None)


def test_evaluation_metrics_empty_arrays():
    """
    Raises error when X_test or y_test are empty.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    X_test = np.empty((0, 1))
    y_test = np.array([])

    with pytest.raises(ValueError, match="X_test is empty"):
        evaluation_metrics(model, X_test, y_test)

    # Non-empty X but empty y
    X_non_empty = np.array([[0], [1]])
    with pytest.raises(ValueError, match="y_test is empty"):
        evaluation_metrics(model, X_non_empty, np.array([]))


def test_evaluation_metrics_mismatched_lengths():
    """
    Raises error when X_test and y_test have different numbers of rows.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    X_test = np.array([[0], [1], [2]])
    y_test = np.array([0, 1])

    with pytest.raises(ValueError, match="same number of rows"):
        evaluation_metrics(model, X_test, y_test)


def test_evaluation_metrics_invalid_model():
    """
    Raises error when model does not implement fit, predict and predict_proba.
    """
    class DummyModel: # Example of invalid model with no required methods
        pass

    dummy_model = DummyModel()
    X_test = np.array([[0], [1]])
    y_test = np.array([0, 1])

    with pytest.raises(ValueError, match="fit', 'predict', and 'predict_proba"):
        evaluation_metrics(dummy_model, X_test, y_test)
        
        
        
def test_evaluation_metrics_with_list_labels():
    """
    Edge case: y_test provided as a Python list instead of a numpy array.
    Function should still run and return valid metrics.
    """
    X_test = np.array([[0], [1], [2], [3]])
    y_test = [0, 0, 1, 1]  # list, not np.array

    model = KNeighborsClassifier(n_neighbors=1)

    metrics = evaluation_metrics(model, X_test, y_test)

    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "y_pred" in metrics
    assert "y_proba" in metrics
    assert len(metrics["y_pred"]) == len(y_test)
    assert len(metrics["y_proba"]) == len(y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0


def test_evaluation_metrics_minimum_valid_size():
    """
    Edge case: smallest dataset with both classes present (2 samples).
    Function should still run and return valid metrics.
    """
    X_test = np.array([[0], [1]])
    y_test = np.array([0, 1])

    model = KNeighborsClassifier(n_neighbors=1)

    metrics = evaluation_metrics(model, X_test, y_test)

    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "fpr" in metrics
    assert "tpr" in metrics
    assert "auc" in metrics
    assert len(metrics["y_pred"]) == 2
    assert len(metrics["y_proba"]) == 2
    assert 0.0 <= metrics["auc"] <= 1.0  