"""Tests for trainer module."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from iris_project.modeling.pipeline_builder import create_sklearn_pipeline
from iris_project.modeling.trainer import train_and_evaluate, train_model


@pytest.fixture
def sample_train_data():
    """Create sample training data."""
    X_train = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 7.0, 6.3, 5.0, 5.4, 6.4, 5.9],
            "sepal_width": [3.5, 3.0, 3.2, 3.3, 3.4, 3.9, 3.2, 3.0],
            "petal_length": [1.4, 1.4, 4.7, 6.0, 1.5, 1.7, 4.5, 5.1],
            "petal_width": [0.2, 0.2, 1.4, 2.5, 0.2, 0.4, 1.5, 1.8],
        }
    )
    y_train = pd.Series(
        [
            "setosa",
            "setosa",
            "versicolor",
            "virginica",
            "setosa",
            "setosa",
            "versicolor",
            "virginica",
        ]
    )
    return X_train, y_train


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    X_test = pd.DataFrame(
        {
            "sepal_length": [5.0, 6.7, 6.0],
            "sepal_width": [3.3, 3.1, 2.2],
            "petal_length": [1.4, 4.4, 5.0],
            "petal_width": [0.2, 1.4, 1.5],
        }
    )
    y_test = pd.Series(["setosa", "versicolor", "virginica"])
    return X_test, y_test


def test_train_model_returns_fitted_pipeline(sample_train_data):
    """Test model training returns a fitted pipeline."""
    X_train, y_train = sample_train_data
    model = LogisticRegression(random_state=42, max_iter=200)
    pipeline = create_sklearn_pipeline(X_train, model)

    trained_pipeline = train_model(pipeline, X_train, y_train)

    assert isinstance(trained_pipeline, Pipeline)
    assert hasattr(trained_pipeline, "classes_")


def test_train_and_evaluate_returns_correct_tuple(sample_train_data, sample_test_data):
    """Test training and evaluation returns correct tuple."""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data

    model = LogisticRegression(random_state=42, max_iter=200)
    pipeline = create_sklearn_pipeline(X_train, model)

    trained_pipeline, metrics, cm, y_proba = train_and_evaluate(
        pipeline, X_train, y_train, X_test, y_test
    )

    assert isinstance(trained_pipeline, Pipeline)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert cm is not None


def test_train_and_evaluate_proba_shape(sample_train_data, sample_test_data):
    """Test that y_proba has shape [n, 3]."""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data

    model = LogisticRegression(random_state=42, max_iter=200)
    pipeline = create_sklearn_pipeline(X_train, model)

    _, _, _, y_proba = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)

    assert y_proba.shape == (len(X_test), 3)
