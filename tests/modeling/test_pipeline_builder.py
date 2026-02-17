"""Tests for pipeline builder module."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from iris_project.modeling.pipeline_builder import create_sklearn_pipeline, get_pipeline_info


@pytest.fixture
def sample_data():
    """Create sample Iris training data."""
    return pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 7.0, 6.3],
            "sepal_width": [3.5, 3.0, 3.2, 3.3],
            "petal_length": [1.4, 1.4, 4.7, 6.0],
            "petal_width": [0.2, 0.2, 1.4, 2.5],
        }
    )


def test_create_sklearn_pipeline_returns_pipeline(sample_data):
    """Test pipeline creation returns Pipeline."""
    model = LogisticRegression(random_state=42)
    pipeline = create_sklearn_pipeline(sample_data, model)
    assert isinstance(pipeline, Pipeline)


def test_create_sklearn_pipeline_has_two_steps(sample_data):
    """Test pipeline has preprocessor and classifier steps."""
    model = LogisticRegression(random_state=42)
    pipeline = create_sklearn_pipeline(sample_data, model)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "preprocessor"
    assert pipeline.steps[1][0] == "classifier"


def test_pipeline_fit_and_predict(sample_data):
    """Test that pipeline can be fitted and make predictions."""
    model = LogisticRegression(random_state=42, max_iter=200)
    pipeline = create_sklearn_pipeline(sample_data, model)

    y = pd.Series(["setosa", "setosa", "versicolor", "virginica"])
    pipeline.fit(sample_data, y)

    predictions = pipeline.predict(sample_data)
    assert len(predictions) == len(sample_data)


def test_pipeline_predict_proba_shape(sample_data):
    """Test that predict_proba returns [n, 3] shape."""
    model = LogisticRegression(random_state=42, max_iter=200)
    pipeline = create_sklearn_pipeline(sample_data, model)

    y = pd.Series(["setosa", "setosa", "versicolor", "virginica"])
    pipeline.fit(sample_data, y)

    proba = pipeline.predict_proba(sample_data)
    assert proba.shape == (4, 3)


def test_get_pipeline_info(sample_data):
    """Test extracting pipeline information."""
    model = LogisticRegression(random_state=42, max_iter=200)
    pipeline = create_sklearn_pipeline(sample_data, model)

    y = pd.Series(["setosa", "setosa", "versicolor", "virginica"])
    pipeline.fit(sample_data, y)

    info = get_pipeline_info(pipeline)
    assert info["steps"] == ["preprocessor", "classifier"]
    assert info["classifier"] == "LogisticRegression"
    assert "num" in info["transformers"]
