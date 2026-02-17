"""Tests for feature engineering functions."""

import pandas as pd
import pytest

from iris_project.features.engineering import engineer_features, get_feature_names


@pytest.fixture
def sample_data():
    """Create sample Iris data for testing."""
    return pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 7.0, 6.3],
            "sepal_width": [3.5, 3.0, 3.2, 3.3],
            "petal_length": [1.4, 1.4, 4.7, 6.0],
            "petal_width": [0.2, 0.2, 1.4, 2.5],
            "species": ["setosa", "setosa", "versicolor", "virginica"],
        }
    )


def test_engineer_features_row_preservation(sample_data):
    """Test that row count is preserved."""
    result = engineer_features(sample_data.copy())
    assert len(result) == len(sample_data)


def test_engineer_features_column_preservation(sample_data):
    """Test that all original columns are preserved."""
    result = engineer_features(sample_data.copy())
    for col in sample_data.columns:
        assert col.lower() in result.columns


def test_engineer_features_numeric_types(sample_data):
    """Test that measurement columns are float64."""
    result = engineer_features(sample_data.copy())
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        assert result[col].dtype == "float64"


def test_engineer_features_lowercase_names(sample_data):
    """Test that column names are lowercase."""
    df = sample_data.copy()
    df.columns = [c.upper() for c in df.columns]
    result = engineer_features(df)
    for col in result.columns:
        assert col == col.lower()


def test_engineer_features_no_nulls(sample_data):
    """Test that no null values are introduced."""
    result = engineer_features(sample_data.copy())
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        assert result[col].notna().all()


def test_engineer_features_idempotency(sample_data):
    """Test that applying engineer_features twice gives same result."""
    result1 = engineer_features(sample_data.copy())
    result2 = engineer_features(result1.copy())
    pd.testing.assert_frame_equal(result1, result2)


def test_get_feature_names():
    """Test getting feature names."""
    feature_names = get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) == 4
    assert "sepal_length" in feature_names
    assert "sepal_width" in feature_names
    assert "petal_length" in feature_names
    assert "petal_width" in feature_names
