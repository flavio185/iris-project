"""Tests for preprocessing functions."""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from iris_project.features.preprocessing import (
    build_preprocessor,
    build_preprocessor_from_config,
    get_preprocessing_config,
)


@pytest.fixture
def sample_data():
    """Create sample Iris feature data for testing."""
    return pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 7.0, 6.3],
            "sepal_width": [3.5, 3.0, 3.2, 3.3],
            "petal_length": [1.4, 1.4, 4.7, 6.0],
            "petal_width": [0.2, 0.2, 1.4, 2.5],
        }
    )


def test_build_preprocessor_returns_column_transformer(sample_data):
    """Test that build_preprocessor returns a ColumnTransformer."""
    preprocessor = build_preprocessor(sample_data)
    assert isinstance(preprocessor, ColumnTransformer)


def test_build_preprocessor_has_num_transformer(sample_data):
    """Test that preprocessor has a numerical transformer."""
    preprocessor = build_preprocessor(sample_data)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names


def test_build_preprocessor_no_cat_transformer(sample_data):
    """Test that preprocessor has no categorical transformer (Iris is all numeric)."""
    preprocessor = build_preprocessor(sample_data)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "cat" not in transformer_names


def test_get_preprocessing_config_structure(sample_data):
    """Test preprocessing config has expected structure."""
    config = get_preprocessing_config(sample_data)
    assert "numerical_columns" in config
    assert "total_features" in config
    assert "created_at" in config
    assert config["total_features"] == 4


def test_get_preprocessing_config_all_four_features(sample_data):
    """Test that all 4 Iris features are in numerical_columns."""
    config = get_preprocessing_config(sample_data)
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        assert col in config["numerical_columns"]


def test_build_preprocessor_from_config(sample_data):
    """Test building preprocessor from saved configuration."""
    config = get_preprocessing_config(sample_data)
    preprocessor = build_preprocessor_from_config(config)

    assert isinstance(preprocessor, ColumnTransformer)
    num_transformer = preprocessor.transformers[0]
    assert num_transformer[2] == config["numerical_columns"]


def test_build_preprocessor_fit_transform_shape(sample_data):
    """Test that preprocessor can fit and transform with correct shape."""
    preprocessor = build_preprocessor(sample_data)
    result = preprocessor.fit_transform(sample_data)
    assert result.shape == (4, 4)
