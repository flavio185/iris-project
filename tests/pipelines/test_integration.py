"""Integration tests for pipelines."""

import pandas as pd
import pytest

from iris_project.features.engineering import engineer_features, get_feature_names
from iris_project.features.preprocessing import build_preprocessor


@pytest.fixture
def sample_raw_data():
    """Create sample raw data similar to Silver layer."""
    return pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 7.0, 6.3],
            "sepal_width": [3.5, 3.0, 3.2, 3.3],
            "petal_length": [1.4, 1.4, 4.7, 6.0],
            "petal_width": [0.2, 0.2, 1.4, 2.5],
            "species": ["setosa", "setosa", "versicolor", "virginica"],
        }
    )


def test_feature_to_preprocessing_integration(sample_raw_data):
    """Test that features created can be used for training preprocessing."""
    df_features = engineer_features(sample_raw_data.copy())
    X = df_features.drop(columns=["species"], errors="ignore")

    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] == 4


def test_train_inference_consistency(sample_raw_data):
    """Test that feature engineering produces consistent results."""
    df_train = engineer_features(sample_raw_data.copy())
    df_inference = engineer_features(sample_raw_data.copy())

    pd.testing.assert_frame_equal(df_train, df_inference)


def test_all_four_features_present(sample_raw_data):
    """Test that all 4 Iris features are present after engineering."""
    df_features = engineer_features(sample_raw_data.copy())
    feature_names = get_feature_names()

    for feature in feature_names:
        assert feature in df_features.columns


def test_no_extra_numeric_features(sample_raw_data):
    """Test that no unexpected numeric features are added."""
    df_features = engineer_features(sample_raw_data.copy())
    X = df_features.drop(columns=["species"], errors="ignore")
    numeric_cols = [c for c in X.columns if X[c].dtype in ["int64", "float64"]]
    assert len(numeric_cols) == 4


def test_no_nulls_in_features(sample_raw_data):
    """Test that no null values are in features."""
    df_features = engineer_features(sample_raw_data.copy())
    for col in get_feature_names():
        assert df_features[col].notna().all()
