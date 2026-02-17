"""Feature engineering functions for Iris classification.

This module contains all feature engineering logic that transforms
raw data into features used for model training and inference.
For Iris, the engineering is minimal (data is already clean numeric features),
but the function exists to preserve the train-serve skew prevention contract.
"""

from loguru import logger
import pandas as pd

from iris_project.config import IRIS_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations.

    This is the main function that orchestrates all feature engineering steps.
    It should be used consistently in both training and inference pipelines
    to prevent train-serve skew.

    Args:
        df: DataFrame with raw features from Silver layer

    Returns:
        DataFrame with engineered features ready for Gold layer
    """
    logger.info("Starting feature engineering...")

    # Ensure all columns are lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Ensure measurement columns are float64
    for col in IRIS_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    logger.info("Feature engineering completed.")
    return df


def get_feature_names() -> list[str]:
    """Get list of all feature names used for modeling.

    Returns:
        List of feature column names
    """
    return list(IRIS_FEATURES)
