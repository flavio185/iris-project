"""Data loading utilities for training pipeline."""

import json

import boto3
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from iris_project.config import IRIS_TARGET


def load_feature_metadata(features_path: str) -> dict:
    """Load feature metadata from S3."""
    metadata_path = features_path.replace(".parquet", "_metadata.json")
    bucket = metadata_path.split("/")[2]
    key = "/".join(metadata_path.split("/")[3:])

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    metadata = json.loads(response["Body"].read())

    logger.info(f"Loaded feature metadata: version {metadata.get('feature_version')}")
    return metadata


def load_features(
    features_path: str,
    target_col: str = IRIS_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Load features from Gold layer and split into train/test.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    logger.info(f"Loading features from: {features_path}")

    df = pd.read_parquet(features_path, storage_options={"anon": False})
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    feature_metadata = load_feature_metadata(features_path)

    X = df.drop(columns=[target_col, "ingestion_time"], errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    feature_metadata["split_strategy"] = {
        "method": "train_test_split",
        "random_state": random_state,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    logger.info(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")

    return X_train, X_test, y_train, y_test, feature_metadata
