"""Feature Pipeline - Orchestrates feature engineering from Silver to Gold layer."""

from datetime import datetime, timezone
import json

import boto3
from loguru import logger
import pandas as pd
import typer

from data_processing.check_s3 import wait_for_s3_object
from iris_project.config import IRIS_TARGET, S3_BUCKET, S3_GOLD_PREFIX, S3_SILVER_PREFIX
from iris_project.features.engineering import engineer_features, get_feature_names
from iris_project.features.preprocessing import get_preprocessing_config, save_preprocessing_config

app = typer.Typer()


def get_dataset_metadata(data_path: str) -> dict:
    """Get metadata for the source dataset from S3."""
    bucket = data_path.split("/")[2]
    key = "/".join(data_path.split("/")[3:])
    s3 = boto3.client("s3")

    versions = s3.list_object_versions(Bucket=bucket, Prefix=key)
    latest_version = versions["Versions"][0]

    metadata = {
        "source_uri": data_path,
        "version_id": latest_version["VersionId"],
        "last_modified": latest_version["LastModified"].isoformat(),
        "size_bytes": latest_version["Size"],
    }
    return metadata


def save_feature_metadata(
    feature_metadata: dict, output_path: str, metadata_suffix: str = "_metadata.json"
) -> None:
    """Save feature metadata to S3 alongside the feature data."""
    metadata_path = output_path.replace(".parquet", metadata_suffix)

    bucket = metadata_path.split("/")[2]
    key = "/".join(metadata_path.split("/")[3:])

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(feature_metadata, indent=2),
        ContentType="application/json",
    )
    logger.info(f"Feature metadata saved to: {metadata_path}")


@app.command()
def run_feature_pipeline(
    input_path: str = f"s3://{S3_BUCKET}/{S3_SILVER_PREFIX}/iris.parquet",
    output_path: str = f"s3://{S3_BUCKET}/{S3_GOLD_PREFIX}/iris_features.parquet",
    feature_version: str = None,
):
    """Run the feature engineering pipeline."""
    logger.info("=" * 60)
    logger.info("FEATURE PIPELINE STARTED")
    logger.info("=" * 60)

    if feature_version is None:
        feature_version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info(f"Feature version: {feature_version}")

    logger.info(f"Loading Silver dataset from: {input_path}")
    wait_for_s3_object(S3_BUCKET, f"{S3_SILVER_PREFIX}/iris.parquet", timeout=60)
    df = pd.read_parquet(input_path, storage_options={"anon": False})
    logger.info(f"Loaded {len(df)} rows from Silver layer")

    source_metadata = get_dataset_metadata(input_path)

    df_features = engineer_features(df)
    logger.info(f"Features: {get_feature_names()}")

    X = df_features.drop(columns=[IRIS_TARGET], errors="ignore")
    preprocessing_config = get_preprocessing_config(X)

    feature_metadata = {
        "feature_version": feature_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": source_metadata,
        "total_rows": len(df_features),
        "total_columns": len(df_features.columns),
        "feature_columns": list(df_features.columns),
        "engineered_features": get_feature_names(),
        "preprocessing_config": preprocessing_config,
    }

    logger.info(f"Saving Gold dataset to {output_path}...")
    df_features.to_parquet(output_path, index=False, storage_options={"anon": False})
    logger.success(f"Features saved successfully: {len(df_features)} rows")

    save_feature_metadata(feature_metadata, output_path)

    preprocessing_config_path = output_path.replace(".parquet", "_preprocessing_config.json")
    save_preprocessing_config(preprocessing_config, preprocessing_config_path)

    logger.info("=" * 60)
    logger.success("FEATURE PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Feature version: {feature_version}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Total features: {len(df_features.columns)}")


if __name__ == "__main__":
    app()
