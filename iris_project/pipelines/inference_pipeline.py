"""Inference Pipeline - Batch inference with consistent feature engineering."""

from datetime import datetime, timezone

from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
import typer

from iris_project.config import IRIS_CLASSES, IRIS_TARGET
from iris_project.features.engineering import engineer_features

app = typer.Typer()


def prepare_inference_data(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    """Prepare data for inference by applying feature engineering."""
    logger.info("Preparing data for inference...")

    df_features = engineer_features(df)

    if target_col and target_col in df_features.columns:
        logger.info(f"Dropping target column: {target_col}")
        df_features = df_features.drop(columns=[target_col])

    df_features = df_features.drop(columns=["ingestion_time"], errors="ignore")

    logger.info(f"Data prepared: {len(df_features)} rows, {len(df_features.columns)} columns")
    return df_features


def load_model_from_mlflow(model_uri: str):
    """Load trained model from MLflow."""
    logger.info(f"Loading model from MLflow: {model_uri}")
    pipeline = mlflow.sklearn.load_model(model_uri)
    logger.success("Model loaded successfully")
    return pipeline


def generate_predictions(
    pipeline, X: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Generate predictions using the loaded pipeline.

    Returns:
        Tuple of (input_features, predictions, probabilities_df [n,3])
    """
    logger.info("Generating predictions...")

    y_pred = pd.Series(pipeline.predict(X), index=X.index)
    y_proba = pd.DataFrame(pipeline.predict_proba(X), index=X.index, columns=IRIS_CLASSES)

    logger.success(f"Predictions generated for {len(X)} rows")
    return X, y_pred, y_proba


def save_predictions(
    X: pd.DataFrame, predictions: pd.Series, probabilities: pd.DataFrame, output_path: str
) -> None:
    """Save predictions to output location."""
    results = X.copy()
    results["prediction"] = predictions
    for cls in IRIS_CLASSES:
        results[f"proba_{cls}"] = probabilities[cls]
    results["inference_timestamp"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info(f"Saving predictions to: {output_path}")
    if output_path.startswith("s3://"):
        results.to_parquet(output_path, index=False, storage_options={"anon": False})
    else:
        results.to_parquet(output_path, index=False)

    logger.success(f"Predictions saved: {len(results)} rows")


@app.command()
def run_batch_inference(
    model_uri: str = typer.Argument(
        ..., help="MLflow model URI (e.g. models:/iris-logisticregression/1)"
    ),
    input_path: str = typer.Argument(..., help="Input data path (S3 or local)"),
    output_path: str = typer.Argument(..., help="Output path for predictions (S3 or local)"),
    target_col: str = typer.Option(IRIS_TARGET, help="Target column to drop if present"),
):
    """Run batch inference pipeline."""
    logger.info("=" * 60)
    logger.info("INFERENCE PIPELINE STARTED")
    logger.info("=" * 60)
    logger.info(f"Model URI: {model_uri}")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")

    logger.info("Loading input data...")
    if input_path.startswith("s3://"):
        df = pd.read_parquet(input_path, storage_options={"anon": False})
    else:
        df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows")

    X = prepare_inference_data(df, target_col=target_col)
    pipeline = load_model_from_mlflow(model_uri)
    X, predictions, probabilities = generate_predictions(pipeline, X)
    save_predictions(X, predictions, probabilities, output_path)

    logger.info("=" * 60)
    logger.success("INFERENCE PIPELINE COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
