"""Training Pipeline - Trains multiple models and logs to MLflow."""

from loguru import logger
import typer

from iris_project.config import IRIS_TARGET, S3_BUCKET, S3_GOLD_PREFIX
from iris_project.modeling.data_loader import load_features
from iris_project.modeling.mlflow_logger import MLflowExperimentLogger
from iris_project.modeling.models import (
    logistic_regression_model,
    random_forest_model,
    svm_model,
)
from iris_project.modeling.pipeline_builder import create_sklearn_pipeline
from iris_project.modeling.trainer import train_and_evaluate

app = typer.Typer()


@app.command()
def run_training_pipeline(
    features_path: str = f"s3://{S3_BUCKET}/{S3_GOLD_PREFIX}/iris_features.parquet",
    target_col: str = IRIS_TARGET,
    experiment_name: str = "iris-baseline-models",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Run the training pipeline."""
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test, feature_metadata = load_features(
        features_path=features_path,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(f"Feature version: {feature_metadata.get('feature_version')}")

    mlflow_logger = MLflowExperimentLogger(experiment_name)

    models = [logistic_regression_model(), random_forest_model(), svm_model()]

    for model in models:
        algorithm = model.__class__.__name__
        logger.info("-" * 60)
        logger.info(f"Training {algorithm}...")

        pipeline = create_sklearn_pipeline(X_train, model)

        trained_pipeline, metrics, cm, y_proba = train_and_evaluate(
            pipeline, X_train, y_train, X_test, y_test
        )

        run_name = f"{algorithm}_{feature_metadata.get('feature_version')}"
        mlflow_logger.log_training_run(
            pipeline=trained_pipeline,
            X_train=X_train,
            X_test=X_test,
            metrics=metrics,
            confusion_matrix=cm,
            feature_metadata=feature_metadata,
            run_name=run_name,
        )

    logger.info("=" * 60)
    logger.success("TRAINING PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Models trained: {len(models)}")
    logger.info(f"Experiment: {experiment_name}")


if __name__ == "__main__":
    app()
