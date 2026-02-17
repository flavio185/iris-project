"""Model training utilities."""

from loguru import logger
import pandas as pd
from sklearn.pipeline import Pipeline

from iris_project.modeling.eval import evaluate_model


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a sklearn pipeline."""
    model_name = pipeline.steps[-1][1].__class__.__name__
    logger.info(f"Training {model_name}...")

    pipeline.fit(X_train, y_train)

    logger.success(f"{model_name} training completed")
    return pipeline


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict, object, object]:
    """Train and evaluate a model pipeline.

    Returns:
        Tuple of (trained_pipeline, metrics, confusion_matrix, y_proba)
    """
    trained_pipeline = train_model(pipeline, X_train, y_train)

    model_name = pipeline.steps[-1][1].__class__.__name__
    logger.info(f"Evaluating {model_name}...")

    metrics, cm, y_proba = evaluate_model(trained_pipeline, X_test, y_test)
    logger.info(f"Metrics: {metrics}")

    return trained_pipeline, metrics, cm, y_proba
