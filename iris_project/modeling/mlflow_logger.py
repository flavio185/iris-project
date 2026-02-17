"""MLflow logging utilities."""

from loguru import logger
from matplotlib import pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline


class MLflowExperimentLogger:
    """Handles MLflow logging for model training experiments."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")

    def log_training_run(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        metrics: dict,
        confusion_matrix,
        feature_metadata: dict,
        run_name: str = None,
    ):
        """Log a complete training run to MLflow."""
        algorithm = pipeline.steps[-1][1].__class__.__name__

        with mlflow.start_run(run_name=run_name or algorithm):
            self._log_data_params(X_train, X_test)
            mlflow.log_param("algorithm", algorithm)
            self._log_feature_params(feature_metadata)
            mlflow.log_metrics(metrics)
            self._log_confusion_matrix(confusion_matrix)
            mlflow.log_dict(feature_metadata, "feature_metadata.json")
            self._log_model(pipeline, X_train, algorithm)

            run_id = mlflow.active_run().info.run_id
            logger.success(f"MLflow run logged successfully. Run ID: {run_id}")

    def _log_data_params(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

    def _log_feature_params(self, feature_metadata: dict):
        mlflow.log_param("feature_version", feature_metadata.get("feature_version", "unknown"))
        mlflow.log_param("total_features", feature_metadata.get("total_columns"))

    def _log_confusion_matrix(self, cm):
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

    def _log_model(self, pipeline: Pipeline, X_train: pd.DataFrame, algorithm: str):
        signature = infer_signature(X_train, pipeline.predict(X_train))

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=algorithm,
            registered_model_name=f"iris-{algorithm.lower()}",
            signature=signature,
            input_example=X_train.head(3),
        )

        logger.info(f"Model registered: iris-{algorithm.lower()}")
