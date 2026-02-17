from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from iris_project.config import IRIS_CLASSES


def plot_confusion_matrix(y_true, y_pred):
    """Generate and plot confusion matrix for multiclass."""
    cm = confusion_matrix(y_true, y_pred, labels=IRIS_CLASSES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=IRIS_CLASSES, yticklabels=IRIS_CLASSES
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    return cm


def evaluate_model(model, X_test, y_test):
    """Comprehensive multiclass model evaluation.

    Returns:
        Tuple of (metrics_dict, confusion_matrix, y_proba [n,3])
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
    }

    # ROC AUC (one-vs-rest, macro) — requires label binarization
    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = np.nan

    cm = plot_confusion_matrix(y_test, y_pred)

    logger.info("\nModel Performance Metrics:")
    logger.info("-" * 50)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.3f}")

    logger.info("\nClassification Report:")
    logger.info("\n %s", classification_report(y_test, y_pred))

    return metrics, cm, y_proba
