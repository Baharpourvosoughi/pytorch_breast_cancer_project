"""Evaluation metrics and plots for the held-out test set."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def evaluate(y_true, probabilities, output_dir="reports"):
    output_path = Path(output_dir)
    figure_path = output_path / "figures"
    figure_path.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "classification_threshold": 0.5,
    }

    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    cm = confusion_matrix(y_true, predictions)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["malignant (0)", "benign (1)"],
    )
    disp.plot(values_format="d")
    plt.title("Test-set confusion matrix")
    plt.tight_layout()
    plt.savefig(figure_path / "confusion_matrix.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.title("Test-set ROC curve")
    plt.tight_layout()
    plt.savefig(figure_path / "roc_curve.png", dpi=150)
    plt.close()

    return metrics
