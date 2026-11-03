"""Evaluation helpers shared by train_models.py and the Streamlit app."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "Class"
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(model, X, y_true):
    """Returns the six assignment metrics plus the predictions.

    With 7 classes, precision / recall / F1 are macro-averaged so that the small
    Bombay class counts as much as the big Dermason one. AUC is one-vs-rest macro.
    """
    y_pred = model.predict(X)
    proba = model.predict_proba(X)

    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=model.classes_
        ),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return {k: float(np.round(v, 4)) for k, v in scores.items()}, y_pred
