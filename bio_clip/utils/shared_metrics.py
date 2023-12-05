import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(labels: np.array, predictions: np.array):
    """Compute a set of classification metrics from a label array
        and a predictions array
    Args:
        labels (np.array): array of labels
        predictions (np.array): array of predictions
    Returns:
        mcc, f1, accuracy, auc_roc, auc_precision_recall,
    """
    auc_roc = roc_auc_score(labels, predictions)
    # Compute PR AUC
    precision, recall, thres = precision_recall_curve(labels, predictions)

    auc_precision_recall = auc(recall, precision)

    f1_max = max(2 * (precision * recall) / (precision + recall))

    predictions = list(map(np.rint, predictions))
    # Compute the MCC score

    mcc = matthews_corrcoef(labels, predictions)
    # Compute the F1 score
    f1 = f1_score(labels, predictions, average="macro")
    # Compute the accuracy score
    acc = accuracy_score(labels, predictions)
    return {
        "mcc": mcc,
        "f1": f1,
        "acc": acc,
        "auc_roc": auc_roc,
        "auc_pr": auc_precision_recall,
        "f1_max": f1_max,
    }
