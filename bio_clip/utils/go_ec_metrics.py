import numpy as np
from sklearn.metrics import auc, precision_recall_curve


def area_under_roc(pred, target):
    """
    Area under receiver operating characteristic curve (ROC).

    Parameters:
        pred (np.ndarray): predictions of shape (n,)
        target (np.ndarray): binary targets of shape (n,)
    """
    order = np.argsort(pred)[::-1]  # To get descending order with numpy
    target = target[order]
    hit = np.cumsum(target)
    all = np.sum(target == 0) * np.sum(target == 1)
    auroc = np.sum(hit[target == 0]) / (all + 1e-10)
    return auroc


def micro_auc(pred, targ):
    return area_under_roc(pred.reshape(-1), targ.reshape(-1))


def optimal_threshold_f1_max(pred, target, eps=1e-6):
    """
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and
    negative samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (np.ndarray): predictions of shape (B, N)
        target (np.ndarray): binary targets of shape (B, N)
    """
    order = np.argsort(pred, axis=1, kind="stable")[
        :, ::-1
    ]  # To get descending order with numpy
    B, N = order.shape
    target = np.take_along_axis(target, order, axis=1)
    precision = np.cumsum(target, axis=1) / np.cumsum(np.ones_like(target), axis=1)
    recall = np.cumsum(target, axis=1) / (np.sum(target, axis=1, keepdims=True) + 1e-10)

    _is_start = np.zeros_like(target, dtype=bool)
    _is_start[:, 0] = True
    rows = np.arange(B)[:, None]
    is_start = np.zeros_like(_is_start, dtype=bool)
    is_start[rows, order] = _is_start

    all_order = np.argsort(pred.ravel())[::-1]
    order = order + np.arange(B)[:, None] * N
    order = order.ravel()
    inv_order = np.argsort(order)
    is_start = is_start.ravel()[all_order]
    all_order = inv_order[all_order]
    precision = precision.ravel()
    recall = recall.ravel()

    all_precision = precision[all_order] - np.where(
        is_start, np.zeros_like(precision), precision[all_order - 1]
    )
    denom = np.cumsum(is_start, axis=0)
    all_precision = np.cumsum(all_precision, axis=0) / (denom + 1e-10)  # * pmask
    all_recall = recall[all_order] - np.where(
        is_start, np.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = np.cumsum(all_recall, axis=0) / pred.shape[0]

    mask = (all_precision + all_recall) > eps
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + (1 - mask))
    return (all_f1 * mask).max()


def term_centric_aucpr_sklearn(predicted, target):
    """
    target [batch, ecgo_terms]
    predicted [batch, ecgo_terms]
    """
    a = []
    for y_true, y_prob in zip(target.T, predicted.T):
        if y_true.sum() > 0:
            p, r, t = precision_recall_curve(y_true, y_prob)
            a.append(auc(r, p))
    term_centric = np.array(a)
    return term_centric


def micro_aucpr_sklearn(predicted, target):
    """
    target [batch, ecgo_terms]
    predicted [batch, ecgo_terms]
    """
    p, r, t = precision_recall_curve(target.reshape(-1), predicted.reshape(-1))
    return auc(r, p)


def get_all_metrics(target, predicted):
    """
    target [batch, ecgo_terms]
    predicted [batch, ecgo_terms]
    """
    aucprs = term_centric_aucpr_sklearn(predicted, target)
    return {
        "mean_term_centric_AUCPR": np.mean(aucprs),
        "median_term_centric_AUCPR": np.median(aucprs),
        "micro_AUCPR": micro_aucpr_sklearn(predicted, target),
        "protein_centric_fmax": float(
            optimal_threshold_f1_max(predicted, target)
        ),  # this is the same as the one in torchdrug
        "micro_AUCROC": float(
            micro_auc(predicted, target)
        ),  # this is the same as the one in torchdrug
    }
