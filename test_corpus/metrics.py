"""
metrics.py
==========
Common evaluation metrics for classification and regression tasks in machine learning.
Provides accuracy, precision, recall, F1-score, and mean squared error computed
directly from lists of predictions and ground-truth labels without external dependencies.
"""


def accuracy(y_true: list, y_pred: list) -> float:
    """
    Compute the fraction of predictions that exactly match the ground-truth labels.
    This is the simplest classification metric but can be misleading on
    imbalanced datasets where the majority class dominates.

    Args:
        y_true: List of ground-truth labels.
        y_pred: List of predicted labels, same length as y_true.

    Returns:
        Accuracy as a float in [0.0, 1.0].
    """
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def precision(y_true: list, y_pred: list, positive_label=1) -> float:
    """
    Compute precision for binary classification: the fraction of positive
    predictions that are actually correct. High precision means few false positives.
    Returns 0.0 when the model makes no positive predictions at all.

    Args:
        y_true:         Ground-truth binary labels.
        y_pred:         Predicted binary labels.
        positive_label: The label value treated as the positive class.

    Returns:
        Precision score as a float in [0.0, 1.0].
    """
    tp = sum(t == positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fp = sum(t != positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true: list, y_pred: list, positive_label=1) -> float:
    """
    Compute recall for binary classification: the fraction of actual positives
    that the model correctly identified. High recall means few false negatives.
    Returns 0.0 when there are no positive examples in y_true.

    Args:
        y_true:         Ground-truth binary labels.
        y_pred:         Predicted binary labels.
        positive_label: The label value treated as the positive class.

    Returns:
        Recall score as a float in [0.0, 1.0].
    """
    tp = sum(t == positive_label and p == positive_label for t, p in zip(y_true, y_pred))
    fn = sum(t == positive_label and p != positive_label for t, p in zip(y_true, y_pred))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true: list, y_pred: list, positive_label=1) -> float:
    """
    Compute the F1-score, the harmonic mean of precision and recall.
    F1 balances the two metrics and is more informative than accuracy
    when class distribution is skewed. Returns 0.0 when both precision
    and recall are zero.

    Args:
        y_true:         Ground-truth binary labels.
        y_pred:         Predicted binary labels.
        positive_label: The label value treated as the positive class.

    Returns:
        F1-score as a float in [0.0, 1.0].
    """
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
    """
    Compute the mean squared error between predicted and true continuous values.
    MSE penalises large errors more heavily than small ones due to squaring,
    making it sensitive to outliers. Commonly used as a loss function for
    regression models.

    Args:
        y_true: List of ground-truth continuous values.
        y_pred: List of predicted continuous values.

    Returns:
        MSE as a non-negative float. Returns 0.0 for empty inputs.
    """
    if not y_true:
        return 0.0
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
