"""
confusion_matrix.py
===================
Tools for building and displaying confusion matrices for multi-class classification.
A confusion matrix shows how many samples of each true class were predicted as
each other class, making it easy to spot which classes the model confuses.
Includes a text-based pretty-printer that works without matplotlib.
"""

from collections import defaultdict


def build_confusion_matrix(
    y_true: list,
    y_pred: list,
) -> tuple[dict[tuple, int], list]:
    """
    Build a confusion matrix as a sparse dictionary mapping (true, predicted)
    pairs to counts. Also returns the sorted list of unique class labels
    found across both y_true and y_pred so the caller can iterate consistently.

    Args:
        y_true: List of ground-truth class labels.
        y_pred: List of predicted class labels, same length as y_true.

    Returns:
        Tuple of (matrix_dict, sorted_labels) where matrix_dict maps
        (true_label, pred_label) → count and sorted_labels is a sorted
        list of all unique labels seen in the data.
    """
    matrix: dict[tuple, int] = defaultdict(int)
    labels: set = set(y_true) | set(y_pred)
    for t, p in zip(y_true, y_pred):
        matrix[(t, p)] += 1
    return dict(matrix), sorted(labels)


def print_confusion_matrix(y_true: list, y_pred: list) -> None:
    """
    Print a formatted confusion matrix to stdout with true labels as rows
    and predicted labels as columns. Each cell shows the count of samples
    with that (true, predicted) combination. Zero counts are printed as dots
    to make non-zero cells stand out visually.

    Args:
        y_true: List of ground-truth class labels.
        y_pred: List of predicted class labels.
    """
    matrix, labels = build_confusion_matrix(y_true, y_pred)
    col_w = max(len(str(l)) for l in labels) + 2

    header = " " * col_w + "".join(str(l).rjust(col_w) for l in labels)
    print(header)
    print("-" * len(header))
    for true_label in labels:
        row = str(true_label).rjust(col_w)
        for pred_label in labels:
            count = matrix.get((true_label, pred_label), 0)
            row  += ("." if count == 0 else str(count)).rjust(col_w)
        print(row)


def per_class_report(y_true: list, y_pred: list) -> dict[str, dict[str, float]]:
    """
    Compute per-class precision, recall, and F1-score from predictions.
    Each class is treated as the positive class in turn while all others
    are treated as negative. Useful for identifying which classes the model
    handles well and which it struggles with.

    Args:
        y_true: List of ground-truth class labels.
        y_pred: List of predicted class labels.

    Returns:
        Dict mapping each class label (as string) to a dict with keys
        'precision', 'recall', and 'f1', each holding a float in [0, 1].
    """
    matrix, labels = build_confusion_matrix(y_true, y_pred)
    report = {}
    for cls in labels:
        tp = matrix.get((cls, cls), 0)
        fp = sum(matrix.get((t, cls), 0) for t in labels if t != cls)
        fn = sum(matrix.get((cls, p), 0) for p in labels if p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        report[str(cls)] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    return report
