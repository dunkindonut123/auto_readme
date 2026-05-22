"""
decision_tree.py
================
A binary decision tree classifier built from scratch using numpy.
Splits nodes by maximising information gain computed from Shannon entropy.
Supports configurable maximum depth to control overfitting. The tree is
stored as a nested dictionary structure and supports recursive prediction.
"""

import numpy as np
from collections import Counter


def entropy(labels: list) -> float:
    """
    Compute the Shannon entropy of a label distribution.
    Entropy measures impurity: a node where all labels are the same has
    entropy 0 (pure), while a node with equal class counts has maximum entropy.
    Used to evaluate the quality of a candidate split during tree construction.

    Args:
        labels: List of class labels at a tree node.

    Returns:
        Shannon entropy as a non-negative float. Returns 0.0 for empty input.
    """
    if not labels:
        return 0.0
    counts = Counter(labels)
    total  = len(labels)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())


def information_gain(parent: list, left: list, right: list) -> float:
    """
    Compute the information gain of splitting a node into two children.
    Information gain is the reduction in entropy achieved by the split,
    weighted by the proportion of samples that go to each child.
    The best split is the one that maximises information gain.

    Args:
        parent: Labels at the parent node before splitting.
        left:   Labels of samples that go to the left child.
        right:  Labels of samples that go to the right child.

    Returns:
        Information gain as a non-negative float.
    """
    n = len(parent)
    weighted = (len(left) / n) * entropy(left) + (len(right) / n) * entropy(right)
    return entropy(parent) - weighted


def best_split(X: np.ndarray, y: list) -> tuple[int, float]:
    """
    Find the feature index and threshold value that produce the highest
    information gain when used to split the current node.
    Evaluates all features and all unique threshold values for each feature.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: List of class labels of length n_samples.

    Returns:
        Tuple of (best_feature_index, best_threshold_value).
    """
    best_gain, best_feat, best_thresh = -1.0, 0, 0.0
    for feat in range(X.shape[1]):
        for thresh in np.unique(X[:, feat]):
            mask  = X[:, feat] <= thresh
            gain  = information_gain(y, [y[i] for i in np.where(mask)[0]],
                                        [y[i] for i in np.where(~mask)[0]])
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh
    return best_feat, best_thresh


def build_tree(X: np.ndarray, y: list, depth: int = 0, max_depth: int = 5) -> dict:
    """
    Recursively build a binary decision tree as a nested dictionary.
    Stops splitting when the node is pure, the maximum depth is reached,
    or there are fewer than 2 samples at the node. Leaf nodes store the
    majority class label as their prediction.

    Args:
        X:         Feature matrix for samples at this node.
        y:         Labels for samples at this node.
        depth:     Current depth in the tree (root starts at 0).
        max_depth: Maximum allowed depth before forcing a leaf node.

    Returns:
        Dict representing either a leaf {"label": class} or an internal
        node {"feature": int, "threshold": float, "left": dict, "right": dict}.
    """
    majority = Counter(y).most_common(1)[0][0]
    if depth >= max_depth or len(set(y)) == 1 or len(y) < 2:
        return {"label": majority}
    feat, thresh = best_split(X, y)
    mask = X[:, feat] <= thresh
    return {
        "feature":   feat,
        "threshold": thresh,
        "left":  build_tree(X[mask],  [y[i] for i in np.where(mask)[0]],  depth + 1, max_depth),
        "right": build_tree(X[~mask], [y[i] for i in np.where(~mask)[0]], depth + 1, max_depth),
    }
