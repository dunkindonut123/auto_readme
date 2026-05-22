"""
knn_classifier.py
=================
A k-Nearest Neighbours classifier implemented from scratch using numpy.
Classifies new samples by finding the k training samples with the smallest
Euclidean distance and returning the majority class among them. No training
step is required — the model simply memorises the training set.
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-Nearest Neighbours classifier that stores all training samples
    and classifies new points by majority vote among the k closest neighbours.
    Distance is measured using the L2 (Euclidean) norm. Ties in voting
    are broken by the label that appears first alphabetically.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._X_train: np.ndarray | None = None
        self._y_train: list | None = None

    def fit(self, X: np.ndarray, y: list) -> "KNNClassifier":
        """
        Store the training data for use during prediction.
        KNN has no explicit training phase — fitting simply memorises
        the dataset so it can be searched at prediction time.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: List of class labels of length n_samples.

        Returns:
            Self, to allow method chaining.
        """
        self._X_train = np.array(X)
        self._y_train = list(y)
        return self

    def predict(self, X: np.ndarray) -> list:
        """
        Predict the class label for each sample in X by majority vote
        among its k nearest training neighbours.
        Euclidean distance is computed between each query point and all
        training points; the k smallest distances determine the vote pool.

        Args:
            X: Query feature matrix of shape (n_queries, n_features).

        Returns:
            List of predicted class labels, one per query sample.
        """
        X = np.array(X)
        predictions = []
        for sample in X:
            distances = np.linalg.norm(self._X_train - sample, axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_labels  = [self._y_train[i] for i in k_indices]
            predictions.append(Counter(k_labels).most_common(1)[0][0])
        return predictions

    def score(self, X: np.ndarray, y: list) -> float:
        """
        Compute classification accuracy on a labelled test set.
        Accuracy is the fraction of test samples whose predicted label
        matches the true label exactly.

        Args:
            X: Test feature matrix of shape (n_samples, n_features).
            y: True class labels of length n_samples.

        Returns:
            Accuracy score as a float in [0.0, 1.0].
        """
        predictions = self.predict(X)
        return sum(p == t for p, t in zip(predictions, y)) / len(y)
