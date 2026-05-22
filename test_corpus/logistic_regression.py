"""
logistic_regression.py
======================
Binary logistic regression trained with mini-batch gradient descent.
Models the probability that a sample belongs to the positive class using
a linear combination of features passed through the sigmoid function.
Supports L2 regularisation to reduce overfitting on small datasets.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid (logistic) function element-wise.
    Maps any real-valued number to the open interval (0, 1),
    which is interpreted as a probability in logistic regression.

    Args:
        z: Pre-activation array of any shape.

    Returns:
        Array of probabilities in (0, 1) with the same shape as z.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the binary cross-entropy loss between true labels and predicted
    probabilities. This is the standard loss function for binary classification.
    A small epsilon is added inside the log to prevent numerical instability
    when predicted probabilities are very close to 0 or 1.

    Args:
        y_true: Array of true binary labels (0 or 1).
        y_pred: Array of predicted probabilities in (0, 1).

    Returns:
        Mean binary cross-entropy loss as a non-negative float.
    """
    eps  = 1e-9
    return -float(np.mean(
        y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
    ))


class LogisticRegression:
    """
    Binary logistic regression classifier trained using gradient descent.
    Weights are updated to minimise binary cross-entropy with optional
    L2 regularisation. The predict_proba method returns raw probabilities;
    predict applies a 0.5 decision threshold to produce hard binary labels.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 100, l2: float = 0.0):
        self.lr     = lr
        self.epochs = epochs
        self.l2     = l2
        self.weights: np.ndarray | None = None
        self.bias:    float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train the logistic regression model on the provided data.
        Weights are initialised to zero and updated via full-batch gradient
        descent for the specified number of epochs. L2 regularisation
        shrinks weights toward zero to prevent overfitting.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Binary label array of shape (n_samples,) with values 0 or 1.

        Returns:
            Self, for method chaining.
        """
        n, d         = X.shape
        self.weights = np.zeros(d)
        self.bias    = 0.0
        for _ in range(self.epochs):
            z    = X @ self.weights + self.bias
            pred = sigmoid(z)
            err  = pred - y
            self.weights -= self.lr * (X.T @ err / n + self.l2 * self.weights)
            self.bias    -= self.lr * err.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted probability of the positive class for each sample.
        Probabilities are derived from the linear combination of features
        and learned weights passed through the sigmoid function.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability array of shape (n_samples,) with values in (0, 1).
        """
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels by thresholding predicted probabilities at 0.5.
        Samples with probability >= 0.5 are assigned label 1; others are assigned 0.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Integer label array of shape (n_samples,) with values 0 or 1.
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
