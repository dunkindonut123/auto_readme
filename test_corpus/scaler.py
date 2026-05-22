"""
scaler.py
=========
Feature scaling transformers that normalise numerical input features before
model training. Unscaled features with very different ranges can cause
gradient-based models to converge slowly or favour high-magnitude features.
Provides MinMax scaling and StandardScaler (z-score normalisation).
"""

import numpy as np


class MinMaxScaler:
    """
    Scales each feature to the range [0, 1] by subtracting the minimum
    and dividing by the range. Features with zero range (constant columns)
    are left as zero to avoid division by zero. Fit on training data only,
    then apply to both training and test data using transform().
    """

    def __init__(self):
        self._min: np.ndarray | None = None
        self._range: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """
        Compute the per-feature minimum and range from the training set.
        These statistics are stored and reused when transform() is called.

        Args:
            X: Training data matrix of shape (n_samples, n_features).

        Returns:
            Self, to allow method chaining (fit_transform pattern).
        """
        X = np.array(X, dtype=float)
        self._min   = X.min(axis=0)
        self._range = X.max(axis=0) - self._min
        self._range[self._range == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply min-max scaling using statistics computed during fit().
        Always call fit() on training data before calling transform()
        on training or test data to prevent data leakage.

        Args:
            X: Data matrix to scale, shape (n_samples, n_features).

        Returns:
            Scaled matrix with feature values in [0, 1].
        """
        return (np.array(X, dtype=float) - self._min) / self._range


class StandardScaler:
    """
    Standardises each feature to zero mean and unit variance by subtracting
    the training mean and dividing by the training standard deviation.
    Features with zero standard deviation are left unchanged. This scaler
    is preferred when the data is approximately Gaussian distributed.
    """

    def __init__(self):
        self._mean: np.ndarray | None = None
        self._std:  np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Compute per-feature mean and standard deviation from the training set.
        Features with zero standard deviation are assigned std=1 to avoid
        division by zero during transform.

        Args:
            X: Training data matrix of shape (n_samples, n_features).

        Returns:
            Self, for method chaining.
        """
        X = np.array(X, dtype=float)
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardise features using mean and std computed during fit().
        The result has approximately zero mean and unit variance per feature
        when applied to data drawn from the same distribution as the training set.

        Args:
            X: Data matrix to standardise, shape (n_samples, n_features).

        Returns:
            Standardised matrix with per-feature zero mean and unit variance.
        """
        return (np.array(X, dtype=float) - self._mean) / self._std
