"""
neural_net.py
=============
A minimal feedforward neural network implemented from scratch using only numpy.
Supports configurable hidden layer sizes, sigmoid and ReLU activations, and
gradient-descent training via backpropagation. Intended for learning purposes
and small tabular datasets rather than production use.
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function element-wise.
    Squashes any real-valued input into the range (0, 1), making it
    suitable for output layers in binary classification problems.

    Args:
        x: Numpy array of any shape.

    Returns:
        Array of the same shape with values in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the sigmoid function, needed during backpropagation.
    If x is already the sigmoid output s, the derivative is s * (1 - s).

    Args:
        x: Numpy array (pre-activation values, not sigmoid outputs).

    Returns:
        Array of the same shape containing sigmoid derivative values.
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply the Rectified Linear Unit activation function element-wise.
    Returns the input directly if positive, otherwise returns zero.
    ReLU is preferred over sigmoid for hidden layers because it avoids
    the vanishing gradient problem for large positive inputs.

    Args:
        x: Numpy array of any shape.

    Returns:
        Array of the same shape with negative values replaced by zero.
    """
    return np.maximum(0.0, x)


class FeedforwardNet:
    """
    A two-layer feedforward neural network with one hidden layer.
    Uses sigmoid activations throughout and trains with full-batch
    gradient descent. Weights are initialised with small random values
    scaled by the square root of the input dimension to prevent
    exploding gradients at the start of training.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * scale
        self.b2 = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network and return output predictions.
        Stores intermediate activations on the instance for use during backprop.

        Args:
            X: Input matrix of shape (n_samples, input_dim).

        Returns:
            Output matrix of shape (n_samples, output_dim) with values in (0, 1).
        """
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """
        Run one step of backpropagation and update weights with gradient descent.
        Uses mean squared error as the loss function. Returns the current loss
        so training progress can be monitored over epochs.

        Args:
            X:  Input matrix of shape (n_samples, input_dim).
            y:  Target matrix of shape (n_samples, output_dim).
            lr: Learning rate controlling the step size of each update.

        Returns:
            Scalar MSE loss for the current batch.
        """
        n     = X.shape[0]
        loss  = float(np.mean((self.a2 - y) ** 2))
        d2    = (self.a2 - y) * sigmoid_derivative(self.z2)
        dW2   = self.a1.T @ d2 / n
        db2   = d2.mean(axis=0)
        d1    = (d2 @ self.W2.T) * sigmoid_derivative(self.z1)
        dW1   = X.T @ d1 / n
        db1   = d1.mean(axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        return loss
