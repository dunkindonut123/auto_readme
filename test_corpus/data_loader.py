"""
data_loader.py
==============
Utilities for loading, splitting, and batching datasets for machine learning experiments.
Provides train/validation/test splits with optional shuffling, and a batch generator
that yields fixed-size chunks of data during model training. All operations work
on plain Python lists so there is no dependency on numpy or pandas.
"""

import random


def train_val_test_split(
    data: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split a dataset into training, validation, and test subsets.
    The test set receives whatever proportion remains after train and val.
    Shuffling is performed before splitting so the order of the input
    does not bias which samples end up in which split.

    Args:
        data:        List of samples to split (any type).
        train_ratio: Fraction of data assigned to the training set.
        val_ratio:   Fraction of data assigned to the validation set.
        shuffle:     Whether to shuffle the data before splitting.
        seed:        Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) lists.
    """
    if shuffle:
        random.seed(seed)
        data = list(data)
        random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)
    return data[:train_end], data[train_end:val_end], data[val_end:]


def batch_generator(data: list, batch_size: int = 32):
    """
    Yield successive non-overlapping batches of batch_size from data.
    The last batch may be smaller than batch_size if the data length
    is not evenly divisible. Useful for mini-batch gradient descent
    without loading the full dataset into memory at once.

    Args:
        data:       List of training samples.
        batch_size: Number of samples per batch.

    Yields:
        Successive list slices of length batch_size (or less for the last batch).
    """
    for start in range(0, len(data), batch_size):
        yield data[start:start + batch_size]


def stratified_split(
    data: list,
    labels: list,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list, list, list, list]:
    """
    Perform a stratified train/test split that preserves the class distribution
    of the original dataset in both subsets. This prevents splits where one
    class is entirely absent from the test set due to random chance.

    Args:
        data:        List of samples.
        labels:      Corresponding list of class labels, same length as data.
        train_ratio: Fraction of each class assigned to training.
        seed:        Random seed for reproducibility.

    Returns:
        Tuple of (train_data, test_data, train_labels, test_labels).
    """
    random.seed(seed)
    class_buckets: dict[any, list[int]] = {}
    for idx, label in enumerate(labels):
        class_buckets.setdefault(label, []).append(idx)

    train_idx, test_idx = [], []
    for indices in class_buckets.values():
        random.shuffle(indices)
        cut = int(len(indices) * train_ratio)
        train_idx.extend(indices[:cut])
        test_idx.extend(indices[cut:])

    return (
        [data[i] for i in train_idx],
        [data[i] for i in test_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in test_idx],
    )
