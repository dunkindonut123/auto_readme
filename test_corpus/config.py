"""
config.py
=========
A lightweight hyperparameter configuration manager for machine learning experiments.
Stores key-value settings, supports loading from and saving to JSON files,
and provides a flat dot-notation accessor for nested configuration trees.
Designed to make experiment settings reproducible and easy to log.
"""

import json
import os


class Config:
    """
    Flat key-value store for experiment hyperparameters and settings.
    Values can be any JSON-serialisable type: strings, numbers, booleans,
    lists, or nested dicts. Supports dot-notation access for nested keys
    (e.g. config.get('model.hidden_dim')) and serialisation to/from JSON.
    """

    def __init__(self, defaults: dict | None = None):
        self._data: dict = defaults.copy() if defaults else {}

    def set(self, key: str, value) -> None:
        """
        Store a hyperparameter value under the given key.
        Overwrites any existing value for that key without warning.

        Args:
            key:   String key, optionally using dot notation for nesting
                   (e.g. 'optimizer.lr' stores under {'optimizer': {'lr': value}}).
            value: Any JSON-serialisable value.
        """
        keys  = key.split(".")
        node  = self._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    def get(self, key: str, default=None):
        """
        Retrieve a hyperparameter value by key, returning default if not found.
        Supports dot-notation traversal of nested dictionaries.

        Args:
            key:     Dot-notation key string.
            default: Value to return if the key is not present.

        Returns:
            The stored value, or default if the key path does not exist.
        """
        keys = key.split(".")
        node = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def save(self, path: str) -> None:
        """
        Serialise the configuration to a JSON file at the given path.
        The parent directory is created if it does not exist.
        Useful for saving the exact hyperparameters used in an experiment
        alongside model checkpoints for reproducibility.

        Args:
            path: File path where the JSON configuration will be written.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def load(self, path: str) -> "Config":
        """
        Load configuration from a JSON file, merging its values into the
        current config. Existing keys are overwritten by values in the file.

        Args:
            path: Path to a JSON configuration file.

        Returns:
            Self, for method chaining.
        """
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self._data.update(loaded)
        return self

    def all(self) -> dict:
        """
        Return a shallow copy of all stored configuration values as a plain dict.
        Useful for logging the full set of hyperparameters at the start of training.

        Returns:
            Dict of all key-value pairs currently stored in the config.
        """
        return self._data.copy()
