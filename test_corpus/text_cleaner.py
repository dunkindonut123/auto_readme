"""
text_cleaner.py
===============
Utilities for cleaning and normalising raw text before feeding it into NLP pipelines.
Handles common noise sources such as HTML tags, special characters, extra whitespace,
and inconsistent casing. Designed to be used as a preprocessing step before tokenization.
"""

import re


def remove_html_tags(text: str) -> str:
    """
    Strip all HTML tags from a string using a regular expression.
    This is a lightweight alternative to a full HTML parser for cases
    where the input is known to contain only simple markup.

    Args:
        text: Raw string potentially containing HTML tags.

    Returns:
        The input string with all HTML tags removed.
    """
    return re.sub(r"<[^>]+>", "", text)


def normalize_whitespace(text: str) -> str:
    """
    Replace all sequences of whitespace characters (spaces, tabs, newlines)
    with a single space and strip leading and trailing whitespace.
    Useful for collapsing multi-line strings into a single clean line.

    Args:
        text: Input string with potentially irregular whitespace.

    Returns:
        A cleaned string with normalized spacing.
    """
    return re.sub(r"\s+", " ", text).strip()


def to_lowercase(text: str) -> str:
    """
    Convert all characters in the input string to lowercase.
    This is the first step in most text normalization pipelines
    to ensure case-insensitive comparison of tokens.

    Args:
        text: Any string.

    Returns:
        The string converted entirely to lowercase.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation characters from the input string.
    Keeps alphanumeric characters and whitespace only.
    Useful for bag-of-words models that do not benefit from punctuation.

    Args:
        text: Input string.

    Returns:
        String with all punctuation removed.
    """
    return re.sub(r"[^\w\s]", "", text)


def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline: strip HTML, normalize whitespace,
    convert to lowercase, and remove punctuation.
    This is the main entry point for cleaning a raw document.

    Args:
        text: Raw input text.

    Returns:
        Fully cleaned and normalized text string.
    """
    text = remove_html_tags(text)
    text = normalize_whitespace(text)
    text = to_lowercase(text)
    text = remove_punctuation(text)
    return text
