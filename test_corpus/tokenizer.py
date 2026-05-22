"""
tokenizer.py
============
Word and sentence tokenization utilities for splitting raw text into
meaningful units before feature extraction or model training.
Supports word-level and sentence-level splitting with configurable
minimum length filters to discard noise tokens.
"""

import re


def word_tokenize(text: str, min_length: int = 2) -> list[str]:
    """
    Split a string into individual word tokens by splitting on non-alphanumeric
    characters. Tokens shorter than min_length are discarded to reduce noise
    from single-character fragments and punctuation artefacts.

    Args:
        text:       Input string to tokenize.
        min_length: Minimum number of characters a token must have to be kept.

    Returns:
        List of word tokens meeting the minimum length requirement.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= min_length]


def sentence_tokenize(text: str) -> list[str]:
    """
    Split a block of text into individual sentences by detecting terminal
    punctuation followed by whitespace. Sentences shorter than 15 characters
    are discarded as fragments unlikely to carry useful information.

    Args:
        text: A paragraph or multi-sentence document string.

    Returns:
        List of sentence strings, each containing at least 15 characters.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= 15]


def ngram_tokenize(tokens: list[str], n: int = 2) -> list[tuple[str, ...]]:
    """
    Generate n-grams from a list of tokens. N-grams capture local word
    co-occurrence patterns that single-word (unigram) models miss.
    For example, bigrams from ['machine', 'learning', 'model'] gives
    [('machine', 'learning'), ('learning', 'model')].

    Args:
        tokens: List of word tokens (output of word_tokenize).
        n:      Size of each n-gram window.

    Returns:
        List of n-gram tuples. Returns an empty list if len(tokens) < n.
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def vocab_from_corpus(corpus: list[str]) -> dict[str, int]:
    """
    Build a vocabulary index mapping each unique token to an integer ID.
    The vocabulary is constructed from all tokens across all documents
    in the corpus. Useful for converting text to numerical feature vectors.

    Args:
        corpus: List of raw document strings.

    Returns:
        Dict mapping token string to a unique integer index, sorted
        alphabetically for deterministic output.
    """
    tokens: set[str] = set()
    for doc in corpus:
        tokens.update(word_tokenize(doc))
    return {token: idx for idx, token in enumerate(sorted(tokens))}
