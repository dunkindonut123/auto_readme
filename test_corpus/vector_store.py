"""
vector_store.py
===============
A lightweight in-memory vector store for nearest-neighbour similarity search.
Stores document embeddings as numpy arrays and retrieves the most similar
documents to a query vector using cosine similarity. Intended for small to
medium corpora where a full-scale index library would be overkill.
"""

import math
import numpy as np


class VectorStore:
    """
    In-memory store that maps document IDs to dense numpy embedding vectors.
    Supports adding documents one at a time and retrieving the top-k most
    similar documents to a query vector via cosine similarity scoring.
    """

    def __init__(self):
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []

    def add(self, doc_id: str, vector: np.ndarray) -> None:
        """
        Add a document and its embedding vector to the store.
        Vectors are stored as-is without normalisation — normalise
        beforehand if you want dot-product search to equal cosine similarity.

        Args:
            doc_id: Unique string identifier for the document.
            vector: Dense numpy array representing the document embedding.
        """
        self._ids.append(doc_id)
        self._vectors.append(vector)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the cosine similarity between two dense vectors.
        Returns a value in [-1, 1] where 1 means identical direction.
        A small epsilon is added to the denominator to prevent division by zero
        when either vector is the zero vector.

        Args:
            a: First dense vector.
            b: Second dense vector.

        Returns:
            Cosine similarity score as a float.
        """
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return dot / norm

    def query(self, vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Retrieve the top_k most similar documents to the query vector.
        Scores all stored vectors against the query using cosine similarity
        and returns results sorted from most to least similar.

        Args:
            vector: Query embedding vector.
            top_k:  Maximum number of results to return.

        Returns:
            List of (doc_id, similarity_score) tuples sorted by descending score.
        """
        if not self._vectors:
            return []
        scores = [(doc_id, self.cosine_similarity(vector, v))
                  for doc_id, v in zip(self._ids, self._vectors)]
        return sorted(scores, key=lambda x: -x[1])[:top_k]

    def size(self) -> int:
        """
        Return the number of documents currently stored in the vector store.

        Returns:
            Integer count of stored document vectors.
        """
        return len(self._ids)
