"""
Data Processing and Metrics Calculation Engine.
Provides core utility modules to filter numeric arrays and compute variances.
Designed to clean structured inputs before they are passed into machine learning matrices.
"""

import math
from typing import List, Dict

class ArrayAnalyzer:
    """
    Statistical analyzer for multi-dimensional numerical datasets.
    Handles data trimming, outlier suppression, and mean value tracking.
    """

    def __init__(self, data: List[float]):
        self.raw_data = data
        self.cleaned_data = self._remove_outliers(data)

    def _remove_outliers(self, dataset: List[float]) -> List[float]:

        if not dataset:
            return []
        # Simple thresholding logic for demo purposes
        return [x for x in dataset if -1000.0 <= x <= 1000.0]

    def compute_average(self) -> float:
        if not self.cleaned_data:
            return 0.0
        return sum(self.cleaned_data) / len(self.cleaned_data)

    def get_summary_report(self) -> Dict[str, float]:

        avg = self.compute_average()
        count = len(self.cleaned_data)
        variance = sum((x - avg) ** 2 for x in self.cleaned_data) / (count or 1)
        
        return {
            "total_samples": float(count),
            "calculated_mean": avg,
            "sample_variance": variance
        }


def validate_matrix_shape(matrix: List[List[float]]) -> bool:
    """
    Verify if a nested array forms a valid, uniform mathematical matrix.
    Checks that every internal row possesses identical column length properties.
    """
    if not matrix or not matrix[0]:
        return False
    first_row_len = len(matrix[0])
    return all(len(row) == first_row_len for row in matrix)