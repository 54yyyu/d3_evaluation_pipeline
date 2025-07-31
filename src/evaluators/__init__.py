"""
Evaluation metrics for D3 generated sequences.

This package contains evaluators for different types of similarity metrics:
- Functional similarity: Oracle model-based metrics
- Sequence similarity: Direct sequence comparison metrics (includes discriminability analysis)
- Compositional similarity: Motif-based composition metrics
"""

from .base_evaluator import BaseEvaluator
from .functional_similarity import FunctionalSimilarityEvaluator
from .sequence_similarity import SequenceSimilarityEvaluator
from .compositional_similarity import CompositionalSimilarityEvaluator

__all__ = [
    'BaseEvaluator',
    'FunctionalSimilarityEvaluator', 
    'SequenceSimilarityEvaluator',
    'CompositionalSimilarityEvaluator'
]