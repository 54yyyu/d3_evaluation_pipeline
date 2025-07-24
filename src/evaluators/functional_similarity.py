"""
Functional similarity evaluation metrics.

This module implements functional similarity metrics that assess how well
synthetic sequences match the functional behavior of test sequences as 
measured by the oracle model.
"""

import numpy as np
import torch
import scipy.stats
from scipy import linalg
from typing import Dict, Any, Union
from .base_evaluator import BaseEvaluator
from ..models.model_utils import ModelWrapper, make_predictions, extract_embeddings


class FunctionalSimilarityEvaluator(BaseEvaluator):
    """Evaluator for functional similarity metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize functional similarity evaluator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "functional_similarity")
        
    def get_required_inputs(self) -> Dict[str, str]:
        """Get required inputs for functional similarity evaluation."""
        return {
            "x_synthetic": "Generated/synthetic sequences (N, L, A) or (N, A, L)",
            "x_test": "Test/observed sequences (N, L, A) or (N, A, L)",
            "oracle_model": "Trained oracle model for making predictions"
        }
    
    def evaluate(self, 
                 x_synthetic: Union[np.ndarray, torch.Tensor],
                 x_test: Union[np.ndarray, torch.Tensor],
                 oracle_model: Any,
                 **kwargs) -> Dict[str, Any]:
        """
        Perform functional similarity evaluation.
        
        Args:
            x_synthetic: Generated sequences
            x_test: Test sequences  
            oracle_model: Oracle model for predictions
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing functional similarity results
        """
        # Validate inputs
        self.validate_inputs(x_synthetic, x_test)
        
        # Convert to tensors and ensure correct format
        x_synthetic_tensor = self._ensure_tensor(x_synthetic)
        x_test_tensor = self._ensure_tensor(x_test)
        
        # Wrap model for consistent interface
        model = ModelWrapper(oracle_model) if not isinstance(oracle_model, ModelWrapper) else oracle_model
        
        # Calculate all functional similarity metrics
        results = {}
        
        # 1. Conditional generation fidelity
        results.update(self._conditional_generation_fidelity(x_test_tensor, x_synthetic_tensor, model))
        
        # 2. Frechet distance
        results.update(self._frechet_distance(x_test_tensor, x_synthetic_tensor, model))
        
        # 3. Predictive distribution shift
        results.update(self._predictive_distribution_shift(x_test_tensor, x_synthetic_tensor))
        
        self.update_results(results)
        return results
    
    def _conditional_generation_fidelity(self, 
                                       x_test: torch.Tensor, 
                                       x_synthetic: torch.Tensor, 
                                       model: ModelWrapper) -> Dict[str, float]:
        """
        Calculate conditional generation fidelity (MSE between predicted activities).
        
        Args:
            x_test: Test sequences
            x_synthetic: Synthetic sequences
            model: Oracle model wrapper
            
        Returns:
            Dictionary with conditional generation fidelity result
        """
        # Make predictions
        y_test, y_synthetic = model.predict(x_test, x_synthetic)
        
        # Calculate MSE
        mse = np.mean((y_test - y_synthetic) ** 2)
        
        return {
            "conditional_generation_fidelity_mse": float(mse)
        }
    
    def _frechet_distance(self, 
                         x_test: torch.Tensor, 
                         x_synthetic: torch.Tensor, 
                         model: ModelWrapper) -> Dict[str, float]:
        """
        Calculate Frechet distance using penultimate layer embeddings.
        
        Args:
            x_test: Test sequences
            x_synthetic: Synthetic sequences  
            model: Oracle model wrapper
            
        Returns:
            Dictionary with Frechet distance result
        """
        # Extract embeddings from penultimate layer
        embeddings_test = model.get_embeddings(x_test)
        embeddings_synthetic = model.get_embeddings(x_synthetic)
        
        # Calculate activation statistics
        mu1, sigma1 = self._calculate_activation_statistics(embeddings_test)
        mu2, sigma2 = self._calculate_activation_statistics(embeddings_synthetic)
        
        # Calculate Frechet distance
        distance = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        return {
            "frechet_distance": float(distance)
        }
    
    def _predictive_distribution_shift(self, 
                                     x_test: torch.Tensor, 
                                     x_synthetic: torch.Tensor) -> Dict[str, float]:
        """
        Calculate predictive distribution shift using Kolmogorov-Smirnov test.
        
        Args:
            x_test: Test sequences
            x_synthetic: Synthetic sequences
            
        Returns:
            Dictionary with distribution shift results
        """
        # Convert to numpy and encode bases using 0,1,2,3 (eliminate a dimension)
        x_test_np = self._ensure_numpy(x_test)
        x_synthetic_np = self._ensure_numpy(x_synthetic)
        
        # Standardize to (N, L, A) format
        x_test_np = self._standardize_shape(x_test_np)
        x_synthetic_np = self._standardize_shape(x_synthetic_np)
        
        base_indices_test = np.argmax(x_test_np, axis=2)
        base_indices_synthetic = np.argmax(x_synthetic_np, axis=2)
        
        # Flatten the arrays
        base_indices_test_flat = base_indices_test.flatten()
        base_indices_synthetic_flat = base_indices_synthetic.flatten()
        
        # Perform Kolmogorov-Smirnov test
        ks_statistic, p_value = scipy.stats.ks_2samp(
            base_indices_synthetic_flat, 
            base_indices_test_flat
        )
        
        return {
            "predictive_distribution_shift_ks_statistic": float(ks_statistic),
            "predictive_distribution_shift_p_value": float(p_value)
        }
    
    def _calculate_activation_statistics(self, embeddings: torch.Tensor) -> tuple:
        """
        Calculate mean and covariance of embeddings.
        
        Args:
            embeddings: Embedding tensor
            
        Returns:
            Tuple of (mean, covariance)
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        mu = np.mean(embeddings_np, axis=0)
        sigma = np.cov(embeddings_np, rowvar=False)
        return mu, sigma
    
    def _calculate_frechet_distance(self, 
                                  mu1: np.ndarray, 
                                  sigma1: np.ndarray, 
                                  mu2: np.ndarray, 
                                  sigma2: np.ndarray, 
                                  eps: float = 1e-6) -> float:
        """
        Calculate Frechet distance between two multivariate Gaussians.
        
        Adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        
        Args:
            mu1, sigma1: Mean and covariance of first distribution
            mu2, sigma2: Mean and covariance of second distribution
            eps: Small value for numerical stability
            
        Returns:
            Frechet distance
        """
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print(f"FID calculation produces singular product; adding {eps} to diagonal of cov estimates")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# Convenience functions for individual metrics

def conditional_generation_fidelity(y_synthetic: np.ndarray, 
                                  y_test: np.ndarray) -> float:
    """
    Calculate conditional generation fidelity (MSE).
    
    Args:
        y_synthetic: Predicted activities for synthetic sequences
        y_test: Predicted activities for test sequences
        
    Returns:
        MSE between predicted activities
    """
    return np.mean((y_synthetic - y_test) ** 2)


def frechet_distance(embeddings_synthetic: torch.Tensor,
                    embeddings_test: torch.Tensor,
                    eps: float = 1e-6) -> float:
    """
    Calculate Frechet distance between embedding distributions.
    
    Args:
        embeddings_synthetic: Embeddings for synthetic sequences
        embeddings_test: Embeddings for test sequences
        eps: Small value for numerical stability
        
    Returns:
        Frechet distance
    """
    evaluator = FunctionalSimilarityEvaluator({})
    
    mu1, sigma1 = evaluator._calculate_activation_statistics(embeddings_test)
    mu2, sigma2 = evaluator._calculate_activation_statistics(embeddings_synthetic)
    
    return evaluator._calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps)


def predictive_distribution_shift(x_synthetic: Union[np.ndarray, torch.Tensor],
                                 x_test: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    Calculate predictive distribution shift using KS test.
    
    Args:
        x_synthetic: Synthetic sequences
        x_test: Test sequences
        
    Returns:
        Dictionary with KS statistic and p-value
    """
    evaluator = FunctionalSimilarityEvaluator({})
    
    if isinstance(x_synthetic, torch.Tensor):
        x_synthetic = x_synthetic.detach().cpu().numpy()
    if isinstance(x_test, torch.Tensor):
        x_test = x_test.detach().cpu().numpy()
        
    return evaluator._predictive_distribution_shift(
        torch.from_numpy(x_test), 
        torch.from_numpy(x_synthetic)
    )