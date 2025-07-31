"""
Discriminability similarity evaluation metrics.

This module implements discriminability metrics that assess how well a binary
classifier can distinguish between synthetic and test sequences. Lower AUROC
scores indicate better generator performance (synthetic sequences are harder
to distinguish from real ones).
"""

import numpy as np
import torch
import os
from typing import Dict, Any, Union, Optional
from .base_evaluator import BaseEvaluator
from models.discriminability_classifier import (
    prepare_discriminability_data,
    save_discriminability_data,
    train_discriminability_classifier,
    PL_DiscriminabilityClassifier
)
import tempfile
from pathlib import Path


class DiscriminabilitySimilarityEvaluator(BaseEvaluator):
    """Evaluator for discriminability similarity metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize discriminability similarity evaluator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "discriminability_similarity")
        
        # Set default discriminability config if not provided
        self.disc_config = config.get('discriminability', {})
        self._set_default_config()
        
    def _set_default_config(self):
        """Set default configuration values for discriminability evaluation."""
        defaults = {
            'validation_split': 0.2,
            'batch_size': 128,
            'train_max_epochs': 50,
            'patience': 10,
            'lr': 0.002,
            'random_seed': 42,
            'save_training_data': True,
            'training_data_path': None  # Will be set automatically if None
        }
        
        for key, value in defaults.items():
            if key not in self.disc_config:
                self.disc_config[key] = value
    
    def get_required_inputs(self) -> Dict[str, str]:
        """Get required inputs for discriminability similarity evaluation."""
        return {
            "x_synthetic": "Generated/synthetic sequences (N, L, A) or (N, A, L)",
            "x_test": "Test/observed sequences (N, L, A) or (N, A, L)"
        }
    
    def evaluate(self, 
                 x_synthetic: Union[np.ndarray, torch.Tensor],
                 x_test: Union[np.ndarray, torch.Tensor],
                 oracle_model: Any = None,
                 save_training_data_path: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Perform discriminability similarity evaluation.
        
        Args:
            x_synthetic: Generated sequences
            x_test: Test sequences  
            oracle_model: Oracle model (not used for discriminability, but kept for interface consistency)
            save_training_data_path: Optional path to save training data H5 file
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing discriminability similarity results
        """
        # Validate inputs
        self.validate_inputs(x_synthetic, x_test)
        
        # Convert to numpy arrays for processing
        x_synthetic_np = self._ensure_numpy(x_synthetic)
        x_test_np = self._ensure_numpy(x_test)
        
        # Standardize shape to ensure compatibility
        x_synthetic_np = self._standardize_shape(x_synthetic_np)
        x_test_np = self._standardize_shape(x_test_np)
        
        print(f"Evaluating discriminability with {len(x_synthetic_np)} synthetic and {len(x_test_np)} test sequences")
        
        # Prepare training data
        print("Preparing discriminability training data...")
        X_train, y_train, X_val, y_val = prepare_discriminability_data(
            x_synthetic_np, x_test_np,
            validation_split=self.disc_config['validation_split'],
            random_seed=self.disc_config['random_seed']
        )
        
        print(f"Training set: {len(X_train)} samples ({np.sum(y_train)} synthetic, {np.sum(y_train == 0)} test)")
        print(f"Validation set: {len(X_val)} samples ({np.sum(y_val)} synthetic, {np.sum(y_val == 0)} test)")
        
        # Save training data if requested
        training_data_saved_path = None
        if self.disc_config['save_training_data']:
            if save_training_data_path:
                training_data_path = save_training_data_path
            elif self.disc_config['training_data_path']:
                training_data_path = self.disc_config['training_data_path']
            else:
                # Create default path in current directory
                training_data_path = "discriminability_training_data.h5"
            
            print(f"Saving training data to {training_data_path}...")
            save_discriminability_data(X_train, y_train, X_val, y_val, training_data_path)
            training_data_saved_path = training_data_path
        
        # Train discriminability classifier
        print("Training discriminability classifier...")
        trained_model, final_auroc = train_discriminability_classifier(
            X_train, y_train, X_val, y_val, self.disc_config
        )
        
        print(f"Discriminability classifier training completed. Final AUROC: {final_auroc:.4f}")
        
        # Calculate additional metrics
        results = self._calculate_discriminability_metrics(
            trained_model, X_val, y_val, final_auroc, training_data_saved_path
        )
        
        self.update_results(results)
        return results
    
    def _calculate_discriminability_metrics(self, 
                                          model: PL_DiscriminabilityClassifier,
                                          X_val: np.ndarray, 
                                          y_val: np.ndarray,
                                          auroc: float,
                                          training_data_path: Optional[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive discriminability metrics.
        
        Args:
            model: Trained discriminability classifier
            X_val: Validation sequences
            y_val: Validation labels
            auroc: AUROC score
            training_data_path: Path where training data was saved
            
        Returns:
            Dictionary with discriminability metrics
        """
        # Get predictions
        X_val_tensor = torch.from_numpy(X_val).float()
        predictions = model.predict_proba(X_val_tensor)
        
        # Calculate various metrics
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, 
            accuracy_score, precision_score, recall_score, f1_score
        )
        
        # Binary predictions using 0.5 threshold
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, binary_predictions)
        precision = precision_score(y_val, binary_predictions, zero_division=0)
        recall = recall_score(y_val, binary_predictions, zero_division=0)
        f1 = f1_score(y_val, binary_predictions, zero_division=0)
        average_precision = average_precision_score(y_val, predictions)
        
        # Class-specific accuracy
        synthetic_mask = y_val == 1
        test_mask = y_val == 0
        
        synthetic_accuracy = accuracy_score(y_val[synthetic_mask], binary_predictions[synthetic_mask]) if np.sum(synthetic_mask) > 0 else 0
        test_accuracy = accuracy_score(y_val[test_mask], binary_predictions[test_mask]) if np.sum(test_mask) > 0 else 0
        
        # Summary statistics
        n_synthetic = np.sum(y_val == 1)
        n_test = np.sum(y_val == 0)
        
        results = {
            # Primary discriminability metric
            "discriminability_auroc": float(auroc),
            
            # Additional classification metrics
            "discriminability_accuracy": float(accuracy),
            "discriminability_precision": float(precision),
            "discriminability_recall": float(recall),
            "discriminability_f1_score": float(f1),
            "discriminability_average_precision": float(average_precision),
            
            # Class-specific metrics
            "discriminability_synthetic_accuracy": float(synthetic_accuracy),
            "discriminability_test_accuracy": float(test_accuracy),
            
            # Data summary
            "discriminability_n_synthetic_val": int(n_synthetic),
            "discriminability_n_test_val": int(n_test),
            "discriminability_validation_split": float(self.disc_config['validation_split']),
            
            # Training info
            "discriminability_training_epochs": int(self.disc_config['train_max_epochs']),
            "discriminability_batch_size": int(self.disc_config['batch_size']),
            "discriminability_learning_rate": float(self.disc_config['lr']),
        }
        
        # Add training data path if saved
        if training_data_path:
            results["discriminability_training_data_path"] = str(training_data_path)
        
        return results
    
    def _standardize_shape(self, x: np.ndarray) -> np.ndarray:
        """
        Ensure sequences are in (N, A, L) format for discriminability classifier.
        
        Args:
            x: Input sequences of shape (N, L, A) or (N, A, L)
            
        Returns:
            Sequences in (N, A, L) format
        """
        if len(x.shape) != 3:
            raise ValueError(f"Input must be 3D, got shape {x.shape}")
            
        # Check if we need to transpose from (N, L, A) to (N, A, L)
        # Assume A=4 for DNA sequences
        if x.shape[2] == 4 and x.shape[1] != 4:
            return np.transpose(x, (0, 2, 1))
        
        return x


# Convenience functions for standalone use

def evaluate_discriminability(x_synthetic: Union[np.ndarray, torch.Tensor],
                            x_test: Union[np.ndarray, torch.Tensor],
                            config: Optional[Dict[str, Any]] = None,
                            save_training_data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate discriminability between synthetic and test sequences.
    
    Args:
        x_synthetic: Synthetic sequences
        x_test: Test sequences
        config: Configuration dictionary
        save_training_data_path: Path to save training data
        
    Returns:
        Dictionary with discriminability results
    """
    if config is None:
        config = {}
    
    evaluator = DiscriminabilitySimilarityEvaluator(config)
    return evaluator.evaluate(
        x_synthetic, x_test, 
        save_training_data_path=save_training_data_path
    )


def quick_discriminability_test(x_synthetic: Union[np.ndarray, torch.Tensor],
                              x_test: Union[np.ndarray, torch.Tensor],
                              max_epochs: int = 20,
                              batch_size: int = 128) -> float:
    """
    Quick discriminability test with minimal configuration.
    
    Args:
        x_synthetic: Synthetic sequences
        x_test: Test sequences
        max_epochs: Maximum training epochs
        batch_size: Batch size for training
        
    Returns:
        AUROC score (lower is better for generator)
    """
    config = {
        'discriminability': {
            'train_max_epochs': max_epochs,
            'batch_size': batch_size,
            'save_training_data': False
        }
    }
    
    results = evaluate_discriminability(x_synthetic, x_test, config)
    return results['discriminability_auroc']


def load_discriminability_training_data(h5_path: str) -> Dict[str, np.ndarray]:
    """
    Load previously saved discriminability training data.
    
    Args:
        h5_path: Path to H5 file with training data
        
    Returns:
        Dictionary with training data arrays
    """
    import h5py
    
    data = {}
    with h5py.File(h5_path, 'r') as f:
        data['X_train'] = f['X_train'][()]
        data['y_train'] = f['y_train'][()]
        data['X_val'] = f['X_val'][()]
        data['y_val'] = f['y_val'][()]
        
        # Load metadata
        data['metadata'] = {
            'synthetic_label': f.attrs.get('synthetic_label', 1),
            'test_label': f.attrs.get('test_label', 0),
            'n_synthetic': f.attrs.get('n_synthetic', 0),
            'n_test': f.attrs.get('n_test', 0),
            'validation_split': f.attrs.get('validation_split', 0.2)
        }
    
    return data