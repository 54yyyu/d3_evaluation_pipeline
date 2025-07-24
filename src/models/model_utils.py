"""
Model utilities for loading and working with oracle models.

This module provides functions for loading oracle models and extracting
various types of information needed for evaluation (predictions, embeddings, etc.).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
from pathlib import Path
import sys


class EmbeddingExtractor:
    """Helper class for extracting intermediate layer embeddings."""
    
    def __init__(self):
        self.embedding = None

    def hook(self, module, input, output):
        """Hook function to capture layer output."""
        self.embedding = output.detach()


class ModelLoader:
    """Class for loading and managing oracle models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model loader with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_cache = {}
        
    def load_oracle_model(self, model_path: str, model_type: str = "deepstarr") -> Any:
        """
        Load oracle model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            model_type: Type of model ("deepstarr", etc.)
            
        Returns:
            Loaded model in evaluation mode
        """
        if model_path in self.model_cache:
            return self.model_cache[model_path]
            
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
        if model_type.lower() == "deepstarr":
            model = self._load_deepstarr_model(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model.eval()
        self.model_cache[model_path] = model
        return model
    
    def _load_deepstarr_model(self, model_path: str) -> Any:
        """Load DeepSTARR model from checkpoint."""
        try:
            # Try to import the DeepSTARR classes
            # This assumes the deepstarr.py module is available
            sys.path.append(str(Path(__file__).parent.parent.parent.parent / "small_data"))
            from deepstarr import PL_DeepSTARR
            
            model = PL_DeepSTARR.load_from_checkpoint(model_path).eval()
            return model
            
        except ImportError as e:
            raise ImportError(f"Could not import DeepSTARR classes: {e}")
        except Exception as e:
            raise RuntimeError(f"Could not load DeepSTARR model: {e}")


class ModelPredictor:
    """Class for making predictions with oracle models."""
    
    def __init__(self, model: Any):
        """
        Initialize predictor with a model.
        
        Args:
            model: Loaded oracle model
        """
        self.model = model
        self.extractor = EmbeddingExtractor()
        
    def predict(self, 
               x_test: torch.Tensor, 
               x_synthetic: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test and synthetic sequences.
        
        Args:
            x_test: Test sequences tensor
            x_synthetic: Synthetic sequences tensor
            
        Returns:
            Tuple of (test_predictions, synthetic_predictions) as numpy arrays
        """
        self.model.eval()
        
        with torch.no_grad():
            y_hat_test = self.model(x_test)
            y_hat_synthetic = self.model(x_synthetic)
            
        return y_hat_test.detach().numpy(), y_hat_synthetic.detach().numpy()
    
    def get_penultimate_embeddings(self, 
                                  x: torch.Tensor, 
                                  layer_name: str = 'model.batchnorm6') -> torch.Tensor:
        """
        Extract penultimate layer embeddings.
        
        Args:
            x: Input sequences tensor
            layer_name: Name of the layer to extract embeddings from
            
        Returns:
            Embeddings tensor
        """
        # Find the specified layer and register hook
        hook_registered = False
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(self.extractor.hook)
                hook_registered = True
                break
        
        if not hook_registered:
            raise ValueError(f"Could not find layer '{layer_name}' in model")
        
        # Forward pass to extract embeddings
        with torch.no_grad():
            _ = self.model(x)
        
        # Remove the hook
        handle.remove()
        
        return self.extractor.embedding
    
    def predict_batch(self, 
                     x: torch.Tensor, 
                     batch_size: int = 128) -> np.ndarray:
        """
        Make predictions in batches to handle large datasets.
        
        Args:
            x: Input sequences tensor
            batch_size: Batch size for processing
            
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                pred = self.model(batch)
                predictions.append(pred.detach().cpu().numpy())
                
        return np.concatenate(predictions, axis=0)


def load_oracle_model(model_path: str, 
                     model_type: str = "deepstarr", 
                     config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Load oracle model (convenience function).
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model
        config: Optional configuration dictionary
        
    Returns:
        Loaded model
    """
    if config is None:
        config = {}
        
    loader = ModelLoader(config)
    return loader.load_oracle_model(model_path, model_type)


def make_predictions(model: Any, 
                    x_test: torch.Tensor, 
                    x_synthetic: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with oracle model (convenience function).
    
    Args:
        model: Oracle model
        x_test: Test sequences
        x_synthetic: Synthetic sequences
        
    Returns:
        Tuple of (test_predictions, synthetic_predictions)
    """
    predictor = ModelPredictor(model)
    return predictor.predict(x_test, x_synthetic)


def extract_embeddings(model: Any, 
                      x: torch.Tensor, 
                      layer_name: str = 'model.batchnorm6') -> torch.Tensor:
    """
    Extract embeddings from specified layer (convenience function).
    
    Args:
        model: Oracle model
        x: Input sequences
        layer_name: Layer to extract from
        
    Returns:
        Embeddings tensor
    """
    predictor = ModelPredictor(model)
    return predictor.get_penultimate_embeddings(x, layer_name)


def convert_to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        x: Input array or tensor
        
    Returns:
        PyTorch tensor
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


def ensure_correct_input_format(x: torch.Tensor, model_type: str = "deepstarr") -> torch.Tensor:
    """
    Ensure input tensor is in correct format for the model.
    
    Args:
        x: Input tensor
        model_type: Type of model
        
    Returns:
        Tensor in correct format
    """
    if model_type.lower() == "deepstarr":
        # DeepSTARR expects (N, A, L) format
        if x.shape[1] != 4 and x.shape[2] == 4:
            # Convert from (N, L, A) to (N, A, L)
            x = x.transpose(1, 2)
    
    return x


def validate_model_compatibility(model: Any, 
                               x_sample: torch.Tensor) -> bool:
    """
    Validate that model can process the given input format.
    
    Args:
        model: Oracle model
        x_sample: Sample input tensor
        
    Returns:
        True if compatible
        
    Raises:
        RuntimeError: If model cannot process input
    """
    try:
        model.eval()
        with torch.no_grad():
            # Try a forward pass with a small sample
            test_input = x_sample[:1] if len(x_sample) > 1 else x_sample
            _ = model(test_input)
        return True
    except Exception as e:
        raise RuntimeError(f"Model compatibility check failed: {e}")


class ModelWrapper:
    """Wrapper class to standardize model interfaces."""
    
    def __init__(self, model: Any, model_type: str = "deepstarr"):
        """
        Initialize wrapper.
        
        Args:
            model: The oracle model
            model_type: Type of model
        """
        self.model = model
        self.model_type = model_type
        self.predictor = ModelPredictor(model)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions (allows using wrapper like original model)."""
        return self.model(x)
        
    def predict(self, x_test: torch.Tensor, x_synthetic: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on both test and synthetic data."""
        return self.predictor.predict(x_test, x_synthetic)
        
    def get_embeddings(self, x: torch.Tensor, layer_name: str = None) -> torch.Tensor:
        """Extract embeddings from specified layer."""
        if layer_name is None:
            layer_name = 'model.batchnorm6'  # Default for DeepSTARR
        return self.predictor.get_penultimate_embeddings(x, layer_name)
        
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self