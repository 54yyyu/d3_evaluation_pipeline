"""
Base evaluator class for all evaluation metrics.

This module provides the abstract base class that all specific evaluators should inherit from.
It defines the common interface and structure for evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
from datetime import datetime
import pickle
import json


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation metrics."""
    
    def __init__(self, config: Dict[str, Any], name: str):
        """
        Initialize the base evaluator.
        
        Args:
            config: Configuration dictionary containing evaluation parameters
            name: Name of the evaluator (e.g., "functional_similarity")
        """
        self.config = config
        self.name = name
        self.results = {}
        self.metadata = {
            "evaluator_name": name,
            "created_at": datetime.now().isoformat(),
            "config": config
        }
    
    @abstractmethod
    def evaluate(self, 
                 x_synthetic: Union[np.ndarray, torch.Tensor],
                 x_test: Union[np.ndarray, torch.Tensor],
                 oracle_model: Any,
                 **kwargs) -> Dict[str, Any]:
        """
        Perform the evaluation.
        
        Args:
            x_synthetic: Generated/synthetic sequences (N, L, A) or (N, A, L)
            x_test: Test/observed sequences (N, L, A) or (N, A, L)  
            oracle_model: Trained oracle model for predictions
            **kwargs: Additional arguments specific to the evaluator
            
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    @abstractmethod
    def get_required_inputs(self) -> Dict[str, str]:
        """
        Get the required inputs for this evaluator.
        
        Returns:
            Dictionary mapping input names to descriptions
        """
        pass
    
    def validate_inputs(self, 
                       x_synthetic: Union[np.ndarray, torch.Tensor],
                       x_test: Union[np.ndarray, torch.Tensor],
                       **kwargs) -> bool:
        """
        Validate that inputs have correct shapes and types.
        
        Args:
            x_synthetic: Generated sequences
            x_test: Test sequences
            **kwargs: Additional inputs to validate
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Convert to numpy for validation
        if isinstance(x_synthetic, torch.Tensor):
            x_synthetic = x_synthetic.detach().numpy()
        if isinstance(x_test, torch.Tensor):
            x_test = x_test.detach().numpy()
            
        # Check shapes
        if len(x_synthetic.shape) != 3:
            raise ValueError(f"x_synthetic must be 3D array, got shape {x_synthetic.shape}")
        if len(x_test.shape) != 3:
            raise ValueError(f"x_test must be 3D array, got shape {x_test.shape}")
            
        # Check if dimensions are compatible
        if x_synthetic.shape[1:] != x_test.shape[1:]:
            raise ValueError(f"Sequence dimensions don't match: {x_synthetic.shape[1:]} vs {x_test.shape[1:]}")
            
        return True
    
    def save_results(self, output_path: str, format: str = "pickle") -> None:
        """
        Save evaluation results to file.
        
        Args:
            output_path: Path to save results
            format: Format to save in ("pickle", "json")
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{self.name}_{timestamp}"
        
        # Add metadata to results
        output_data = {
            "results": self.results,
            "metadata": self.metadata
        }
        
        if format == "pickle":
            filepath = f"{output_path}/{filename}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(output_data, f)
        elif format == "json":
            filepath = f"{output_path}/{filename}.json"
            # Convert numpy arrays to lists for JSON serialization
            json_data = self._prepare_for_json(output_data)
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Results saved to: {filepath}")
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Convert numpy arrays and other non-JSON types for serialization."""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.float64):
            return float(data)
        elif isinstance(data, np.int64):
            return int(data)
        else:
            return data
    
    def _ensure_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert numpy array to torch tensor if needed."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float()
    
    def _ensure_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert torch tensor to numpy array if needed."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    def _standardize_shape(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Standardize input shape to (N, L, A) format.
        
        Args:
            x: Input sequences of shape (N, L, A) or (N, A, L)
            
        Returns:
            Sequences in (N, L, A) format
        """
        if len(x.shape) != 3:
            raise ValueError(f"Input must be 3D, got shape {x.shape}")
            
        # Check if we need to transpose from (N, A, L) to (N, L, A)
        # Assume A=4 for DNA sequences
        if x.shape[1] == 4 and x.shape[2] != 4:
            if isinstance(x, torch.Tensor):
                return x.transpose(1, 2)
            else:
                return np.transpose(x, (0, 2, 1))
        
        return x
    
    def update_results(self, new_results: Dict[str, Any]) -> None:
        """Update the results dictionary with new results."""
        self.results.update(new_results)
        self.metadata["updated_at"] = datetime.now().isoformat()