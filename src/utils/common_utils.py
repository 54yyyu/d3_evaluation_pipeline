"""
Common utilities for the evaluation pipeline.

This module provides general utility functions used throughout the pipeline.
"""

import os
import logging
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional file to save logs
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("evaluation_pipeline")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    """
    Get torch device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda")
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return torch.device(device)


def format_results_for_display(results: Dict[str, Any]) -> str:
    """
    Format evaluation results for display.
    
    Args:
        results: Results dictionary
        
    Returns:
        Formatted string
    """
    lines = ["Evaluation Results:"]
    lines.append("=" * 50)
    
    for key, value in results.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        elif isinstance(value, (int, str)):
            lines.append(f"{key}: {value}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, float):
                    lines.append(f"  {subkey}: {subvalue:.6f}")
                else:
                    lines.append(f"  {subkey}: {subvalue}")
    
    return "\n".join(lines)


def create_run_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create metadata for an evaluation run.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Metadata dictionary
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "system_info": {
            "python_version": f"{torch.__version__}",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }


def validate_file_exists(file_path: Union[str, Path], 
                        description: str = "File") -> Path:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to file
        description: Description of file for error messages
        
    Returns:
        Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage info
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
            "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        }
    except ImportError:
        return {"error": "psutil not available"}


def profile_function_time(func):
    """
    Decorator to profile function execution time.
    
    Args:
        func: Function to profile
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger = logging.getLogger("evaluation_pipeline")
        logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
        
        return result
    return wrapper


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            description: Description of the operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        self.logger = logging.getLogger("evaluation_pipeline")
        
    def update(self, step: Optional[int] = None) -> None:
        """
        Update progress.
        
        Args:
            step: Current step number (if None, increment by 1)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress = self.current_step / self.total_steps
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if progress > 0:
            eta = elapsed / progress - elapsed
            self.logger.info(
                f"{self.description}: {self.current_step}/{self.total_steps} "
                f"({progress*100:.1f}%) - ETA: {eta:.1f}s"
            )
    
    def finish(self) -> None:
        """Mark progress as finished."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{self.description} completed in {elapsed:.1f}s")


def chunks(lst: list, chunk_size: int):
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of the list
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def convert_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Numpy array
    """
    return tensor.detach().cpu().numpy()


def load_numpy_safely(file_path: Union[str, Path]) -> np.ndarray:
    """
    Safely load numpy array from file.
    
    Args:
        file_path: Path to numpy file
        
    Returns:
        Loaded numpy array
        
    Raises:
        ValueError: If file cannot be loaded
    """
    try:
        return np.load(file_path)
    except Exception as e:
        raise ValueError(f"Could not load numpy file {file_path}: {e}")


def summarize_array(arr: np.ndarray, name: str = "Array") -> str:
    """
    Create a summary string for a numpy array.
    
    Args:
        arr: Numpy array
        name: Name for the array
        
    Returns:
        Summary string
    """
    return (
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}"
    )