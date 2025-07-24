"""
Configuration utilities for the evaluation pipeline.

This module provides functions for loading, merging, and validating
configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle base configuration inheritance
    if '_base_' in config:
        base_config_path = config_path.parent / config['_base_']
        base_config = load_config(base_config_path)
        config = merge_configs(base_config, config)
        # Remove the _base_ key after merging
        config.pop('_base_', None)
    
    return config


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['data', 'model', 'evaluation', 'output']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section missing: {section}")
    
    # Validate data section
    required_data_keys = ['samples_path', 'dataset_path', 'oracle_path']
    for key in required_data_keys:
        if key not in config['data']:
            raise ValueError(f"Required data configuration missing: {key}")
    
    # Validate model section
    if 'type' not in config['model']:
        raise ValueError("Model type must be specified")
    
    # Validate evaluation section
    evaluation_keys = ['run_functional_similarity', 'run_sequence_similarity', 'run_compositional_similarity']
    if not any(config['evaluation'].get(key, False) for key in evaluation_keys):
        raise ValueError("At least one evaluation type must be enabled")
    
    return True


def resolve_paths(config: Dict[str, Any], base_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in configuration to absolute paths.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for resolving relative paths
        
    Returns:
        Configuration with resolved paths
    """
    if base_path is None:
        base_path = Path.cwd()
    
    config = copy.deepcopy(config)
    
    # Resolve data paths
    path_keys = ['samples_path', 'dataset_path', 'oracle_path', 'motif_database_path']
    for key in path_keys:
        if key in config['data']:
            path = Path(config['data'][key])
            if not path.is_absolute():
                config['data'][key] = str(base_path / path)
    
    # Resolve output directory
    if 'results_dir' in config['output']:
        path = Path(config['output']['results_dir'])
        if not path.is_absolute():
            config['output']['results_dir'] = str(base_path / path)
    
    return config


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Default configuration
    """
    return {
        'data': {
            'samples_path': 'samples.npz',
            'dataset_path': 'dataset.h5', 
            'oracle_path': 'oracle_model.ckpt',
            'motif_database_path': 'motifs.txt'
        },
        'model': {
            'type': 'deepstarr',
            'embedding_layer': 'model.batchnorm6'
        },
        'evaluation': {
            'run_functional_similarity': True,
            'run_sequence_similarity': True,
            'run_compositional_similarity': True,
            'batch_size': 2000,
            'prediction_batch_size': 128,
            'kmer_lengths': [3, 4, 5],
            'attribution_max_samples': 1000
        },
        'output': {
            'results_dir': 'results',
            'format': 'pickle',
            'include_metadata': True
        },
        'compute': {
            'device': 'auto',
            'num_workers': 4,
            'random_seed': 42
        }
    }


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_config_for_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Configuration dictionary for the dataset
    """
    config_dir = Path(__file__).parent.parent.parent / "configs"
    
    # Map dataset names to config files
    dataset_configs = {
        'deepstarr': 'deepstarr_config.yaml',
        'DeepSTARR': 'deepstarr_config.yaml'
    }
    
    if dataset_name.lower() in dataset_configs:
        config_file = dataset_configs[dataset_name.lower()]
        config_path = config_dir / config_file
        
        if config_path.exists():
            return load_config(config_path)
    
    # Fall back to base config
    base_config_path = config_dir / "base_config.yaml"
    if base_config_path.exists():
        return load_config(base_config_path)
    
    # Final fallback to default config
    return create_default_config()


def update_config_from_args(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Update configuration with command-line arguments or other overrides.
    
    Args:
        config: Base configuration
        **kwargs: Key-value pairs to override in config
        
    Returns:
        Updated configuration
    """
    config = copy.deepcopy(config)
    
    # Map flat argument names to nested config paths
    arg_mapping = {
        'samples_path': ['data', 'samples_path'],
        'dataset_path': ['data', 'dataset_path'],
        'oracle_path': ['data', 'oracle_path'],
        'motif_database_path': ['data', 'motif_database_path'],
        'results_dir': ['output', 'results_dir'],
        'batch_size': ['evaluation', 'batch_size'],
        'device': ['compute', 'device'],
        'random_seed': ['compute', 'random_seed']
    }
    
    for arg_name, value in kwargs.items():
        if value is not None and arg_name in arg_mapping:
            keys = arg_mapping[arg_name]
            
            # Navigate to the nested dictionary
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
    
    return config


class ConfigManager:
    """Class for managing configuration throughout the pipeline."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = create_default_config()
        
        self.validate()
    
    def validate(self) -> None:
        """Validate the current configuration."""
        validate_config(self.config)
    
    def resolve_paths(self, base_path: Optional[Path] = None) -> None:
        """Resolve relative paths in configuration."""
        self.config = resolve_paths(self.config, base_path)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        self.config = update_config_from_args(self.config, **kwargs)
        self.validate()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration key (e.g., 'data.samples_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration key
            value: Value to set
        """
        keys = key_path.split('.')
        current = self.config
        
        # Navigate to parent dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        save_config(self.config, output_path)