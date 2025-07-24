"""
Data utilities for loading and processing common inputs.

This module provides functions for loading the standard inputs required
by the evaluation pipeline: synthetic sequences, test datasets, and oracle models.
"""

import os
import h5py
import numpy as np
import torch
from typing import Dict, Tuple, Union, Any, Optional
from pathlib import Path


class DataLoader:
    """Class for loading and managing evaluation data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader with configuration.
        
        Args:
            config: Configuration dictionary containing data paths
        """
        self.config = config
        self.data_cache = {}
    
    def load_synthetic_sequences(self, samples_path: str) -> np.ndarray:
        """
        Load synthetic/generated sequences from NPZ file.
        
        Args:
            samples_path: Path to the samples.npz file
            
        Returns:
            Synthetic sequences as numpy array with shape (N, L, A)
        """
        if samples_path in self.data_cache:
            return self.data_cache[samples_path]
            
        if not os.path.exists(samples_path):
            raise FileNotFoundError(f"Samples file not found: {samples_path}")
            
        data = np.load(samples_path)
        samples = []
        
        # Load all arrays from the npz file
        for key in data.files:
            samples.append(data[key])
        
        # Transpose samples to get shape (N, L, A) from (N, A, L)
        x_synthetic = np.transpose(samples[0], (0, 2, 1))
        
        self.data_cache[samples_path] = x_synthetic
        return x_synthetic
    
    def load_dataset(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """
        Load dataset from H5 file.
        
        Args:
            dataset_path: Path to the dataset H5 file
            
        Returns:
            Dictionary containing X_test, X_train, X_valid, Y_test, Y_train, Y_valid
        """
        if dataset_path in self.data_cache:
            return self.data_cache[dataset_path]
            
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        dataset = {}
        
        with h5py.File(dataset_path, 'r') as f:
            # Load sequence data
            dataset['X_test'] = f['X_test'][()]
            dataset['X_train'] = f['X_train'][()]
            
            # Load labels if they exist
            if 'Y_test' in f:
                dataset['Y_test'] = f['Y_test'][()]
            if 'Y_train' in f:
                dataset['Y_train'] = f['Y_train'][()]
            if 'X_valid' in f:
                dataset['X_valid'] = f['X_valid'][()]
            if 'Y_valid' in f:
                dataset['Y_valid'] = f['Y_valid'][()]
        
        self.data_cache[dataset_path] = dataset
        return dataset
    
    def extract_evaluation_data(self, 
                               samples_path: str, 
                               dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the main data needed for evaluation.
        
        Args:
            samples_path: Path to synthetic sequences
            dataset_path: Path to dataset
            
        Returns:
            Tuple of (x_test, x_synthetic, x_train)
        """
        # Load synthetic sequences
        x_synthetic = self.load_synthetic_sequences(samples_path)
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        x_test = dataset['X_test']
        x_train = dataset['X_train']
        
        return x_test, x_synthetic, x_train
    
    def convert_to_tensors(self, 
                          x_test: np.ndarray, 
                          x_synthetic: np.ndarray, 
                          x_train: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, ...]:
        """
        Convert numpy arrays to PyTorch tensors.
        
        Args:
            x_test: Test sequences
            x_synthetic: Synthetic sequences  
            x_train: Training sequences (optional)
            
        Returns:
            Tuple of tensors
        """
        tensors = [
            torch.from_numpy(x_test).float(),
            torch.from_numpy(x_synthetic).float()
        ]
        
        if x_train is not None:
            tensors.append(torch.from_numpy(x_train).float())
            
        return tuple(tensors)
    
    def validate_data_compatibility(self, 
                                  x_synthetic: np.ndarray, 
                                  x_test: np.ndarray) -> bool:
        """
        Validate that synthetic and test data have compatible shapes.
        
        Args:
            x_synthetic: Synthetic sequences
            x_test: Test sequences
            
        Returns:
            True if compatible
            
        Raises:
            ValueError: If shapes are incompatible
        """
        if len(x_synthetic.shape) != 3 or len(x_test.shape) != 3:
            raise ValueError("Sequences must be 3-dimensional")
            
        # Check sequence length and alphabet size compatibility
        if x_synthetic.shape[1:] != x_test.shape[1:]:
            raise ValueError(
                f"Sequence dimensions incompatible: "
                f"synthetic {x_synthetic.shape[1:]} vs test {x_test.shape[1:]}"
            )
            
        return True
    
    def get_data_summary(self, 
                        x_synthetic: np.ndarray, 
                        x_test: np.ndarray, 
                        x_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get summary statistics of the loaded data.
        
        Args:
            x_synthetic: Synthetic sequences
            x_test: Test sequences
            x_train: Training sequences (optional)
            
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            "synthetic_sequences": {
                "count": x_synthetic.shape[0],
                "shape": x_synthetic.shape,
                "sequence_length": x_synthetic.shape[1],
                "alphabet_size": x_synthetic.shape[2]
            },
            "test_sequences": {
                "count": x_test.shape[0], 
                "shape": x_test.shape,
                "sequence_length": x_test.shape[1],
                "alphabet_size": x_test.shape[2]
            }
        }
        
        if x_train is not None:
            summary["train_sequences"] = {
                "count": x_train.shape[0],
                "shape": x_train.shape,
                "sequence_length": x_train.shape[1],
                "alphabet_size": x_train.shape[2]
            }
            
        return summary


def load_standard_inputs(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the standard inputs for evaluation pipeline.
    
    Args:
        config: Configuration dictionary with data paths
        
    Returns:
        Tuple of (x_test, x_synthetic, x_train)
    """
    loader = DataLoader(config)
    
    samples_path = config['data']['samples_path']
    dataset_path = config['data']['dataset_path']
    
    return loader.extract_evaluation_data(samples_path, dataset_path)


def prepare_sequences_for_analysis(sequences: Union[np.ndarray, torch.Tensor], 
                                 target_format: str = "NLA") -> np.ndarray:
    """
    Prepare sequences for different types of analysis by ensuring correct format.
    
    Args:
        sequences: Input sequences 
        target_format: Target format ("NLA" for (N,L,A) or "NAL" for (N,A,L))
        
    Returns:
        Sequences in the target format
    """
    # Convert to numpy if tensor
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.detach().cpu().numpy()
    
    # Determine current format and convert if needed
    if target_format == "NLA":
        # Want (N, L, A) format
        if sequences.shape[1] == 4:  # Currently (N, A, L)
            sequences = np.transpose(sequences, (0, 2, 1))
    elif target_format == "NAL":
        # Want (N, A, L) format  
        if sequences.shape[2] == 4:  # Currently (N, L, A)
            sequences = np.transpose(sequences, (0, 2, 1))
    else:
        raise ValueError(f"Unsupported target format: {target_format}")
        
    return sequences


def one_hot_to_sequences(one_hot: np.ndarray, 
                        dna_dict: Dict[int, str] = None) -> list:
    """
    Convert one-hot encoded sequences to ACGT string format.
    
    Args:
        one_hot: One-hot encoded sequences (N, L, A)
        dna_dict: Mapping from indices to nucleotides
        
    Returns:
        List of DNA sequence strings
    """
    if dna_dict is None:
        dna_dict = {0: "A", 1: "C", 2: "G", 3: "T"}
    
    sequences = []
    for seq in one_hot:
        # Convert each position to nucleotide
        seq_str = "".join([dna_dict[np.where(pos)[0][0]] for pos in seq])
        sequences.append(seq_str)
        
    return sequences


def create_fasta_file(sequences: list, output_path: str) -> None:
    """
    Create a FASTA file from a list of sequences.
    
    Args:
        sequences: List of DNA sequence strings
        output_path: Path to output FASTA file
    """
    with open(output_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">Seq{i}\n{seq}\n")