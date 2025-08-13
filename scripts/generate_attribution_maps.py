#!/usr/bin/env python3
"""
Generate GradientSHAP attribution maps for visualization data.

This script computes GradientSHAP attribution maps for all sequences across all timesteps
in visualization data files and adds the results back to the H5 file with '_shap.h5' suffix.
"""

import argparse
import sys
import h5py
import numpy as np
import torch
from pathlib import Path
import logging
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.model_utils import load_oracle_model, ModelWrapper
from utils.common_utils import setup_logging, validate_file_exists, set_random_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate GradientSHAP attribution maps for visualization data"
    )
    
    # Required arguments
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input H5 visualization file"
    )
    parser.add_argument(
        "oracle_model_path",
        type=str,
        help="Path to oracle model checkpoint"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file path (default: {input}_shap.h5)"
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=0,
        help="Target class index for attribution (default: 0)"
    )
    parser.add_argument(
        "--k-mer-size",
        type=int,
        default=6,
        help="K-mer size for smoothing (default: 6)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for computation (default: auto)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="deepstarr",
        help="Type of oracle model (default: deepstarr)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Logging arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="File to save logs"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Check input files exist
    validate_file_exists(args.input_file, "Input H5 file")
    validate_file_exists(args.oracle_model_path, "Oracle model checkpoint")
    
    # Set output file if not provided
    if not args.output_file:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.parent / f"{input_path.stem}_shap{input_path.suffix}")
    
    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args


def convert_sequences_to_onehot(sequences: np.ndarray) -> np.ndarray:
    """
    Convert token sequences to one-hot format.
    
    Args:
        sequences: Token sequences with shape (batch_size, seq_length)
        
    Returns:
        One-hot encoded sequences with shape (batch_size, seq_length, 4)
    """
    batch_size, seq_length = sequences.shape
    onehot = np.zeros((batch_size, seq_length, 4), dtype=np.float32)
    
    for i in range(batch_size):
        for j in range(seq_length):
            token = int(sequences[i, j])
            if 0 <= token <= 3:  # Valid nucleotide token
                onehot[i, j, token] = 1.0
    
    return onehot


def _process_attribution_map_4d(saliency_map: np.ndarray, k: int = 6) -> np.ndarray:
    """
    Process attribution map with gradient correction, normalization and k-mer smoothing.
    Keeps result in 4D nucleotide space.
    
    Args:
        saliency_map: Raw attribution scores with shape (batch_size, seq_length, 4)
        k: K-mer size for smoothing
        
    Returns:
        Processed attribution map with shape (batch_size, seq_length, 4)
    """
    # Apply gradient correction
    saliency_map = saliency_map - np.mean(saliency_map, axis=-1, keepdims=True)
    
    # Apply normalization
    norm_factor = np.sum(np.sqrt(np.sum(np.square(saliency_map), axis=-1, keepdims=True)), axis=-2, keepdims=True)
    saliency_map = saliency_map / (norm_factor + 1e-8)
    
    # Apply k-mer smoothing
    saliency_special = saliency_map.copy()
    for i in range(k-1):
        rolled = np.roll(saliency_map, -i-1, axis=-2)
        saliency_special += rolled
    
    return saliency_special


def _gradient_shap(x_seq: torch.Tensor, model: ModelWrapper, class_index: int = 0) -> np.ndarray:
    """
    Calculate Gradient SHAP attribution scores.
    
    Args:
        x_seq: Input sequences with shape (N, L, A)
        model: Wrapped oracle model
        class_index: Target class index
        
    Returns:
        Attribution scores with shape (N, L, A)
    """
    try:
        from captum.attr import GradientShap
    except ImportError:
        raise ImportError("Captum required for attribution analysis")
    
    N, L, A = x_seq.shape
    score_cache = []
    
    for i, x in enumerate(tqdm(x_seq, desc="Computing gradient attributions", unit="seq")):
        # Process single sequence
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.transpose(1, 2)  # Convert to (N, A, L) format for model
        x.requires_grad_(True)
        
        # Create random background - match original parameters
        num_background = 1000
        null_index = np.random.randint(0, A, size=(num_background, L))
        x_null = torch.zeros((num_background, A, L))
        for n in range(num_background):
            for l in range(L):
                x_null[n, null_index[n, l], l] = 1.0
        x_null.requires_grad_(True)
        
        # Calculate Gradient SHAP - match original parameters
        gradient_shap = GradientShap(model)
        grad = gradient_shap.attribute(
            x,
            n_samples=100,
            stdevs=0.1,
            baselines=x_null,
            target=class_index
        )
        
        grad = grad.data.cpu().numpy()
        # Apply gradient correction
        grad -= np.mean(grad, axis=1, keepdims=True)
        score_cache.append(np.squeeze(grad))
    
    score_cache = np.array(score_cache)
    # Convert back to (N, L, A) format
    return np.transpose(score_cache, (0, 2, 1))


def load_visualization_data(file_path: str):
    """
    Load visualization data from H5 file.
    
    Args:
        file_path: Path to H5 visualization file
        
    Returns:
        Tuple of (h5_file, metadata, step_keys)
    """
    h5_file = h5py.File(file_path, 'r')
    
    # Read metadata
    metadata = {}
    for key in h5_file['metadata'].attrs:
        metadata[key] = h5_file['metadata'].attrs[key]
    
    # Get step keys
    step_keys = sorted([k for k in h5_file['steps'].keys() if k.startswith('step_')])
    
    return h5_file, metadata, step_keys


def process_step_attributions(step_group, model, class_index, k_mer_size, batch_size, device):
    """
    Process attributions for a single timestep.
    
    Args:
        step_group: H5 group for the timestep
        model: Oracle model
        class_index: Target class index
        k_mer_size: K-mer size for smoothing
        batch_size: Batch size for processing
        device: Computation device
        
    Returns:
        Processed attribution matrix
    """
    # Load sequences
    sequences = np.array(step_group['sequence'])  # (batch_size, seq_length)
    
    # Convert to one-hot
    sequences_onehot = convert_sequences_to_onehot(sequences)  # (batch_size, seq_length, 4)
    
    # Convert to tensor and move to device
    sequences_tensor = torch.from_numpy(sequences_onehot).float().to(device)
    
    # Compute attributions in batches
    all_attributions = []
    num_sequences = len(sequences_tensor)
    
    for i in range(0, num_sequences, batch_size):
        end_idx = min(i + batch_size, num_sequences)
        batch_sequences = sequences_tensor[i:end_idx]
        
        # Compute raw attributions
        batch_attributions = _gradient_shap(batch_sequences, model, class_index)
        all_attributions.append(batch_attributions)
    
    # Concatenate all batches
    raw_attributions = np.concatenate(all_attributions, axis=0)
    
    # Process attributions
    processed_attributions = _process_attribution_map_4d(raw_attributions, k=k_mer_size)
    
    return processed_attributions


def copy_h5_structure(input_file, output_file):
    """
    Copy H5 file structure and data to output file.
    
    Args:
        input_file: Input H5 file handle
        output_file: Output H5 file handle
    """
    # Copy metadata
    metadata_group = output_file.create_group('metadata')
    for key in input_file['metadata'].attrs:
        metadata_group.attrs[key] = input_file['metadata'].attrs[key]
    
    # Copy metadata datasets if they exist
    for key in input_file['metadata'].keys():
        input_file.copy(f'metadata/{key}', metadata_group)
    
    # Copy steps structure
    steps_group = output_file.create_group('steps')
    
    return steps_group


def main():
    """Main function."""
    args = parse_arguments()
    args = validate_arguments(args)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, args.log_file)
    
    try:
        # Set random seed
        set_random_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
        
        # Load oracle model
        logger.info(f"Loading oracle model from {args.oracle_model_path}")
        oracle_model = load_oracle_model(args.oracle_model_path, args.model_type)
        model_wrapper = ModelWrapper(oracle_model, args.model_type)
        model_wrapper.eval()
        
        # Move model to device
        device = torch.device(args.device)
        oracle_model.to(device)
        logger.info(f"Using device: {device}")
        
        # Load visualization data
        logger.info(f"Loading visualization data from {args.input_file}")
        input_h5, metadata, step_keys = load_visualization_data(args.input_file)
        
        logger.info(f"Found {len(step_keys)} timesteps to process")
        logger.info(f"Dataset: {metadata.get('dataset', 'unknown')}")
        logger.info(f"Num samples: {metadata.get('num_samples', 'unknown')}")
        logger.info(f"Sequence length: {metadata.get('sequence_length', 'unknown')}")
        
        # Create output file
        logger.info(f"Creating output file: {args.output_file}")
        with h5py.File(args.output_file, 'w') as output_h5:
            # Copy original structure
            steps_group = copy_h5_structure(input_h5, output_h5)
            
            # Process each timestep
            for step_key in tqdm(step_keys, desc="Processing timesteps", unit="step"):
                logger.info(f"Processing {step_key}")
                
                # Get input step group
                input_step = input_h5['steps'][step_key]
                
                # Create output step group
                output_step = steps_group.create_group(step_key)
                
                # Copy all existing data
                for key in input_step.keys():
                    input_step.copy(key, output_step)
                
                # Copy attributes
                for key in input_step.attrs:
                    output_step.attrs[key] = input_step.attrs[key]
                
                # Compute and add attribution matrix
                attribution_matrix = process_step_attributions(
                    input_step, model_wrapper, args.class_index, 
                    args.k_mer_size, args.batch_size, device
                )
                
                # Add attribution matrix to output
                output_step.create_dataset(
                    'attribution_matrix', 
                    data=attribution_matrix,
                    dtype=np.float32,
                    compression='gzip'
                )
                
                logger.debug(f"Added attribution matrix with shape {attribution_matrix.shape}")
        
        # Close input file
        input_h5.close()
        
        logger.info(f"Successfully generated attribution maps and saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Attribution generation failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()