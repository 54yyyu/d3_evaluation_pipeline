import os
import pandas as pd
import numpy as np
import h5py
import torch
from datetime import datetime
import glob
import sys
from pathlib import Path

# Import from helpers for index-to-onehot conversion
from .helpers import is_index_encoded, index_to_onehot

def discover_batch_samples(batch_dir, csv_filename="metadata.csv"):
    """
    Discover samples in batch directory and handle CSV metadata.

    Args:
        batch_dir: Path to directory containing sample files (.npz, .h5, .pt) or subdirectories
        csv_filename: Name of CSV file for metadata

    Returns:
        List of dictionaries with sample metadata, or exits if CSV template created
    """
    batch_dir = Path(batch_dir)
    csv_path = batch_dir / csv_filename

    # Check if CSV already exists
    if csv_path.exists():
        print(f"Using existing metadata file: {csv_path}")
        df = pd.read_csv(csv_path)
        return df.to_dict('records')

    # CSV doesn't exist - need to create template
    print(f"CSV metadata file not found: {csv_path}")
    print("Creating template...")

    # Detect directory structure - look for multiple file types
    sample_files = []
    for pattern in ["*.npz", "*.h5", "*.hdf5", "*.pt", "*.pth"]:
        sample_files.extend(batch_dir.glob(pattern))

    subdirs = [d for d in batch_dir.iterdir() if d.is_dir()]

    samples = []

    if sample_files and not subdirs:
        # Flat structure: folder/*.{npz,h5,pt}
        print(f"Detected flat structure with {len(sample_files)} sample files")
        for sample_file in sorted(sample_files):
            sample_name = sample_file.stem  # filename without extension
            samples.append({
                'sample_name': sample_name,
                'file_path': str(sample_file.relative_to(batch_dir)),
                'created_date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            })

    elif subdirs:
        # Nested structure: folder/subfolder/*.{npz,h5,pt}
        print(f"Detected nested structure with {len(subdirs)} subdirectories")
        for subdir in sorted(subdirs):
            subdir_samples = []
            for pattern in ["*.npz", "*.h5", "*.hdf5", "*.pt", "*.pth"]:
                subdir_samples.extend(subdir.glob(pattern))

            if subdir_samples:
                for sample_file in sorted(subdir_samples):
                    # Create unique sample name using subfolder and filename
                    sample_name = f"{subdir.name}_{sample_file.stem}"
                    samples.append({
                        'sample_name': sample_name,
                        'file_path': str(sample_file.relative_to(batch_dir)),
                        'created_date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    })
            else:
                print(f"Warning: No sample files found in {subdir}")

    else:
        print("Error: No sample files (.npz, .h5, .pt) found in batch directory")
        sys.exit(1)

    # Create template CSV
    df = pd.DataFrame(samples)
    df.to_csv(csv_path, index=False)

    print(f"Template CSV created: {csv_path}")
    print(f"Found {len(samples)} sample files")
    print("CSV wasn't found, template created, can modify the sample names before running again (if using default names just run again)")

    # Exit the program
    sys.exit(0)

def load_batch_sample(batch_dir, sample_record):
    """
    Load a single sample from batch directory.

    Args:
        batch_dir: Path to batch directory
        sample_record: Dictionary with sample metadata from CSV

    Returns:
        Tuple of (sample_name, data) or None if failed
        Data can be npz object, numpy array, or torch tensor depending on file type
    """
    batch_dir = Path(batch_dir)
    file_path = batch_dir / sample_record['file_path']
    sample_name = sample_record['sample_name']

    try:
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            return None

        # Determine file type and load accordingly
        if file_path.suffix == '.npz':
            data = np.load(file_path)
            print(f"Loaded NPZ sample '{sample_name}' from {file_path}")
            # Note: NPZ files are returned as-is, conversion happens in main.py
            return sample_name, data

        elif file_path.suffix in ['.pt', '.pth']:
            # Load PyTorch tensor
            tensor_data = torch.load(file_path, map_location='cpu')
            if isinstance(tensor_data, torch.Tensor):
                data = tensor_data.numpy()
            else:
                print(f"Warning: Expected tensor in {file_path}, got {type(tensor_data)}")
                return None

            # Check if index-encoded and convert to one-hot if needed
            if is_index_encoded(data):
                print(f"  Detected index-encoded sequences (0123=ACGT). Converting to one-hot encoding...")
                data = index_to_onehot(data)
                print(f"  Converted to one-hot with shape: {data.shape}")

            print(f"Loaded PyTorch sample '{sample_name}' from {file_path}")
            return sample_name, data

        elif file_path.suffix in ['.h5', '.hdf5']:
            # Try loading as PyTorch first (some .h5 files are actually PyTorch)
            try:
                tensor_data = torch.load(file_path, map_location='cpu')
                if isinstance(tensor_data, torch.Tensor):
                    data = tensor_data.numpy()

                    # Check if index-encoded and convert to one-hot if needed
                    if is_index_encoded(data):
                        print(f"  Detected index-encoded sequences (0123=ACGT). Converting to one-hot encoding...")
                        data = index_to_onehot(data)
                        print(f"  Converted to one-hot with shape: {data.shape}")

                    print(f"Loaded PyTorch (h5 extension) sample '{sample_name}' from {file_path}")
                    return sample_name, data
            except:
                pass

            # If that fails, try as HDF5
            try:
                with h5py.File(file_path, 'r') as f:
                    # Load first key as data
                    if 'arr_0' in f.keys():
                        data = f['arr_0'][()]
                    elif 'samples' in f.keys():
                        data = f['samples'][()]
                    elif 'x_synthetic' in f.keys():
                        data = f['x_synthetic'][()]
                    else:
                        first_key = list(f.keys())[0]
                        data = f[first_key][()]
                        print(f"Warning: Using key '{first_key}' for sample data")

                # Check if index-encoded and convert to one-hot if needed
                if is_index_encoded(data):
                    print(f"  Detected index-encoded sequences (0123=ACGT). Converting to one-hot encoding...")
                    data = index_to_onehot(data)
                    print(f"  Converted to one-hot with shape: {data.shape}")

                print(f"Loaded HDF5 sample '{sample_name}' from {file_path}")
                return sample_name, data
            except Exception as h5_error:
                print(f"Error loading as HDF5: {h5_error}")
                return None
        else:
            print(f"Warning: Unsupported file type: {file_path.suffix}")
            return None

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def write_concise_csv(output_dir, analysis_name, sample_name, metric_values):
    """
    Write concise metrics to analysis-specific CSV file.
    
    Args:
        output_dir: Output directory path
        analysis_name: Name of analysis (e.g., 'motif_enrichment')
        sample_name: Name of the sample
        metric_values: Dictionary of metric names and values
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{analysis_name}.csv"
    
    # Read existing CSV or create new one
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.DataFrame()
    
    # Add metrics for this sample
    for metric_name, value in metric_values.items():
        df.loc[metric_name, sample_name] = value
    
    # Save CSV without index name to avoid extra comma
    df.index.name = None
    df.to_csv(csv_path)
    print(f"Concise metrics saved to: {csv_path}")

def write_full_h5(output_dir, analysis_name, sample_name, full_results):
    """
    Write comprehensive results to analysis-specific HDF5 file.
    
    Args:
        output_dir: Output directory path
        analysis_name: Name of analysis (e.g., 'motif_enrichment')
        sample_name: Name of the sample
        full_results: Dictionary of all results including arrays/matrices
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    h5_path = output_dir / f"{analysis_name}.h5"
    
    with h5py.File(h5_path, 'a') as f:  # 'a' for append mode
        # Create group for this sample
        if sample_name in f:
            # Sample already exists, create subgroup for multiple data points
            existing_keys = list(f[sample_name].keys())
            next_index = len([k for k in existing_keys if k.startswith('run_')])
            group = f[sample_name].create_group(f'run_{next_index}')
        else:
            group = f.create_group(sample_name)
        
        # Store all results
        for key, value in full_results.items():
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, (int, float, str, bool)):
                group.attrs[key] = value
            elif isinstance(value, list):
                # Convert list to numpy array if possible
                try:
                    arr = np.array(value)
                    group.create_dataset(key, data=arr)
                except:
                    # Store as string if can't convert to array
                    group.attrs[key] = str(value)
            else:
                # Store as string representation
                group.attrs[key] = str(value)
    
    print(f"Full results saved to: {h5_path}")

def get_concise_metrics(analysis_name, results):
    """
    Extract concise metrics based on analysis type.
    
    Args:
        analysis_name: Name of the analysis
        results: Full results dictionary
    
    Returns:
        Dictionary of concise metrics
    """
    metric_mapping = {
        'attribution_consistency': ['KLD', 'KLD_concat'],
        'motif_cooccurrence': ['frobenius_norm'],
        'motif_enrichment': ['pearson_r_statistic'],
        'cond_gen_fidelity': ['conditional_generation_fidelity_mse'],
        'frechet_distance': ['frechet_distance'],
        'predictive_dist_shift': ['predictive_distribution_shift_ks_statistic'],
        'discriminability': ['auroc'],
        'kmer_spectrum_shift': ['js_distance'],
        'percent_identity': ['average_max_percent_identity_samples_vs_training']
    }
    
    if analysis_name not in metric_mapping:
        print(f"Warning: Unknown analysis name '{analysis_name}' for concise metrics")
        return {}
    
    concise_metrics = {}
    for metric_key in metric_mapping[analysis_name]:
        if metric_key in results:
            value = results[metric_key]
            # Handle numpy scalars
            if hasattr(value, 'item'):
                value = value.item()
            concise_metrics[metric_key] = value
        else:
            print(f"Warning: Metric '{metric_key}' not found in results for {analysis_name}")
    
    return concise_metrics