import numpy as np
import torch
from datetime import datetime
import pickle
from tqdm import tqdm

print("=" * 60)
print("PERCENT IDENTITY ANALYSIS")
print("=" * 60)

def calculate_sequence_identity_batch(X_source, X_target, batch_size=256):
    """Calculate percent identity using normalized Hamming distance.
    
    Percent identity: PID(xgen,i, xreal) = 1 − dH(xgen,i, xreal) / L
    where dH is Hamming distance and L is sequence length.
    
    Args:
        X_source: Source sequences with shape (num_source, 4, seq_length)
        X_target: Target sequences with shape (num_target, 4, seq_length)
        batch_size: Batch size for processing
        
    Returns:
        Percent identity matrix with shape (num_source, num_target)
        Values range from 0.0 (no similarity) to 1.0 (identical)
    """
    num_source, alphabet_size, seq_length = X_source.shape    
    num_target = X_target.shape[0]
    L = seq_length  # Actual sequence length for normalization (249)
    
    # Reshape the matrices for dot product computation
    X_source_flat = np.reshape(X_source, [-1, alphabet_size * seq_length])
    X_target_flat = np.reshape(X_target, [-1, alphabet_size * seq_length])
    
    # Initialize the matrix to store percent identity results
    percent_identity = np.zeros((num_source, num_target), dtype=np.float32)
    
    # Process the source data in batches
    total_batches = (num_source + batch_size - 1) // batch_size
    for start_idx in tqdm(range(0, num_source, batch_size), desc="Computing percent identity", total=total_batches):
        end_idx = min(start_idx + batch_size, num_source)
        
        # Compute dot product (number of matching positions)
        matching_positions = np.dot(X_source_flat[start_idx:end_idx], X_target_flat.T)
        
        # Convert to percent identity by normalizing with sequence length L
        # PID = matching_positions / L (equivalent to 1 - dH/L)
        batch_pid = matching_positions.astype(np.float32) / L
        
        # Store the result in the corresponding slice of the output matrix
        percent_identity[start_idx:end_idx, :] = batch_pid
    
    return percent_identity

def run_percent_identity_analysis(x_synthetic_tensor, x_train_tensor, output_dir=".", sample_name=None):
    """
    Run percent identity analysis.
    
    Quantifies similarity using normalized Hamming distance. Measures both memorization 
    (similarity to training data) and diversity (similarity within generated sequences).
    
    Args:
        x_synthetic_tensor: Synthetic sequences tensor
        x_train_tensor: Training sequences tensor
        output_dir: Directory to save results
        sample_name: Name of sample for batch processing (optional)
        
    Returns:
        dict: Results dictionary with percent identity metrics
    """
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("✓ Sequence identity function defined")
    
    print("Computing sequence identity (synthetic vs synthetic)...")
    x_synthetic_tensor2 = x_synthetic_tensor
    percent_identity_1 = calculate_sequence_identity_batch(x_synthetic_tensor, x_synthetic_tensor2, batch_size=256)
    max_percent_identity_1 = np.max(percent_identity_1, axis=1)
    average_max_percent_identity_1 = np.mean(max_percent_identity_1)
    global_max_percent_identity_1 = np.max(max_percent_identity_1)

    print("Computing sequence identity (synthetic vs training)...")
    percent_identity_2 = calculate_sequence_identity_batch(x_synthetic_tensor, x_train_tensor, batch_size=2000)
    max_percent_identity_2 = np.max(percent_identity_2, axis=1)
    average_max_percent_identity_2 = np.mean(max_percent_identity_2) 
    global_max_percent_identity_2 = np.max(max_percent_identity_2)
    
    results = {
        'global_max_percent_identity_samples_vs_samples': global_max_percent_identity_1,
        'global_max_percent_identity_samples_vs_training': global_max_percent_identity_2,
        'average_max_percent_identity_samples_vs_samples': average_max_percent_identity_1,
        'average_max_percent_identity_samples_vs_training': average_max_percent_identity_2,
        'percent_identity_matrix_samples_vs_samples': percent_identity_1,
        'percent_identity_matrix_samples_vs_training': percent_identity_2
    }
    
    # Handle batch vs single mode
    if sample_name is not None:
        # Batch mode - use new format
        from utils.batch_helpers import write_concise_csv, write_full_h5, get_concise_metrics
        
        # Write concise metrics
        concise_metrics = get_concise_metrics('percent_identity', results)
        write_concise_csv(output_dir, 'percent_identity', sample_name, concise_metrics)
        
        # Write full results
        write_full_h5(output_dir, 'percent_identity', sample_name, results)
        
        print(f"Percent identity results saved for sample '{sample_name}'")
    else:
        # Single mode - keep original format
        filename = f'{output_dir}/percent_identity_{current_date}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Percent identity results saved to '{filename}'")
    
    return results