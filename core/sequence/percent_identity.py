import numpy as np
import torch
from datetime import datetime
import pickle
from tqdm import tqdm

def calculate_cross_sequence_identity_batch(X_train, X_test, batch_size):
    """Calculate cross-sequence identity using batched dot products."""
    num_train, seq_length, alphabet_size = X_train.shape    
    num_test = X_test.shape[0]
    
    # Reshape the matrices for dot product computation
    X_train = np.reshape(X_train, [-1, seq_length * alphabet_size])
    X_test = np.reshape(X_test, [-1, seq_length * alphabet_size])
    
    # Initialize the matrix to store the results
    seq_identity = np.zeros((num_train, num_test)).astype(np.int8)
    
    # Process the training data in batches
    total_batches = (num_train + batch_size - 1) // batch_size
    for start_idx in tqdm(range(0, num_train, batch_size), desc="Computing sequence identity", total=total_batches):
        end_idx = min(start_idx + batch_size, num_train)
        
        # Compute the dot product for this batch
        batch_result = np.dot(X_train[start_idx:end_idx], X_test.T) 
        
        # Store the result in the corresponding slice of the output matrix
        seq_identity[start_idx:end_idx, :] = batch_result.astype(np.int8)
    
    return seq_identity

def run_percent_identity_analysis(x_synthetic_tensor, x_train_tensor, output_dir="."):
    """
    Run percent identity analysis.
    
    Quantifies similarity using normalized Hamming distance. Measures both memorization 
    (similarity to training data) and diversity (similarity within generated sequences).
    
    Args:
        x_synthetic_tensor: Synthetic sequences tensor
        x_train_tensor: Training sequences tensor
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with percent identity metrics
    """
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("Computing sequence identity (synthetic vs synthetic)...")
    x_synthetic_tensor2 = x_synthetic_tensor
    percent_identity_1 = calculate_cross_sequence_identity_batch(x_synthetic_tensor, x_synthetic_tensor2, batch_size=256)
    max_percent_identity_1 = np.max(percent_identity_1, axis=1)
    average_max_percent_identity_1 = np.mean(max_percent_identity_1)
    global_max_percent_identity_1 = np.max(max_percent_identity_1)

    print("Computing sequence identity (synthetic vs training)...")
    percent_identity_2 = calculate_cross_sequence_identity_batch(x_synthetic_tensor, x_train_tensor, batch_size=2000)
    max_percent_identity_2 = np.max(percent_identity_2, axis=1)
    average_max_percent_identity_2 = np.mean(max_percent_identity_2) 
    global_max_percent_identity_2 = np.max(max_percent_identity_2)
    
    results = {
        'global_max_percent_identity_samples_vs_samples': global_max_percent_identity_1,
        'global_max_percent_identity_samples_vs_training': global_max_percent_identity_2,
        'average_max_percent_identity_samples_vs_samples': average_max_percent_identity_1,
        'average_max_percent_identity_samples_vs_training': average_max_percent_identity_2
    }
    
    # Save results
    filename = f'{output_dir}/percent_identity_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Percent identity results saved to '{filename}'")
    return results