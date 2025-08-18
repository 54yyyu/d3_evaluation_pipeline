import numpy as np
import torch
from datetime import datetime
import pickle

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
    from utils.seq_evals_improved import calculate_cross_sequence_identity_batch
    
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