import numpy as np
import torch
from datetime import datetime
import pickle
import scipy.stats

def predictive_distribution_shift(x_synthetic_tensor, x_test_tensor):
    """Compute Kolmogorov-Smirnov test statistic between sequence distributions."""
    # encode bases using 0,1,2,3 (eliminate a dimension)
    base_indices_test = np.argmax(x_test_tensor.detach().cpu().numpy(), axis=1)
    base_indices_syn = np.argmax(x_synthetic_tensor.detach().cpu().numpy(), axis=1)

    # flatten the arrays (now they are one dimension)
    base_indices_test_f = base_indices_test.flatten()
    base_indices_syn_f = base_indices_syn.flatten()

    # return ks test statistic
    return scipy.stats.ks_2samp(base_indices_syn_f, base_indices_test_f).statistic

def run_predictive_distribution_shift_analysis(x_test_tensor, x_synthetic_tensor, output_dir="."):
    """
    Run predictive distribution shift analysis.
    
    Uses the Kolmogorov-Smirnov statistic to compare empirical cumulative distribution 
    functions of oracle predictions for generated and real sequences.
    
    Args:
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with distribution shift metric
    """
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("Computing predictive distribution shift...")
    hamming_distance = predictive_distribution_shift(x_synthetic_tensor, x_test_tensor)
    
    results = {
        'predictive_distribution_shift_hamming_distance': hamming_distance
    }
    
    # Save results
    filename = f'{output_dir}/predictive_dist_shift_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Predictive distribution shift results saved to '{filename}'")
    return results