import numpy as np
import torch
from datetime import datetime
import pickle
import scipy.stats

def predictive_distribution_shift(y_hat_test, y_hat_syn):
    """Compute Kolmogorov-Smirnov test statistic between predicted distributions."""
    # Calculate KS statistic for each output dimension and take the mean
    ks_statistic = scipy.stats.kstest(y_hat_test, y_hat_syn).statistic.mean()
    return ks_statistic

def run_predictive_distribution_shift_analysis(deepstarr, x_test_tensor, x_synthetic_tensor, output_dir="."):
    """
    Run predictive distribution shift analysis.
    
    Uses the Kolmogorov-Smirnov statistic to compare empirical cumulative distribution 
    functions of oracle predictions for generated and real sequences.
    
    Args:
        deepstarr: The DeepSTARR oracle model
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with distribution shift metric
    """
    from utils.helpers import load_predictions
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("Computing model predictions for distribution shift analysis...")
    y_hat_test, y_hat_syn = load_predictions(x_test_tensor, x_synthetic_tensor, deepstarr)
    
    print("Computing predictive distribution shift...")
    ks_statistic = predictive_distribution_shift(y_hat_test, y_hat_syn)
    
    results = {
        'predictive_distribution_shift_ks_statistic': ks_statistic
    }
    
    # Save results
    filename = f'{output_dir}/predictive_dist_shift_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Predictive distribution shift results saved to '{filename}'")
    return results