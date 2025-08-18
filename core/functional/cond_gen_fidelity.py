import numpy as np
import torch
from datetime import datetime
import pickle

def run_conditional_generation_fidelity_analysis(deepstarr, x_test_tensor, x_synthetic_tensor, output_dir="."):
    """
    Run conditional generation fidelity analysis.
    
    Measures how well generated sequences achieve functional activities similar to real sequences
    by computing MSE between oracle predictions.
    
    Args:
        deepstarr: The DeepSTARR oracle model
        x_test_tensor: Test sequences tensor  
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with fidelity MSE
    """
    from utils.helpers import load_predictions
    from utils.seq_evals_improved import conditional_generation_fidelity
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("Computing model predictions for fidelity analysis...")
    y_hat_test, y_hat_syn = load_predictions(x_test_tensor, x_synthetic_tensor, deepstarr)
    mse = conditional_generation_fidelity(y_hat_syn, y_hat_test)
    
    results = {
        'conditional_generation_fidelity_mse': mse
    }
    
    # Save results
    filename = f'{output_dir}/cond_gen_fidelity_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Conditional generation fidelity results saved to '{filename}'")
    return results