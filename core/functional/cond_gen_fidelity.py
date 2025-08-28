import numpy as np
import torch
from datetime import datetime
import pickle

def conditional_generation_fidelity(activity1, activity2):
    """Compute MSE between predicted activities."""
    return np.mean((activity1 - activity2)**2)

def run_conditional_generation_fidelity_analysis(deepstarr, x_test_tensor, x_synthetic_tensor, output_dir=".", sample_name=None):
    """
    Run conditional generation fidelity analysis.
    
    Measures how well generated sequences achieve functional activities similar to real sequences
    by computing MSE between oracle predictions.
    
    Args:
        deepstarr: The DeepSTARR oracle model
        x_test_tensor: Test sequences tensor  
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        sample_name: Name of sample for batch processing (optional)
        
    Returns:
        dict: Results dictionary with fidelity MSE
    """
    from utils.helpers import load_predictions
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("Computing model predictions for fidelity analysis...")
    y_hat_test, y_hat_syn = load_predictions(x_test_tensor, x_synthetic_tensor, deepstarr)
    mse = conditional_generation_fidelity(y_hat_syn, y_hat_test)
    
    results = {
        'conditional_generation_fidelity_mse': mse
    }
    
    # Handle batch vs single mode
    if sample_name is not None:
        # Batch mode - use new format
        from utils.batch_helpers import write_concise_csv, write_full_h5, get_concise_metrics
        
        # Write concise metrics
        concise_metrics = get_concise_metrics('cond_gen_fidelity', results)
        write_concise_csv(output_dir, 'cond_gen_fidelity', sample_name, concise_metrics)
        
        # Write full results
        write_full_h5(output_dir, 'cond_gen_fidelity', sample_name, results)
        
        print(f"Conditional generation fidelity results saved for sample '{sample_name}'")
    else:
        # Single mode - keep original format
        filename = f'{output_dir}/cond_gen_fidelity_{current_date}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Conditional generation fidelity results saved to '{filename}'")
    
    return results