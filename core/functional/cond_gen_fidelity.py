import numpy as np
import torch
from datetime import datetime
import pickle

def conditional_generation_fidelity(activity1, activity2):
    """Compute MSE between predicted activities."""
    return np.mean((activity1 - activity2)**2)

def run_conditional_generation_fidelity_analysis(oracle_model, x_test_tensor, x_synthetic_tensor, output_dir=".", sample_name=None, model_type='deepstarr', per_dimension=False):
    """
    Run conditional generation fidelity analysis.

    Measures how well generated sequences achieve functional activities similar to real sequences
    by computing MSE between oracle predictions.

    Args:
        oracle_model: The oracle model (DeepSTARR, MPRALegNet, or tuple of 3 models for multi-oracle)
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        sample_name: Name of sample for batch processing (optional)
        model_type: Type of model ('deepstarr', 'lentimpra', 'multi-oracle')
        per_dimension: If True and model_type is 'multi-oracle', compute metrics separately for each oracle (default: False)

    Returns:
        dict: Results dictionary with fidelity MSE (3 separate values if per_dimension=True for multi-oracle)
    """
    from utils.helpers import load_predictions, load_multi_oracle_predictions

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Computing model predictions for fidelity analysis...")
    if model_type == 'multi-oracle':
        y_hat_test, y_hat_syn = load_multi_oracle_predictions(x_test_tensor, x_synthetic_tensor, oracle_model)
    else:
        y_hat_test, y_hat_syn = load_predictions(x_test_tensor, x_synthetic_tensor, oracle_model)

    # Compute metrics
    if per_dimension and model_type == 'multi-oracle':
        # Compute separate MSE for each of the 3 oracles
        mse_oracle_1 = conditional_generation_fidelity(y_hat_syn[:, 0], y_hat_test[:, 0])
        mse_oracle_2 = conditional_generation_fidelity(y_hat_syn[:, 1], y_hat_test[:, 1])
        mse_oracle_3 = conditional_generation_fidelity(y_hat_syn[:, 2], y_hat_test[:, 2])

        results = {
            'conditional_generation_fidelity_mse_oracle_1': mse_oracle_1,
            'conditional_generation_fidelity_mse_oracle_2': mse_oracle_2,
            'conditional_generation_fidelity_mse_oracle_3': mse_oracle_3
        }
    else:
        # Default behavior: compute single MSE (averaged over all dimensions if multi-oracle)
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