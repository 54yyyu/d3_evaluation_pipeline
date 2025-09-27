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

def run_predictive_distribution_shift_analysis(oracle_model, x_test_tensor, x_synthetic_tensor, output_dir=".", sample_name=None, model_type='deepstarr', batch_size=32):
    """
    Run predictive distribution shift analysis.

    Uses the Kolmogorov-Smirnov statistic to compare empirical cumulative distribution
    functions of oracle predictions for generated and real sequences.

    Args:
        oracle_model: The oracle model (DeepSTARR, MPRALegNet, or SEI)
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        sample_name: Name of sample for batch processing (optional)
        model_type: Type of oracle model ('deepstarr', 'mpralegnet', 'lentimpra', 'sei')
        batch_size: Batch size for inference to manage GPU memory

    Returns:
        dict: Results dictionary with distribution shift metric
    """
    from utils.helpers import load_predictions_batched

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Computing model predictions for distribution shift analysis...")
    y_hat_test, y_hat_syn = load_predictions_batched(x_test_tensor, x_synthetic_tensor, oracle_model, model_type, batch_size)
    
    print("Computing predictive distribution shift...")
    ks_statistic = predictive_distribution_shift(y_hat_test, y_hat_syn)
    
    results = {
        'predictive_distribution_shift_ks_statistic': ks_statistic,
        'y_hat_test': y_hat_test,
        'y_hat_syn': y_hat_syn
    }
    
    # Handle batch vs single mode
    if sample_name is not None:
        # Batch mode - use new format
        from utils.batch_helpers import write_concise_csv, write_full_h5, get_concise_metrics
        
        # Write concise metrics
        concise_metrics = get_concise_metrics('predictive_dist_shift', results)
        write_concise_csv(output_dir, 'predictive_dist_shift', sample_name, concise_metrics)
        
        # Write full results
        write_full_h5(output_dir, 'predictive_dist_shift', sample_name, results)
        
        print(f"Predictive distribution shift results saved for sample '{sample_name}'")
    else:
        # Single mode - keep original format
        filename = f'{output_dir}/predictive_dist_shift_{current_date}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Predictive distribution shift results saved to '{filename}'")
    
    return results