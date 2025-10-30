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

def run_predictive_distribution_shift_analysis(oracle_model, x_test_tensor, x_synthetic_tensor, output_dir=".", sample_name=None, model_type='deepstarr', per_dimension=False):
    """
    Run predictive distribution shift analysis.

    Uses the Kolmogorov-Smirnov statistic to compare empirical cumulative distribution
    functions of oracle predictions for generated and real sequences.

    Args:
        oracle_model: The oracle model (DeepSTARR, MPRALegNet, or tuple of 3 models for multi-oracle)
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        sample_name: Name of sample for batch processing (optional)
        model_type: Type of model ('deepstarr', 'lentimpra', 'multi-oracle')
        per_dimension: If True and model_type is 'multi-oracle', compute metrics separately for each oracle (default: False)

    Returns:
        dict: Results dictionary with distribution shift metric (3 separate values if per_dimension=True for multi-oracle)
    """
    from utils.helpers import load_predictions, load_multi_oracle_predictions

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Computing model predictions for distribution shift analysis...")
    if model_type == 'multi-oracle':
        y_hat_test, y_hat_syn = load_multi_oracle_predictions(x_test_tensor, x_synthetic_tensor, oracle_model)
    else:
        y_hat_test, y_hat_syn = load_predictions(x_test_tensor, x_synthetic_tensor, oracle_model)

    print("Computing predictive distribution shift...")

    # Compute metrics
    if per_dimension and model_type == 'multi-oracle':
        # Compute separate KS statistic for each of the 3 oracles
        ks_stat_oracle_1 = scipy.stats.ks_2samp(y_hat_test[:, 0], y_hat_syn[:, 0]).statistic
        ks_stat_oracle_2 = scipy.stats.ks_2samp(y_hat_test[:, 1], y_hat_syn[:, 1]).statistic
        ks_stat_oracle_3 = scipy.stats.ks_2samp(y_hat_test[:, 2], y_hat_syn[:, 2]).statistic

        results = {
            'predictive_distribution_shift_ks_statistic_oracle_1': ks_stat_oracle_1,
            'predictive_distribution_shift_ks_statistic_oracle_2': ks_stat_oracle_2,
            'predictive_distribution_shift_ks_statistic_oracle_3': ks_stat_oracle_3,
            'y_hat_test': y_hat_test,
            'y_hat_syn': y_hat_syn
        }
    else:
        # Default behavior: compute single KS statistic (averaged over all dimensions if multi-oracle)
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