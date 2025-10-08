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

def run_predictive_distribution_shift_analysis(oracle_model, x_test_tensor, x_synthetic_tensor, output_dir=".", sample_name=None, model_type='deepstarr'):
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

    Returns:
        dict: Results dictionary with distribution shift metric
    """
    import torch
    import numpy as np
    from tqdm import tqdm

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Computing model predictions for distribution shift analysis...")

    # Use batch processing to avoid GPU memory issues
    batch_size = 8  # Very conservative batch size for large SEI models

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU for batch processing with batch_size={batch_size}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU for batch processing with batch_size={batch_size}")

    # Move model to device
    oracle_model = oracle_model.to(device)
    oracle_model.eval()

    def process_in_batches(tensor, desc="Processing"):
        """Process tensor in batches to avoid memory issues."""
        predictions = []

        with torch.no_grad():
            for i in tqdm(range(0, len(tensor), batch_size), desc=desc):
                batch = tensor[i:i+batch_size].to(device)
                batch_pred = oracle_model(batch)

                predictions.append(batch_pred.detach().cpu())

                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return torch.cat(predictions, dim=0).numpy()

    # Process test and synthetic sequences in batches
    y_hat_test = process_in_batches(x_test_tensor, "Processing test sequences")
    y_hat_syn = process_in_batches(x_synthetic_tensor, "Processing synthetic sequences")
    
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