import numpy as np
import torch
from datetime import datetime
import pickle

def conditional_generation_fidelity(activity1, activity2):
    """Compute MSE between predicted activities."""
    return np.mean((activity1 - activity2)**2)

def run_conditional_generation_fidelity_analysis(oracle_model, x_test_tensor, x_synthetic_tensor, output_dir=".", sample_name=None, model_type='deepstarr'):
    """
    Run conditional generation fidelity analysis.

    Measures how well generated sequences achieve functional activities similar to real sequences
    by computing MSE between oracle predictions.

    Args:
        oracle_model: The oracle model (DeepSTARR, MPRALegNet, or SEI)
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        sample_name: Name of sample for batch processing (optional)
        model_type: Type of oracle model ('deepstarr', 'mpralegnet', 'lentimpra', 'sei')

    Returns:
        dict: Results dictionary with fidelity MSE
    """
    import torch
    import numpy as np
    from tqdm import tqdm

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Computing model predictions for fidelity analysis...")

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