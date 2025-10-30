import numpy as np
import torch
from datetime import datetime
import pickle
from scipy import linalg

def calculate_activation_statistics(embeddings):
    """Calculate mean and covariance of embeddings."""
    embeddings_d = embeddings.detach().cpu().numpy()
    mu = np.mean(embeddings_d, axis=0)
    sigma = np.cov(embeddings_d, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet distance between two multivariate Gaussians."""
    # adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    # Frechet distance: d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def run_frechet_distance_analysis(oracle_model, x_test_tensor, x_synthetic_tensor, output_dir=".", sample_name=None, model_type='deepstarr', per_dimension=False):
    """
    Run Fréchet distance analysis.

    Compares the distribution of oracle-predicted embeddings between real and generated sequences.
    Lower values indicate closer alignment in oracle embedding space.

    Args:
        oracle_model: The oracle model (DeepSTARR, MPRALegNet, or tuple of 3 models for multi-oracle)
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        sample_name: Name of sample for batch processing (optional)
        model_type: Type of model ('deepstarr', 'mpralegnet', 'lentimpra', 'multi-oracle')
        per_dimension: If True and model_type is 'multi-oracle', compute metrics separately for each oracle (default: False)

    Returns:
        dict: Results dictionary with Fréchet distance (3 separate values if per_dimension=True for multi-oracle)
    """
    from utils.helpers import get_penultimate_embeddings, get_multi_oracle_embeddings

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Extracting embeddings for Fréchet distance...")

    # Compute metrics
    if per_dimension and model_type == 'multi-oracle':
        # Compute separate Frechet distance for each of the 3 oracles
        model1, model2, model3 = oracle_model

        # Oracle 1
        embeddings1_oracle1 = get_penultimate_embeddings(model1, x_test_tensor, 'lentimpra')
        embeddings2_oracle1 = get_penultimate_embeddings(model1, x_synthetic_tensor, 'lentimpra')
        mu1_o1, sigma1_o1 = calculate_activation_statistics(embeddings1_oracle1)
        mu2_o1, sigma2_o1 = calculate_activation_statistics(embeddings2_oracle1)
        frechet_distance_oracle_1 = calculate_frechet_distance(mu1_o1, sigma1_o1, mu2_o1, sigma2_o1)

        # Oracle 2
        embeddings1_oracle2 = get_penultimate_embeddings(model2, x_test_tensor, 'lentimpra')
        embeddings2_oracle2 = get_penultimate_embeddings(model2, x_synthetic_tensor, 'lentimpra')
        mu1_o2, sigma1_o2 = calculate_activation_statistics(embeddings1_oracle2)
        mu2_o2, sigma2_o2 = calculate_activation_statistics(embeddings2_oracle2)
        frechet_distance_oracle_2 = calculate_frechet_distance(mu1_o2, sigma1_o2, mu2_o2, sigma2_o2)

        # Oracle 3
        embeddings1_oracle3 = get_penultimate_embeddings(model3, x_test_tensor, 'lentimpra')
        embeddings2_oracle3 = get_penultimate_embeddings(model3, x_synthetic_tensor, 'lentimpra')
        mu1_o3, sigma1_o3 = calculate_activation_statistics(embeddings1_oracle3)
        mu2_o3, sigma2_o3 = calculate_activation_statistics(embeddings2_oracle3)
        frechet_distance_oracle_3 = calculate_frechet_distance(mu1_o3, sigma1_o3, mu2_o3, sigma2_o3)

        results = {
            'frechet_distance_oracle_1': frechet_distance_oracle_1,
            'frechet_distance_oracle_2': frechet_distance_oracle_2,
            'frechet_distance_oracle_3': frechet_distance_oracle_3
        }
    else:
        # Default behavior: use concatenated embeddings for multi-oracle
        if model_type == 'multi-oracle':
            embeddings1 = get_multi_oracle_embeddings(oracle_model, x_test_tensor, 'lentimpra')
            embeddings2 = get_multi_oracle_embeddings(oracle_model, x_synthetic_tensor, 'lentimpra')
        else:
            embeddings1 = get_penultimate_embeddings(oracle_model, x_test_tensor, model_type)
            embeddings2 = get_penultimate_embeddings(oracle_model, x_synthetic_tensor, model_type)

        print("Computing activation statistics...")
        mu1, sigma1 = calculate_activation_statistics(embeddings1)
        mu2, sigma2 = calculate_activation_statistics(embeddings2)
        frechet_distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        results = {
            'frechet_distance': frechet_distance,
            'mu1': mu1,
            'sigma1': sigma1,
            'mu2': mu2,
            'sigma2': sigma2
        }
    
    # Handle batch vs single mode
    if sample_name is not None:
        # Batch mode - use new format
        from utils.batch_helpers import write_concise_csv, write_full_h5, get_concise_metrics
        
        # Write concise metrics
        concise_metrics = get_concise_metrics('frechet_distance', results)
        write_concise_csv(output_dir, 'frechet_distance', sample_name, concise_metrics)
        
        # Write full results
        write_full_h5(output_dir, 'frechet_distance', sample_name, results)
        
        print(f"Fréchet distance results saved for sample '{sample_name}'")
    else:
        # Single mode - keep original format
        filename = f'{output_dir}/frechet_distance_{current_date}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Fréchet distance results saved to '{filename}'")
    
    return results