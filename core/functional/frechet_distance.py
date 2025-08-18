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

def run_frechet_distance_analysis(deepstarr, x_test_tensor, x_synthetic_tensor, output_dir="."):
    """
    Run Fréchet distance analysis.
    
    Compares the distribution of oracle-predicted embeddings between real and generated sequences.
    Lower values indicate closer alignment in oracle embedding space.
    
    Args:
        deepstarr: The DeepSTARR oracle model
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with Fréchet distance
    """
    from utils.helpers import get_penultimate_embeddings
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("Extracting embeddings for Fréchet distance...")
    embeddings1 = get_penultimate_embeddings(deepstarr, x_test_tensor)
    embeddings2 = get_penultimate_embeddings(deepstarr, x_synthetic_tensor)
    
    print("Computing activation statistics...")
    mu1, sigma1 = calculate_activation_statistics(embeddings1)
    mu2, sigma2 = calculate_activation_statistics(embeddings2)
    frechet_distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    results = {
        'frechet_distance': frechet_distance
    }
    
    # Save results
    filename = f'{output_dir}/frechet_distance_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Fréchet distance results saved to '{filename}'")
    return results