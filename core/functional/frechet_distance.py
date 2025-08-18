import numpy as np
import torch
from datetime import datetime
import pickle

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
    from utils.seq_evals_improved import calculate_activation_statistics, calculate_frechet_distance
    
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