import numpy as np
import torch
from datetime import datetime
import pickle

def run_functional_similarity_analysis(deepstarr, x_test_tensor, x_synthetic_tensor, x_train_tensor, output_dir="."):
    """
    Run functional similarity analysis including fidelity, Frechet distance, and distribution shift.
    
    Args:
        deepstarr: The DeepSTARR model
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        x_train_tensor: Training sequences tensor
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with all functional similarity metrics
    """
    # Import required functions from utils
    from utils.helpers import load_predictions, get_penultimate_embeddings, put_deepstarr_into_NLA
    from utils.seq_evals_improved import (
        conditional_generation_fidelity,
        calculate_activation_statistics, calculate_frechet_distance,
        predictive_distribution_shift, calculate_cross_sequence_identity_batch,
        kmer_statistics
    )
    
    # Get current timestamp
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Functional similarity: Conditional generation fidelity
    print("Computing model predictions for fidelity analysis...")
    y_hat_test, y_hat_syn = load_predictions(x_test_tensor, x_synthetic_tensor, deepstarr)
    mse = conditional_generation_fidelity(y_hat_syn, y_hat_test)

    # Functional similarity: Frechet distance
    print("Extracting embeddings for Frechet distance...")
    embeddings1 = get_penultimate_embeddings(deepstarr, x_test_tensor)
    embeddings2 = get_penultimate_embeddings(deepstarr, x_synthetic_tensor)
    print("Computing activation statistics...")
    mu1, sigma1 = calculate_activation_statistics(embeddings1)
    mu2, sigma2 = calculate_activation_statistics(embeddings2)
    frechet_distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    # Functional similarity: Predictive distribution shift
    print("Computing predictive distribution shift...")
    hamming_distance = predictive_distribution_shift(x_synthetic_tensor, x_test_tensor)

    # Sequence similarity: Percent identity
    print("Computing sequence identity (synthetic vs synthetic)...")
    x_synthetic_tensor2 = x_synthetic_tensor
    percent_identity_1 = calculate_cross_sequence_identity_batch(x_synthetic_tensor, x_synthetic_tensor2, batch_size=256)
    max_percent_identity_1 = np.max(percent_identity_1, axis=1)
    average_max_percent_identity_1 = np.mean(max_percent_identity_1)
    global_max_percent_identity_1 = np.max(max_percent_identity_1)

    print("Computing sequence identity (synthetic vs training)...")
    percent_identity_2 = calculate_cross_sequence_identity_batch(x_synthetic_tensor, x_train_tensor, batch_size=2000)
    max_percent_identity_2 = np.max(percent_identity_2, axis=1)
    average_max_percent_identity_2 = np.mean(max_percent_identity_2) 
    global_max_percent_identity_2 = np.max(max_percent_identity_2)

    # Sequence similarity: k-mer spectrum shift
    print("Computing k-mer spectrum statistics...")
    X_test, X_syn = put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor)
    kmer_length = 3
    Kullback_Leibler_divergence = kmer_statistics(kmer_length, X_test, X_syn)[0]
    Jensen_Shannon_distance = kmer_statistics(kmer_length, X_test, X_syn)[1]

    # Create results dictionary
    results = {
        'Conditional generation fidelity - mse': mse,
        'Frechet distance': frechet_distance,
        'Predictive distribution shift - Hamming_distance': hamming_distance,
        'Global max percent identity (samples v samples)': global_max_percent_identity_1,
        'Global max percent identity (samples v training)': global_max_percent_identity_2,
        'Average max percent identity (samples v samples)': average_max_percent_identity_1,
        'Average max percent identity (samples v training)': average_max_percent_identity_2,
        'kmer_spectra (Kullback Leibler divergence)': Kullback_Leibler_divergence, 
        'kmer_spectra (Jensen-Shannon distance)': Jensen_Shannon_distance
    }

    # Save results
    filename = f'{output_dir}/functional_similarity_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print(f"Functional similarity results saved to '{filename}'")
    return results