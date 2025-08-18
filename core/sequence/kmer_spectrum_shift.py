import numpy as np
import torch
from datetime import datetime
import pickle

def run_kmer_spectrum_shift_analysis(x_test_tensor, x_synthetic_tensor, kmer_length=3, output_dir="."):
    """
    Run k-mer spectrum shift analysis.
    
    Compares k-mer frequency distributions between generated and real sequences
    using Jensen-Shannon divergence.
    
    Args:
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        kmer_length: Length of k-mers to analyze (default: 3)
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with k-mer statistics
    """
    from utils.helpers import put_deepstarr_into_NLA
    from utils.seq_evals_improved import kmer_statistics
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print(f"Computing k-mer spectrum statistics (k={kmer_length})...")
    X_test, X_syn = put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor)
    Kullback_Leibler_divergence = kmer_statistics(kmer_length, X_test, X_syn)[0]
    Jensen_Shannon_distance = kmer_statistics(kmer_length, X_test, X_syn)[1]
    
    results = {
        'kmer_spectra_kullback_leibler_divergence': Kullback_Leibler_divergence,
        'kmer_spectra_jensen_shannon_distance': Jensen_Shannon_distance,
        'kmer_length': kmer_length
    }
    
    # Save results
    filename = f'{output_dir}/kmer_spectrum_shift_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"k-mer spectrum shift results saved to '{filename}'")
    return results