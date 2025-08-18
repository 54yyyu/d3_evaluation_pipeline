import numpy as np
import torch
from datetime import datetime
import pickle

def run_motif_cooccurrence_analysis(x_test_tensor, x_synthetic_tensor, output_dir="."):
    """
    Run motif co-occurrence analysis.
    
    Evaluates whether generated sequences preserve motif co-occurrence patterns found in
    real sequences using Frobenius norm of covariance matrices.
    
    Args:
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with motif co-occurrence statistics
    """
    from utils.helpers import put_deepstarr_into_NLA, one_hot_to_seq, create_fasta_file
    from utils.seq_evals_improved import make_occurrence_matrix, frobenius_norm
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Convert tensors to appropriate format
    x_test, x_synthetic = put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor)

    print("Converting sequences to text format...")
    x_synthetic_e = one_hot_to_seq(x_synthetic)
    x_test_e = one_hot_to_seq(x_test)

    print("Creating FASTA files...")
    create_fasta_file(x_synthetic_e, 'sub_synthetic_seq.txt')
    create_fasta_file(x_test_e, 'sub_test_seq.txt')

    print("Creating motif occurrence matrix for test sequences...")
    test_motif_matrix = make_occurrence_matrix('sub_test_seq.txt')
    print("Creating motif occurrence matrix for synthetic sequences...")
    synthetic_motif_matrix = make_occurrence_matrix('sub_synthetic_seq.txt')

    mm_1 = np.array(test_motif_matrix).T
    mm_2 = np.array(synthetic_motif_matrix).T

    C = np.cov(mm_1)
    C2 = np.cov(mm_2) 

    fn = frobenius_norm(C, C2)
    
    results = {
        'frobenius_norm': fn
    }
    
    # Save results
    filename = f'{output_dir}/motif_cooccurrence_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Motif co-occurrence results saved to '{filename}'")
    return results