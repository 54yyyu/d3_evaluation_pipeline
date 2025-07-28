import numpy as np
from datetime import datetime
import pickle

def run_motif_analysis(x_test_tensor, x_synthetic_tensor, output_dir="."):
    """
    Run motif enrichment and co-occurrence analysis.
    
    Args:
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor  
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with motif statistics
    """
    # Import required functions from utils
    from utils.seq_evals_improved import (
        put_deepstarr_into_NLA, one_hot_to_seq, create_fasta_file,
        motif_count, enrich_pr, make_occurrence_matrix, frobenius_norm
    )
    
    # Get current timestamp
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Convert tensors to appropriate format
    x_test, x_synthetic = put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor)

    # Motif Enrichment
    x_synthetic_e = one_hot_to_seq(x_synthetic)
    x_test_e = one_hot_to_seq(x_test)

    create_fasta_file(x_synthetic_e, 'sub_sythetic_seq.txt')
    create_fasta_file(x_test_e, 'sub_test_seq.txt')

    motif_count_1 = motif_count('sub_test_seq.txt', 'JASPAR2024_CORE_non-redundant_pfms_meme.txt')
    motif_count_2 = motif_count('sub_sythetic_seq.txt', 'JASPAR2024_CORE_non-redundant_pfms_meme.txt')
    pr = enrich_pr(motif_count_1, motif_count_2)

    # Motif co-occurrence
    motif_matrix_1 = make_occurrence_matrix('sub_test_seq.txt')
    motif_matrix_2 = make_occurrence_matrix('sub_sythetic_seq.txt')

    mm_1 = np.array(motif_matrix_1).T
    mm_2 = np.array(motif_matrix_2).T

    C = np.cov(mm_1)
    C2 = np.cov(mm_2) 

    fn = frobenius_norm(C, C2)

    # Create results dictionary
    results = {
        'Pearson R Statistic': pr,
        'Frobenius Norm': fn
    }

    # Save results
    filename = f'{output_dir}/motifs_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print(f"Motif analysis results saved to '{filename}'")
    return results