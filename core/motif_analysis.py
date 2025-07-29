import numpy as np
from datetime import datetime
import pickle
from tqdm import tqdm

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
    from utils.helpers import put_deepstarr_into_NLA, one_hot_to_seq, create_fasta_file
    from utils.seq_evals_improved import (
        motif_count, enrich_pr, make_occurrence_matrix, frobenius_norm
    )
    
    # Get current timestamp
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Convert tensors to appropriate format
    x_test, x_synthetic = put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor)

    # Motif Enrichment
    print("Converting sequences to text format...")
    x_synthetic_e = one_hot_to_seq(x_synthetic)
    x_test_e = one_hot_to_seq(x_test)

    print("Creating FASTA files...")
    create_fasta_file(x_synthetic_e, 'sub_sythetic_seq.txt')
    create_fasta_file(x_test_e, 'sub_test_seq.txt')

    print("Scanning test sequences for motifs...")
    test_motif_counts = motif_count('sub_test_seq.txt', 'JASPAR2024_CORE_non-redundant_pfms_meme.txt')
    print("Scanning synthetic sequences for motifs...")
    synthetic_motif_counts = motif_count('sub_sythetic_seq.txt', 'JASPAR2024_CORE_non-redundant_pfms_meme.txt')
    print("Computing enrichment statistics...")
    pr = enrich_pr(test_motif_counts, synthetic_motif_counts)

    # Motif co-occurrence
    print("Creating motif occurrence matrix for test sequences...")
    test_motif_matrix = make_occurrence_matrix('sub_test_seq.txt')
    print("Creating motif occurrence matrix for synthetic sequences...")
    synthetic_motif_matrix = make_occurrence_matrix('sub_sythetic_seq.txt')

    mm_1 = np.array(test_motif_matrix).T
    mm_2 = np.array(synthetic_motif_matrix).T

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