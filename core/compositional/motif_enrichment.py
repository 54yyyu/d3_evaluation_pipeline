import numpy as np
import torch
from datetime import datetime
import pickle

def run_motif_enrichment_analysis(x_test_tensor, x_synthetic_tensor, output_dir="."):
    """
    Run motif enrichment analysis.
    
    Measures whether generated sequences recapitulate key motif content found in real genomic 
    sequences using Pearson correlation of motif occurrence counts.
    
    Args:
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        
    Returns:
        dict: Results dictionary with motif enrichment statistics
    """
    from utils.helpers import put_deepstarr_into_NLA, one_hot_to_seq, create_fasta_file
    from utils.seq_evals_improved import motif_count, enrich_pr
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Convert tensors to appropriate format
    x_test, x_synthetic = put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor)

    print("Converting sequences to text format...")
    x_synthetic_e = one_hot_to_seq(x_synthetic)
    x_test_e = one_hot_to_seq(x_test)

    print("Creating FASTA files...")
    create_fasta_file(x_synthetic_e, 'sub_synthetic_seq.txt')
    create_fasta_file(x_test_e, 'sub_test_seq.txt')

    print("Scanning test sequences for motifs...")
    test_motif_counts = motif_count('sub_test_seq.txt', 'JASPAR2024_CORE_non-redundant_pfms_meme.txt')
    print("Scanning synthetic sequences for motifs...")
    synthetic_motif_counts = motif_count('sub_synthetic_seq.txt', 'JASPAR2024_CORE_non-redundant_pfms_meme.txt')
    print("Computing enrichment statistics...")
    pr = enrich_pr(test_motif_counts, synthetic_motif_counts)
    
    results = {
        'pearson_r_statistic': pr
    }
    
    # Save results
    filename = f'{output_dir}/motif_enrichment_{current_date}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Motif enrichment results saved to '{filename}'")
    return results