import numpy as np
import torch
from datetime import datetime
import pickle
from tqdm import tqdm
import os

# Try to import memelite first, fallback to pymemesuite
try:
    # Try relative import first
    from utils.seq_evals_func_motifs import motif_count_memelite, make_occurrence_matrix_memelite
    USE_MEMELITE = True
except (ImportError, ValueError):
    try:
        # Try absolute import as fallback
        from utils.seq_evals_func_motifs import motif_count_memelite, make_occurrence_matrix_memelite
        USE_MEMELITE = True
    except ImportError:
        try:
            # Try direct import from same directory
            import sys
            sys.path.append('utils')
            import seq_evals_func_motifs
            motif_count_memelite = seq_evals_func_motifs.motif_count_memelite
            make_occurrence_matrix_memelite = seq_evals_func_motifs.make_occurrence_matrix_memelite
            USE_MEMELITE = True
        except ImportError:
            USE_MEMELITE = False

# Always import pymemesuite as fallback
try:
    from pymemesuite import fimo
    from pymemesuite.common import MotifFile, Sequence
    from pymemesuite.fimo import FIMO
    PYMEMESUITE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import pymemesuite: {e}")
    PYMEMESUITE_AVAILABLE = False
    # Define dummy classes to avoid NameError
    class FIMO:
        def __init__(self):
            raise ImportError("pymemesuite not available")
    class MotifFile:
        def __init__(self, *args):
            raise ImportError("pymemesuite not available")
    class Sequence:
        def __init__(self, *args):
            raise ImportError("pymemesuite not available")

from Bio import SeqIO
import Bio

def sequences_to_onehot(sequences):
    """Convert DNA sequences to one-hot encoding for memelite."""
    # DNA nucleotide mapping
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Get max sequence length
    max_length = max(len(seq) for seq in sequences)
    
    # Initialize one-hot array (N, A, L) where N=num_sequences, A=4 nucleotides, L=sequence_length
    onehot = np.zeros((len(sequences), 4, max_length))
    
    for i, seq in tqdm(enumerate(sequences), desc="Converting to one-hot", total=len(sequences)):
        for j, nucleotide in enumerate(seq.upper()):
            if nucleotide in nucleotide_to_index:
                onehot[i, nucleotide_to_index[nucleotide], j] = 1
    
    return onehot

def make_occurrence_matrix(path, motif_db_path='JASPAR2024_CORE_non-redundant_pfms_meme.txt'):
    """
    path is the filepath to the list of sequences in fasta format
    returns a matrix containing the motif counts for all the sequences
    """
    
    if USE_MEMELITE:
        try:
            # Read sequences and convert to one-hot for memelite
            sequences = []
            for record in Bio.SeqIO.parse(path, "fasta"):
                sequences.append(str(record.seq))
            
            onehot_seqs = sequences_to_onehot(sequences)
            return make_occurrence_matrix_memelite(motif_db_path, onehot_seqs)
        except Exception as e:
            # Fall back to pymemesuite if memelite fails
            print(f"Warning: memelite failed ({e}), falling back to pymemesuite")
            pass

    # Check if pymemesuite is available
    if not PYMEMESUITE_AVAILABLE:
        raise ImportError("Neither memelite nor pymemesuite is available for motif analysis")

    # Original pymemesuite implementation
    motif_ids = []
    occurrence = []

    sequences = [
        Sequence(str(record.seq), name=record.id.encode())
        for record in Bio.SeqIO.parse(path, "fasta")
        ]

    
    try:
        fimo = FIMO() 
    except Exception as e:
        raise RuntimeError(f"Failed to initialize FIMO: {e}. Check that FIMO class is properly imported.")
    
    #matrix with m rows and n columns
    occurrence_matrix = []
    for sequence in tqdm(sequences, desc="Processing sequences for occurrence matrix"):
        sequence = [sequence]
        occurrence = []
        motif_ids = []
        with MotifFile(motif_db_path) as motif_file:
            motifs_list = list(motif_file)
            motif_file.seek(0)  # Reset to beginning for actual processing
            for motif in motifs_list:
                pattern = fimo.score_motif(motif, sequence, motif_file.background)
                motif_ids.append(motif.accession.decode())
                occurrence.append(len(pattern.matched_elements))
        occurrence_matrix.append(occurrence)

    return occurrence_matrix

def covariance_matrix(x):
    """Compute covariance matrix."""
    return np.cov(x)

def frobenius_norm(cov, cov2):
    """Compute Frobenius norm between two covariance matrices."""
    return np.sqrt(np.sum((cov - cov2)**2))

def run_motif_cooccurrence_analysis(x_test_tensor, x_synthetic_tensor, output_dir=".", motif_db_path='JASPAR2024_CORE_non-redundant_pfms_meme.txt', sample_name=None):
    """
    Run motif co-occurrence analysis.
    
    Evaluates whether generated sequences preserve motif co-occurrence patterns found in
    real sequences using Frobenius norm of covariance matrices.
    
    Args:
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        output_dir: Directory to save results
        motif_db_path: Path to motif database file (default: JASPAR2024_CORE_non-redundant_pfms_meme.txt)
        sample_name: Name of sample for batch processing (optional)
        
    Returns:
        dict: Results dictionary with motif co-occurrence statistics
    """
    from utils.helpers import put_deepstarr_into_NLA, one_hot_to_seq, create_fasta_file
    
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
    test_motif_matrix = make_occurrence_matrix('sub_test_seq.txt', motif_db_path)
    print("Creating motif occurrence matrix for synthetic sequences...")
    synthetic_motif_matrix = make_occurrence_matrix('sub_synthetic_seq.txt', motif_db_path)

    mm_1 = np.array(test_motif_matrix).T
    mm_2 = np.array(synthetic_motif_matrix).T

    C = np.cov(mm_1)
    C2 = np.cov(mm_2) 

    fn = frobenius_norm(C, C2)
    
    results = {
        'frobenius_norm': fn,
        'test_motif_matrix': np.array(test_motif_matrix),
        'synthetic_motif_matrix': np.array(synthetic_motif_matrix),
        'test_covariance_matrix': C,
        'synthetic_covariance_matrix': C2
    }
    
    # Handle batch vs single mode
    if sample_name is not None:
        # Batch mode - use new format
        from utils.batch_helpers import write_concise_csv, write_full_h5, get_concise_metrics
        
        # Write concise metrics
        concise_metrics = get_concise_metrics('motif_cooccurrence', results)
        write_concise_csv(output_dir, 'motif_cooccurrence', sample_name, concise_metrics)
        
        # Write full results
        write_full_h5(output_dir, 'motif_cooccurrence', sample_name, results)
        
        print(f"Motif co-occurrence results saved for sample '{sample_name}'")
    else:
        # Single mode - keep original format
        filename = f'{output_dir}/motif_cooccurrence_{current_date}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Motif co-occurrence results saved to '{filename}'")
    
    return results