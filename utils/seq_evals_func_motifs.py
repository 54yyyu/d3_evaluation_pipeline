import numpy as np
import pandas as pd
# following are new to seq_evals_improved.py
from memelite.fimo import fimo as memelite_fimo
from memelite.io import read_meme


# no longer need to make fastq file for sequences

def motif_count_memelite(meme_file_path, onehot_seqs, return_list = False):
    """
    Input:
        meme_file_path: JASPAR meme file of motifs
        onehot_seqs: one-hot coded sequences in the shape of N,A,L
        return_list: boolean, if True return an additional list, defaluted to be False
    Output:
        a dictionary containing the motif counts for all the sequences
        optional: an additional output, a list of data frames for each motif's scanning results
    """
    motifs = read_meme(meme_file_path) # a dictionary
    
    hits = memelite_fimo(meme_file_path, onehot_seqs, dim=0) # dim=0(1): each motif (sequence) has a data frame in the output
    motif_names = list(motifs.keys())
    occurrence = [df.shape[0] for df in hits]
    
    motif_counts = dict(zip(motif_names, occurrence))
    
    if return_list:
        hits_filtered = [df for df in hits if not df.empty]
        return motif_counts, hits_filtered
    else:
        return motif_counts

def make_occurrence_matrix_memelite(meme_file_path, onehot_seqs):
    """
    Input:
        meme_file_path: JASPAR meme file of motifs
        onehot_seqs: one-hot coded sequences in the shape of N,A,L
    Output:
        a matrix of motif occurrence in each sequences, sequences in rows and motifs in columns
    """
    motifs = read_meme(meme_file_path)
    motif_names = list(motifs.keys())
    hits_by_seq = memelite_fimo(meme_file_path, onehot_seqs, dim=1) # output is list of data frames for each sequence
    occurrence_matrix = np.zeros((onehot_seqs.shape[0], len(motif_names))) # sequences in rows, motifs in columns
    for i in range(onehot_seqs.shape[0]):
        i_hits_df = hits_by_seq[i]
        # skip the sequence if it has no motif matched
        if not i_hits_df.empty:
            i_motif_counts_dict = i_hits_df['motif_name'].value_counts().to_dict()
            for j_motif in i_motif_counts_dict:
                # find the index of the motif
                j = motif_names.index(j_motif)
                occurrence_matrix[i,j] = i_motif_counts_dict[j_motif]
    return occurrence_matrix