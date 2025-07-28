import numpy as np
from numpy import load
import h5py
import pandas as pd
import torch
import lightning as L
from pytorch_lightning import LightningModule
from scipy import linalg
import scipy
from tqdm import tqdm
from itertools import product
from sklearn.metrics import mean_squared_error
import gc
from Bio import motifs
import os
# Try to import memelite first, fallback to pymemesuite
try:
    # Try relative import first
    from .seq_evals_func_motifs import motif_count_memelite, make_occurrence_matrix_memelite
    USE_MEMELITE = True
except (ImportError, ValueError):
    try:
        # Try absolute import as fallback
        from utils.seq_evals_func_motifs import motif_count_memelite, make_occurrence_matrix_memelite
        USE_MEMELITE = True
    except ImportError:
        try:
            # Try direct import from same directory
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
import glob
import tempfile
import Bio
from captum.attr import GradientShap
import matplotlib.pyplot as plt
import logomaker


#############################################################################################
# Functional similarity: Conditional generation fidelity
#############################################################################################
'''
prerequisite: 
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A)) 
    - oracle (inference model that maps x to activities)

example:
    activity1 = oracle.predict(x_synthetic)
    activity2 = oracle.predict(x_test)
    mse = conditional_generation_fidelity(activity1, activity2)
'''

def conditional_generation_fidelity(activity1, activity2):
    return np.mean((activity1 - activity2)**2)

#############################################################################################
# Functional similarity: Frechet distance
#############################################################################################
'''
prerequisite: 
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A)) 
    - oracle_embedding_fun (function that acquires the penultimate embeddings)

example:
    embeddings1 = oracle_embedding_fun(x_synthetic)
    embeddings2 = oracle_embedding_fun(x_test)
    mu1, sigma1 = calculate_activation_statistics(embeddings1)
    mu2, sigma2 = calculate_activation_statistics(embeddings2)
    distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
'''

def calculate_activation_statistics(embeddings):
    embeddings_d = embeddings.detach().cpu().numpy()
    mu = np.mean(embeddings_d, axis=0)
    sigma = np.cov(embeddings_d, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    #Frechet distance: d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

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

#############################################################################################
# Functional similarity: Predictive distribution shift
#############################################################################################
'''
prerequisite:
    - x_synthetic_tensor (generated seqeunces as a tensor with shapes (N,L,A)) 
    - x_test_tensor (observed sequences as a tensor with shapes (N,L,A))

example:
    activity1 = oracle.predict(x_synthetic)
    activity2 = oracle.predict(x_test)
    mse = conditional_generation_fidelity(activity1, activity2)
'''

def predictive_distribution_shift(x_synthetic_tensor, x_test_tensor):
    
    #encode bases using 0,1,2,3 (eliminate a dimension)
    base_indices_test = np.argmax(x_test_tensor.detach().cpu().numpy(), axis=1)
    base_indices_syn = np.argmax(x_synthetic_tensor.detach().cpu().numpy(), axis=1)

    #flatten the arrays (now they are one dimension)
    base_indices_test_f = base_indices_test.flatten()
    base_indices_syn_f = base_indices_syn.flatten()

    #return ks test statistic
    return scipy.stats.ks_2samp(base_indices_syn_f, base_indices_test_f).statistic

#############################################################################################
# Sequence similarity: Percent identity
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))

example:
    percent_identity = calculate_cross_sequence_identity_batch(x_synthetic, x_test, batch_size)
    max_percent_identity = np.max(percent_identity, axis=1)
    global_max_percent_identity = np.max(max_percent_identity)
'''

def calculate_cross_sequence_identity_batch(X_train, X_test, batch_size):
    num_train, seq_length, alphabet_size = X_train.shape    
    num_test = X_test.shape[0]
    
    # Reshape the matrices for dot product computation
    X_train = np.reshape(X_train, [-1, seq_length * alphabet_size])
    X_test = np.reshape(X_test, [-1, seq_length * alphabet_size])
    
    # Initialize the matrix to store the results
    seq_identity = np.zeros((num_train, num_test)).astype(np.int8)
    
    # Process the training data in batches
    total_batches = (num_train + batch_size - 1) // batch_size
    for start_idx in tqdm(range(0, num_train, batch_size), desc="Computing sequence identity", total=total_batches):
        end_idx = min(start_idx + batch_size, num_train)
        
        # Compute the dot product for this batch
        batch_result = np.dot(X_train[start_idx:end_idx], X_test.T) 
        
        # Store the result in the corresponding slice of the output matrix
        seq_identity[start_idx:end_idx, :] = batch_result.astype(np.int8)
    
    return seq_identity

#############################################################################################
# Sequence similarity: k-mer spectrum shift
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))

example:
    kld, jsd = kmer_statistics(kmer_len, data1, data2)
'''

def kmer_statistics(kmer_length, data1, data2):

    #generate kmer distributions 
    dist1 = compute_kmer_spectra(data1, kmer_length)
    dist2 = compute_kmer_spectra(data2, kmer_length)

    #computer KLD
    kld = np.round(np.sum(scipy.special.kl_div(dist1, dist2)), 6)

    #computer jensen-shannon 
    jsd = np.round(np.sum(scipy.spatial.distance.jensenshannon(dist1, dist2)), 6)

    return kld, jsd

def compute_kmer_spectra(
    X,
    kmer_length=3,
    dna_dict = {
        0: "A",
        1: "C",
        2: "G",
        3: "T"
      }
    ):
    # convert one hot to A,C,G,T
    seq_list = []

    for index in tqdm(range(len(X)), desc="Converting sequences to text"): #for loop is what actually converts a list of one-hot encoded sequences into ACGT

        seq = X[index]

        seq_list += ["".join([dna_dict[np.where(i)[0][0]] for i in seq])]

    obj = kmer_featurization(kmer_length)  # initialize a kmer_featurization object
    kmer_features = obj.obtain_kmer_feature_for_a_list_of_sequences(seq_list, write_number_of_occurrences=True)

    kmer_permutations = ["".join(p) for p in product(["A", "C", "G", "T"], repeat=kmer_length)] #list of all kmer permutations, length specified by repeat=

    kmer_dict = {}
    for kmer in kmer_permutations:
        n = obj.kmer_numbering_for_one_kmer(kmer)
        kmer_dict[n] = kmer

    global_counts = np.sum(np.array(kmer_features), axis=0)

    # what to compute entropy against
    global_counts_normalized = global_counts / sum(global_counts) # this is the distribution of kmers in the testset
    # print(global_counts_normalized)
    return global_counts_normalized

class kmer_featurization:

    def __init__(self, k):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = ['A', 'C', 'G', 'T']
        self.multiplyBy = 4 ** np.arange(k-1, -1, -1) # the multiplying number for each digit position in the k-number system
        self.n = 4**k # number of possible k-mers

    def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
        """
        Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.
        Args:
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        kmer_features = [] #a list containing the one-hot representation of kmers for each sequence in the list of sequences given
        for seq in seqs: #first obtain the one-hot representation of the kmers in a sequence
            this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences=write_number_of_occurrences)
            kmer_features.append(this_kmer_feature) #append this one-hot list into another list

        kmer_features = np.array(kmer_features)

        return kmer_features

    def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False): #
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.
        Args:
          seq:
            a string, a DNA sequence
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(seq) - self.k + 1

        kmer_feature = np.zeros(self.n) #array of zeroes the same length of all possible kmers

        for i in range(number_of_kmers): #for each kmer feature, turn the corresponding index in the list of all kmer features to 1
            this_kmer = seq[i:(i+self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1

        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers

        return kmer_feature

    def kmer_numbering_for_one_kmer(self, kmer): #returns the corresponding index of a kmer in the larger list of all possible kmers?
        """
        Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
        """
        digits = []
        for letter in kmer:
            digits.append(self.letters.index(letter))

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering

#############################################################################################
# Sequence similarity: Discriminatability
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_train (observed training sequences with shapes (N,L,A))
    - classification model 

example:
    data_dict = prep_data_for_classification(x_train, x_synthetic)
    write_to_h5('Discriminatability.h5', data_dict)

    run train_deepstarr_ood_score.py on elzar
'''

def prep_data_for_classification(x_test_tensor, x_synthetic_tensor):
    x_train = np.vstack([x_test_tensor.detach().cpu().numpy(), x_synthetic_tensor.detach().cpu().numpy()])
    y_train = np.vstack([np.ones((x_test_tensor.shape[0],1)), np.zeros((x_synthetic_tensor.shape[0],1))])
    x_train = np.transpose(x_train, (0, 2, 1)) 

    #write x_train and y_train into dict to create .h5 file
    data_dict = {
        'x_train': x_train,
        'y_train': y_train,
    }

    return data_dict

#############################################################################################
# Compositional similarity: Motif enrichment
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))
    - JASPAR_file (Datasbase for motif search)

example:
	x_synthetic = one_hot_to_seq(x_synthetic)
	x_test = one_hot_to_seq(x_test)
	create_fasta_file(x_synthetic,'sythetic_seq.txt')
	create_fasta_file(x_test,'test_seq.txt')
	motif_count = motif_count('test_seq.txt', JASPAR_file)
	motif_count_2 = motif_count('synthetic_seq.txt', JASPAR_file)
	pr = enrich_pr(motif_count,motif_count_2)
'''

def sequences_to_onehot(sequences):
    '''
    Convert DNA sequences to one-hot encoding for memelite
    '''
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

def motif_count(path, path_to_database):
    '''
    path is the filepath to the list of sequences in fasta format

    returns a dictionary containing the motif counts for all the sequences
    '''
    
    if USE_MEMELITE:
        try:
            # Read sequences and convert to one-hot for memelite
            sequences = []
            for record in Bio.SeqIO.parse(path, "fasta"):
                sequences.append(str(record.seq))
            
            onehot_seqs = sequences_to_onehot(sequences)
            return motif_count_memelite(path_to_database, onehot_seqs)
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
        
    with MotifFile("JASPAR2024_CORE_non-redundant_pfms_meme.txt") as motif_file:
        motifs_list = list(motif_file)
        motif_file.seek(0)  # Reset to beginning for actual processing
        for motif in tqdm(motifs_list, desc="Scanning motifs"):
            pattern = fimo.score_motif(motif, sequences, motif_file.background)
            motif_ids.append(motif.accession.decode())
            occurrence.append(len(pattern.matched_elements))
    
    motif_counts = dict(zip(motif_ids,occurrence))

    return motif_counts

def enrich_pr(count_1,count_2):
	c_1 = list(count_1.values())
	c_2 = list(count_2.values())

	return scipy.stats.pearsonr(c_1,c_2)

#############################################################################################
# Compositional similarity: Motif co-occurrence
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))
    - JASPAR_file (Datasbase for motif search)

example:

	motif_matrix_test = make_occurrence_matrix('test_seq.txt')
    motif_matrix_synthetic = make_occurrence_matrix('synthetic_seq.txt')
    C = np.cov(motif_matrix_test)
    C2 = np.cov(motif_matrix_synthetic)
    distance = frobenius_norm(C, C2)
'''

def make_occurrence_matrix(path):
    '''
    path is the filepath to the list of sequences in fasta format

    returns a matrix containing the motif counts for all the sequences
    '''
    
    if USE_MEMELITE:
        try:
            # Read sequences and convert to one-hot for memelite
            sequences = []
            for record in Bio.SeqIO.parse(path, "fasta"):
                sequences.append(str(record.seq))
            
            onehot_seqs = sequences_to_onehot(sequences)
            return make_occurrence_matrix_memelite("JASPAR2024_CORE_non-redundant_pfms_meme.txt", onehot_seqs)
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
        with MotifFile("JASPAR2024_CORE_non-redundant_pfms_meme.txt") as motif_file:
            motifs_list = list(motif_file)
            motif_file.seek(0)  # Reset to beginning for actual processing
            for motif in motifs_list:
                pattern = fimo.score_motif(motif, sequence, motif_file.background)
                motif_ids.append(motif.accession.decode())
                occurrence.append(len(pattern.matched_elements))
        occurrence_matrix.append(occurrence)

    return occurrence_matrix


def covariance_matrix(x):
    return np.cov(x)


def frobenius_norm(cov, cov2):
    return np.sqrt(np.sum((cov - cov2)**2))




#############################################################################################
# Compositional similarity: Attribution analysis
#############################################################################################
'''
prerequisite:
    - x_seq (generated or observed seqeunces with shapes (N,L,A)) 
    - oracle model 

example:
    shap_score = gradient_shap(x_seq, oracle, task_idx)
    plot_attribution_map(x_seq, shap_score)
'''

def gradient_shap(x_seq, model, class_index=0, trim_end=None):
    # Detect device of the model
    device = next(model.parameters()).device
    
    x_seq = np.swapaxes(x_seq,1,2)
    N,A,L = x_seq.shape
    score_cache = []
    for i,x in tqdm(enumerate(x_seq), desc="Computing SHAP scores", total=len(x_seq)):
        # process sequences so that they are right shape (based on insertions)
        x = np.expand_dims(x, axis=0)
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32).to(device)
        #x_tensor = model._pad_end(x_tensor)
        x = x_tensor.cpu().detach().numpy()
        # random background
        num_background = 1000
        null_index = np.random.randint(0,3, size=(num_background,L))
        x_null = np.zeros((num_background,A,L))
        for n in range(num_background):
            for l in range(L):
                x_null[n,null_index[n,l],l] = 1.0
        x_null_tensor = torch.tensor(x_null, requires_grad=True, dtype=torch.float32).to(device)
        #x_null_tensor = model._pad_end(x_null_tensor)
        # calculate gradient shap
        gradient_shap = GradientShap(model)
        grad = gradient_shap.attribute(x_tensor,
                                      n_samples=100,
                                      stdevs=0.1,
                                      baselines=x_null_tensor,
                                      target=class_index)
        grad = grad.data.cpu().numpy()
        # process gradients with gradient correction (Majdandzic et al. 2022)
        grad -= np.mean(grad, axis=1, keepdims=True)
        score_cache.append(np.squeeze(grad))
    score_cache = np.array(score_cache)
    if len(score_cache.shape)<3:
        score_cache=np.expand_dims(score_cache,axis=0)
    if trim_end:
        score_cache = score_cache[:,:,:-trim_end]
    return np.swapaxes(score_cache,1,2)

def plot_attribution_map(x_seq, shap_score, alphabet='ACGT', figsize=(20,1)):
    num_plot = len(x_seq)
    fig = plt.figure(figsize=(20,2*num_plot))
    i = 0
    i_subplot = 0
    for (x,grad) in zip(x_seq,shap_score):
        
        x_index = np.argmax(np.squeeze(x), axis=1)
        grad = np.squeeze(grad)
        L, A = grad.shape

        seq = ''
        saliency = np.zeros((L))
        for i in range(L):
            seq += alphabet[x_index[i]]
            saliency[i] = grad[i,x_index[i]]
        # create saliency matrix
        saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)

        ax = plt.subplot(num_plot,1,i_subplot+1)
        i_subplot+=1
        logomaker.Logo(saliency_df, figsize=figsize, ax=ax)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])


#############################################################################################
# Compositional similarity: Attribution consistency
#############################################################################################

'''
prerequisite:
    - x_seq (generated or observed seqeunces with shapes (N,L,A)) 
    - oracle model 

example:
    shap_score = gradient_shap(x_seq, oracle, task_idx)
    attributino_map = process_attribution_map(shap_score)
    mask = unit_mask(x_seq)
    phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad([attribution_map], x_seq, mask, radius_count_cutoff)
    LIM, box_length, box_volume, n_bins, n_bins_half = initialize_integration_2(0.1)
	entropic_information = calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, 0.1, box_volume, prior_range=3)
'''

def process_attribution_map(saliency_map_raw, k=6):
    saliency_map_raw = saliency_map_raw - np.mean(saliency_map_raw, axis=-1, keepdims=True) # gradient correction
    saliency_map_raw = saliency_map_raw / np.sum(np.sqrt(np.sum(np.square(saliency_map_raw), axis=-1, keepdims=True)), axis=-2, keepdims=True) #normaliz
    saliency_special_map_k = saliency_map_raw.copy()
    for i in range(k-1): 
        saliency_map_raw_rolled_i = np.roll(saliency_map_raw, -i-1, axis=-2)
        saliency_special_map_k += saliency_map_raw_rolled_i
    saliency_special = orthonormal_coordinates(saliency_special_map_k)
    return saliency_special



def orthonormal_coordinates(attr_map):
    """reduce 4d array to 3d"""

    attr_map_on = np.zeros((attr_map.shape[0], attr_map.shape[1], 3))

    x = attr_map[:, :, 0]
    y = attr_map[:, :, 1]
    z = attr_map[:, :, 2]
    w = attr_map[:, :, 3]

    # Now convert to new coordinates
    e1 = 1 / np.sqrt(2) * (-x + y)
    e2 = np.sqrt(2 / 3) * (-1/2*x -1/2*y)
    e3 = np.sqrt(3 / 4) * (-1/3*x -1/3*y -1/3*z + w)
    attr_map_on[:, :, 0] = e1
    attr_map_on[:, :, 1] = e2
    attr_map_on[:, :, 2] = e3

    return attr_map_on


def unit_mask(x_seq):
    return np.sum(np.ones(x_seq.shape),axis=-1) / 4

def spherical_coordinates_process_2_trad(saliency_map_raw_s, X, mask, radius_count_cutoff=0.04):
    global N_EXP
    N_EXP = len(saliency_map_raw_s)
    radius_count=int(radius_count_cutoff * np.prod(X.shape)/4)
    cutoff=[]
    x_s, y_s, z_s, r_s, phi_1_s, phi_2_s = [], [], [], [], [], []
    PI = 3.1416
    for s in range (0, N_EXP):
        saliency_map_raw = saliency_map_raw_s[s]
        xxx_motif=saliency_map_raw[:,:,0]
        yyy_motif=(saliency_map_raw[:,:,1])
        zzz_motif=(saliency_map_raw[:,:,2])
        xxx_motif_pattern=saliency_map_raw[:,:,0]*mask
        yyy_motif_pattern=(saliency_map_raw[:,:,1])*mask
        zzz_motif_pattern=(saliency_map_raw[:,:,2])*mask
        r=np.sqrt(xxx_motif*xxx_motif+yyy_motif*yyy_motif+zzz_motif*zzz_motif)
        resh = X.shape[0] * X.shape[1]
        x=np.array(xxx_motif_pattern.reshape(resh,))
        y=np.array(yyy_motif_pattern.reshape(resh,))
        z=np.array(zzz_motif_pattern.reshape(resh,))
        r=np.array(r.reshape(resh,))
        #Take care of any NANs.
        x=np.nan_to_num(x)
        y=np.nan_to_num(y)
        z=np.nan_to_num(z)
        r=np.nan_to_num(r)
        cutoff.append( np.sort(r)[-radius_count] )
        R_cuttof_index = np.sqrt(x*x+y*y+z*z) > cutoff[s]
        #Cut off
        x=x[R_cuttof_index]
        y=y[R_cuttof_index]
        z=z[R_cuttof_index]
        r=np.array(r[R_cuttof_index])
        x_s.append(x)
        y_s.append(y)
        z_s.append(z)
        r_s.append(r)
        #rotate axis
        x__ = np.array(y)
        y__ = np.array(z)
        z__ = np.array(x)
        x = x__
        y = y__
        z = z__
        #"phi"
        phi_1 = np.arctan(y/x) #default
        phi_1 = np.where((x<0) & (y>=0), np.arctan(y/x) + PI, phi_1)   #overwrite
        phi_1 = np.where((x<0) & (y<0), np.arctan(y/x) - PI, phi_1)   #overwrite
        phi_1 = np.where (x==0, PI/2, phi_1) #overwrite
        #Renormalize temorarily to have both angles in [0,PI]:
        phi_1 = phi_1/2 + PI/2
        #"theta"
        phi_2=np.arccos(z/r)
        #back to list
        phi_1 = list(phi_1)
        phi_2 = list(phi_2)
        phi_1_s.append(phi_1)
        phi_2_s.append(phi_2)
    #print(cutoff)
    return phi_1_s, phi_2_s, r_s

def initialize_integration_2(box_length):
    LIM = 3.1416
    global volume_border_correction
    box_volume = box_length*box_length
    n_bins = int(LIM/box_length)
    volume_border_correction =  (LIM/box_length/n_bins)*(LIM/box_length/n_bins)
    #print('volume_border_correction = ', volume_border_correction)
    n_bins_half = int(n_bins/2)
    return LIM, box_length, box_volume, n_bins, n_bins_half




def calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, box_length, box_volume, prior_range):
    global Empirical_box_pdf_s
    global Empirical_box_count_s
    global Empirical_box_count_plain_s
    Empirical_box_pdf_s=[]
    Empirical_box_count_s = []
    Empirical_box_count_plain_s = []
    prior_correction_s = []
    Spherical_box_prior_pdf_s=[]
    for s in range (0,N_EXP):
        #print(s)
        Empirical_box_pdf_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[0])
        Empirical_box_count_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[1])
        Empirical_box_count_plain_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[2])
    Entropic_information = []
    for s in range (0,N_EXP):
        Entropic_information.append ( KL_divergence_2 (Empirical_box_pdf_s[s], Empirical_box_count_s[s], Empirical_box_count_plain_s[s], n_bins, box_volume, prior_range)  )
    return list(Entropic_information)



def KL_divergence_2(Empirical_box_pdf, Empirical_box_count, Empirical_box_count_plain, n_bins, box_volume, prior_range):  #, correction2)
    # p= empirical distribution, q=prior spherical distribution
    # Notice that the prior distribution is never 0! So it is safe to divide by q.
    # L'Hospital rule provides that p*log(p) --> 0 when p->0. When we encounter p=0, we would just set the contribution of that term to 0, i.e. ignore it in the sum.
    Relative_entropy = 0
    PI = 3.1416
    for i in range (1, n_bins-1):
        for j in range(1,n_bins-1):
            if (Empirical_box_pdf[i,j] > 0  ):
                phi_1 = i/n_bins*PI
                phi_2 = j/n_bins*PI
                correction3 = 0
                prior_counter = 0
                prior=0
                for ii in range(-prior_range,prior_range):
                    for jj in range(-prior_range,prior_range):
                        if(i+ii>0 and i+ii<n_bins and j+jj>0 and j+jj<n_bins):
                            prior+=Empirical_box_pdf[i+ii,j+jj]
                            prior_counter+=1
                prior=prior/prior_counter
                if(prior>0) : KL_divergence_contribution = Empirical_box_pdf[i,j] * np.log (Empirical_box_pdf[i,j]  /  prior )
                if(np.sin(phi_1)>0 and prior>0 ): Relative_entropy+=KL_divergence_contribution  #and Empirical_box_count_plain[i,j]>1
    Relative_entropy = Relative_entropy * box_volume #(volume differential in the "integral")
    return np.round(Relative_entropy,3)


def Empiciral_box_pdf_func_2 (phi_1, phi_2, r_s, n_bins, box_length, box_volume):
    N_points = len(phi_1) #Number of points
    Empirical_box_count = np.zeros((n_bins, n_bins))
    Empirical_box_count_plain = np.zeros((n_bins, n_bins))
    #Now populate the box. Go over every single point.
    for i in range (0, N_points):
        # k, l are box numbers of the (phi_1, phi_2) point
        k=np.minimum(int(phi_1[i]/box_length), n_bins-1)
        l=np.minimum(int(phi_2[i]/box_length), n_bins-1)
        #Increment count in (k,l,m) box:
        Empirical_box_count[k,l]+=1*r_s[i]*r_s[i]
        Empirical_box_count_plain[k,l]+=1
    #To get the probability distribution, divide the Empirical_box_count by the total number of points.
    Empirical_box_pdf = Empirical_box_count / N_points / box_volume
    #Check that it integrates to around 1:
    #print('Integral of the empirical_box_pdf (before first renormalization) = ' , np.sum(Empirical_box_pdf*box_volume), '(should be 1.0 if OK) \n')
    correction = 1 / np.sum(Empirical_box_pdf*box_volume)
    #Another, optional correction 
    count_empty_boxes = 0
    count_single_points = 0
    for k in range (1, n_bins-1):
        for l in range(1,n_bins-1):
            if(Empirical_box_count[k,l] ==1):
                count_empty_boxes+=1
                count_single_points+=1
    return Empirical_box_pdf * correction * 1 , Empirical_box_count *correction , Empirical_box_count_plain #, correction2


#############################################################################################
# Informativeness: Added information
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - y_synthetic (activities for sequences)
    - x_train (observed sequences with shapes (N,L,A))
    - y_train (activities for sequences)
    - x_valid (observed sequences with shapes (N,L,A))
    - y_valid (activities for sequences)
    - x_test (observed sequences with shapes (N,L,A))
    - y_test (activities for sequences)

example:
    downsample = 0.25
    N = x_train.shape[0]
    num_downsample = int(N*downsample)
    x_train = np.vstack([x_train[:num_downsample], x_synthetic])
    y_train = np.vstack([y_train[:num_downsample], y_synthetic])

    # train model using x_train, y_train

    # evaluate model on x_test, y_test
'''


#############################################################################################
# Conditional sampling diversity: sequence diversity
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 

example:
    max_percent_identity = sequence diversity(x_synthetic)
'''

def sequence_diversity(x, batch_size):
    percent_identity = calculate_cross_sequence_identity_batch(x, x, batch_size)
    val = []
    for i in range(len(percent_identity)):
        sort = np.sort(percent_identity[i])[::-1] 
        val.append(sort[1]) # <-- take second highest percent identity due to match w/ self
    return np.array(val)



#############################################################################################
# Conditional sampling diversity: mechanistic diversity
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - y_test (observed functional activity used for conditional generation)
    - oracle model

example:
    shap_scores = gradient_shap(x_seq, oracle, task_idx)
    max_self_similarity = mechanistic_diversity(shap_scores)
    
'''

def mechanistic_diversity(attr_scores):
    N, L, A = attr_scores.shape    
    
    # Reshape the matrices for dot product computation
    attr_scores = np.reshape(attr_scores, [-1, L * A])
    
    # Initialize the matrix to store the results
    max_similarity = []    
    # Process the training data in batches
    for i in range(N):
        val = np.dot(np.expand_dims(attr_scores[i,:], axis=0), attr_scores.T) 
        max_similarity.append(np.sort(val)[::-1][1])
    
    return np.array(max_similarity) 