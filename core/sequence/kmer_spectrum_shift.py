import numpy as np
import torch
from datetime import datetime
import pickle
import scipy.special
import scipy.spatial.distance
from tqdm import tqdm
from itertools import product

def kmer_statistics(kmer_length, data1, data2):
    """Compute KL divergence and Jensen-Shannon distance between k-mer distributions."""
    # generate kmer distributions 
    dist1 = compute_kmer_spectra(data1, kmer_length)
    dist2 = compute_kmer_spectra(data2, kmer_length)

    # computer KLD
    kld = np.round(np.sum(scipy.special.kl_div(dist1, dist2)), 6)

    # computer jensen-shannon 
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
    """Convert one-hot sequences to k-mer frequency distributions."""
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
    """Class for k-mer featurization of DNA sequences."""
    
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

def run_kmer_spectrum_shift_analysis(x_test_tensor, x_synthetic_tensor, kmer_length=6, output_dir=".", sample_name=None):
    """
    Run k-mer spectrum shift analysis.
    
    Compares k-mer frequency distributions between generated and real sequences
    using Jensen-Shannon divergence.
    
    Args:
        x_test_tensor: Test sequences tensor
        x_synthetic_tensor: Synthetic sequences tensor
        kmer_length: Length of k-mers to analyze (default: 6)
        output_dir: Directory to save results
        sample_name: Name of sample for batch processing (optional)
        
    Returns:
        dict: Results dictionary with k-mer statistics
    """
    from utils.helpers import put_deepstarr_into_NLA
    
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print(f"Computing k-mer spectrum statistics (k={kmer_length})...")
    X_test, X_syn = put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor)
    Kullback_Leibler_divergence = kmer_statistics(kmer_length, X_test, X_syn)[0]
    Jensen_Shannon_distance = kmer_statistics(kmer_length, X_test, X_syn)[1]
    
    results = {
        'kmer_spectra_kullback_leibler_divergence': Kullback_Leibler_divergence,
        'kmer_spectra_jensen_shannon_distance': Jensen_Shannon_distance,
        'js_distance': Jensen_Shannon_distance,  # Alias for concise metrics
        'kmer_length': kmer_length
    }
    
    # Handle batch vs single mode
    if sample_name is not None:
        # Batch mode - use new format
        from utils.batch_helpers import write_concise_csv, write_full_h5, get_concise_metrics
        
        # Write concise metrics
        concise_metrics = get_concise_metrics('kmer_spectrum_shift', results)
        write_concise_csv(output_dir, 'kmer_spectrum_shift', sample_name, concise_metrics)
        
        # Write full results
        write_full_h5(output_dir, 'kmer_spectrum_shift', sample_name, results)
        
        print(f"k-mer spectrum shift results saved for sample '{sample_name}'")
    else:
        # Single mode - keep original format
        filename = f'{output_dir}/kmer_spectrum_shift_{current_date}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"k-mer spectrum shift results saved to '{filename}'")
    
    return results