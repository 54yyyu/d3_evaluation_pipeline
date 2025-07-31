"""
Sequence similarity evaluation metrics.

This module implements sequence-level similarity metrics that assess how well
synthetic sequences match test sequences at the sequence level (percent identity,
k-mer spectrum, discriminatability).
"""

import numpy as np
import torch
import scipy.stats
import scipy.special
import scipy.spatial.distance
import h5py
from itertools import product
from tqdm import tqdm
from typing import Dict, Any, Union, List, Optional
from .base_evaluator import BaseEvaluator
from models.discriminability_classifier import (
    prepare_discriminability_data,
    save_discriminability_data,
    train_discriminability_classifier
)


class SequenceSimilarityEvaluator(BaseEvaluator):
    """Evaluator for sequence similarity metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sequence similarity evaluator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "sequence_similarity")
        
        # Set up discriminability configuration
        self.disc_config = config.get('discriminability', {})
        self._set_default_discriminability_config()
        
        # Control which analyses to run
        self.run_discriminability = config.get('evaluation', {}).get('run_discriminability_similarity', True)
        
    def _set_default_discriminability_config(self):
        """Set default configuration values for discriminability evaluation."""
        defaults = {
            'validation_split': 0.2,
            'batch_size': 128,
            'train_max_epochs': 50,
            'patience': 10,
            'lr': 0.002,
            'random_seed': 42,
            'save_training_data': True,
            'training_data_path': None
        }
        
        for key, value in defaults.items():
            if key not in self.disc_config:
                self.disc_config[key] = value
        
    def get_required_inputs(self) -> Dict[str, str]:
        """Get required inputs for sequence similarity evaluation."""
        return {
            "x_synthetic": "Generated/synthetic sequences (N, L, A) or (N, A, L)",
            "x_test": "Test/observed sequences (N, L, A) or (N, A, L)",
            "x_train": "Training sequences (optional, for percent identity analysis)"
        }
    
    def evaluate(self, 
                 x_synthetic: Union[np.ndarray, torch.Tensor],
                 x_test: Union[np.ndarray, torch.Tensor],
                 oracle_model: Any = None,
                 x_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Perform sequence similarity evaluation.
        
        Args:
            x_synthetic: Generated sequences
            x_test: Test sequences
            oracle_model: Oracle model (not used for sequence similarity)
            x_train: Training sequences (optional)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing sequence similarity results
        """
        # Validate inputs
        self.validate_inputs(x_synthetic, x_test)
        
        # Convert to numpy and standardize format
        x_synthetic_np = self._ensure_numpy(x_synthetic)
        x_test_np = self._ensure_numpy(x_test)
        
        x_synthetic_np = self._standardize_shape(x_synthetic_np)
        x_test_np = self._standardize_shape(x_test_np)
        
        if x_train is not None:
            x_train_np = self._ensure_numpy(x_train)
            x_train_np = self._standardize_shape(x_train_np)
        else:
            x_train_np = None
        
        # Calculate all sequence similarity metrics
        results = {}
        
        # 1. Percent identity analysis
        results.update(self._percent_identity_analysis(x_synthetic_np, x_test_np, x_train_np))
        
        # 2. K-mer spectrum analysis
        results.update(self._kmer_spectrum_analysis(x_synthetic_np, x_test_np))
        
        # 3. Discriminability analysis (train classifier to distinguish synthetic vs test)
        if self.run_discriminability:
            results.update(self._discriminability_analysis(x_synthetic_np, x_test_np))
        
        # 4. Sequence diversity (self-similarity)
        results.update(self._sequence_diversity_analysis(x_synthetic_np))
        
        self.update_results(results)
        return results
    
    def _percent_identity_analysis(self, 
                                 x_synthetic: np.ndarray,
                                 x_test: np.ndarray,
                                 x_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate percent identity metrics.
        
        Args:
            x_synthetic: Synthetic sequences
            x_test: Test sequences  
            x_train: Training sequences (optional)
            
        Returns:
            Dictionary with percent identity results
        """
        results = {}
        
        # Get batch size from config or use default
        batch_size = self.config.get('batch_size', 2000)
        
        # Synthetic vs synthetic (self-similarity)
        percent_identity_self = self._calculate_cross_sequence_identity_batch(
            x_synthetic, x_synthetic, batch_size=min(256, len(x_synthetic))
        )
        max_identity_self = np.max(percent_identity_self, axis=1)
        
        # Take second highest due to perfect self-match
        second_highest_self = []
        for i in range(len(percent_identity_self)):
            sorted_vals = np.sort(percent_identity_self[i])[::-1]
            second_highest_self.append(sorted_vals[1] if len(sorted_vals) > 1 else sorted_vals[0])
        
        results.update({
            "max_percent_identity_self": float(np.max(max_identity_self)),
            "mean_percent_identity_self": float(np.mean(max_identity_self)),
            "mean_second_highest_identity_self": float(np.mean(second_highest_self))
        })
        
        # Synthetic vs test
        percent_identity_test = self._calculate_cross_sequence_identity_batch(
            x_synthetic, x_test, batch_size=batch_size
        )
        max_identity_test = np.max(percent_identity_test, axis=1)
        
        results.update({
            "max_percent_identity_vs_test": float(np.max(max_identity_test)),
            "mean_percent_identity_vs_test": float(np.mean(max_identity_test))
        })
        
        # Synthetic vs training (if provided)
        if x_train is not None:
            percent_identity_train = self._calculate_cross_sequence_identity_batch(
                x_synthetic, x_train, batch_size=batch_size
            )
            max_identity_train = np.max(percent_identity_train, axis=1)
            
            results.update({
                "max_percent_identity_vs_train": float(np.max(max_identity_train)),
                "mean_percent_identity_vs_train": float(np.mean(max_identity_train))
            })
        
        return results
    
    def _kmer_spectrum_analysis(self, 
                              x_synthetic: np.ndarray,
                              x_test: np.ndarray,
                              kmer_lengths: List[int] = None) -> Dict[str, Any]:
        """
        Calculate k-mer spectrum similarity metrics.
        
        Args:
            x_synthetic: Synthetic sequences
            x_test: Test sequences
            kmer_lengths: List of k-mer lengths to analyze
            
        Returns:
            Dictionary with k-mer spectrum results
        """
        if kmer_lengths is None:
            kmer_lengths = self.config.get('kmer_lengths', [3, 4, 5])
        
        results = {}
        
        for k in kmer_lengths:
            # Calculate k-mer distributions
            dist_test = self._compute_kmer_spectra(x_test, k)
            dist_synthetic = self._compute_kmer_spectra(x_synthetic, k)
            
            # Calculate Kullback-Leibler divergence - match original order: (test, synthetic)
            kld = np.round(np.sum(scipy.special.kl_div(dist_test, dist_synthetic)), 6)
            
            # Calculate Jensen-Shannon distance  
            jsd = np.round(scipy.spatial.distance.jensenshannon(dist_test, dist_synthetic), 6)
            
            results.update({
                f"kmer_{k}_kullback_leibler_divergence": float(kld),
                f"kmer_{k}_jensen_shannon_distance": float(jsd)
            })
        
        return results
    
    def _discriminability_analysis(self, 
                                  x_synthetic: np.ndarray,
                                  x_test: np.ndarray) -> Dict[str, Any]:
        """
        Perform discriminability analysis by training a binary classifier.
        
        This analysis measures how well a neural network can distinguish between
        synthetic and test sequences. Lower AUROC indicates better generator
        performance (synthetic sequences are harder to distinguish from real ones).
        
        Args:
            x_synthetic: Synthetic sequences (N, L, A)
            x_test: Test sequences (N, L, A)
            
        Returns:
            Dictionary with discriminability metrics
        """
        print(f"Running discriminability analysis with {len(x_synthetic)} synthetic and {len(x_test)} test sequences...")
        
        # Prepare training data
        X_train, y_train, X_val, y_val = prepare_discriminability_data(
            x_synthetic, x_test,
            validation_split=self.disc_config['validation_split'],
            random_seed=self.disc_config['random_seed']
        )
        
        print(f"Training set: {len(X_train)} samples ({np.sum(y_train)} synthetic, {np.sum(y_train == 0)} test)")
        print(f"Validation set: {len(X_val)} samples ({np.sum(y_val)} synthetic, {np.sum(y_val == 0)} test)")
        
        # Save training data if requested
        training_data_saved_path = None
        if self.disc_config['save_training_data']:
            training_data_path = self.disc_config.get('training_data_path')
            if training_data_path is None:
                # Create default path
                training_data_path = "discriminability_training_data.h5"
            
            print(f"Saving discriminability training data to {training_data_path}...")
            save_discriminability_data(X_train, y_train, X_val, y_val, training_data_path)
            training_data_saved_path = training_data_path
        
        # Train discriminability classifier
        print("Training discriminability classifier...")
        trained_model, final_auroc = train_discriminability_classifier(
            X_train, y_train, X_val, y_val, self.disc_config
        )
        
        print(f"Discriminability classifier training completed. Final AUROC: {final_auroc:.4f}")
        
        # Calculate additional metrics
        results = self._calculate_discriminability_metrics(
            trained_model, X_val, y_val, final_auroc, training_data_saved_path
        )
        
        return results
    
    def _calculate_discriminability_metrics(self, 
                                          model: Any,
                                          X_val: np.ndarray, 
                                          y_val: np.ndarray,
                                          auroc: float,
                                          training_data_path: Optional[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive discriminability metrics.
        
        Args:
            model: Trained discriminability classifier
            X_val: Validation sequences
            y_val: Validation labels
            auroc: AUROC score
            training_data_path: Path where training data was saved
            
        Returns:
            Dictionary with discriminability metrics
        """
        # Get predictions
        X_val_tensor = torch.from_numpy(X_val).float()
        predictions = model.predict_proba(X_val_tensor)
        
        # Calculate various metrics
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, 
            accuracy_score, precision_score, recall_score, f1_score
        )
        
        # Binary predictions using 0.5 threshold
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, binary_predictions)
        precision = precision_score(y_val, binary_predictions, zero_division=0)
        recall = recall_score(y_val, binary_predictions, zero_division=0)
        f1 = f1_score(y_val, binary_predictions, zero_division=0)
        average_precision = average_precision_score(y_val, predictions)
        
        # Class-specific accuracy
        synthetic_mask = y_val == 1
        test_mask = y_val == 0
        
        synthetic_accuracy = accuracy_score(y_val[synthetic_mask], binary_predictions[synthetic_mask]) if np.sum(synthetic_mask) > 0 else 0
        test_accuracy = accuracy_score(y_val[test_mask], binary_predictions[test_mask]) if np.sum(test_mask) > 0 else 0
        
        # Summary statistics
        n_synthetic = np.sum(y_val == 1)
        n_test = np.sum(y_val == 0)
        
        results = {
            # Primary discriminability metric
            "discriminability_auroc": float(auroc),
            
            # Additional classification metrics
            "discriminability_accuracy": float(accuracy),
            "discriminability_precision": float(precision),
            "discriminability_recall": float(recall),
            "discriminability_f1_score": float(f1),
            "discriminability_average_precision": float(average_precision),
            
            # Class-specific metrics
            "discriminability_synthetic_accuracy": float(synthetic_accuracy),
            "discriminability_test_accuracy": float(test_accuracy),
            
            # Data summary
            "discriminability_n_synthetic_val": int(n_synthetic),
            "discriminability_n_test_val": int(n_test),
            "discriminability_validation_split": float(self.disc_config['validation_split']),
            
            # Training info
            "discriminability_training_epochs": int(self.disc_config['train_max_epochs']),
            "discriminability_batch_size": int(self.disc_config['batch_size']),
            "discriminability_learning_rate": float(self.disc_config['lr']),
        }
        
        # Add training data path if saved
        if training_data_path:
            results["discriminability_training_data_path"] = str(training_data_path)
        
        return results
    
    def _sequence_diversity_analysis(self, x_synthetic: np.ndarray) -> Dict[str, Any]:
        """
        Analyze sequence diversity within synthetic sequences.
        
        Args:
            x_synthetic: Synthetic sequences
            
        Returns:
            Dictionary with diversity metrics
        """
        batch_size = min(256, len(x_synthetic))
        
        # Calculate pairwise similarities
        similarities = self._calculate_cross_sequence_identity_batch(
            x_synthetic, x_synthetic, batch_size=batch_size
        )
        
        # For each sequence, find its highest similarity to others (excluding self)
        diversity_scores = []
        for i in range(len(similarities)):
            # Sort similarities and take second highest (first is self-match)
            sorted_sims = np.sort(similarities[i])[::-1]
            if len(sorted_sims) > 1:
                diversity_scores.append(sorted_sims[1])
            else:
                diversity_scores.append(sorted_sims[0])
        
        diversity_scores = np.array(diversity_scores)
        
        return {
            "sequence_diversity_max_similarity": float(np.max(diversity_scores)),
            "sequence_diversity_mean_similarity": float(np.mean(diversity_scores)),
            "sequence_diversity_std_similarity": float(np.std(diversity_scores))
        }
    
    def _calculate_cross_sequence_identity_batch(self, 
                                               X_train: np.ndarray, 
                                               X_test: np.ndarray, 
                                               batch_size: int = 1000) -> np.ndarray:
        """
        Calculate cross-sequence identity in batches.
        
        Args:
            X_train: First set of sequences (N1, L, A)
            X_test: Second set of sequences (N2, L, A)  
            batch_size: Batch size for processing
            
        Returns:
            Identity matrix (N1, N2)
        """
        num_train, seq_length, alphabet_size = X_train.shape
        num_test = X_test.shape[0]
        
        # Reshape for dot product computation
        X_train_flat = X_train.reshape(num_train, -1)
        X_test_flat = X_test.reshape(num_test, -1)
        
        # Initialize result matrix - match original data type
        seq_identity = np.zeros((num_train, num_test), dtype=np.int8)
        
        # Process in batches
        for start_idx in tqdm(range(0, num_train, batch_size), desc="Computing sequence identity"):
            end_idx = min(start_idx + batch_size, num_train)
            
            # Compute dot product for this batch
            batch_result = np.dot(X_train_flat[start_idx:end_idx], X_test_flat.T)
            
            # Store result - cast to int8 to match original
            seq_identity[start_idx:end_idx, :] = batch_result.astype(np.int8)
        
        return seq_identity
    
    def _compute_kmer_spectra(self, 
                             X: np.ndarray, 
                             kmer_length: int = 3) -> np.ndarray:
        """
        Compute k-mer spectra for sequences.
        
        Args:
            X: Input sequences (N, L, A)
            kmer_length: Length of k-mers
            
        Returns:
            Normalized k-mer distribution
        """
        # Convert one-hot to sequence strings
        dna_dict = {0: "A", 1: "C", 2: "G", 3: "T"}
        seq_list = []
        
        for seq in tqdm(X, desc=f"Converting to {kmer_length}-mers"):
            seq_str = "".join([dna_dict[np.where(pos)[0][0]] for pos in seq])
            seq_list.append(seq_str)
        
        # Initialize k-mer featurizer
        kmer_obj = KmerFeaturization(kmer_length)
        kmer_features = kmer_obj.obtain_kmer_feature_for_a_list_of_sequences(
            seq_list, write_number_of_occurrences=True
        )
        
        # Calculate global k-mer counts
        global_counts = np.sum(np.array(kmer_features), axis=0)
        
        # Normalize to get distribution
        global_counts_normalized = global_counts / np.sum(global_counts)
        
        return global_counts_normalized


class KmerFeaturization:
    """Class for k-mer featurization of DNA sequences."""
    
    def __init__(self, k: int):
        """
        Initialize k-mer featurizer.
        
        Args:
            k: Length of k-mers
        """
        self.k = k
        self.letters = ['A', 'C', 'G', 'T']
        self.multiplyBy = 4 ** np.arange(k-1, -1, -1)
        self.n = 4**k
    
    def obtain_kmer_feature_for_a_list_of_sequences(self, 
                                                   seqs: List[str], 
                                                   write_number_of_occurrences: bool = False) -> List[np.ndarray]:
        """
        Get k-mer features for a list of sequences.
        
        Args:
            seqs: List of DNA sequence strings
            write_number_of_occurrences: If True, count occurrences; if False, use frequencies
            
        Returns:
            List of k-mer feature vectors
        """
        kmer_features = []
        for seq in seqs:
            feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences)
            kmer_features.append(feature)
        return kmer_features
    
    def obtain_kmer_feature_for_one_sequence(self, 
                                           seq: str, 
                                           write_number_of_occurrences: bool = False) -> np.ndarray:
        """
        Get k-mer feature vector for one sequence.
        
        Args:
            seq: DNA sequence string
            write_number_of_occurrences: If True, count occurrences; if False, use frequencies
            
        Returns:
            K-mer feature vector
        """
        number_of_kmers = len(seq) - self.k + 1
        kmer_feature = np.zeros(self.n)
        
        for i in range(number_of_kmers):
            this_kmer = seq[i:(i+self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1
        
        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers
        
        return kmer_feature
    
    def kmer_numbering_for_one_kmer(self, kmer: str) -> int:
        """
        Get numerical index for a k-mer.
        
        Args:
            kmer: K-mer string
            
        Returns:
            Numerical index
        """
        digits = []
        for letter in kmer:
            digits.append(self.letters.index(letter))
        
        digits = np.array(digits)
        numbering = (digits * self.multiplyBy).sum()
        
        return numbering


# Convenience functions

def calculate_percent_identity(x1: np.ndarray, 
                             x2: np.ndarray, 
                             batch_size: int = 1000) -> np.ndarray:
    """
    Calculate percent identity between two sets of sequences.
    
    Args:
        x1: First set of sequences (N1, L, A)
        x2: Second set of sequences (N2, L, A)
        batch_size: Batch size for computation
        
    Returns:
        Identity matrix (N1, N2)
    """
    evaluator = SequenceSimilarityEvaluator({})
    return evaluator._calculate_cross_sequence_identity_batch(x1, x2, batch_size)


def kmer_statistics(kmer_length: int, 
                   data1: np.ndarray, 
                   data2: np.ndarray) -> tuple:
    """
    Calculate k-mer statistics between two datasets.
    
    Args:
        kmer_length: Length of k-mers
        data1: First dataset (test data)
        data2: Second dataset (synthetic data)
        
    Returns:
        Tuple of (KLD, JSD)
    """
    evaluator = SequenceSimilarityEvaluator({})
    
    dist1 = evaluator._compute_kmer_spectra(data1, kmer_length)
    dist2 = evaluator._compute_kmer_spectra(data2, kmer_length)
    
    # Match original order and rounding
    kld = np.round(np.sum(scipy.special.kl_div(dist1, dist2)), 6)
    jsd = np.round(scipy.spatial.distance.jensenshannon(dist1, dist2), 6)
    
    return kld, jsd