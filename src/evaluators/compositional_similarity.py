"""
Compositional similarity evaluation metrics.

This module implements compositional similarity metrics that assess how well
synthetic sequences match test sequences in terms of motif content, co-occurrence
patterns, and attribution consistency.
"""

import numpy as np
import torch
import scipy.stats
from typing import Dict, Any, Union, List, Optional, Tuple
import tempfile
import os
from pathlib import Path
from .base_evaluator import BaseEvaluator
from src.data.data_utils import one_hot_to_sequences, create_fasta_file
from src.models.model_utils import ModelWrapper


class CompositionalSimilarityEvaluator(BaseEvaluator):
    """Evaluator for compositional similarity metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize compositional similarity evaluator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "compositional_similarity")
        
    def get_required_inputs(self) -> Dict[str, str]:
        """Get required inputs for compositional similarity evaluation."""
        return {
            "x_synthetic": "Generated/synthetic sequences (N, L, A) or (N, A, L)",
            "x_test": "Test/observed sequences (N, L, A) or (N, A, L)",
            "oracle_model": "Trained oracle model (for attribution analysis)",
            "motif_database_path": "Path to motif database file (e.g., JASPAR MEME format)"
        }
    
    def evaluate(self, 
                 x_synthetic: Union[np.ndarray, torch.Tensor],
                 x_test: Union[np.ndarray, torch.Tensor],
                 oracle_model: Any,
                 motif_database_path: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Perform compositional similarity evaluation.
        
        Args:
            x_synthetic: Generated sequences
            x_test: Test sequences
            oracle_model: Oracle model for attribution analysis
            motif_database_path: Path to motif database
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing compositional similarity results
        """
        # Validate inputs
        self.validate_inputs(x_synthetic, x_test)
        
        # Convert to numpy and standardize format
        x_synthetic_np = self._ensure_numpy(x_synthetic)
        x_test_np = self._ensure_numpy(x_test)
        
        x_synthetic_np = self._standardize_shape(x_synthetic_np)
        x_test_np = self._standardize_shape(x_test_np)
        
        results = {}
        
        # 1. Motif enrichment analysis
        if motif_database_path:
            results.update(self._motif_enrichment_analysis(
                x_synthetic_np, x_test_np, motif_database_path
            ))
            
            # 2. Motif co-occurrence analysis
            results.update(self._motif_cooccurrence_analysis(
                x_synthetic_np, x_test_np, motif_database_path
            ))
        else:
            results["motif_analysis_note"] = "Motif database path not provided, skipping motif analysis"
        
        # 3. Attribution consistency analysis (if oracle model provided)
        if oracle_model:
            results.update(self._attribution_consistency_analysis(
                x_synthetic_np, x_test_np, oracle_model
            ))
        else:
            results["attribution_analysis_note"] = "Oracle model not provided, skipping attribution analysis"
        
        self.update_results(results)
        return results
    
    def _motif_enrichment_analysis(self, 
                                 x_synthetic: np.ndarray,
                                 x_test: np.ndarray,
                                 motif_database_path: str) -> Dict[str, Any]:
        """
        Analyze motif enrichment using FIMO or TangerMeme.
        
        Args:
            x_synthetic: Synthetic sequences
            x_test: Test sequences
            motif_database_path: Path to motif database
            
        Returns:
            Dictionary with motif enrichment results
        """
        try:
            # Try TangerMeme first (more efficient)
            return self._motif_enrichment_tangermeme(x_synthetic, x_test, motif_database_path)
        except ImportError:
            # Fall back to FIMO if TangerMeme not available
            return self._motif_enrichment_fimo(x_synthetic, x_test, motif_database_path)
    
    def _motif_enrichment_tangermeme(self, 
                                   x_synthetic: np.ndarray,
                                   x_test: np.ndarray,
                                   motif_database_path: str) -> Dict[str, Any]:
        """Motif enrichment analysis using TangerMeme."""
        try:
            from tangermeme.tools.fimo import fimo as tangermeme_fimo
            from tangermeme.io import read_meme
        except ImportError:
            raise ImportError("TangerMeme not available")
        
        # Convert sequences to (N, A, L) format for TangerMeme
        x_test_nal = np.transpose(x_test, (0, 2, 1)) if x_test.shape[2] == 4 else x_test
        x_synthetic_nal = np.transpose(x_synthetic, (0, 2, 1)) if x_synthetic.shape[2] == 4 else x_synthetic
        
        # Get motif counts
        motif_counts_test = self._motif_count_tangermeme(motif_database_path, x_test_nal)
        motif_counts_synthetic = self._motif_count_tangermeme(motif_database_path, x_synthetic_nal)
        
        # Calculate Pearson correlation
        counts_test = list(motif_counts_test.values())
        counts_synthetic = list(motif_counts_synthetic.values())
        
        correlation, p_value = scipy.stats.pearsonr(counts_test, counts_synthetic)
        
        return {
            "motif_enrichment_pearson_r": float(correlation),
            "motif_enrichment_p_value": float(p_value),
            "motif_analysis_method": "tangermeme"
        }
    
    def _motif_enrichment_fimo(self, 
                             x_synthetic: np.ndarray,
                             x_test: np.ndarray,
                             motif_database_path: str) -> Dict[str, Any]:
        """Motif enrichment analysis using FIMO."""
        try:
            from pymemesuite import fimo
            from pymemesuite.common import MotifFile, Sequence
            from pymemesuite.fimo import FIMO
            import Bio.SeqIO
        except ImportError:
            raise ImportError("pymemesuite or Bio not available")
        
        # Convert to sequence strings and create temporary FASTA files
        test_sequences = one_hot_to_sequences(x_test)
        synthetic_sequences = one_hot_to_sequences(x_synthetic)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as test_file:
            create_fasta_file(test_sequences, test_file.name)
            test_fasta_path = test_file.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as syn_file:
            create_fasta_file(synthetic_sequences, syn_file.name)
            syn_fasta_path = syn_file.name
        
        try:
            # Get motif counts
            motif_counts_test = self._motif_count_fimo(test_fasta_path, motif_database_path)
            motif_counts_synthetic = self._motif_count_fimo(syn_fasta_path, motif_database_path)
            
            # Calculate correlation
            counts_test = list(motif_counts_test.values())
            counts_synthetic = list(motif_counts_synthetic.values())
            
            correlation, p_value = scipy.stats.pearsonr(counts_test, counts_synthetic)
            
            return {
                "motif_enrichment_pearson_r": float(correlation),
                "motif_enrichment_p_value": float(p_value),
                "motif_analysis_method": "fimo"
            }
            
        finally:
            # Clean up temporary files
            os.unlink(test_fasta_path)
            os.unlink(syn_fasta_path)
    
    def _motif_cooccurrence_analysis(self, 
                                   x_synthetic: np.ndarray,
                                   x_test: np.ndarray,
                                   motif_database_path: str) -> Dict[str, Any]:
        """
        Analyze motif co-occurrence patterns.
        
        Args:
            x_synthetic: Synthetic sequences
            x_test: Test sequences
            motif_database_path: Path to motif database
            
        Returns:
            Dictionary with co-occurrence results
        """
        try:
            # Try TangerMeme approach
            return self._motif_cooccurrence_tangermeme(x_synthetic, x_test, motif_database_path)
        except ImportError:
            # Fall back to FIMO
            return self._motif_cooccurrence_fimo(x_synthetic, x_test, motif_database_path)
    
    def _motif_cooccurrence_tangermeme(self, 
                                     x_synthetic: np.ndarray,
                                     x_test: np.ndarray,
                                     motif_database_path: str) -> Dict[str, Any]:
        """Motif co-occurrence analysis using TangerMeme."""
        try:
            from tangermeme.tools.fimo import fimo as tangermeme_fimo
            from tangermeme.io import read_meme
        except ImportError:
            raise ImportError("TangerMeme not available")
        
        # Convert to (N, A, L) format
        x_test_nal = np.transpose(x_test, (0, 2, 1)) if x_test.shape[2] == 4 else x_test
        x_synthetic_nal = np.transpose(x_synthetic, (0, 2, 1)) if x_synthetic.shape[2] == 4 else x_synthetic
        
        # Get occurrence matrices
        motif_matrix_test = self._make_occurrence_matrix_tangermeme(motif_database_path, x_test_nal)
        motif_matrix_synthetic = self._make_occurrence_matrix_tangermeme(motif_database_path, x_synthetic_nal)
        
        # Calculate covariance matrices
        cov_test = np.cov(motif_matrix_test.T)
        cov_synthetic = np.cov(motif_matrix_synthetic.T)
        
        # Calculate Frobenius norm of difference
        frobenius_norm = np.sqrt(np.sum((cov_test - cov_synthetic)**2))
        
        return {
            "motif_cooccurrence_frobenius_norm": float(frobenius_norm),
            "cooccurrence_method": "tangermeme"
        }
    
    def _motif_cooccurrence_fimo(self, 
                               x_synthetic: np.ndarray,
                               x_test: np.ndarray,
                               motif_database_path: str) -> Dict[str, Any]:
        """Motif co-occurrence analysis using FIMO."""
        # Convert to sequences and create temporary files
        test_sequences = one_hot_to_sequences(x_test)
        synthetic_sequences = one_hot_to_sequences(x_synthetic)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as test_file:
            create_fasta_file(test_sequences, test_file.name)
            test_fasta_path = test_file.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as syn_file:
            create_fasta_file(synthetic_sequences, syn_file.name)
            syn_fasta_path = syn_file.name
        
        try:
            # Get occurrence matrices
            motif_matrix_test = self._make_occurrence_matrix_fimo(test_fasta_path, motif_database_path)
            motif_matrix_synthetic = self._make_occurrence_matrix_fimo(syn_fasta_path, motif_database_path)
            
            # Calculate covariance matrices
            cov_test = np.cov(np.array(motif_matrix_test).T)
            cov_synthetic = np.cov(np.array(motif_matrix_synthetic).T)
            
            # Calculate Frobenius norm
            frobenius_norm = np.sqrt(np.sum((cov_test - cov_synthetic)**2))
            
            return {
                "motif_cooccurrence_frobenius_norm": float(frobenius_norm),
                "cooccurrence_method": "fimo"
            }
            
        finally:
            # Clean up
            os.unlink(test_fasta_path)
            os.unlink(syn_fasta_path)
    
    def _attribution_consistency_analysis(self, 
                                        x_synthetic: np.ndarray,
                                        x_test: np.ndarray,
                                        oracle_model: Any) -> Dict[str, Any]:
        """
        Analyze attribution consistency using gradient-based attribution.
        
        Args:
            x_synthetic: Synthetic sequences
            x_test: Test sequences
            oracle_model: Oracle model
            
        Returns:
            Dictionary with attribution consistency results
        """
        try:
            from captum.attr import GradientShap
        except ImportError:
            return {"attribution_analysis_error": "Captum not available for attribution analysis"}
        
        # Convert to tensors
        x_synthetic_tensor = self._ensure_tensor(x_synthetic)
        x_test_tensor = self._ensure_tensor(x_test)
        
        # Wrap model
        model = ModelWrapper(oracle_model) if not isinstance(oracle_model, ModelWrapper) else oracle_model
        
        # Subsample for computational efficiency
        max_samples = self.config.get('attribution_max_samples', 1000)
        if len(x_synthetic_tensor) > max_samples:
            indices = np.random.choice(len(x_synthetic_tensor), max_samples, replace=False)
            x_synthetic_subset = x_synthetic_tensor[indices]
        else:
            x_synthetic_subset = x_synthetic_tensor
            
        if len(x_test_tensor) > max_samples:
            indices = np.random.choice(len(x_test_tensor), max_samples, replace=False)
            x_test_subset = x_test_tensor[indices]
        else:
            x_test_subset = x_test_tensor
        
        # Calculate attribution scores
        shap_scores_synthetic = self._gradient_shap(x_synthetic_subset, model)
        shap_scores_test = self._gradient_shap(x_test_subset, model)
        
        # Process attribution maps
        attribution_map_synthetic = self._process_attribution_map(shap_scores_synthetic)
        attribution_map_test = self._process_attribution_map(shap_scores_test)
        
        # Calculate consistency metrics
        consistency_score = self._calculate_attribution_consistency(
            attribution_map_synthetic, attribution_map_test
        )
        
        return {
            "attribution_consistency_score": float(consistency_score),
            "attribution_samples_used": min(len(x_synthetic_tensor), max_samples)
        }
    
    def _gradient_shap(self, x_seq: torch.Tensor, model: ModelWrapper, class_index: int = 0) -> np.ndarray:
        """Calculate Gradient SHAP attribution scores."""
        try:
            from captum.attr import GradientShap
        except ImportError:
            raise ImportError("Captum required for attribution analysis")
        
        N, L, A = x_seq.shape
        score_cache = []
        
        for i, x in enumerate(x_seq):
            # Process single sequence
            x = x.unsqueeze(0)  # Add batch dimension
            x = x.transpose(1, 2)  # Convert to (N, A, L) format for model
            x.requires_grad_(True)
            
            # Create random background
            num_background = 100
            null_index = np.random.randint(0, A, size=(num_background, L))
            x_null = torch.zeros((num_background, A, L))
            for n in range(num_background):
                for l in range(L):
                    x_null[n, null_index[n, l], l] = 1.0
            x_null.requires_grad_(True)
            
            # Calculate Gradient SHAP
            gradient_shap = GradientShap(model)
            grad = gradient_shap.attribute(
                x,
                n_samples=50,
                stdevs=0.1,
                baselines=x_null,
                target=class_index
            )
            
            grad = grad.data.cpu().numpy()
            # Apply gradient correction
            grad -= np.mean(grad, axis=1, keepdims=True)
            score_cache.append(np.squeeze(grad))
        
        score_cache = np.array(score_cache)
        # Convert back to (N, L, A) format
        return np.transpose(score_cache, (0, 2, 1))
    
    def _process_attribution_map(self, saliency_map: np.ndarray, k: int = 6) -> np.ndarray:
        """Process attribution map for consistency analysis."""
        # Apply gradient correction and normalization
        saliency_map = saliency_map - np.mean(saliency_map, axis=-1, keepdims=True)
        norm_factor = np.sum(np.sqrt(np.sum(np.square(saliency_map), axis=-1, keepdims=True)), axis=-2, keepdims=True)
        saliency_map = saliency_map / (norm_factor + 1e-8)
        
        # Apply k-mer smoothing
        saliency_special = saliency_map.copy()
        for i in range(k-1):
            rolled = np.roll(saliency_map, -i-1, axis=-2)
            saliency_special += rolled
        
        # Convert to orthonormal coordinates
        return self._orthonormal_coordinates(saliency_special)
    
    def _orthonormal_coordinates(self, attr_map: np.ndarray) -> np.ndarray:
        """Convert 4D attribution map to 3D orthonormal coordinates."""
        attr_map_on = np.zeros((attr_map.shape[0], attr_map.shape[1], 3))
        
        x = attr_map[:, :, 0]
        y = attr_map[:, :, 1]
        z = attr_map[:, :, 2]
        w = attr_map[:, :, 3]
        
        # Convert to orthonormal basis
        e1 = 1 / np.sqrt(2) * (-x + y)
        e2 = np.sqrt(2 / 3) * (-0.5*x - 0.5*y + z)
        e3 = np.sqrt(3 / 4) * (-x/3 - y/3 - z/3 + w)
        
        attr_map_on[:, :, 0] = e1
        attr_map_on[:, :, 1] = e2
        attr_map_on[:, :, 2] = e3
        
        return attr_map_on
    
    def _calculate_attribution_consistency(self, 
                                         attr_synthetic: np.ndarray,
                                         attr_test: np.ndarray) -> float:
        """Calculate consistency score between attribution maps."""
        # Flatten attribution maps
        attr_synthetic_flat = attr_synthetic.reshape(-1, 3)
        attr_test_flat = attr_test.reshape(-1, 3)
        
        # Calculate correlation between attribution distributions
        correlations = []
        for i in range(3):
            corr, _ = scipy.stats.pearsonr(attr_synthetic_flat[:, i], attr_test_flat[:, i])
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    # Helper methods for motif analysis
    def _motif_count_tangermeme(self, meme_file_path: str, onehot_seqs: np.ndarray) -> Dict[str, int]:
        """Get motif counts using TangerMeme."""
        from tangermeme.tools.fimo import fimo as tangermeme_fimo
        from tangermeme.io import read_meme
        
        motifs = read_meme(meme_file_path)
        hits = tangermeme_fimo(meme_file_path, onehot_seqs, dim=0)
        motif_names = list(motifs.keys())
        occurrence = [df.shape[0] for df in hits]
        
        return dict(zip(motif_names, occurrence))
    
    def _make_occurrence_matrix_tangermeme(self, meme_file_path: str, onehot_seqs: np.ndarray) -> np.ndarray:
        """Create occurrence matrix using TangerMeme."""
        from tangermeme.tools.fimo import fimo as tangermeme_fimo
        from tangermeme.io import read_meme
        
        motifs = read_meme(meme_file_path)
        motif_names = list(motifs.keys())
        hits_by_seq = tangermeme_fimo(meme_file_path, onehot_seqs, dim=1)
        
        occurrence_matrix = np.zeros((onehot_seqs.shape[0], len(motif_names)))
        
        for i in range(onehot_seqs.shape[0]):
            i_hits_df = hits_by_seq[i]
            if not i_hits_df.empty:
                i_motif_counts = i_hits_df['motif_name'].value_counts().to_dict()
                for j, motif_name in enumerate(motif_names):
                    if motif_name in i_motif_counts:
                        occurrence_matrix[i, j] = i_motif_counts[motif_name]
        
        return occurrence_matrix
    
    def _motif_count_fimo(self, fasta_path: str, motif_database_path: str) -> Dict[str, int]:
        """Get motif counts using FIMO."""
        from pymemesuite.common import MotifFile, Sequence
        from pymemesuite.fimo import FIMO
        import Bio.SeqIO
        
        motif_ids = []
        occurrence = []
        
        sequences = [
            Sequence(str(record.seq), name=record.id.encode())
            for record in Bio.SeqIO.parse(fasta_path, "fasta")
        ]
        
        fimo_obj = FIMO()
        with MotifFile(motif_database_path) as motif_file:
            for motif in motif_file:
                pattern = fimo_obj.score_motif(motif, sequences, motif_file.background)
                motif_ids.append(motif.accession.decode())
                occurrence.append(len(pattern.matched_elements))
        
        return dict(zip(motif_ids, occurrence))
    
    def _make_occurrence_matrix_fimo(self, fasta_path: str, motif_database_path: str) -> List[List[int]]:
        """Create occurrence matrix using FIMO."""
        from pymemesuite.common import MotifFile, Sequence
        from pymemesuite.fimo import FIMO
        import Bio.SeqIO
        
        sequences = [
            Sequence(str(record.seq), name=record.id.encode())
            for record in Bio.SeqIO.parse(fasta_path, "fasta")
        ]
        
        fimo_obj = FIMO()
        occurrence_matrix = []
        
        for sequence in sequences:
            seq_list = [sequence]
            occurrence = []
            
            with MotifFile(motif_database_path) as motif_file:
                for motif in motif_file:
                    pattern = fimo_obj.score_motif(motif, seq_list, motif_file.background)
                    occurrence.append(len(pattern.matched_elements))
            
            occurrence_matrix.append(occurrence)
        
        return occurrence_matrix