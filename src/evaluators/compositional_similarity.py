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
from tqdm import tqdm
from .base_evaluator import BaseEvaluator
from data.data_utils import one_hot_to_sequences, create_fasta_file
from models.model_utils import ModelWrapper


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
        
        # Count total steps for progress tracking
        total_steps = 0
        if motif_database_path:
            total_steps += 2  # motif enrichment + co-occurrence
        if oracle_model:
            total_steps += 1  # attribution analysis
            
        progress_bar = tqdm(total=total_steps, desc="Compositional similarity evaluation", 
                           unit="analysis", disable=total_steps == 0)
        
        try:
            # 1. Motif enrichment analysis
            if motif_database_path:
                progress_bar.set_description("Motif enrichment analysis")
                results.update(self._motif_enrichment_analysis(
                    x_synthetic_np, x_test_np, motif_database_path
                ))
                progress_bar.update(1)
                
                # 2. Motif co-occurrence analysis
                progress_bar.set_description("Motif co-occurrence analysis")
                results.update(self._motif_cooccurrence_analysis(
                    x_synthetic_np, x_test_np, motif_database_path
                ))
                progress_bar.update(1)
            else:
                results["motif_analysis_note"] = "Motif database path not provided, skipping motif analysis"
            
            # 3. Attribution consistency analysis (if oracle model provided)
            if oracle_model:
                progress_bar.set_description("Attribution consistency analysis")
                results.update(self._attribution_consistency_analysis(
                    x_synthetic_np, x_test_np, oracle_model
                ))
                progress_bar.update(1)
            else:
                results["attribution_analysis_note"] = "Oracle model not provided, skipping attribution analysis"
                
        finally:
            progress_bar.close()
        
        self.update_results(results)
        return results
    
    def _motif_enrichment_analysis(self, 
                                 x_synthetic: np.ndarray,
                                 x_test: np.ndarray,
                                 motif_database_path: str) -> Dict[str, Any]:
        """
        Analyze motif enrichment using FIMO or MemeLite.
        
        Args:
            x_synthetic: Synthetic sequences
            x_test: Test sequences
            motif_database_path: Path to motif database
            
        Returns:
            Dictionary with motif enrichment results
        """
        try:
            # Try MemeLite first (more efficient)
            return self._motif_enrichment_memelite(x_synthetic, x_test, motif_database_path)
        except ImportError:
            # Fall back to FIMO if MemeLite not available
            return self._motif_enrichment_fimo(x_synthetic, x_test, motif_database_path)
    
    def _motif_enrichment_memelite(self, 
                                   x_synthetic: np.ndarray,
                                   x_test: np.ndarray,
                                   motif_database_path: str) -> Dict[str, Any]:
        """Motif enrichment analysis using MemeLite."""
        try:
            from memelite.fimo import fimo as memelite_fimo
            from memelite.io import read_meme
        except ImportError:
            raise ImportError("MemeLite not available")
        
        # Convert sequences to (N, A, L) format for MemeLite
        x_test_nal = np.transpose(x_test, (0, 2, 1)) if x_test.shape[2] == 4 else x_test
        x_synthetic_nal = np.transpose(x_synthetic, (0, 2, 1)) if x_synthetic.shape[2] == 4 else x_synthetic
        
        # Get motif counts
        with tqdm(total=2, desc="Computing motif counts", unit="dataset", leave=False) as pbar:
            pbar.set_description("Computing test motif counts")
            motif_counts_test = self._motif_count_memelite(motif_database_path, x_test_nal)
            pbar.update(1)
            
            pbar.set_description("Computing synthetic motif counts")
            motif_counts_synthetic = self._motif_count_memelite(motif_database_path, x_synthetic_nal)
            pbar.update(1)
        
        # Calculate Pearson correlation
        counts_test = list(motif_counts_test.values())
        counts_synthetic = list(motif_counts_synthetic.values())
        
        correlation, p_value = scipy.stats.pearsonr(counts_test, counts_synthetic)
        
        return {
            "motif_enrichment_pearson_r": float(correlation),
            "motif_enrichment_p_value": float(p_value),
            "motif_analysis_method": "memelite"
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
            with tqdm(total=2, desc="Computing FIMO motif counts", unit="dataset", leave=False) as pbar:
                pbar.set_description("Computing test motif counts")
                motif_counts_test = self._motif_count_fimo(test_fasta_path, motif_database_path)
                pbar.update(1)
                
                pbar.set_description("Computing synthetic motif counts")
                motif_counts_synthetic = self._motif_count_fimo(syn_fasta_path, motif_database_path)
                pbar.update(1)
            
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
            # Try MemeLite approach
            return self._motif_cooccurrence_memelite(x_synthetic, x_test, motif_database_path)
        except ImportError:
            # Fall back to FIMO
            return self._motif_cooccurrence_fimo(x_synthetic, x_test, motif_database_path)
    
    def _motif_cooccurrence_memelite(self, 
                                     x_synthetic: np.ndarray,
                                     x_test: np.ndarray,
                                     motif_database_path: str) -> Dict[str, Any]:
        """Motif co-occurrence analysis using MemeLite."""
        try:
            from memelite.fimo import fimo as memelite_fimo
            from memelite.io import read_meme
        except ImportError:
            raise ImportError("MemeLite not available")
        
        # Convert to (N, A, L) format
        x_test_nal = np.transpose(x_test, (0, 2, 1)) if x_test.shape[2] == 4 else x_test
        x_synthetic_nal = np.transpose(x_synthetic, (0, 2, 1)) if x_synthetic.shape[2] == 4 else x_synthetic
        
        # Get occurrence matrices
        with tqdm(total=2, desc="Building occurrence matrices", unit="dataset", leave=False) as pbar:
            pbar.set_description("Building test occurrence matrix")
            motif_matrix_test = self._make_occurrence_matrix_memelite(motif_database_path, x_test_nal)
            pbar.update(1)
            
            pbar.set_description("Building synthetic occurrence matrix") 
            motif_matrix_synthetic = self._make_occurrence_matrix_memelite(motif_database_path, x_synthetic_nal)
            pbar.update(1)
        
        # Calculate covariance matrices - match original transpose operations
        mm_test = np.array(motif_matrix_test).T
        mm_synthetic = np.array(motif_matrix_synthetic).T
        
        cov_test = np.cov(mm_test)
        cov_synthetic = np.cov(mm_synthetic)
        
        # Calculate Frobenius norm of difference
        frobenius_norm = np.sqrt(np.sum((cov_test - cov_synthetic)**2))
        
        return {
            "motif_cooccurrence_frobenius_norm": float(frobenius_norm),
            "cooccurrence_method": "memelite"
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
            with tqdm(total=2, desc="Building FIMO occurrence matrices", unit="dataset", leave=False) as pbar:
                pbar.set_description("Building test occurrence matrix")
                motif_matrix_test = self._make_occurrence_matrix_fimo(test_fasta_path, motif_database_path)
                pbar.update(1)
                
                pbar.set_description("Building synthetic occurrence matrix")
                motif_matrix_synthetic = self._make_occurrence_matrix_fimo(syn_fasta_path, motif_database_path)
                pbar.update(1)
            
            # Calculate covariance matrices - match original transpose operations
            mm_test = np.array(motif_matrix_test).T
            mm_synthetic = np.array(motif_matrix_synthetic).T
            
            cov_test = np.cov(mm_test)
            cov_synthetic = np.cov(mm_synthetic)
            
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
        Analyze attribution consistency using the original spherical coordinate processing.
        
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
        
        # Subsample for computational efficiency (original uses top 2000 activity sequences)
        max_samples = self.config.get('attribution_max_samples', 2000)
        
        # For synthetic sequences, select top activity sequences like in original
        if len(x_synthetic_tensor) > max_samples:
            # Get activities and select top sequences
            y_synthetic = model.predict(x_synthetic_tensor, x_synthetic_tensor)[1]  # Get synthetic predictions
            total_activity = np.sum(y_synthetic, axis=1)
            sorted_indices = np.argsort(total_activity)[::-1]  # Descending order
            top_indices = sorted_indices[:max_samples]
            x_synthetic_top = x_synthetic_tensor[top_indices]
        else:
            x_synthetic_top = x_synthetic_tensor
            
        # Calculate SHAP scores for top activity sequences
        shap_scores_synthetic = self._gradient_shap(x_synthetic_top, model)
        attribution_map_synthetic = self._process_attribution_map(shap_scores_synthetic, k=6)
        mask_synthetic = self._unit_mask(x_synthetic_top)
        
        # Calculate entropic information for synthetic sequences
        phi_1_s, phi_2_s, r_s = self._spherical_coordinates_process_2_trad(
            [attribution_map_synthetic], 
            x_synthetic_top, 
            mask_synthetic, 
            radius_count_cutoff=0.04
        )
        
        LIM, box_length, box_volume, n_bins, n_bins_half = self._initialize_integration_2(0.1)
        entropic_info_synthetic = self._calculate_entropy_2(
            phi_1_s, phi_2_s, r_s, n_bins, 0.1, box_volume, prior_range=3
        )
        
        # For consistency analysis, concatenate test and synthetic sequences
        x_concatenated = torch.cat((x_test_tensor, x_synthetic_tensor), dim=0)
        shap_scores_concatenated = self._gradient_shap(x_concatenated, model)
        attribution_map_concatenated = self._process_attribution_map(shap_scores_concatenated, k=6)
        mask_concatenated = self._unit_mask(x_concatenated)
        
        phi_1_s_concat, phi_2_s_concat, r_s_concat = self._spherical_coordinates_process_2_trad(
            [attribution_map_concatenated],
            x_concatenated,
            mask_concatenated,
            radius_count_cutoff=0.04
        )
        
        entropic_info_concatenated = self._calculate_entropy_2(
            phi_1_s_concat, phi_2_s_concat, r_s_concat, n_bins, 0.1, box_volume, prior_range=3
        )
        
        return {
            "entropic_information_top_synthetic": float(entropic_info_synthetic[0]),
            "entropic_information_concatenated": float(entropic_info_concatenated[0]),
            "attribution_samples_used": len(x_synthetic_top),
            "attribution_method": "original_spherical_coordinates"
        }
    
    def _gradient_shap(self, x_seq: torch.Tensor, model: ModelWrapper, class_index: int = 0) -> np.ndarray:
        """Calculate Gradient SHAP attribution scores."""
        try:
            from captum.attr import GradientShap
        except ImportError:
            raise ImportError("Captum required for attribution analysis")
        
        N, L, A = x_seq.shape
        score_cache = []
        
        for i, x in tqdm(enumerate(x_seq), total=len(x_seq), 
                        desc="Computing gradient attributions", unit="seq"):
            # Process single sequence
            x = x.unsqueeze(0)  # Add batch dimension
            x = x.transpose(1, 2)  # Convert to (N, A, L) format for model
            x.requires_grad_(True)
            
            # Create random background - match original parameters
            num_background = 1000
            null_index = np.random.randint(0, A, size=(num_background, L))
            x_null = torch.zeros((num_background, A, L))
            for n in range(num_background):
                for l in range(L):
                    x_null[n, null_index[n, l], l] = 1.0
            x_null.requires_grad_(True)
            
            # Calculate Gradient SHAP - match original parameters
            gradient_shap = GradientShap(model)
            grad = gradient_shap.attribute(
                x,
                n_samples=100,
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
    
    def _unit_mask(self, x_seq: torch.Tensor) -> np.ndarray:
        """Create unit mask for sequences."""
        x_np = self._ensure_numpy(x_seq)
        return np.sum(np.ones(x_np.shape), axis=-1) / 4
    
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
    
    # Original spherical coordinate processing functions
    
    def _spherical_coordinates_process_2_trad(self, saliency_map_raw_s, X, mask, radius_count_cutoff=0.04):
        """Process attribution maps into spherical coordinates (original implementation)."""
        N_EXP = len(saliency_map_raw_s)
        radius_count = int(radius_count_cutoff * np.prod(X.shape) / 4)
        cutoff = []
        x_s, y_s, z_s, r_s, phi_1_s, phi_2_s = [], [], [], [], [], []
        PI = 3.1416
        
        for s in range(N_EXP):
            saliency_map_raw = saliency_map_raw_s[s]
            xxx_motif = saliency_map_raw[:, :, 0]
            yyy_motif = saliency_map_raw[:, :, 1] 
            zzz_motif = saliency_map_raw[:, :, 2]
            xxx_motif_pattern = saliency_map_raw[:, :, 0] * mask
            yyy_motif_pattern = saliency_map_raw[:, :, 1] * mask
            zzz_motif_pattern = saliency_map_raw[:, :, 2] * mask
            r = np.sqrt(xxx_motif * xxx_motif + yyy_motif * yyy_motif + zzz_motif * zzz_motif)
            resh = X.shape[0] * X.shape[1]
            x = np.array(xxx_motif_pattern.reshape(resh,))
            y = np.array(yyy_motif_pattern.reshape(resh,))
            z = np.array(zzz_motif_pattern.reshape(resh,))
            r = np.array(r.reshape(resh,))
            
            # Take care of any NANs
            x = np.nan_to_num(x)
            y = np.nan_to_num(y)
            z = np.nan_to_num(z)
            r = np.nan_to_num(r)
            cutoff.append(np.sort(r)[-radius_count])
            R_cutoff_index = np.sqrt(x*x + y*y + z*z) > cutoff[s]
            
            # Cut off
            x = x[R_cutoff_index]
            y = y[R_cutoff_index]
            z = z[R_cutoff_index]
            r = np.array(r[R_cutoff_index])
            x_s.append(x)
            y_s.append(y)
            z_s.append(z)
            r_s.append(r)
            
            # Rotate axis
            x__ = np.array(y)
            y__ = np.array(z)
            z__ = np.array(x)
            x = x__
            y = y__
            z = z__
            
            # "phi"
            phi_1 = np.arctan(y/x)  # default
            phi_1 = np.where((x<0) & (y>=0), np.arctan(y/x) + PI, phi_1)   # overwrite
            phi_1 = np.where((x<0) & (y<0), np.arctan(y/x) - PI, phi_1)   # overwrite
            phi_1 = np.where(x==0, PI/2, phi_1)  # overwrite
            # Renormalize temporarily to have both angles in [0,PI]:
            phi_1 = phi_1/2 + PI/2
            # "theta"
            phi_2 = np.arccos(z/r)
            # back to list
            phi_1 = list(phi_1)
            phi_2 = list(phi_2)
            phi_1_s.append(phi_1)
            phi_2_s.append(phi_2)
            
        return phi_1_s, phi_2_s, r_s
    
    def _initialize_integration_2(self, box_length):
        """Initialize integration parameters (original implementation)."""
        LIM = 3.1416
        box_volume = box_length * box_length
        n_bins = int(LIM / box_length)
        volume_border_correction = (LIM / box_length / n_bins) * (LIM / box_length / n_bins)
        n_bins_half = int(n_bins / 2)
        return LIM, box_length, box_volume, n_bins, n_bins_half
    
    def _calculate_entropy_2(self, phi_1_s, phi_2_s, r_s, n_bins, box_length, box_volume, prior_range):
        """Calculate entropy using original method."""
        N_EXP = len(phi_1_s)
        Empirical_box_pdf_s = []
        Empirical_box_count_s = []
        Empirical_box_count_plain_s = []
        
        for s in range(N_EXP):
            pdf, count, count_plain = self._empirical_box_pdf_func_2(
                phi_1_s[s], phi_2_s[s], r_s[s], n_bins, box_length, box_volume
            )
            Empirical_box_pdf_s.append(pdf)
            Empirical_box_count_s.append(count)
            Empirical_box_count_plain_s.append(count_plain)
        
        Entropic_information = []
        for s in range(N_EXP):
            entropy = self._kl_divergence_2(
                Empirical_box_pdf_s[s], 
                Empirical_box_count_s[s], 
                Empirical_box_count_plain_s[s], 
                n_bins, 
                box_volume, 
                prior_range
            )
            Entropic_information.append(entropy)
        
        return Entropic_information
    
    def _empirical_box_pdf_func_2(self, phi_1, phi_2, r_s, n_bins, box_length, box_volume):
        """Calculate empirical box PDF (original implementation)."""
        N_points = len(phi_1)  # Number of points
        Empirical_box_count = np.zeros((n_bins, n_bins))
        Empirical_box_count_plain = np.zeros((n_bins, n_bins))
        
        # Now populate the box. Go over every single point.
        for i in range(N_points):
            # k, l are box numbers of the (phi_1, phi_2) point
            k = np.minimum(int(phi_1[i] / box_length), n_bins-1)
            l = np.minimum(int(phi_2[i] / box_length), n_bins-1)
            # Increment count in (k,l) box:
            Empirical_box_count[k, l] += 1 * r_s[i] * r_s[i]
            Empirical_box_count_plain[k, l] += 1
        
        # To get the probability distribution, divide the Empirical_box_count by the total number of points.
        Empirical_box_pdf = Empirical_box_count / N_points / box_volume
        # Check that it integrates to around 1:
        correction = 1 / np.sum(Empirical_box_pdf * box_volume)
        
        return Empirical_box_pdf * correction, Empirical_box_count * correction, Empirical_box_count_plain
    
    def _kl_divergence_2(self, Empirical_box_pdf, Empirical_box_count, Empirical_box_count_plain, 
                        n_bins, box_volume, prior_range):
        """Calculate KL divergence (original implementation)."""
        # p= empirical distribution, q=prior spherical distribution
        # Notice that the prior distribution is never 0! So it is safe to divide by q.
        # L'Hospital rule provides that p*log(p) --> 0 when p->0. When we encounter p=0, 
        # we would just set the contribution of that term to 0, i.e. ignore it in the sum.
        Relative_entropy = 0
        PI = 3.1416
        
        for i in range(1, n_bins-1):
            for j in range(1, n_bins-1):
                if Empirical_box_pdf[i, j] > 0:
                    phi_1 = i / n_bins * PI
                    phi_2 = j / n_bins * PI
                    prior_counter = 0
                    prior = 0
                    
                    for ii in range(-prior_range, prior_range):
                        for jj in range(-prior_range, prior_range):
                            if (i+ii > 0 and i+ii < n_bins and j+jj > 0 and j+jj < n_bins):
                                prior += Empirical_box_pdf[i+ii, j+jj]
                                prior_counter += 1
                    
                    prior = prior / prior_counter
                    
                    if prior > 0:
                        KL_divergence_contribution = Empirical_box_pdf[i, j] * np.log(
                            Empirical_box_pdf[i, j] / prior
                        )
                    
                    if np.sin(phi_1) > 0 and prior > 0:
                        Relative_entropy += KL_divergence_contribution
        
        Relative_entropy = Relative_entropy * box_volume  # (volume differential in the "integral")
        return np.round(Relative_entropy, 3)
    
    # Helper methods for motif analysis
    def _motif_count_memelite(self, meme_file_path: str, onehot_seqs: np.ndarray) -> Dict[str, int]:
        """Get motif counts using MemeLite."""
        from memelite.fimo import fimo as memelite_fimo
        from memelite.io import read_meme
        
        motifs = read_meme(meme_file_path)
        hits = memelite_fimo(meme_file_path, onehot_seqs, dim=0)
        motif_names = list(motifs.keys())
        occurrence = [df.shape[0] for df in hits]
        
        return dict(zip(motif_names, occurrence))
    
    def _make_occurrence_matrix_memelite(self, meme_file_path: str, onehot_seqs: np.ndarray) -> np.ndarray:
        """Create occurrence matrix using MemeLite."""
        from memelite.fimo import fimo as memelite_fimo
        from memelite.io import read_meme
        
        motifs = read_meme(meme_file_path)
        motif_names = list(motifs.keys())
        hits_by_seq = memelite_fimo(meme_file_path, onehot_seqs, dim=1)
        
        occurrence_matrix = np.zeros((onehot_seqs.shape[0], len(motif_names)))
        
        for i in tqdm(range(onehot_seqs.shape[0]), desc="Building motif occurrence matrix", unit="seq"):
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
            motifs = list(motif_file)  # Convert to list for progress tracking
            
        with MotifFile(motif_database_path) as motif_file:
            for motif in tqdm(motifs, desc="Computing FIMO motif counts", unit="motif"):
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
        
        for sequence in tqdm(sequences, desc="Building FIMO occurrence matrix", unit="seq"):
            seq_list = [sequence]
            occurrence = []
            
            with MotifFile(motif_database_path) as motif_file:
                for motif in motif_file:
                    pattern = fimo_obj.score_motif(motif, seq_list, motif_file.background)
                    occurrence.append(len(pattern.matched_elements))
            
            occurrence_matrix.append(occurrence)
        
        return occurrence_matrix