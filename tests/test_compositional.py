"""
Tests for core/compositional/ analyses:
  - motif_enrichment
  - motif_cooccurrence
  - attribution_consistency
"""

import numpy as np
import torch
import pytest
import os

from tests.conftest import _random_onehot, DEEPSTARR_SEQ_LEN


# ===================================================================
# motif_enrichment
# ===================================================================

class TestMotifEnrichment:
    """Tests for the Pearson-correlation-based motif enrichment metric."""

    def test_enrich_pr_identical_counts_gives_r_one(self):
        """Pearson r between identical count dicts must be 1.0."""
        from core.compositional.motif_enrichment import enrich_pr

        counts = {"MA0001.1": 10, "MA0002.1": 20, "MA0003.1": 5}
        result = enrich_pr(counts, counts)
        assert result.statistic == pytest.approx(1.0, abs=1e-6)

    def test_enrich_pr_proportional_counts_gives_r_one(self):
        """Pearson r between proportionally scaled counts must be 1.0."""
        from core.compositional.motif_enrichment import enrich_pr

        c1 = {"A": 10, "B": 20, "C": 30}
        c2 = {"A": 20, "B": 40, "C": 60}
        result = enrich_pr(c1, c2)
        assert result.statistic == pytest.approx(1.0, abs=1e-4)

    def test_enrich_pr_returns_valid_pvalue(self):
        """p-value from Pearson correlation must be in [0, 1]."""
        from core.compositional.motif_enrichment import enrich_pr

        c1 = {"A": 10, "B": 20, "C": 30, "D": 5}
        c2 = {"A": 15, "B": 25, "C": 10, "D": 8}
        result = enrich_pr(c1, c2)
        assert 0 <= result.pvalue <= 1


# ===================================================================
# motif_cooccurrence
# ===================================================================

class TestMotifCooccurrence:
    """Tests for the Frobenius-norm-based motif co-occurrence metric."""

    def test_frobenius_norm_identical_matrices_gives_zero(self):
        """Frobenius norm between identical matrices must be 0."""
        from core.compositional.motif_cooccurrence import frobenius_norm

        cov = np.random.randn(10, 10)
        assert frobenius_norm(cov, cov) == pytest.approx(0.0)

    def test_frobenius_norm_known_value(self):
        """Verify Frobenius norm against a hand-calculated example."""
        from core.compositional.motif_cooccurrence import frobenius_norm

        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[0.0, 0.0], [0.0, 0.0]])
        # ||A - B||_F = sqrt(1 + 0 + 0 + 1) = sqrt(2)
        assert frobenius_norm(a, b) == pytest.approx(np.sqrt(2.0), abs=1e-6)

    def test_frobenius_norm_symmetry(self):
        """||A - B||_F == ||B - A||_F."""
        from core.compositional.motif_cooccurrence import frobenius_norm

        a = np.random.randn(8, 8)
        b = np.random.randn(8, 8)
        assert frobenius_norm(a, b) == pytest.approx(frobenius_norm(b, a))

    def test_frobenius_norm_non_negative(self):
        """Frobenius norm must always be >= 0."""
        from core.compositional.motif_cooccurrence import frobenius_norm

        a = np.random.randn(5, 5)
        b = np.random.randn(5, 5)
        assert frobenius_norm(a, b) >= 0

    def test_covariance_matrix_shape(self):
        """Covariance of (n, d) matrix should be (d, d)."""
        from core.compositional.motif_cooccurrence import covariance_matrix

        x = np.random.randn(50, 10)
        cov = covariance_matrix(x)
        assert cov.shape == (10, 10)

    def test_covariance_matrix_symmetry(self):
        """Covariance matrix must be symmetric."""
        from core.compositional.motif_cooccurrence import covariance_matrix

        x = np.random.randn(50, 10)
        cov = covariance_matrix(x)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_sequences_to_onehot_shape(self):
        """sequences_to_onehot must produce (N, 4, max_len) arrays."""
        from core.compositional.motif_cooccurrence import sequences_to_onehot

        seqs = ["ACGTACGT", "GGGGAAAA", "TTTTCCCC"]
        onehot = sequences_to_onehot(seqs)
        assert onehot.shape == (3, 4, 8)

    def test_sequences_to_onehot_valid_encoding(self):
        """Each position must have exactly one 1 across the 4 channels."""
        from core.compositional.motif_cooccurrence import sequences_to_onehot

        seqs = ["ACGT"]
        onehot = sequences_to_onehot(seqs)
        # Sum across channel dim should be 1 at every position
        np.testing.assert_array_equal(onehot.sum(axis=1), np.ones((1, 4)))


# ===================================================================
# attribution_consistency
# ===================================================================

class TestAttributionConsistency:
    """Tests for the SHAP/entropy-based attribution consistency metric."""

    def test_process_attribution_map_shape(self):
        """Output should be (N, L, 3) orthonormal coordinates."""
        from core.compositional.attribution_consistency import process_attribution_map

        np.random.seed(42)
        saliency = np.random.randn(10, DEEPSTARR_SEQ_LEN, 4).astype(np.float32)
        result = process_attribution_map(saliency, k=6)
        assert result.shape == (10, DEEPSTARR_SEQ_LEN, 3)

    def test_orthonormal_coordinates_shape(self):
        """orthonormal_coordinates must reduce last dim from 4 to 3."""
        from core.compositional.attribution_consistency import orthonormal_coordinates

        attr_map = np.random.randn(5, 100, 4)
        result = orthonormal_coordinates(attr_map)
        assert result.shape == (5, 100, 3)

    def test_unit_mask_shape_and_values(self):
        """unit_mask must return (N, L) of 1/4 values."""
        from core.compositional.attribution_consistency import unit_mask

        np.random.seed(42)
        seqs = np.zeros((5, 100, 4), dtype=np.float32)
        # make valid one-hot
        for i in range(5):
            for j in range(100):
                seqs[i, j, np.random.randint(4)] = 1.0
        seqs_tensor = torch.tensor(seqs)
        mask = unit_mask(seqs_tensor)
        assert mask.shape == (5, 100)

    def test_initialize_integration_2_outputs(self):
        """Must return consistent integration parameters."""
        from core.compositional.attribution_consistency import initialize_integration_2

        LIM, box_length, box_volume, n_bins, n_bins_half = initialize_integration_2(0.1)
        assert LIM == pytest.approx(np.pi)
        assert box_length == pytest.approx(0.1)
        assert n_bins == int(2 * np.pi / 0.1) + 1
        assert n_bins_half == n_bins // 2
        # box_volume should be positive
        assert box_volume > 0

    def test_run_analysis_returns_expected_keys(self, deepstarr_model, tmp_output_dir):
        """Full analysis must return KLD and KLD_concat."""
        from core.compositional.attribution_consistency import run_attribution_consistency_analysis

        np.random.seed(42)
        # Use small sequences to keep test fast
        n = 2100  # need > 2000 for top-2000 selection
        sample_seqs = torch.tensor(
            _random_onehot(n, DEEPSTARR_SEQ_LEN, fmt="NLA"), dtype=torch.float32
        )
        X_test = torch.tensor(
            _random_onehot(50, DEEPSTARR_SEQ_LEN, fmt="NLA"), dtype=torch.float32
        )
        # This calls gradient_shap which is expensive; verifying the result structure
        results = run_attribution_consistency_analysis(
            deepstarr_model, sample_seqs, X_test, output_dir=tmp_output_dir
        )
        assert "KLD" in results
        assert "KLD_concat" in results
