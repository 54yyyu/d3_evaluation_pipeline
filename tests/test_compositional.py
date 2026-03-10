"""
Tests for core/compositional/ — attribution consistency,
motif enrichment, and motif co-occurrence.

Note: attribution_consistency tests use the mock oracle for SHAP computation.
Motif enrichment and motif co-occurrence tests cover the pure-math helpers
(enrich_pr, frobenius_norm, covariance_matrix, sequences_to_onehot) and
the integration paths where memelite/pymemesuite are not required.
"""

import numpy as np
import torch
import pytest


# =====================================================================
# attribution_consistency — helper functions
# =====================================================================

class TestOrthonormalCoordinates:
    def test_output_shape(self):
        from core.compositional.attribution_consistency import orthonormal_coordinates
        attr_map = np.random.randn(10, 50, 4)
        result = orthonormal_coordinates(attr_map)
        assert result.shape == (10, 50, 3)

    def test_zero_input(self):
        from core.compositional.attribution_consistency import orthonormal_coordinates
        attr_map = np.zeros((5, 20, 4))
        result = orthonormal_coordinates(attr_map)
        np.testing.assert_array_equal(result, 0)


class TestProcessAttributionMap:
    def test_output_shape(self):
        from core.compositional.attribution_consistency import process_attribution_map
        saliency = np.random.randn(10, 50, 4).astype(np.float32)
        result = process_attribution_map(saliency, k=6)
        # Should reduce 4 → 3 in last dim
        assert result.shape == (10, 50, 3)

    def test_different_k_values(self):
        from core.compositional.attribution_consistency import process_attribution_map
        saliency = np.random.randn(5, 30, 4).astype(np.float32)
        r3 = process_attribution_map(saliency, k=3)
        r6 = process_attribution_map(saliency, k=6)
        assert r3.shape == r6.shape
        # Different k should give different results
        assert not np.allclose(r3, r6)


class TestUnitMask:
    def test_from_tensor(self):
        from core.compositional.attribution_consistency import unit_mask
        x = torch.randn(10, 50, 4)
        mask = unit_mask(x)
        assert mask.shape == (10, 50)
        # Each element should be 1.0 (sum of ones / 4)
        np.testing.assert_allclose(mask, 1.0)

    def test_from_numpy(self):
        from core.compositional.attribution_consistency import unit_mask
        x = np.random.randn(10, 50, 4)
        mask = unit_mask(x)
        assert mask.shape == (10, 50)


class TestSphericalCoordinates:
    def test_output_lengths_match(self):
        from core.compositional.attribution_consistency import (
            process_attribution_map, unit_mask,
            spherical_coordinates_process_2_trad
        )
        np.random.seed(0)
        N, L = 10, 50
        saliency = np.random.randn(N, L, 4).astype(np.float32)
        X = np.random.randn(N, L, 4).astype(np.float32)
        attr = process_attribution_map(saliency, k=6)
        mask = unit_mask(X)
        phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad(
            [attr], X, mask, radius_count_cutoff=0.04
        )
        assert len(phi_1_s) == 1
        assert len(phi_2_s) == 1
        assert len(r_s) == 1
        # phi_1 and phi_2 should have same length within each experiment
        assert len(phi_1_s[0]) == len(phi_2_s[0])


class TestEntropyCalculation:
    def test_initialize_integration(self):
        from core.compositional.attribution_consistency import initialize_integration_2
        LIM, bl, bv, n_bins, n_bins_half = initialize_integration_2(0.1)
        assert LIM == pytest.approx(3.1416)
        assert bl == 0.1
        assert n_bins == int(3.1416 / 0.1)
        assert n_bins_half == n_bins // 2

    def test_entropy_returns_list(self):
        from core.compositional.attribution_consistency import (
            process_attribution_map, unit_mask,
            spherical_coordinates_process_2_trad,
            initialize_integration_2, calculate_entropy_2
        )
        np.random.seed(0)
        N, L = 20, 50
        saliency = np.random.randn(N, L, 4).astype(np.float32)
        X = np.random.randn(N, L, 4).astype(np.float32)
        attr = process_attribution_map(saliency, k=6)
        mask = unit_mask(X)
        phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad(
            [attr], X, mask, radius_count_cutoff=0.04
        )
        LIM, bl, bv, n_bins, n_bins_half = initialize_integration_2(0.1)
        entropy = calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, 0.1, bv, prior_range=3)
        assert isinstance(entropy, list)
        assert len(entropy) == 1  # one experiment


# =====================================================================
# motif_enrichment — helper functions
# =====================================================================

class TestEnrichPr:
    def test_perfect_correlation(self):
        from core.compositional.motif_enrichment import enrich_pr
        counts = {'motif_a': 10, 'motif_b': 20, 'motif_c': 30}
        result = enrich_pr(counts, counts)
        assert result.statistic == pytest.approx(1.0)

    def test_negative_correlation(self):
        from core.compositional.motif_enrichment import enrich_pr
        c1 = {'a': 10, 'b': 20, 'c': 30}
        c2 = {'a': 30, 'b': 20, 'c': 10}
        result = enrich_pr(c1, c2)
        assert result.statistic == pytest.approx(-1.0)

    def test_returns_pvalue(self):
        from core.compositional.motif_enrichment import enrich_pr
        c1 = {'a': 10, 'b': 20, 'c': 30, 'd': 40}
        c2 = {'a': 12, 'b': 18, 'c': 28, 'd': 42}
        result = enrich_pr(c1, c2)
        assert hasattr(result, 'pvalue')
        assert 0 <= result.pvalue <= 1


class TestSequencesToOnehotEnrichment:
    def test_shape(self):
        from core.compositional.motif_enrichment import sequences_to_onehot
        seqs = ["ACGT", "TGCA"]
        onehot = sequences_to_onehot(seqs)
        assert onehot.shape == (2, 4, 4)

    def test_single_sequence(self):
        from core.compositional.motif_enrichment import sequences_to_onehot
        onehot = sequences_to_onehot(["AAAA"])
        # All A → index 0 should be 1 for all positions
        assert onehot.shape == (1, 4, 4)
        np.testing.assert_array_equal(onehot[0, 0, :], 1.0)
        np.testing.assert_array_equal(onehot[0, 1:, :], 0.0)

    def test_handles_lowercase(self):
        from core.compositional.motif_enrichment import sequences_to_onehot
        onehot = sequences_to_onehot(["acgt"])
        assert onehot.shape == (1, 4, 4)
        # Should still have exactly one 1 per position
        assert np.all(onehot.sum(axis=1) == 1.0)


# =====================================================================
# motif_cooccurrence — helper functions
# =====================================================================

class TestCovarianceMatrix:
    def test_square_output(self):
        from core.compositional.motif_cooccurrence import covariance_matrix
        x = np.random.randn(10, 50)  # 10 features, 50 observations
        cov = covariance_matrix(x)
        assert cov.shape == (10, 10)

    def test_symmetric(self):
        from core.compositional.motif_cooccurrence import covariance_matrix
        x = np.random.randn(8, 30)
        cov = covariance_matrix(x)
        np.testing.assert_allclose(cov, cov.T)

    def test_diagonal_nonnegative(self):
        from core.compositional.motif_cooccurrence import covariance_matrix
        x = np.random.randn(5, 20)
        cov = covariance_matrix(x)
        assert np.all(np.diag(cov) >= 0)


class TestFrobeniusNorm:
    def test_identical_matrices_zero(self):
        from core.compositional.motif_cooccurrence import frobenius_norm
        A = np.random.randn(5, 5)
        assert frobenius_norm(A, A) == pytest.approx(0.0)

    def test_known_value(self):
        from core.compositional.motif_cooccurrence import frobenius_norm
        A = np.array([[1.0, 0], [0, 1]])
        B = np.array([[0.0, 0], [0, 0]])
        # ||A - B||_F = sqrt(1 + 1) = sqrt(2)
        assert frobenius_norm(A, B) == pytest.approx(np.sqrt(2))

    def test_symmetric(self):
        from core.compositional.motif_cooccurrence import frobenius_norm
        A = np.random.randn(4, 4)
        B = np.random.randn(4, 4)
        assert frobenius_norm(A, B) == pytest.approx(frobenius_norm(B, A))

    def test_nonnegative(self):
        from core.compositional.motif_cooccurrence import frobenius_norm
        A = np.random.randn(5, 5)
        B = np.random.randn(5, 5)
        assert frobenius_norm(A, B) >= 0


class TestSequencesToOnehotCooccurrence:
    def test_shape(self):
        from core.compositional.motif_cooccurrence import sequences_to_onehot
        seqs = ["ACGT", "TGCA", "GGGG"]
        onehot = sequences_to_onehot(seqs)
        assert onehot.shape == (3, 4, 4)

    def test_one_hot_property(self):
        from core.compositional.motif_cooccurrence import sequences_to_onehot
        seqs = ["ACGTACGT"]
        onehot = sequences_to_onehot(seqs)
        # Each position should have exactly one 1
        assert np.all(onehot.sum(axis=1) == 1.0)
