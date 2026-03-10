"""
Tests for core/functional/ — conditional generation fidelity,
Fréchet distance, and predictive distribution shift.
"""

import numpy as np
import torch
import pickle
import pytest
import os
from pathlib import Path


# =====================================================================
# conditional_generation_fidelity (pure math)
# =====================================================================

class TestConditionalGenerationFidelity:
    def test_identical_inputs_zero_mse(self):
        from core.functional.cond_gen_fidelity import conditional_generation_fidelity
        a = np.array([1.0, 2.0, 3.0])
        assert conditional_generation_fidelity(a, a) == 0.0

    def test_known_mse(self):
        from core.functional.cond_gen_fidelity import conditional_generation_fidelity
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        # MSE = mean([1, 1, 1]) = 1.0
        assert conditional_generation_fidelity(a, b) == pytest.approx(1.0)

    def test_symmetric(self):
        from core.functional.cond_gen_fidelity import conditional_generation_fidelity
        np.random.seed(0)
        a = np.random.randn(100)
        b = np.random.randn(100)
        assert conditional_generation_fidelity(a, b) == pytest.approx(
            conditional_generation_fidelity(b, a)
        )

    def test_nonnegative(self):
        from core.functional.cond_gen_fidelity import conditional_generation_fidelity
        np.random.seed(1)
        a, b = np.random.randn(50), np.random.randn(50)
        assert conditional_generation_fidelity(a, b) >= 0


# =====================================================================
# run_conditional_generation_fidelity_analysis (integration)
# =====================================================================

class TestRunCondGenFidelityAnalysis:
    def test_single_mode_returns_dict(self, mock_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis
        results = run_conditional_generation_fidelity_analysis(
            mock_oracle, x_test_tensor, x_synthetic_tensor, output_dir=tmp_output_dir
        )
        assert isinstance(results, dict)
        assert 'conditional_generation_fidelity_mse' in results
        assert isinstance(results['conditional_generation_fidelity_mse'], (float, np.floating))

    def test_single_mode_writes_pickle(self, mock_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis
        run_conditional_generation_fidelity_analysis(
            mock_oracle, x_test_tensor, x_synthetic_tensor, output_dir=tmp_output_dir
        )
        pkl_files = list(Path(tmp_output_dir).glob("cond_gen_fidelity_*.pkl"))
        assert len(pkl_files) == 1
        with open(pkl_files[0], 'rb') as f:
            loaded = pickle.load(f)
        assert 'conditional_generation_fidelity_mse' in loaded

    def test_batch_mode_writes_csv_and_h5(self, mock_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis
        run_conditional_generation_fidelity_analysis(
            mock_oracle, x_test_tensor, x_synthetic_tensor,
            output_dir=tmp_output_dir, sample_name="sample_A"
        )
        assert (Path(tmp_output_dir) / "cond_gen_fidelity.csv").exists()
        assert (Path(tmp_output_dir) / "cond_gen_fidelity.h5").exists()

    def test_multi_oracle_default(self, mock_multi_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis
        results = run_conditional_generation_fidelity_analysis(
            mock_multi_oracle, x_test_tensor, x_synthetic_tensor,
            output_dir=tmp_output_dir, model_type='multi-oracle'
        )
        # Default (per_dimension=False) should give single MSE
        assert 'conditional_generation_fidelity_mse' in results

    def test_multi_oracle_per_dimension(self, mock_multi_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis
        results = run_conditional_generation_fidelity_analysis(
            mock_multi_oracle, x_test_tensor, x_synthetic_tensor,
            output_dir=tmp_output_dir, model_type='multi-oracle', per_dimension=True
        )
        assert 'conditional_generation_fidelity_mse_oracle_1' in results
        assert 'conditional_generation_fidelity_mse_oracle_2' in results
        assert 'conditional_generation_fidelity_mse_oracle_3' in results


# =====================================================================
# Fréchet distance (pure math)
# =====================================================================

class TestFrechetDistanceMath:
    def test_identical_distributions_zero(self):
        from core.functional.frechet_distance import calculate_activation_statistics, calculate_frechet_distance
        np.random.seed(0)
        emb = torch.randn(100, 16)
        mu, sigma = calculate_activation_statistics(emb)
        fd = calculate_frechet_distance(mu, sigma, mu, sigma)
        assert fd == pytest.approx(0.0, abs=1e-4)

    def test_different_distributions_positive(self):
        from core.functional.frechet_distance import calculate_activation_statistics, calculate_frechet_distance
        emb1 = torch.randn(100, 16)
        emb2 = torch.randn(100, 16) + 5.0  # shifted mean
        mu1, sigma1 = calculate_activation_statistics(emb1)
        mu2, sigma2 = calculate_activation_statistics(emb2)
        fd = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        assert fd > 0

    def test_symmetric(self):
        from core.functional.frechet_distance import calculate_activation_statistics, calculate_frechet_distance
        emb1 = torch.randn(100, 8)
        emb2 = torch.randn(100, 8) + 2.0
        mu1, sigma1 = calculate_activation_statistics(emb1)
        mu2, sigma2 = calculate_activation_statistics(emb2)
        fd_12 = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        fd_21 = calculate_frechet_distance(mu2, sigma2, mu1, sigma1)
        assert fd_12 == pytest.approx(fd_21, rel=1e-3)

    def test_activation_statistics_shapes(self):
        from core.functional.frechet_distance import calculate_activation_statistics
        emb = torch.randn(50, 16)
        mu, sigma = calculate_activation_statistics(emb)
        assert mu.shape == (16,)
        assert sigma.shape == (16, 16)


# =====================================================================
# run_frechet_distance_analysis (integration)
# =====================================================================

class TestRunFrechetDistanceAnalysis:
    def test_single_mode(self, mock_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.frechet_distance import run_frechet_distance_analysis
        results = run_frechet_distance_analysis(
            mock_oracle, x_test_tensor, x_synthetic_tensor, output_dir=tmp_output_dir
        )
        assert 'frechet_distance' in results
        assert 'mu1' in results
        assert 'sigma1' in results
        assert results['frechet_distance'] >= 0

    def test_writes_pickle(self, mock_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.frechet_distance import run_frechet_distance_analysis
        run_frechet_distance_analysis(
            mock_oracle, x_test_tensor, x_synthetic_tensor, output_dir=tmp_output_dir
        )
        assert len(list(Path(tmp_output_dir).glob("frechet_distance_*.pkl"))) == 1

    def test_multi_oracle_per_dimension(self, mock_multi_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.frechet_distance import run_frechet_distance_analysis
        results = run_frechet_distance_analysis(
            mock_multi_oracle, x_test_tensor, x_synthetic_tensor,
            output_dir=tmp_output_dir, model_type='multi-oracle', per_dimension=True
        )
        assert 'frechet_distance_oracle_1' in results
        assert 'frechet_distance_oracle_2' in results
        assert 'frechet_distance_oracle_3' in results

    def test_multi_oracle_concatenated(self, mock_multi_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.frechet_distance import run_frechet_distance_analysis
        results = run_frechet_distance_analysis(
            mock_multi_oracle, x_test_tensor, x_synthetic_tensor,
            output_dir=tmp_output_dir, model_type='multi-oracle', per_dimension=False
        )
        assert 'frechet_distance' in results


# =====================================================================
# predictive_distribution_shift (pure math)
# =====================================================================

class TestPredictiveDistributionShift:
    def test_identical_distributions(self):
        from core.functional.predictive_dist_shift import predictive_distribution_shift
        np.random.seed(0)
        a = np.random.randn(200)
        ks = predictive_distribution_shift(a, a)
        assert ks == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_positive(self):
        from core.functional.predictive_dist_shift import predictive_distribution_shift
        a = np.random.randn(200)
        b = np.random.randn(200) + 5.0
        ks = predictive_distribution_shift(a, b)
        assert ks > 0

    def test_bounded_0_1(self):
        from core.functional.predictive_dist_shift import predictive_distribution_shift
        a = np.random.randn(200)
        b = np.random.randn(200)
        ks = predictive_distribution_shift(a, b)
        assert 0 <= ks <= 1


# =====================================================================
# run_predictive_distribution_shift_analysis (integration)
# =====================================================================

class TestRunPredictiveDistShiftAnalysis:
    def test_single_mode(self, mock_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.predictive_dist_shift import run_predictive_distribution_shift_analysis
        results = run_predictive_distribution_shift_analysis(
            mock_oracle, x_test_tensor, x_synthetic_tensor, output_dir=tmp_output_dir
        )
        assert 'predictive_distribution_shift_ks_statistic' in results
        assert 'y_hat_test' in results
        assert 'y_hat_syn' in results

    def test_multi_oracle_per_dimension(self, mock_multi_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.predictive_dist_shift import run_predictive_distribution_shift_analysis
        results = run_predictive_distribution_shift_analysis(
            mock_multi_oracle, x_test_tensor, x_synthetic_tensor,
            output_dir=tmp_output_dir, model_type='multi-oracle', per_dimension=True
        )
        assert 'predictive_distribution_shift_ks_statistic_oracle_1' in results
        assert 'predictive_distribution_shift_ks_statistic_oracle_2' in results
        assert 'predictive_distribution_shift_ks_statistic_oracle_3' in results

    def test_batch_mode(self, mock_oracle, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.functional.predictive_dist_shift import run_predictive_distribution_shift_analysis
        run_predictive_distribution_shift_analysis(
            mock_oracle, x_test_tensor, x_synthetic_tensor,
            output_dir=tmp_output_dir, sample_name="batch_sample_1"
        )
        assert (Path(tmp_output_dir) / "predictive_dist_shift.csv").exists()
        assert (Path(tmp_output_dir) / "predictive_dist_shift.h5").exists()
