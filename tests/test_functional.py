"""
Tests for core/functional/ analyses:
  - conditional_generation_fidelity
  - frechet_distance
  - predictive_dist_shift
"""

import numpy as np
import torch
import pytest
import os
import pickle


# ===================================================================
# conditional_generation_fidelity
# ===================================================================

class TestConditionalGenerationFidelity:
    """Tests for the MSE-based fidelity metric."""

    def test_identical_predictions_give_zero_mse(self):
        """MSE between identical arrays must be exactly 0."""
        from core.functional.cond_gen_fidelity import conditional_generation_fidelity

        a = np.random.randn(100, 2).astype(np.float32)
        assert conditional_generation_fidelity(a, a) == pytest.approx(0.0)

    def test_known_mse_value(self):
        """Verify MSE against a hand-calculated example."""
        from core.functional.cond_gen_fidelity import conditional_generation_fidelity

        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 2.0], [1.0, 2.0]])
        # differences: [0, 0, 2, 2], squares: [0, 0, 4, 4], mean = 2.0
        assert conditional_generation_fidelity(a, b) == pytest.approx(2.0)

    def test_mse_is_non_negative(self):
        """MSE must always be >= 0."""
        from core.functional.cond_gen_fidelity import conditional_generation_fidelity

        a = np.random.randn(50, 2).astype(np.float32)
        b = np.random.randn(50, 2).astype(np.float32)
        assert conditional_generation_fidelity(a, b) >= 0

    def test_mse_symmetry(self):
        """MSE(a, b) == MSE(b, a)."""
        from core.functional.cond_gen_fidelity import conditional_generation_fidelity

        a = np.random.randn(50, 2).astype(np.float32)
        b = np.random.randn(50, 2).astype(np.float32)
        assert conditional_generation_fidelity(a, b) == pytest.approx(
            conditional_generation_fidelity(b, a)
        )

    def test_run_analysis_returns_expected_keys(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Full analysis must return dict with 'conditional_generation_fidelity_mse'."""
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis

        x_test, x_syn, _ = deepstarr_tensors
        results = run_conditional_generation_fidelity_analysis(
            deepstarr_model, x_test, x_syn, output_dir=tmp_output_dir
        )
        assert "conditional_generation_fidelity_mse" in results
        assert isinstance(results["conditional_generation_fidelity_mse"], (float, np.floating))

    def test_run_analysis_saves_pickle(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Single-sample mode must write a .pkl file."""
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis

        x_test, x_syn, _ = deepstarr_tensors
        run_conditional_generation_fidelity_analysis(
            deepstarr_model, x_test, x_syn, output_dir=tmp_output_dir
        )
        pkl_files = [f for f in os.listdir(tmp_output_dir) if f.endswith(".pkl")]
        assert len(pkl_files) == 1

    def test_run_analysis_batch_mode_writes_csv_and_h5(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Batch mode must write CSV and H5 files."""
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis

        x_test, x_syn, _ = deepstarr_tensors
        run_conditional_generation_fidelity_analysis(
            deepstarr_model, x_test, x_syn,
            output_dir=tmp_output_dir, sample_name="test_sample"
        )
        assert os.path.exists(os.path.join(tmp_output_dir, "cond_gen_fidelity.csv"))
        assert os.path.exists(os.path.join(tmp_output_dir, "cond_gen_fidelity.h5"))

    def test_run_analysis_with_model_type_deepstarr(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Explicit model_type='deepstarr' should work."""
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis

        x_test, x_syn, _ = deepstarr_tensors
        results = run_conditional_generation_fidelity_analysis(
            deepstarr_model, x_test, x_syn,
            output_dir=tmp_output_dir, model_type="deepstarr"
        )
        assert results["conditional_generation_fidelity_mse"] >= 0


# ===================================================================
# frechet_distance
# ===================================================================

class TestFrechetDistance:
    """Tests for the Fréchet distance metric."""

    def test_identical_embeddings_give_zero_distance(self):
        """FD between identical embedding sets must be ~0."""
        from core.functional.frechet_distance import (
            calculate_activation_statistics,
            calculate_frechet_distance,
        )

        emb = torch.randn(100, 16)
        mu1, sig1 = calculate_activation_statistics(emb)
        mu2, sig2 = calculate_activation_statistics(emb)
        fd = calculate_frechet_distance(mu1, sig1, mu2, sig2)
        assert fd == pytest.approx(0.0, abs=1e-4)

    def test_different_embeddings_give_positive_distance(self):
        """FD between different distributions must be > 0."""
        from core.functional.frechet_distance import (
            calculate_activation_statistics,
            calculate_frechet_distance,
        )

        emb1 = torch.randn(100, 16)
        emb2 = torch.randn(100, 16) + 5.0  # shifted distribution
        mu1, sig1 = calculate_activation_statistics(emb1)
        mu2, sig2 = calculate_activation_statistics(emb2)
        fd = calculate_frechet_distance(mu1, sig1, mu2, sig2)
        assert fd > 0

    def test_activation_statistics_shapes(self):
        """mu should be (d,) and sigma should be (d, d)."""
        from core.functional.frechet_distance import calculate_activation_statistics

        d = 16
        emb = torch.randn(50, d)
        mu, sigma = calculate_activation_statistics(emb)
        assert mu.shape == (d,)
        assert sigma.shape == (d, d)

    def test_frechet_distance_symmetry(self):
        """FD(A, B) == FD(B, A)."""
        from core.functional.frechet_distance import (
            calculate_activation_statistics,
            calculate_frechet_distance,
        )

        emb1 = torch.randn(80, 16)
        emb2 = torch.randn(80, 16) + 2.0
        mu1, sig1 = calculate_activation_statistics(emb1)
        mu2, sig2 = calculate_activation_statistics(emb2)
        fd_ab = calculate_frechet_distance(mu1, sig1, mu2, sig2)
        fd_ba = calculate_frechet_distance(mu2, sig2, mu1, sig1)
        assert fd_ab == pytest.approx(fd_ba, rel=1e-4)

    def test_run_analysis_returns_expected_keys(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Full analysis must return dict with FD and statistics."""
        from core.functional.frechet_distance import run_frechet_distance_analysis

        x_test, x_syn, _ = deepstarr_tensors
        results = run_frechet_distance_analysis(
            deepstarr_model, x_test, x_syn, output_dir=tmp_output_dir
        )
        for key in ["frechet_distance", "mu1", "sigma1", "mu2", "sigma2"]:
            assert key in results

    def test_run_analysis_saves_pickle(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Single-sample mode must write a .pkl file."""
        from core.functional.frechet_distance import run_frechet_distance_analysis

        x_test, x_syn, _ = deepstarr_tensors
        run_frechet_distance_analysis(
            deepstarr_model, x_test, x_syn, output_dir=tmp_output_dir
        )
        pkl_files = [f for f in os.listdir(tmp_output_dir) if f.endswith(".pkl")]
        assert len(pkl_files) == 1


# ===================================================================
# predictive_distribution_shift
# ===================================================================

class TestPredictiveDistributionShift:
    """Tests for the KS-statistic-based distribution shift metric."""

    def test_identical_distributions_give_zero_ks(self):
        """KS stat between identical samples should be ~0."""
        from core.functional.predictive_dist_shift import predictive_distribution_shift

        y = np.random.randn(200, 2).astype(np.float32)
        ks = predictive_distribution_shift(y, y)
        assert ks == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distributions_give_positive_ks(self):
        """KS stat between different distributions should be > 0."""
        from core.functional.predictive_dist_shift import predictive_distribution_shift

        y1 = np.random.randn(200, 2).astype(np.float32)
        y2 = (np.random.randn(200, 2) + 5.0).astype(np.float32)
        ks = predictive_distribution_shift(y1, y2)
        assert ks > 0

    def test_ks_stat_bounded_zero_one(self):
        """KS statistic should be in [0, 1]."""
        from core.functional.predictive_dist_shift import predictive_distribution_shift

        y1 = np.random.randn(100, 2).astype(np.float32)
        y2 = np.random.randn(100, 2).astype(np.float32)
        ks = predictive_distribution_shift(y1, y2)
        assert 0 <= ks <= 1.0

    def test_run_analysis_returns_expected_keys(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Full analysis must return dict with KS statistic and predictions."""
        from core.functional.predictive_dist_shift import run_predictive_distribution_shift_analysis

        x_test, x_syn, _ = deepstarr_tensors
        results = run_predictive_distribution_shift_analysis(
            deepstarr_model, x_test, x_syn, output_dir=tmp_output_dir
        )
        assert "predictive_distribution_shift_ks_statistic" in results
        assert "y_hat_test" in results
        assert "y_hat_syn" in results

    def test_run_analysis_saves_pickle(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Single-sample mode must write a .pkl file."""
        from core.functional.predictive_dist_shift import run_predictive_distribution_shift_analysis

        x_test, x_syn, _ = deepstarr_tensors
        run_predictive_distribution_shift_analysis(
            deepstarr_model, x_test, x_syn, output_dir=tmp_output_dir
        )
        pkl_files = [f for f in os.listdir(tmp_output_dir) if f.endswith(".pkl")]
        assert len(pkl_files) == 1

    def test_run_analysis_batch_mode(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Batch mode must write CSV and H5."""
        from core.functional.predictive_dist_shift import run_predictive_distribution_shift_analysis

        x_test, x_syn, _ = deepstarr_tensors
        run_predictive_distribution_shift_analysis(
            deepstarr_model, x_test, x_syn,
            output_dir=tmp_output_dir, sample_name="s1"
        )
        assert os.path.exists(os.path.join(tmp_output_dir, "predictive_dist_shift.csv"))
        assert os.path.exists(os.path.join(tmp_output_dir, "predictive_dist_shift.h5"))
