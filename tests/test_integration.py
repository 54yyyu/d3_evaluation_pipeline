"""
Integration tests for the full pipeline:
  - End-to-end single-sample mode (each analysis type)
  - Batch mode output structure
  - Result file contents and consistency
  - Cross-analysis result aggregation
"""

import numpy as np
import torch
import pytest
import os
import pickle
import h5py

from tests.conftest import _random_onehot, DEEPSTARR_SEQ_LEN, MockDeepSTARR


# ===================================================================
# Single-sample end-to-end
# ===================================================================

class TestSingleSamplePipeline:
    """End-to-end tests for single-sample analysis mode."""

    def test_cond_gen_fidelity_pickle_contents(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Pickle file must contain loadable results with correct key."""
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis

        x_test, x_syn, _ = deepstarr_tensors
        run_conditional_generation_fidelity_analysis(
            deepstarr_model, x_test, x_syn, output_dir=tmp_output_dir
        )
        pkl_files = [f for f in os.listdir(tmp_output_dir) if f.endswith(".pkl")]
        with open(os.path.join(tmp_output_dir, pkl_files[0]), "rb") as f:
            data = pickle.load(f)
        assert "conditional_generation_fidelity_mse" in data
        assert isinstance(data["conditional_generation_fidelity_mse"], (float, np.floating))

    def test_percent_identity_pickle_contents(self, deepstarr_tensors, tmp_output_dir):
        """Pickle must contain identity matrices and summary stats."""
        from core.sequence.percent_identity import run_percent_identity_analysis

        _, x_syn, x_train = deepstarr_tensors
        run_percent_identity_analysis(x_syn, x_train, output_dir=tmp_output_dir)
        pkl_files = [f for f in os.listdir(tmp_output_dir) if f.endswith(".pkl")]
        with open(os.path.join(tmp_output_dir, pkl_files[0]), "rb") as f:
            data = pickle.load(f)
        assert "average_max_percent_identity_samples_vs_training" in data
        # Value should be between 0 and 1
        val = data["average_max_percent_identity_samples_vs_training"]
        assert 0 <= val <= 1

    def test_kmer_pickle_contents(self, deepstarr_tensors, tmp_output_dir):
        """Pickle must contain JSD, KLD, and kmer_length."""
        from core.sequence.kmer_spectrum_shift import run_kmer_spectrum_shift_analysis

        x_test, x_syn, _ = deepstarr_tensors
        run_kmer_spectrum_shift_analysis(
            x_test, x_syn, kmer_length=3, output_dir=tmp_output_dir
        )
        pkl_files = [f for f in os.listdir(tmp_output_dir) if f.endswith(".pkl")]
        with open(os.path.join(tmp_output_dir, pkl_files[0]), "rb") as f:
            data = pickle.load(f)
        assert data["kmer_length"] == 3
        assert data["js_distance"] >= 0

    def test_predictive_dist_shift_pickle_contains_predictions(
        self, deepstarr_model, deepstarr_tensors, tmp_output_dir
    ):
        """Pickle must include the raw predictions arrays."""
        from core.functional.predictive_dist_shift import run_predictive_distribution_shift_analysis

        x_test, x_syn, _ = deepstarr_tensors
        run_predictive_distribution_shift_analysis(
            deepstarr_model, x_test, x_syn, output_dir=tmp_output_dir
        )
        pkl_files = [f for f in os.listdir(tmp_output_dir) if f.endswith(".pkl")]
        with open(os.path.join(tmp_output_dir, pkl_files[0]), "rb") as f:
            data = pickle.load(f)
        assert data["y_hat_test"].shape[0] == len(x_test)
        assert data["y_hat_syn"].shape[0] == len(x_syn)


# ===================================================================
# Batch mode end-to-end
# ===================================================================

class TestBatchModePipeline:
    """End-to-end tests for batch mode output structure."""

    def test_batch_csv_accumulates_samples(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """Running analysis for multiple samples should accumulate columns in CSV."""
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis

        x_test, x_syn, _ = deepstarr_tensors
        for name in ["sample_A", "sample_B", "sample_C"]:
            run_conditional_generation_fidelity_analysis(
                deepstarr_model, x_test, x_syn,
                output_dir=tmp_output_dir, sample_name=name
            )
        csv_path = os.path.join(tmp_output_dir, "cond_gen_fidelity.csv")
        assert os.path.exists(csv_path)
        import csv
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        # All 3 sample names should appear in header
        for name in ["sample_A", "sample_B", "sample_C"]:
            assert name in header

    def test_batch_h5_has_all_sample_groups(self, deepstarr_model, deepstarr_tensors, tmp_output_dir):
        """H5 file should have a group per sample."""
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis

        x_test, x_syn, _ = deepstarr_tensors
        for name in ["s1", "s2"]:
            run_conditional_generation_fidelity_analysis(
                deepstarr_model, x_test, x_syn,
                output_dir=tmp_output_dir, sample_name=name
            )
        h5_path = os.path.join(tmp_output_dir, "cond_gen_fidelity.h5")
        with h5py.File(h5_path, "r") as f:
            assert "s1" in f
            assert "s2" in f

    def test_batch_percent_identity_csv(self, deepstarr_tensors, tmp_output_dir):
        """percent_identity batch mode should produce CSV with the key metric."""
        from core.sequence.percent_identity import run_percent_identity_analysis

        _, x_syn, x_train = deepstarr_tensors
        run_percent_identity_analysis(
            x_syn, x_train, output_dir=tmp_output_dir, sample_name="batch_s1"
        )
        csv_path = os.path.join(tmp_output_dir, "percent_identity.csv")
        assert os.path.exists(csv_path)

    def test_batch_kmer_csv(self, deepstarr_tensors, tmp_output_dir):
        """kmer batch mode should produce CSV."""
        from core.sequence.kmer_spectrum_shift import run_kmer_spectrum_shift_analysis

        x_test, x_syn, _ = deepstarr_tensors
        run_kmer_spectrum_shift_analysis(
            x_test, x_syn, kmer_length=3,
            output_dir=tmp_output_dir, sample_name="batch_s1"
        )
        csv_path = os.path.join(tmp_output_dir, "kmer_spectrum_shift.csv")
        assert os.path.exists(csv_path)


# ===================================================================
# Cross-analysis consistency
# ===================================================================

class TestCrossAnalysisConsistency:
    """Tests that verify results are consistent across runs."""

    def test_deterministic_mse(self, deepstarr_model, deepstarr_tensors, tmp_path):
        """Same inputs should produce identical MSE across two runs."""
        from core.functional.cond_gen_fidelity import run_conditional_generation_fidelity_analysis

        x_test, x_syn, _ = deepstarr_tensors
        dir1 = str(tmp_path / "run1")
        dir2 = str(tmp_path / "run2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        r1 = run_conditional_generation_fidelity_analysis(
            deepstarr_model, x_test, x_syn, output_dir=dir1
        )
        r2 = run_conditional_generation_fidelity_analysis(
            deepstarr_model, x_test, x_syn, output_dir=dir2
        )
        assert r1["conditional_generation_fidelity_mse"] == pytest.approx(
            r2["conditional_generation_fidelity_mse"]
        )

    def test_deterministic_kmer(self, deepstarr_tensors, tmp_path):
        """Same inputs should produce identical k-mer JSD across two runs."""
        from core.sequence.kmer_spectrum_shift import run_kmer_spectrum_shift_analysis

        x_test, x_syn, _ = deepstarr_tensors
        dir1 = str(tmp_path / "run1")
        dir2 = str(tmp_path / "run2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        r1 = run_kmer_spectrum_shift_analysis(x_test, x_syn, kmer_length=3, output_dir=dir1)
        r2 = run_kmer_spectrum_shift_analysis(x_test, x_syn, kmer_length=3, output_dir=dir2)
        assert r1["js_distance"] == pytest.approx(r2["js_distance"])

    def test_deterministic_percent_identity(self, deepstarr_tensors, tmp_path):
        """Same inputs should produce identical percent identity."""
        from core.sequence.percent_identity import run_percent_identity_analysis

        _, x_syn, x_train = deepstarr_tensors
        dir1 = str(tmp_path / "run1")
        dir2 = str(tmp_path / "run2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        r1 = run_percent_identity_analysis(x_syn, x_train, output_dir=dir1)
        r2 = run_percent_identity_analysis(x_syn, x_train, output_dir=dir2)
        assert r1["average_max_percent_identity_samples_vs_training"] == pytest.approx(
            r2["average_max_percent_identity_samples_vs_training"]
        )
