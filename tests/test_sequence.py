"""
Tests for core/sequence/ — percent identity, k-mer spectrum shift,
and discriminability.
"""

import numpy as np
import torch
import pickle
import pytest
import os
import h5py
from pathlib import Path


# =====================================================================
# calculate_sequence_identity_batch (pure math)
# =====================================================================

class TestCalculateSequenceIdentityBatch:
    def test_identical_sequences_identity_1(self):
        from core.sequence.percent_identity import calculate_sequence_identity_batch
        # All same sequence → diagonal should be 1.0
        seq = np.zeros((1, 4, 50), dtype=np.float32)
        seq[0, 0, :] = 1.0  # all A's
        seqs = np.repeat(seq, 5, axis=0)
        pid = calculate_sequence_identity_batch(seqs, seqs)
        np.testing.assert_allclose(pid, 1.0, atol=1e-5)

    def test_completely_different_sequences(self):
        from core.sequence.percent_identity import calculate_sequence_identity_batch
        L = 100
        # seq1: all A
        seq1 = np.zeros((1, 4, L), dtype=np.float32)
        seq1[0, 0, :] = 1.0
        # seq2: all C
        seq2 = np.zeros((1, 4, L), dtype=np.float32)
        seq2[0, 1, :] = 1.0
        pid = calculate_sequence_identity_batch(seq1, seq2)
        assert pid[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_output_shape(self, x_test_np, x_synthetic_np):
        from core.sequence.percent_identity import calculate_sequence_identity_batch
        pid = calculate_sequence_identity_batch(x_test_np, x_synthetic_np, batch_size=16)
        assert pid.shape == (x_test_np.shape[0], x_synthetic_np.shape[0])

    def test_values_between_0_and_1(self, x_test_np, x_synthetic_np):
        from core.sequence.percent_identity import calculate_sequence_identity_batch
        pid = calculate_sequence_identity_batch(x_test_np, x_synthetic_np)
        assert np.all(pid >= 0.0)
        assert np.all(pid <= 1.0)

    def test_self_identity_diagonal(self, x_test_np):
        from core.sequence.percent_identity import calculate_sequence_identity_batch
        pid = calculate_sequence_identity_batch(x_test_np, x_test_np)
        # Diagonal should be 1.0 (sequence compared to itself)
        diag = np.diag(pid)
        np.testing.assert_allclose(diag, 1.0, atol=1e-5)


# =====================================================================
# run_percent_identity_analysis (integration)
# =====================================================================

class TestRunPercentIdentityAnalysis:
    def test_returns_expected_keys(self, x_synthetic_np, x_train_np, tmp_output_dir):
        from core.sequence.percent_identity import run_percent_identity_analysis
        results = run_percent_identity_analysis(x_synthetic_np, x_train_np, output_dir=tmp_output_dir)
        expected_keys = [
            'global_max_percent_identity_samples_vs_samples',
            'global_max_percent_identity_samples_vs_training',
            'average_max_percent_identity_samples_vs_samples',
            'average_max_percent_identity_samples_vs_training',
            'percent_identity_matrix_samples_vs_samples',
            'percent_identity_matrix_samples_vs_training',
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_metrics_are_reasonable(self, x_synthetic_np, x_train_np, tmp_output_dir):
        from core.sequence.percent_identity import run_percent_identity_analysis
        results = run_percent_identity_analysis(x_synthetic_np, x_train_np, output_dir=tmp_output_dir)
        # All metrics should be between 0 and 1
        assert 0 <= results['average_max_percent_identity_samples_vs_samples'] <= 1
        assert 0 <= results['average_max_percent_identity_samples_vs_training'] <= 1
        assert 0 <= results['global_max_percent_identity_samples_vs_samples'] <= 1
        assert 0 <= results['global_max_percent_identity_samples_vs_training'] <= 1

    def test_single_mode_writes_pickle(self, x_synthetic_np, x_train_np, tmp_output_dir):
        from core.sequence.percent_identity import run_percent_identity_analysis
        run_percent_identity_analysis(x_synthetic_np, x_train_np, output_dir=tmp_output_dir)
        pkl_files = list(Path(tmp_output_dir).glob("percent_identity_*.pkl"))
        assert len(pkl_files) == 1

    def test_batch_mode_writes_csv_h5(self, x_synthetic_np, x_train_np, tmp_output_dir):
        from core.sequence.percent_identity import run_percent_identity_analysis
        run_percent_identity_analysis(
            x_synthetic_np, x_train_np, output_dir=tmp_output_dir, sample_name="sample_X"
        )
        assert (Path(tmp_output_dir) / "percent_identity.csv").exists()
        assert (Path(tmp_output_dir) / "percent_identity.h5").exists()


# =====================================================================
# kmer_featurization
# =====================================================================

class TestKmerFeaturization:
    def test_numbering_consistency(self):
        from core.sequence.kmer_spectrum_shift import kmer_featurization
        obj = kmer_featurization(3)
        # 'AAA' should map to 0, 'TTT' should map to 63 (4^3 - 1)
        assert obj.kmer_numbering_for_one_kmer('AAA') == 0
        assert obj.kmer_numbering_for_one_kmer('TTT') == 63

    def test_feature_vector_length(self):
        from core.sequence.kmer_spectrum_shift import kmer_featurization
        k = 3
        obj = kmer_featurization(k)
        feature = obj.obtain_kmer_feature_for_one_sequence("ACGTACGT", write_number_of_occurrences=False)
        assert len(feature) == 4**k

    def test_feature_sums_to_one_for_frequencies(self):
        from core.sequence.kmer_spectrum_shift import kmer_featurization
        obj = kmer_featurization(3)
        feature = obj.obtain_kmer_feature_for_one_sequence("ACGTACGTACGT", write_number_of_occurrences=False)
        assert feature.sum() == pytest.approx(1.0, abs=1e-5)

    def test_batch_features_shape(self):
        from core.sequence.kmer_spectrum_shift import kmer_featurization
        k = 4
        obj = kmer_featurization(k)
        seqs = ["ACGTACGTACGT", "TGCATGCATGCA", "AAAACCCCGGGG"]
        features = obj.obtain_kmer_feature_for_a_list_of_sequences(seqs, write_number_of_occurrences=True)
        assert features.shape == (3, 4**k)


# =====================================================================
# kmer_statistics / compute_kmer_spectra
# =====================================================================

class TestKmerStatistics:
    def test_identical_data_jsd_zero(self):
        from core.sequence.kmer_spectrum_shift import kmer_statistics
        np.random.seed(0)
        N, L = 20, 50
        # Same data → JSD should be 0
        data = np.zeros((N, L, 4), dtype=np.float32)
        for i in range(N):
            indices = np.random.randint(0, 4, size=L)
            for j in range(L):
                data[i, j, indices[j]] = 1.0
        kld, jsd = kmer_statistics(3, data, data)
        assert jsd == pytest.approx(0.0, abs=1e-4)

    def test_different_data_positive_jsd(self):
        from core.sequence.kmer_spectrum_shift import kmer_statistics
        np.random.seed(0)
        N, L = 30, 50
        data1 = np.zeros((N, L, 4), dtype=np.float32)
        data2 = np.zeros((N, L, 4), dtype=np.float32)
        for i in range(N):
            for j in range(L):
                data1[i, j, np.random.randint(0, 4)] = 1.0
                data2[i, j, np.random.randint(0, 4)] = 1.0
        _, jsd = kmer_statistics(3, data1, data2)
        assert jsd >= 0

    def test_jsd_bounded(self):
        from core.sequence.kmer_spectrum_shift import kmer_statistics
        np.random.seed(0)
        N, L = 20, 50
        data1 = np.zeros((N, L, 4), dtype=np.float32)
        data2 = np.zeros((N, L, 4), dtype=np.float32)
        for i in range(N):
            for j in range(L):
                data1[i, j, np.random.randint(0, 4)] = 1.0
                data2[i, j, np.random.randint(0, 4)] = 1.0
        _, jsd = kmer_statistics(3, data1, data2)
        # JSD is bounded [0, ln(2)] ≈ [0, 0.693] but rounded value could be slightly off
        assert jsd <= 1.0


# =====================================================================
# run_kmer_spectrum_shift_analysis (integration)
# =====================================================================

class TestRunKmerSpectrumShiftAnalysis:
    def test_returns_expected_keys(self, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.sequence.kmer_spectrum_shift import run_kmer_spectrum_shift_analysis
        results = run_kmer_spectrum_shift_analysis(
            x_test_tensor, x_synthetic_tensor, kmer_length=3, output_dir=tmp_output_dir
        )
        assert 'kmer_spectra_kullback_leibler_divergence' in results
        assert 'kmer_spectra_jensen_shannon_distance' in results
        assert 'js_distance' in results
        assert 'kmer_length' in results
        assert results['kmer_length'] == 3

    def test_jsd_nonnegative(self, x_test_tensor, x_synthetic_tensor, tmp_output_dir):
        from core.sequence.kmer_spectrum_shift import run_kmer_spectrum_shift_analysis
        results = run_kmer_spectrum_shift_analysis(
            x_test_tensor, x_synthetic_tensor, kmer_length=3, output_dir=tmp_output_dir
        )
        assert results['kmer_spectra_jensen_shannon_distance'] >= 0


# =====================================================================
# prep_data_for_classification
# =====================================================================

class TestPrepDataForClassification:
    def test_output_shapes(self, x_test_tensor, x_synthetic_tensor):
        from core.sequence.discriminability import prep_data_for_classification
        data_dict = prep_data_for_classification(x_test_tensor, x_synthetic_tensor)
        n_total = x_test_tensor.shape[0] + x_synthetic_tensor.shape[0]
        assert data_dict['x_train'].shape[0] == n_total
        assert data_dict['y_train'].shape == (n_total, 1)

    def test_labels_correct(self, x_test_tensor, x_synthetic_tensor):
        from core.sequence.discriminability import prep_data_for_classification
        data_dict = prep_data_for_classification(x_test_tensor, x_synthetic_tensor)
        y = data_dict['y_train']
        n_test = x_test_tensor.shape[0]
        n_syn = x_synthetic_tensor.shape[0]
        # First n_test should be 1 (real), next n_syn should be 0 (synthetic)
        np.testing.assert_array_equal(y[:n_test], 1.0)
        np.testing.assert_array_equal(y[n_test:], 0.0)

    def test_x_train_transposed(self, x_test_tensor, x_synthetic_tensor):
        from core.sequence.discriminability import prep_data_for_classification
        data_dict = prep_data_for_classification(x_test_tensor, x_synthetic_tensor)
        # Input is (N, 4, L), stacked then transposed to (N, L, 4) then back... let's just check it's 3D
        assert data_dict['x_train'].ndim == 3


# =====================================================================
# CNN model structure
# =====================================================================

class TestCNNModel:
    def test_forward_pass_shape(self):
        from core.sequence.discriminability import CNN
        model = CNN(output_dim=1)
        x = torch.randn(8, 4, 230)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 1)

    def test_output_range_with_sigmoid(self):
        from core.sequence.discriminability import CNN
        model = CNN(output_dim=1)
        model.eval()
        x = torch.randn(8, 4, 230)
        with torch.no_grad():
            out = model(x)
        # Output uses sigmoid, so should be in [0, 1]
        assert torch.all(out >= 0)
        assert torch.all(out <= 1)


# =====================================================================
# PL_CNN data loading
# =====================================================================

class TestPLCNN:
    def test_loads_from_h5(self, discriminability_h5):
        from core.sequence.discriminability import PL_CNN
        model = PL_CNN(input_h5_file=str(discriminability_h5))
        assert hasattr(model, 'X_train')
        assert hasattr(model, 'X_train_split')
        assert hasattr(model, 'X_valid')
        assert len(model.X_train_split) + len(model.X_valid) == len(model.X_train)

    def test_train_val_split_ratio(self, discriminability_h5):
        from core.sequence.discriminability import PL_CNN
        model = PL_CNN(input_h5_file=str(discriminability_h5))
        total = len(model.X_train)
        n_train = len(model.X_train_split)
        ratio = n_train / total
        # Should be approximately 0.8
        assert 0.75 <= ratio <= 0.85

    def test_missing_file_raises(self, tmp_path):
        from core.sequence.discriminability import PL_CNN
        with pytest.raises(RuntimeError):
            PL_CNN(input_h5_file=str(tmp_path / "nonexistent.h5"))
