"""
Tests for core/sequence/ analyses:
  - percent_identity
  - kmer_spectrum_shift
  - discriminability
"""

import numpy as np
import torch
import pytest
import os
import h5py

from tests.conftest import _random_onehot, _random_onehot_with_padding, DEEPSTARR_SEQ_LEN


# ===================================================================
# percent_identity
# ===================================================================

class TestPercentIdentity:
    """Tests for the normalized Hamming distance identity metric."""

    def test_identical_sequences_give_100_percent(self):
        """A sequence compared to itself must have 100% identity."""
        from core.sequence.percent_identity import calculate_sequence_identity_batch

        np.random.seed(42)
        seqs = _random_onehot(10, DEEPSTARR_SEQ_LEN)
        identity = calculate_sequence_identity_batch(seqs, seqs)
        # diagonal should be 1.0
        np.testing.assert_allclose(np.diag(identity), 1.0, atol=1e-6)

    def test_identity_values_in_zero_one(self):
        """All percent identity values must be in [0, 1]."""
        from core.sequence.percent_identity import calculate_sequence_identity_batch

        np.random.seed(42)
        source = _random_onehot(20, DEEPSTARR_SEQ_LEN)
        target = _random_onehot(30, DEEPSTARR_SEQ_LEN)
        identity = calculate_sequence_identity_batch(source, target)
        assert np.all(identity >= 0.0)
        assert np.all(identity <= 1.0)

    def test_identity_matrix_shape(self):
        """Output shape must be (n_source, n_target)."""
        from core.sequence.percent_identity import calculate_sequence_identity_batch

        np.random.seed(42)
        source = _random_onehot(15, DEEPSTARR_SEQ_LEN)
        target = _random_onehot(25, DEEPSTARR_SEQ_LEN)
        identity = calculate_sequence_identity_batch(source, target)
        assert identity.shape == (15, 25)

    def test_identity_symmetry(self):
        """identity(A, B)[i, j] == identity(B, A)[j, i]."""
        from core.sequence.percent_identity import calculate_sequence_identity_batch

        np.random.seed(42)
        a = _random_onehot(10, DEEPSTARR_SEQ_LEN)
        b = _random_onehot(10, DEEPSTARR_SEQ_LEN)
        id_ab = calculate_sequence_identity_batch(a, b)
        id_ba = calculate_sequence_identity_batch(b, a)
        np.testing.assert_allclose(id_ab, id_ba.T, atol=1e-6)

    def test_run_analysis_returns_expected_keys(self, deepstarr_tensors, tmp_output_dir):
        """Full analysis must return all summary statistics."""
        from core.sequence.percent_identity import run_percent_identity_analysis

        _, x_syn, x_train = deepstarr_tensors
        results = run_percent_identity_analysis(x_syn, x_train, output_dir=tmp_output_dir)
        expected_keys = [
            "average_max_percent_identity_samples_vs_training",
            "average_max_percent_identity_samples_vs_samples",
        ]
        for key in expected_keys:
            assert key in results

    def test_run_analysis_saves_pickle(self, deepstarr_tensors, tmp_output_dir):
        """Single mode must write a .pkl file."""
        from core.sequence.percent_identity import run_percent_identity_analysis

        _, x_syn, x_train = deepstarr_tensors
        run_percent_identity_analysis(x_syn, x_train, output_dir=tmp_output_dir)
        pkl_files = [f for f in os.listdir(tmp_output_dir) if f.endswith(".pkl")]
        assert len(pkl_files) == 1

    def test_run_analysis_batch_mode(self, deepstarr_tensors, tmp_output_dir):
        """Batch mode must write CSV and H5."""
        from core.sequence.percent_identity import run_percent_identity_analysis

        _, x_syn, x_train = deepstarr_tensors
        run_percent_identity_analysis(
            x_syn, x_train, output_dir=tmp_output_dir, sample_name="s1"
        )
        assert os.path.exists(os.path.join(tmp_output_dir, "percent_identity.csv"))
        assert os.path.exists(os.path.join(tmp_output_dir, "percent_identity.h5"))


# ===================================================================
# kmer_spectrum_shift
# ===================================================================

class TestKmerSpectrumShift:
    """Tests for the k-mer frequency distribution shift metric."""

    def test_identical_sequences_give_zero_jsd(self):
        """JSD between identical k-mer distributions must be ~0."""
        from core.sequence.kmer_spectrum_shift import kmer_statistics

        np.random.seed(42)
        seqs = _random_onehot(50, DEEPSTARR_SEQ_LEN, fmt="NLA")
        kld, jsd = kmer_statistics(3, seqs, seqs)
        assert jsd == pytest.approx(0.0, abs=1e-6)
        assert kld == pytest.approx(0.0, abs=1e-6)

    def test_different_sequences_give_positive_jsd(self):
        """JSD between different distributions must be > 0."""
        from core.sequence.kmer_spectrum_shift import kmer_statistics

        np.random.seed(42)
        # Use very different sequences by biasing nucleotide composition
        seqs1 = _random_onehot(50, DEEPSTARR_SEQ_LEN, fmt="NLA")
        # Create biased sequences (mostly A's)
        seqs2 = np.zeros((50, DEEPSTARR_SEQ_LEN, 4), dtype=np.float32)
        seqs2[:, :, 0] = 1.0
        kld, jsd = kmer_statistics(3, seqs1, seqs2)
        assert jsd > 0

    def test_compute_kmer_spectra_sums_to_one(self):
        """Normalized k-mer spectrum must sum to 1."""
        from core.sequence.kmer_spectrum_shift import compute_kmer_spectra

        np.random.seed(42)
        seqs = _random_onehot(30, DEEPSTARR_SEQ_LEN, fmt="NLA")
        spectrum = compute_kmer_spectra(seqs, kmer_length=3)
        assert spectrum.sum() == pytest.approx(1.0, abs=1e-6)

    def test_compute_kmer_spectra_length(self):
        """Spectrum vector length must be 4^k."""
        from core.sequence.kmer_spectrum_shift import compute_kmer_spectra

        np.random.seed(42)
        seqs = _random_onehot(20, DEEPSTARR_SEQ_LEN, fmt="NLA")
        for k in [3, 4, 6]:
            spectrum = compute_kmer_spectra(seqs, kmer_length=k)
            assert len(spectrum) == 4**k

    def test_compute_kmer_spectra_handles_padding(self):
        """Sequences with zero-padded positions should not crash."""
        from core.sequence.kmer_spectrum_shift import compute_kmer_spectra

        np.random.seed(42)
        seqs = _random_onehot_with_padding(20, DEEPSTARR_SEQ_LEN, pad_len=50, fmt="NLA")
        spectrum = compute_kmer_spectra(seqs, kmer_length=3)
        assert spectrum.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.isfinite(spectrum))

    def test_kmer_featurization_single_sequence(self):
        """kmer_featurization must count occurrences correctly for a simple case."""
        from core.sequence.kmer_spectrum_shift import kmer_featurization

        obj = kmer_featurization(2)
        features = obj.obtain_kmer_feature_for_one_sequence("ACGT", write_number_of_occurrences=True)
        # 2-mers: AC, CG, GT → 3 distinct 2-mers
        assert features.sum() == 3
        # Check specific k-mer: "AC" should be at numbering 0*4+1 = 1
        ac_idx = obj.kmer_numbering_for_one_kmer("AC")
        assert features[ac_idx] == 1

    def test_kmer_numbering_uniqueness(self):
        """Each k-mer must map to a unique index."""
        from core.sequence.kmer_spectrum_shift import kmer_featurization
        from itertools import product as iter_product

        obj = kmer_featurization(3)
        all_kmers = ["".join(p) for p in iter_product("ACGT", repeat=3)]
        indices = [obj.kmer_numbering_for_one_kmer(k) for k in all_kmers]
        assert len(set(indices)) == 64  # 4^3 unique indices

    def test_run_analysis_returns_expected_keys(self, deepstarr_tensors, tmp_output_dir):
        """Full analysis must return dict with KLD, JSD, and aliases."""
        from core.sequence.kmer_spectrum_shift import run_kmer_spectrum_shift_analysis

        x_test, x_syn, _ = deepstarr_tensors
        results = run_kmer_spectrum_shift_analysis(
            x_test, x_syn, kmer_length=3, output_dir=tmp_output_dir
        )
        for key in [
            "kmer_spectra_kullback_leibler_divergence",
            "kmer_spectra_jensen_shannon_distance",
            "js_distance",
            "kmer_length",
        ]:
            assert key in results
        assert results["kmer_length"] == 3

    def test_run_analysis_saves_pickle(self, deepstarr_tensors, tmp_output_dir):
        """Single mode must write a .pkl file."""
        from core.sequence.kmer_spectrum_shift import run_kmer_spectrum_shift_analysis

        x_test, x_syn, _ = deepstarr_tensors
        run_kmer_spectrum_shift_analysis(
            x_test, x_syn, kmer_length=3, output_dir=tmp_output_dir
        )
        pkl_files = [f for f in os.listdir(tmp_output_dir) if f.endswith(".pkl")]
        assert len(pkl_files) == 1


# ===================================================================
# discriminability
# ===================================================================

class TestDiscriminability:
    """Tests for the binary classifier AUROC discriminability metric."""

    def test_prep_data_shapes(self, deepstarr_tensors):
        """prep_data_for_classification must return correct shapes and labels."""
        from core.sequence.discriminability import prep_data_for_classification

        x_test, x_syn, _ = deepstarr_tensors
        data_dict = prep_data_for_classification(x_test, x_syn)
        x_train = data_dict["x_train"]
        y_train = data_dict["y_train"]
        n_total = len(x_test) + len(x_syn)
        assert x_train.shape[0] == n_total
        assert y_train.shape[0] == n_total
        # x_train should be (N, L, 4) format after transpose
        assert x_train.shape[2] == 4

    def test_prep_data_labels_balanced(self, deepstarr_tensors):
        """Labels must have equal counts of 0 and 1 when inputs are same size."""
        from core.sequence.discriminability import prep_data_for_classification

        x_test, x_syn, _ = deepstarr_tensors
        data_dict = prep_data_for_classification(x_test, x_syn)
        y = data_dict["y_train"]
        n_real = (y == 1).sum()
        n_syn = (y == 0).sum()
        assert n_real == len(x_test)
        assert n_syn == len(x_syn)

    def test_prep_data_writes_to_h5(self, deepstarr_tensors, tmp_path):
        """Writing prep data to H5 should produce a valid file."""
        from core.sequence.discriminability import prep_data_for_classification
        from utils.helpers import write_to_h5

        x_test, x_syn, _ = deepstarr_tensors
        data_dict = prep_data_for_classification(x_test, x_syn)
        h5_path = str(tmp_path / "disc.h5")
        write_to_h5(h5_path, data_dict)
        assert os.path.exists(h5_path)
        with h5py.File(h5_path, "r") as f:
            assert "x_train" in f
            assert "y_train" in f
