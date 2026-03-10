"""
Tests for utils/helpers.py:
  - Data format detection and loading
  - Tensor conversion
  - Prediction utilities
  - Sequence processing utilities
  - File I/O
"""

import numpy as np
import torch
import pytest
import os
import h5py

from tests.conftest import (
    _random_onehot,
    _random_onehot_with_padding,
    DEEPSTARR_SEQ_LEN,
    LENTIMPRA_SEQ_LEN,
    SEI_SEQ_LEN,
)


# ===================================================================
# Data format detection
# ===================================================================

class TestDetectDataFormat:
    """Tests for detect_data_format()."""

    def test_detect_npz(self, deepstarr_data_npz):
        from utils.helpers import detect_data_format
        assert detect_data_format(deepstarr_data_npz) == "npz"

    def test_detect_h5(self, deepstarr_data_h5):
        from utils.helpers import detect_data_format
        assert detect_data_format(deepstarr_data_h5) == "h5"

    def test_detect_nonexistent_returns_unknown(self, tmp_path):
        from utils.helpers import detect_data_format
        assert detect_data_format(str(tmp_path / "nope.xyz")) == "unknown"


# ===================================================================
# extract_data (DeepSTARR format)
# ===================================================================

class TestExtractData:
    """Tests for extract_data() — DeepSTARR format loading."""

    def test_extract_from_h5(self, sample_npz_file, deepstarr_data_h5):
        from utils.helpers import extract_data

        x_test, x_syn, x_train = extract_data(sample_npz_file, deepstarr_data_h5)
        assert x_test.shape[1] == 4  # (N, 4, L)
        assert x_syn.shape[1] == 4
        assert x_train.shape[1] == 4
        assert x_test.shape[2] == DEEPSTARR_SEQ_LEN

    def test_extract_from_npz(self, sample_npz_file, deepstarr_data_npz):
        from utils.helpers import extract_data

        x_test, x_syn, x_train = extract_data(sample_npz_file, deepstarr_data_npz)
        assert x_test.shape[1] == 4
        assert x_syn.shape[1] == 4
        assert x_train.shape[1] == 4

    def test_extract_h5_samples(self, sample_h5_file, deepstarr_data_h5):
        """Samples in H5 format should also work."""
        from utils.helpers import extract_data

        x_test, x_syn, x_train = extract_data(sample_h5_file, deepstarr_data_h5)
        assert x_syn.shape[1] == 4

    def test_extract_data_shapes_consistent(self, sample_npz_file, deepstarr_data_h5):
        """All returned arrays must have same seq length dim."""
        from utils.helpers import extract_data

        x_test, x_syn, x_train = extract_data(sample_npz_file, deepstarr_data_h5)
        assert x_test.shape[2] == x_syn.shape[2] == x_train.shape[2]


# ===================================================================
# extract_lentimpra_data
# ===================================================================

class TestExtractLentimpradata:
    """Tests for extract_lentimpra_data()."""

    def test_extract_onehot_format(self, tmp_path):
        """Should handle onehot_test/onehot_train naming."""
        from utils.helpers import extract_lentimpra_data

        np.random.seed(42)
        # Create sample file
        sample_seqs = _random_onehot(30, LENTIMPRA_SEQ_LEN, fmt="NLA")
        sample_path = str(tmp_path / "samples.npz")
        np.savez(sample_path, arr_0=sample_seqs)

        # Create data file with onehot_test/onehot_train
        x_test = _random_onehot(40, LENTIMPRA_SEQ_LEN, fmt="NLA")  # (N, 230, 4)
        x_train = _random_onehot(80, LENTIMPRA_SEQ_LEN, fmt="NLA")
        data_path = str(tmp_path / "data.h5")
        with h5py.File(data_path, "w") as f:
            f.create_dataset("onehot_test", data=x_test)
            f.create_dataset("onehot_train", data=x_train)

        x_test_out, x_syn_out, x_train_out = extract_lentimpra_data(sample_path, data_path)
        # Must be in (N, 4, 230) format
        assert x_test_out.shape[1] == 4
        assert x_test_out.shape[2] == LENTIMPRA_SEQ_LEN
        assert x_train_out.shape[1] == 4


# ===================================================================
# extract_sei_data
# ===================================================================

class TestExtractSeiData:
    """Tests for extract_sei_data() — SEI format with padding."""

    def test_extract_promoter_format(self, tmp_path):
        """Promoter format: (N, seq_len, 6) with padding to 4096."""
        from utils.helpers import extract_sei_data

        np.random.seed(42)
        seq_len = 200

        # Create sample file
        sample_seqs = _random_onehot(10, seq_len, fmt="NLA")
        sample_path = str(tmp_path / "samples.npz")
        np.savez(sample_path, arr_0=sample_seqs)

        # Create promoter data (N, seq_len, 6)
        test_data = np.zeros((20, seq_len, 6), dtype=np.float32)
        train_data = np.zeros((40, seq_len, 6), dtype=np.float32)
        for arr in [test_data, train_data]:
            indices = np.random.randint(0, 4, size=(arr.shape[0], seq_len))
            for i in range(arr.shape[0]):
                arr[i, np.arange(seq_len), indices[i]] = 1.0
            arr[:, :, 4:] = np.random.randn(arr.shape[0], seq_len, 2).astype(np.float32)
        data_path = str(tmp_path / "data.npz")
        np.savez(data_path, train=train_data, test=test_data, valid=test_data)

        x_test, x_syn, x_train = extract_sei_data(sample_path, data_path)
        # All outputs must be padded to 4096
        assert x_test.shape == (20, 4, SEI_SEQ_LEN)
        assert x_syn.shape[2] == SEI_SEQ_LEN
        assert x_train.shape == (40, 4, SEI_SEQ_LEN)

    def test_padding_uses_uniform_background(self, tmp_path):
        """Padded regions must use 0.25 uniform values."""
        from utils.helpers import extract_sei_data

        np.random.seed(42)
        seq_len = 100  # much shorter than 4096

        sample_seqs = _random_onehot(5, seq_len, fmt="NLA")
        sample_path = str(tmp_path / "samples.npz")
        np.savez(sample_path, arr_0=sample_seqs)

        data = np.zeros((10, seq_len, 6), dtype=np.float32)
        indices = np.random.randint(0, 4, size=(10, seq_len))
        for i in range(10):
            data[i, np.arange(seq_len), indices[i]] = 1.0
        data_path = str(tmp_path / "data.npz")
        np.savez(data_path, train=data, test=data, valid=data)

        x_test, _, _ = extract_sei_data(sample_path, data_path)
        # Check padding region (first few positions should be 0.25)
        pad_left = (SEI_SEQ_LEN - seq_len) // 2
        if pad_left > 0:
            np.testing.assert_allclose(x_test[0, :, :pad_left], 0.25, atol=1e-6)


# ===================================================================
# Tensor conversion
# ===================================================================

class TestNumpyToTensor:
    """Tests for numpy_to_tensor()."""

    def test_converts_to_float32_tensor(self):
        from utils.helpers import numpy_to_tensor

        arr = np.random.randn(10, 4, 100).astype(np.float64)
        t = numpy_to_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32

    def test_preserves_shape(self):
        from utils.helpers import numpy_to_tensor

        arr = np.random.randn(5, 4, 249).astype(np.float32)
        t = numpy_to_tensor(arr)
        assert t.shape == (5, 4, 249)


# ===================================================================
# Sequence processing utilities
# ===================================================================

class TestSequenceProcessing:
    """Tests for sequence conversion and filtering utilities."""

    def test_put_deepstarr_into_NLA_transposes(self):
        """Must transpose from (N, 4, L) to (N, L, 4)."""
        from utils.helpers import put_deepstarr_into_NLA

        x1 = torch.randn(10, 4, 249)
        x2 = torch.randn(10, 4, 249)
        out1, out2 = put_deepstarr_into_NLA(x1, x2)
        assert out1.shape == (10, 249, 4)
        assert out2.shape == (10, 249, 4)

    def test_detect_sequences_with_zero_padding_NAL(self):
        """Detect padding in (N, 4, L) format."""
        from utils.helpers import detect_sequences_with_zero_padding

        np.random.seed(42)
        seqs = _random_onehot(10, 100)  # (N, 4, L), no padding
        valid, invalid = detect_sequences_with_zero_padding(seqs)
        assert len(valid) == 10
        assert len(invalid) == 0

    def test_detect_sequences_with_zero_padding_finds_padded(self):
        """Must find sequences that have zero-padded positions."""
        from utils.helpers import detect_sequences_with_zero_padding

        np.random.seed(42)
        seqs = _random_onehot_with_padding(10, 100, pad_len=20)  # (N, 4, L)
        valid, invalid = detect_sequences_with_zero_padding(seqs)
        assert len(invalid) == 10  # all have padding
        assert len(valid) == 0

    def test_filter_sequences_for_kmer_analysis_removes_padded(self):
        """Must remove sequences with zero padding."""
        from utils.helpers import filter_sequences_for_kmer_analysis

        np.random.seed(42)
        x_test = torch.tensor(_random_onehot(20, 100), dtype=torch.float32)  # no padding
        x_syn = torch.tensor(
            _random_onehot_with_padding(20, 100, pad_len=20), dtype=torch.float32
        )
        X_test_f, X_syn_f, n_rm_test, n_rm_syn = filter_sequences_for_kmer_analysis(x_test, x_syn)
        assert n_rm_test == 0
        assert n_rm_syn == 20  # all synthetic have padding
        assert X_test_f.shape[0] == 20
        assert X_syn_f.shape[0] == 0

    def test_one_hot_to_seq_roundtrip(self):
        """Converting one-hot to seq should produce valid ACGT strings."""
        from utils.helpers import one_hot_to_seq

        np.random.seed(42)
        seqs_oh = _random_onehot(5, 10, fmt="NLA")  # (N, L, 4)
        seq_strs = one_hot_to_seq(seqs_oh)
        assert len(seq_strs) == 5
        for s in seq_strs:
            assert len(s) == 10
            assert all(c in "ACGT" for c in s)


# ===================================================================
# File I/O
# ===================================================================

class TestWriteToH5:
    """Tests for write_to_h5()."""

    def test_writes_datasets(self, tmp_path):
        from utils.helpers import write_to_h5

        path = str(tmp_path / "test.h5")
        data = {"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6])}
        write_to_h5(path, data)
        with h5py.File(path, "r") as f:
            assert "x" in f
            assert "y" in f
            np.testing.assert_array_equal(f["x"][()], [1, 2, 3])

    def test_overwrites_existing_file(self, tmp_path):
        from utils.helpers import write_to_h5

        path = str(tmp_path / "test.h5")
        write_to_h5(path, {"a": np.array([1])})
        write_to_h5(path, {"b": np.array([2])})
        with h5py.File(path, "r") as f:
            assert "b" in f


# ===================================================================
# EmbeddingExtractor
# ===================================================================

class TestEmbeddingExtractor:
    """Tests for the EmbeddingExtractor hook class."""

    def test_hook_captures_output(self):
        from utils.helpers import EmbeddingExtractor

        extractor = EmbeddingExtractor()
        dummy_output = torch.randn(5, 16)
        extractor.hook(None, None, dummy_output)
        assert extractor.embedding is not None
        assert extractor.embedding.shape == (5, 16)


# ===================================================================
# load_predictions_batched
# ===================================================================

class TestLoadPredictionsBatched:
    """Tests for batched prediction loading."""

    def test_output_shapes(self, deepstarr_model, deepstarr_tensors):
        from utils.helpers import load_predictions_batched

        x_test, x_syn, _ = deepstarr_tensors
        y_test, y_syn = load_predictions_batched(x_test, x_syn, deepstarr_model, batch_size=16)
        assert y_test.shape[0] == len(x_test)
        assert y_syn.shape[0] == len(x_syn)

    def test_output_is_numpy(self, deepstarr_model, deepstarr_tensors):
        from utils.helpers import load_predictions_batched

        x_test, x_syn, _ = deepstarr_tensors
        y_test, y_syn = load_predictions_batched(x_test, x_syn, deepstarr_model, batch_size=16)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(y_syn, np.ndarray)

    def test_deterministic_across_batch_sizes(self, deepstarr_model, deepstarr_tensors):
        """Results should be identical regardless of batch size."""
        from utils.helpers import load_predictions_batched

        x_test, x_syn, _ = deepstarr_tensors
        y1_test, y1_syn = load_predictions_batched(x_test, x_syn, deepstarr_model, batch_size=8)
        y2_test, y2_syn = load_predictions_batched(x_test, x_syn, deepstarr_model, batch_size=32)
        np.testing.assert_allclose(y1_test, y2_test, atol=1e-5)
        np.testing.assert_allclose(y1_syn, y2_syn, atol=1e-5)


# ===================================================================
# get_penultimate_embeddings
# ===================================================================

class TestGetPenultimateEmbeddings:
    """Tests for penultimate layer embedding extraction."""

    def test_deepstarr_embeddings_shape(self, deepstarr_model, deepstarr_tensors):
        from utils.helpers import get_penultimate_embeddings

        x_test, _, _ = deepstarr_tensors
        emb = get_penultimate_embeddings(deepstarr_model, x_test, model_type="deepstarr", batch_size=16)
        assert emb.shape[0] == len(x_test)
        assert emb.ndim == 2

    def test_sei_raises_not_implemented(self, sei_model, sei_tensors):
        from utils.helpers import get_penultimate_embeddings

        x_test, _, _ = sei_tensors
        with pytest.raises(NotImplementedError):
            get_penultimate_embeddings(sei_model, x_test, model_type="sei")

    def test_embeddings_are_numpy(self, deepstarr_model, deepstarr_tensors):
        from utils.helpers import get_penultimate_embeddings

        x_test, _, _ = deepstarr_tensors
        emb = get_penultimate_embeddings(deepstarr_model, x_test, model_type="deepstarr", batch_size=16)
        assert isinstance(emb, np.ndarray)
