"""
Tests for utils/helpers.py — data loading, encoding detection, shape handling,
prediction loading, and embedding extraction.
"""

import numpy as np
import torch
import h5py
import pytest
import os


# ===== is_index_encoded =====

class TestIsIndexEncoded:
    def test_integer_2d_0123(self):
        from utils.helpers import is_index_encoded
        seqs = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64)
        assert is_index_encoded(seqs) is True

    def test_float_2d_effectively_integers(self):
        from utils.helpers import is_index_encoded
        seqs = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
        assert is_index_encoded(seqs) is True

    def test_onehot_3d_is_not_index(self):
        from utils.helpers import is_index_encoded
        seqs = np.random.rand(10, 4, 100).astype(np.float32)
        assert is_index_encoded(seqs) is False

    def test_1d_returns_false(self):
        from utils.helpers import is_index_encoded
        seqs = np.array([0, 1, 2, 3])
        assert is_index_encoded(seqs) is False

    def test_2d_with_values_beyond_3(self):
        from utils.helpers import is_index_encoded
        seqs = np.array([[0, 1, 5, 3]], dtype=np.int64)
        assert is_index_encoded(seqs) is False


# ===== index_to_onehot =====

class TestIndexToOnehot:
    def test_basic_conversion(self):
        from utils.helpers import index_to_onehot
        seqs = np.array([[0, 1, 2, 3]], dtype=np.int64)
        onehot = index_to_onehot(seqs)
        assert onehot.shape == (1, 4, 4)
        # Position 0 should be A (index 0)
        assert onehot[0, 0, 0] == 1.0
        assert onehot[0, 0, 1] == 0.0
        # Position 1 should be C (index 1)
        assert onehot[0, 1, 1] == 1.0

    def test_roundtrip_shape(self):
        from utils.helpers import index_to_onehot
        N, L = 10, 230
        seqs = np.random.randint(0, 4, size=(N, L))
        onehot = index_to_onehot(seqs)
        assert onehot.shape == (N, L, 4)
        # Each position should have exactly one 1
        assert np.allclose(onehot.sum(axis=2), 1.0)


# ===== ensure_correct_shape =====

class TestEnsureCorrectShape:
    def test_already_correct_NAL(self):
        from utils.helpers import ensure_correct_shape
        data = np.random.rand(10, 4, 230).astype(np.float32)
        result = ensure_correct_shape(data)
        assert result.shape == (10, 4, 230)
        np.testing.assert_array_equal(result, data)

    def test_transposes_NLA_to_NAL(self):
        from utils.helpers import ensure_correct_shape
        data = np.random.rand(10, 230, 4).astype(np.float32)
        result = ensure_correct_shape(data)
        assert result.shape == (10, 4, 230)

    def test_converts_index_encoded(self):
        from utils.helpers import ensure_correct_shape
        data = np.random.randint(0, 4, size=(10, 230))
        result = ensure_correct_shape(data)
        assert result.shape == (10, 4, 230)

    def test_raises_on_wrong_dims(self):
        from utils.helpers import ensure_correct_shape
        data = np.random.rand(10, 5, 230).astype(np.float32)
        with pytest.raises(ValueError):
            ensure_correct_shape(data)


# ===== resolve_key_from_file / KEY_PRIORITIES =====

class TestResolveKeyFromFile:
    def test_user_key_override(self, tmp_path):
        from utils.helpers import resolve_key_from_file
        p = tmp_path / "test.h5"
        with h5py.File(str(p), 'w') as f:
            f.create_dataset('custom_key', data=np.zeros((5, 4, 10)))
            f.create_dataset('X_test', data=np.ones((5, 4, 10)))
        with h5py.File(str(p), 'r') as f:
            data, key = resolve_key_from_file(f, 'h5', 'test', user_keys=['custom_key'])
        assert key == 'custom_key'
        np.testing.assert_array_equal(data, np.zeros((5, 4, 10)))

    def test_default_priority_deepstarr(self, tmp_path):
        from utils.helpers import resolve_key_from_file
        p = tmp_path / "test.h5"
        with h5py.File(str(p), 'w') as f:
            f.create_dataset('X_test', data=np.ones((5, 4, 10)))
            f.create_dataset('other', data=np.zeros((5, 4, 10)))
        with h5py.File(str(p), 'r') as f:
            data, key = resolve_key_from_file(f, 'h5', 'test')
        assert key == 'X_test'

    def test_lentimpra_priority(self, tmp_path):
        from utils.helpers import resolve_key_from_file
        p = tmp_path / "test.h5"
        with h5py.File(str(p), 'w') as f:
            f.create_dataset('onehot_test', data=np.ones((5, 4, 10)))
            f.create_dataset('X_test', data=np.zeros((5, 4, 10)))
        with h5py.File(str(p), 'r') as f:
            data, key = resolve_key_from_file(f, 'h5', 'test', model_type='lentimpra')
        assert key == 'onehot_test'

    def test_user_key_not_found_raises(self, tmp_path):
        from utils.helpers import resolve_key_from_file
        p = tmp_path / "test.h5"
        with h5py.File(str(p), 'w') as f:
            f.create_dataset('X_test', data=np.zeros((5,)))
        with h5py.File(str(p), 'r') as f:
            with pytest.raises(KeyError):
                resolve_key_from_file(f, 'h5', 'test', user_keys=['nonexistent'])

    def test_fallback_when_no_priority_match(self, tmp_path):
        from utils.helpers import resolve_key_from_file
        p = tmp_path / "test.h5"
        with h5py.File(str(p), 'w') as f:
            f.create_dataset('weird_key', data=np.zeros((5,)))
        with h5py.File(str(p), 'r') as f:
            data, key = resolve_key_from_file(f, 'h5', 'test')
        assert key == 'weird_key'


# ===== load_file_by_type =====

class TestLoadFileByType:
    def test_load_npz(self, tmp_npz_samples):
        from utils.helpers import load_file_by_type
        data, key, ftype = load_file_by_type(tmp_npz_samples, 'samples')
        assert ftype == 'npz'

    def test_load_h5(self, tmp_h5_data):
        from utils.helpers import load_file_by_type
        data, key, ftype = load_file_by_type(tmp_h5_data, 'test')
        assert ftype == 'h5'
        assert key == 'X_test'

    def test_unsupported_format_raises(self, tmp_path):
        from utils.helpers import load_file_by_type
        p = tmp_path / "data.csv"
        p.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported"):
            load_file_by_type(str(p), 'test')

    def test_load_pt_file(self, tmp_path):
        from utils.helpers import load_file_by_type
        t = torch.randn(10, 4, 230)
        p = tmp_path / "data.pt"
        torch.save(t, str(p))
        data, key, ftype = load_file_by_type(str(p), 'samples')
        assert ftype == 'pt'
        assert data.shape == (10, 4, 230)


# ===== extract_sequences =====

class TestExtractSequences:
    def test_npz_samples_h5_data(self, tmp_npz_samples, tmp_h5_data):
        from utils.helpers import extract_sequences
        x_test, x_syn, x_train = extract_sequences(tmp_npz_samples, tmp_h5_data)
        assert x_test.shape[1] == 4   # (N, 4, L)
        assert x_syn.shape[1] == 4
        assert x_train.shape[1] == 4

    def test_h5_samples_h5_data(self, tmp_h5_samples, tmp_h5_data):
        from utils.helpers import extract_sequences
        x_test, x_syn, x_train = extract_sequences(tmp_h5_samples, tmp_h5_data)
        assert x_syn.shape[1] == 4

    def test_lentimpra_key_priority(self, tmp_h5_samples, tmp_h5_lentimpra_data):
        from utils.helpers import extract_sequences
        x_test, x_syn, x_train = extract_sequences(
            tmp_h5_samples, tmp_h5_lentimpra_data, model_type='lentimpra'
        )
        assert x_test.shape[1] == 4


# ===== detect_data_format =====

class TestDetectDataFormat:
    def test_npz(self, tmp_npz_samples):
        from utils.helpers import detect_data_format
        assert detect_data_format(tmp_npz_samples) == 'npz'

    def test_h5(self, tmp_h5_data):
        from utils.helpers import detect_data_format
        assert detect_data_format(tmp_h5_data) == 'h5'

    def test_nonexistent(self):
        from utils.helpers import detect_data_format
        assert detect_data_format('/nonexistent/path.xyz') == 'unknown'


# ===== numpy_to_tensor =====

class TestNumpyToTensor:
    def test_basic(self):
        from utils.helpers import numpy_to_tensor
        arr = np.array([1.0, 2.0, 3.0])
        t = numpy_to_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32


# ===== load_predictions =====

class TestLoadPredictions:
    def test_returns_numpy_arrays(self, mock_oracle, x_test_tensor, x_synthetic_tensor):
        from utils.helpers import load_predictions
        y_test, y_syn = load_predictions(x_test_tensor, x_synthetic_tensor, mock_oracle)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(y_syn, np.ndarray)
        assert y_test.shape[0] == x_test_tensor.shape[0]
        assert y_syn.shape[0] == x_synthetic_tensor.shape[0]

    def test_output_dim_matches_model(self, mock_oracle, x_test_tensor, x_synthetic_tensor):
        from utils.helpers import load_predictions
        y_test, y_syn = load_predictions(x_test_tensor, x_synthetic_tensor, mock_oracle)
        # mock_oracle has output_dim=2
        assert y_test.shape[1] == 2
        assert y_syn.shape[1] == 2

    def test_batched_inference_same_result(self, mock_oracle, x_test_tensor, x_synthetic_tensor):
        """Batched inference should produce the same results regardless of batch size."""
        from utils.helpers import load_predictions
        os.environ['D3_INFER_BATCH_SIZE'] = '10'
        y_test_small, y_syn_small = load_predictions(x_test_tensor, x_synthetic_tensor, mock_oracle)
        os.environ['D3_INFER_BATCH_SIZE'] = '1024'
        y_test_big, y_syn_big = load_predictions(x_test_tensor, x_synthetic_tensor, mock_oracle)
        np.testing.assert_allclose(y_test_small, y_test_big, atol=1e-5)
        np.testing.assert_allclose(y_syn_small, y_syn_big, atol=1e-5)
        os.environ.pop('D3_INFER_BATCH_SIZE', None)


# ===== load_multi_oracle_predictions =====

class TestLoadMultiOraclePredictions:
    def test_output_shape(self, mock_multi_oracle, x_test_tensor, x_synthetic_tensor):
        from utils.helpers import load_multi_oracle_predictions
        y_test, y_syn = load_multi_oracle_predictions(
            x_test_tensor, x_synthetic_tensor, mock_multi_oracle
        )
        # Should be (N, 3) — one column per oracle
        assert y_test.shape == (x_test_tensor.shape[0], 3)
        assert y_syn.shape == (x_synthetic_tensor.shape[0], 3)

    def test_each_column_differs(self, mock_multi_oracle, x_test_tensor, x_synthetic_tensor):
        """Each oracle should produce different predictions (different seeds)."""
        from utils.helpers import load_multi_oracle_predictions
        y_test, _ = load_multi_oracle_predictions(
            x_test_tensor, x_synthetic_tensor, mock_multi_oracle
        )
        # At least one pair of columns should differ
        assert not np.allclose(y_test[:, 0], y_test[:, 1]) or \
               not np.allclose(y_test[:, 1], y_test[:, 2])


# ===== get_penultimate_embeddings =====

class TestGetPenultimateEmbeddings:
    def test_deepstarr_type(self, mock_oracle, x_test_tensor):
        from utils.helpers import get_penultimate_embeddings
        emb = get_penultimate_embeddings(mock_oracle, x_test_tensor, model_type='deepstarr')
        assert isinstance(emb, torch.Tensor)
        assert emb.shape[0] == x_test_tensor.shape[0]

    def test_lentimpra_type(self, mock_lentimpra_model, x_test_tensor):
        from utils.helpers import get_penultimate_embeddings
        emb = get_penultimate_embeddings(mock_lentimpra_model, x_test_tensor, model_type='lentimpra')
        assert isinstance(emb, torch.Tensor)
        assert emb.shape[0] == x_test_tensor.shape[0]

    def test_unsupported_type_raises(self, mock_oracle, x_test_tensor):
        from utils.helpers import get_penultimate_embeddings
        with pytest.raises(ValueError, match="Unsupported"):
            get_penultimate_embeddings(mock_oracle, x_test_tensor, model_type='nonexistent')

    def test_default_is_deepstarr(self, mock_oracle, x_test_tensor):
        from utils.helpers import get_penultimate_embeddings
        emb = get_penultimate_embeddings(mock_oracle, x_test_tensor)
        assert emb.shape[0] == x_test_tensor.shape[0]


# ===== get_multi_oracle_embeddings =====

class TestGetMultiOracleEmbeddings:
    def test_concatenated_shape(self, mock_multi_oracle, x_test_tensor):
        from utils.helpers import get_multi_oracle_embeddings
        emb = get_multi_oracle_embeddings(mock_multi_oracle, x_test_tensor)
        # Should be 3x the single-model embedding dim
        single_emb_dim = 16  # from _MockLentimprModel
        assert emb.shape == (x_test_tensor.shape[0], single_emb_dim * 3)


# ===== put_deepstarr_into_NLA =====

class TestPutDeepstarrIntoNLA:
    def test_transposes_correctly(self, x_test_tensor, x_synthetic_tensor):
        from utils.helpers import put_deepstarr_into_NLA
        x_test_nla, x_syn_nla = put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor)
        assert isinstance(x_test_nla, np.ndarray)
        # Input (N, 4, L) -> output (N, L, 4)
        assert x_test_nla.shape == (x_test_tensor.shape[0], x_test_tensor.shape[2], 4)
        assert x_syn_nla.shape == (x_synthetic_tensor.shape[0], x_synthetic_tensor.shape[2], 4)


# ===== one_hot_to_seq =====

class TestOneHotToSeq:
    def test_known_sequence(self):
        from utils.helpers import one_hot_to_seq
        # A single sequence: A, C, G, T
        onehot = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1],  # T
        ], dtype=np.float32)[np.newaxis, :]  # (1, 4, 4)
        seqs = one_hot_to_seq(onehot)
        assert len(seqs) == 1
        assert seqs[0] == "ACGT"

    def test_output_length(self, x_test_np):
        from utils.helpers import one_hot_to_seq, put_deepstarr_into_NLA
        # Need NLA format for one_hot_to_seq
        x_nla = np.transpose(x_test_np[:3], (0, 2, 1))
        seqs = one_hot_to_seq(x_nla)
        assert len(seqs) == 3
        assert all(len(s) == x_test_np.shape[2] for s in seqs)
        assert all(set(s) <= {'A', 'C', 'G', 'T'} for s in seqs)


# ===== write_to_h5 =====

class TestWriteToH5:
    def test_roundtrip(self, tmp_path):
        from utils.helpers import write_to_h5
        p = str(tmp_path / "test.h5")
        data = {'arr1': np.array([1, 2, 3]), 'arr2': np.array([4.0, 5.0])}
        write_to_h5(p, data)
        with h5py.File(p, 'r') as f:
            np.testing.assert_array_equal(f['arr1'][()], [1, 2, 3])
            np.testing.assert_array_equal(f['arr2'][()], [4.0, 5.0])


# ===== create_fasta_file =====

class TestCreateFastaFile:
    def test_writes_correct_format(self, tmp_path):
        from utils.helpers import create_fasta_file
        p = str(tmp_path / "test.fasta")
        seqs = ["ACGT", "TGCA", "AAAA"]
        create_fasta_file(seqs, p)
        with open(p) as f:
            content = f.read()
        assert ">Seq0\nACGT\n" in content
        assert ">Seq1\nTGCA\n" in content
        assert ">Seq2\nAAAA\n" in content
