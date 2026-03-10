"""
Shared fixtures for the D3 evaluation pipeline test suite.

Provides synthetic DNA sequence data and mock oracle models so that
every core analysis module can be tested without loading real checkpoints.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import h5py
import os
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers to generate random one-hot DNA sequences
# ---------------------------------------------------------------------------

def _random_onehot(n, length, fmt="NAL"):
    """Generate random one-hot DNA sequences.

    Args:
        n: number of sequences
        length: sequence length
        fmt: 'NAL' -> (N, 4, L),  'NLA' -> (N, L, 4)
    """
    indices = np.random.randint(0, 4, size=(n, length))
    onehot = np.zeros((n, length, 4), dtype=np.float32)
    for i in range(n):
        for j in range(length):
            onehot[i, j, indices[i, j]] = 1.0
    if fmt == "NAL":
        return np.transpose(onehot, (0, 2, 1))  # (N, 4, L)
    return onehot  # (N, L, 4)


# ---------------------------------------------------------------------------
# Sequence fixtures  (small sizes for fast tests)
# ---------------------------------------------------------------------------

SEQ_LEN = 230   # lentimpra sequence length
N_TEST = 50
N_SYNTH = 50
N_TRAIN = 80


@pytest.fixture
def seq_len():
    return SEQ_LEN


@pytest.fixture
def x_test_np():
    """Test sequences as numpy array (N, 4, L)."""
    np.random.seed(42)
    return _random_onehot(N_TEST, SEQ_LEN)


@pytest.fixture
def x_synthetic_np():
    """Synthetic sequences as numpy array (N, 4, L)."""
    np.random.seed(43)
    return _random_onehot(N_SYNTH, SEQ_LEN)


@pytest.fixture
def x_train_np():
    """Training sequences as numpy array (N, 4, L)."""
    np.random.seed(44)
    return _random_onehot(N_TRAIN, SEQ_LEN)


@pytest.fixture
def x_test_tensor(x_test_np):
    return torch.from_numpy(x_test_np).float()


@pytest.fixture
def x_synthetic_tensor(x_synthetic_np):
    return torch.from_numpy(x_synthetic_np).float()


@pytest.fixture
def x_train_tensor(x_train_np):
    return torch.from_numpy(x_train_np).float()


@pytest.fixture
def sample_seqs_NLA():
    """Sample sequences in (N, L, A) format for attribution analysis."""
    np.random.seed(45)
    return torch.from_numpy(_random_onehot(N_SYNTH, SEQ_LEN, fmt="NLA")).float()


@pytest.fixture
def x_test_NLA():
    """Test sequences in (N, L, A) format for attribution analysis."""
    np.random.seed(46)
    return torch.from_numpy(_random_onehot(N_TEST, SEQ_LEN, fmt="NLA")).float()


# ---------------------------------------------------------------------------
# Mock oracle model — mimics forward() and named_modules()
# ---------------------------------------------------------------------------

class _MockOracleModel(nn.Module):
    """Tiny model that accepts (N, 4, L) and returns (N, 2) predictions.

    Has a named submodule 'model.batchnorm6' so that
    get_penultimate_embeddings() can attach its hook.
    """

    def __init__(self, seq_len=SEQ_LEN, output_dim=2, embedding_dim=16):
        super().__init__()
        # Create nested structure so named_modules yields 'model.batchnorm6'
        self.model = nn.Module()
        self.model.conv = nn.Conv1d(4, embedding_dim, kernel_size=3, padding=1)
        self.model.batchnorm6 = nn.BatchNorm1d(embedding_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        # x: (N, 4, L)
        h = self.model.conv(x)
        h = self.model.batchnorm6(h)
        h = self.pool(h).squeeze(-1)  # (N, embedding_dim)
        return self.head(h)            # (N, output_dim)


class _MockLentimprModel(nn.Module):
    """Tiny model that mimics MPRALegNet structure.

    Has 'model.head.2' submodule for lentimpra embedding extraction.
    Returns (N, 1) predictions.
    """

    def __init__(self, seq_len=SEQ_LEN, embedding_dim=16):
        super().__init__()
        self.model = nn.Module()
        self.model.conv = nn.Conv1d(4, embedding_dim, kernel_size=3, padding=1)
        self.model.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.ReLU(),        # index 2 -> 'model.head.2'
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x):
        h = self.model.conv(x)
        return self.model.head(h)


@pytest.fixture
def mock_oracle():
    """A small DeepSTARR-like oracle model (CPU, eval mode)."""
    torch.manual_seed(0)
    model = _MockOracleModel()
    model.eval()
    return model


@pytest.fixture
def mock_lentimpra_model():
    """A small MPRALegNet-like oracle model (CPU, eval mode)."""
    torch.manual_seed(0)
    model = _MockLentimprModel()
    model.eval()
    return model


@pytest.fixture
def mock_multi_oracle():
    """Tuple of 3 lentimpra-like models for multi-oracle tests."""
    torch.manual_seed(0)
    m1 = _MockLentimprModel(); m1.eval()
    torch.manual_seed(1)
    m2 = _MockLentimprModel(); m2.eval()
    torch.manual_seed(2)
    m3 = _MockLentimprModel(); m3.eval()
    return (m1, m2, m3)


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for file-writing tests."""
    d = tmp_path / "output"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# Temporary data files (NPZ / H5) for data-loading tests
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_npz_samples(tmp_path, x_synthetic_np):
    """Write synthetic sequences to an NPZ file and return its path."""
    p = tmp_path / "samples.npz"
    # Save in (N, L, 4) format — the loader transposes
    data_NLA = np.transpose(x_synthetic_np, (0, 2, 1))
    np.savez(str(p), arr_0=data_NLA)
    return str(p)


@pytest.fixture
def tmp_h5_data(tmp_path, x_test_np, x_train_np):
    """Write test/train data to an H5 file with X_test and X_train keys."""
    p = tmp_path / "data.h5"
    with h5py.File(str(p), 'w') as f:
        f.create_dataset('X_test', data=x_test_np)
        f.create_dataset('X_train', data=x_train_np)
    return str(p)


@pytest.fixture
def tmp_h5_lentimpra_data(tmp_path):
    """H5 file with lentimpra-style keys (onehot_test, onehot_train)."""
    np.random.seed(47)
    x_test = _random_onehot(30, SEQ_LEN)
    x_train = _random_onehot(60, SEQ_LEN)
    p = tmp_path / "lentimpra_data.h5"
    with h5py.File(str(p), 'w') as f:
        f.create_dataset('onehot_test', data=x_test)
        f.create_dataset('onehot_train', data=x_train)
    return str(p)


@pytest.fixture
def tmp_h5_samples(tmp_path, x_synthetic_np):
    """Write synthetic sequences to an H5 file."""
    p = tmp_path / "samples.h5"
    with h5py.File(str(p), 'w') as f:
        f.create_dataset('sequences_onehot', data=x_synthetic_np)
    return str(p)


@pytest.fixture
def tmp_index_encoded_npz(tmp_path):
    """NPZ file with index-encoded sequences (N, L) values 0-3."""
    np.random.seed(48)
    index_seqs = np.random.randint(0, 4, size=(40, SEQ_LEN))
    p = tmp_path / "index_samples.npz"
    np.savez(str(p), arr_0=index_seqs)
    return str(p)


@pytest.fixture
def discriminability_h5(tmp_path, x_test_np, x_synthetic_np):
    """Create a Discriminatability.h5 file for discriminability tests."""
    x_combined = np.concatenate([x_test_np, x_synthetic_np], axis=0)
    # Transpose to (N, A, L) as prep_data_for_classification does
    x_combined_transposed = np.transpose(x_combined, (0, 2, 1))
    y = np.concatenate([
        np.ones((x_test_np.shape[0], 1)),
        np.zeros((x_synthetic_np.shape[0], 1))
    ], axis=0)
    p = tmp_path / "Discriminatability.h5"
    with h5py.File(str(p), 'w') as f:
        f.create_dataset('x_train', data=x_combined_transposed)
        f.create_dataset('y_train', data=y)
    return str(p)
