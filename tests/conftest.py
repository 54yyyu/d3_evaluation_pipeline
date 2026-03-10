"""
Shared fixtures for the D3 evaluation pipeline test suite.

Provides mock data tensors, mock oracle models, and temporary directories
that mirror the shapes and conventions used throughout the pipeline.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import os
import tempfile
import h5py


# ---------------------------------------------------------------------------
# Sequence length constants (must match model architectures)
# ---------------------------------------------------------------------------
DEEPSTARR_SEQ_LEN = 249
LENTIMPRA_SEQ_LEN = 230
SEI_SEQ_LEN = 4096


# ---------------------------------------------------------------------------
# One-hot sequence generators
# ---------------------------------------------------------------------------

def _random_onehot(n, seq_len, fmt="NAL"):
    """Generate random one-hot DNA sequences.

    Args:
        n: number of sequences
        seq_len: length of each sequence
        fmt: "NAL" for (N, 4, L) or "NLA" for (N, L, 4)
    """
    indices = np.random.randint(0, 4, size=(n, seq_len))
    onehot = np.zeros((n, seq_len, 4), dtype=np.float32)
    for i in range(n):
        onehot[i, np.arange(seq_len), indices[i]] = 1.0
    if fmt == "NAL":
        return onehot.transpose(0, 2, 1)  # (N, 4, L)
    return onehot  # (N, L, 4)


def _random_onehot_with_padding(n, seq_len, pad_len=50, fmt="NAL"):
    """Generate one-hot sequences with zero-padding at the end."""
    seqs = _random_onehot(n, seq_len, fmt="NLA")  # (N, L, 4)
    seqs[:, -pad_len:, :] = 0.0  # zero out last pad_len positions
    if fmt == "NAL":
        return seqs.transpose(0, 2, 1)
    return seqs


# ---------------------------------------------------------------------------
# Mock oracle models
# ---------------------------------------------------------------------------

class MockDeepSTARR(nn.Module):
    """Mimics DeepSTARR: (batch, 4, 249) -> (batch, 2)."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4, 2)
        # Expose a named layer so get_penultimate_embeddings can hook into it
        self.model = nn.Module()
        self.model.batchnorm6 = nn.BatchNorm1d(4)

    def forward(self, x):
        x = self.model.batchnorm6(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class MockMPRALegNet(nn.Module):
    """Mimics MPRALegNet: (batch, 4, 230) -> (batch, 1)."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.model = nn.Module()
        head = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),  # index 2 — this is the hook target
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        self.model.head = head
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        x = self.pool(x).squeeze(-1)
        return self.model.head(x)


class MockSEI(nn.Module):
    """Mimics SEI: (batch, 4, 4096) -> (batch, 21907)."""

    def __init__(self, n_features=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4, n_features)
        self.n_features = n_features

    def forward(self, x):
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded numpy random generator for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def deepstarr_model():
    return MockDeepSTARR().eval()


@pytest.fixture
def mpralegnet_model():
    return MockMPRALegNet().eval()


@pytest.fixture
def sei_model():
    return MockSEI().eval()


@pytest.fixture
def deepstarr_tensors():
    """Return (x_test, x_synthetic, x_train) as torch tensors for DeepSTARR."""
    np.random.seed(42)
    x_test = torch.tensor(_random_onehot(50, DEEPSTARR_SEQ_LEN), dtype=torch.float32)
    x_syn = torch.tensor(_random_onehot(50, DEEPSTARR_SEQ_LEN), dtype=torch.float32)
    x_train = torch.tensor(_random_onehot(100, DEEPSTARR_SEQ_LEN), dtype=torch.float32)
    return x_test, x_syn, x_train


@pytest.fixture
def lentimpra_tensors():
    """Return (x_test, x_synthetic, x_train) as torch tensors for LentIMPRA."""
    np.random.seed(42)
    x_test = torch.tensor(_random_onehot(50, LENTIMPRA_SEQ_LEN), dtype=torch.float32)
    x_syn = torch.tensor(_random_onehot(50, LENTIMPRA_SEQ_LEN), dtype=torch.float32)
    x_train = torch.tensor(_random_onehot(100, LENTIMPRA_SEQ_LEN), dtype=torch.float32)
    return x_test, x_syn, x_train


@pytest.fixture
def sei_tensors():
    """Return (x_test, x_synthetic, x_train) as torch tensors for SEI."""
    np.random.seed(42)
    x_test = torch.tensor(_random_onehot(10, SEI_SEQ_LEN), dtype=torch.float32)
    x_syn = torch.tensor(_random_onehot(10, SEI_SEQ_LEN), dtype=torch.float32)
    x_train = torch.tensor(_random_onehot(20, SEI_SEQ_LEN), dtype=torch.float32)
    return x_test, x_syn, x_train


@pytest.fixture
def padded_tensors():
    """Sequences with zero-padding (last 50 positions)."""
    np.random.seed(42)
    x_test = torch.tensor(_random_onehot(30, DEEPSTARR_SEQ_LEN), dtype=torch.float32)
    x_syn = torch.tensor(
        _random_onehot_with_padding(30, DEEPSTARR_SEQ_LEN, pad_len=50),
        dtype=torch.float32,
    )
    return x_test, x_syn


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for test results."""
    out = tmp_path / "results"
    out.mkdir()
    return str(out)


@pytest.fixture
def sample_npz_file(tmp_path):
    """Create a temporary NPZ file mimicking generated samples."""
    np.random.seed(42)
    seqs = _random_onehot(50, DEEPSTARR_SEQ_LEN, fmt="NLA")  # (N, L, 4)
    path = str(tmp_path / "samples.npz")
    np.savez(path, arr_0=seqs)
    return path


@pytest.fixture
def sample_h5_file(tmp_path):
    """Create a temporary H5 file mimicking generated samples."""
    np.random.seed(42)
    seqs = _random_onehot(50, DEEPSTARR_SEQ_LEN, fmt="NLA")
    path = str(tmp_path / "samples.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("arr_0", data=seqs)
    return path


@pytest.fixture
def deepstarr_data_h5(tmp_path):
    """Create a temporary H5 data file with X_test and X_train."""
    np.random.seed(42)
    x_test = _random_onehot(50, DEEPSTARR_SEQ_LEN)  # (N, 4, L)
    x_train = _random_onehot(100, DEEPSTARR_SEQ_LEN)
    path = str(tmp_path / "data.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("X_test", data=x_test)
        f.create_dataset("X_train", data=x_train)
    return path


@pytest.fixture
def deepstarr_data_npz(tmp_path):
    """Create a temporary NPZ data file with X_test and X_train."""
    np.random.seed(42)
    x_test = _random_onehot(50, DEEPSTARR_SEQ_LEN)
    x_train = _random_onehot(100, DEEPSTARR_SEQ_LEN)
    path = str(tmp_path / "data.npz")
    np.savez(path, x_test=x_test, x_train=x_train)
    return path


@pytest.fixture
def lentimpra_data_h5(tmp_path):
    """Create a temporary H5 data file with onehot_test/onehot_train (NLA format)."""
    np.random.seed(42)
    x_test = _random_onehot(50, LENTIMPRA_SEQ_LEN, fmt="NLA")  # (N, 230, 4)
    x_train = _random_onehot(100, LENTIMPRA_SEQ_LEN, fmt="NLA")
    path = str(tmp_path / "lentimpra_data.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("onehot_test", data=x_test)
        f.create_dataset("onehot_train", data=x_train)
    return path


@pytest.fixture
def promoter_data_npz(tmp_path):
    """Create a temporary NPZ in promoter format: (N, seq_len, 6)."""
    np.random.seed(42)
    seq_len = 200
    n_train, n_test, n_valid = 80, 20, 20
    # channels 0-3: one-hot, channels 4-5: activity
    train = np.zeros((n_train, seq_len, 6), dtype=np.float32)
    test = np.zeros((n_test, seq_len, 6), dtype=np.float32)
    valid = np.zeros((n_valid, seq_len, 6), dtype=np.float32)
    for arr in [train, test, valid]:
        indices = np.random.randint(0, 4, size=(arr.shape[0], seq_len))
        for i in range(arr.shape[0]):
            arr[i, np.arange(seq_len), indices[i]] = 1.0
        arr[:, :, 4:] = np.random.randn(arr.shape[0], seq_len, 2).astype(np.float32)
    path = str(tmp_path / "promoter_data.npz")
    np.savez(path, train=train, test=test, valid=valid)
    return path


@pytest.fixture
def batch_dir(tmp_path):
    """Create a batch directory with multiple sample NPZ files."""
    np.random.seed(42)
    batch = tmp_path / "batch"
    batch.mkdir()
    for i in range(3):
        seqs = _random_onehot(20, DEEPSTARR_SEQ_LEN, fmt="NLA")
        np.savez(str(batch / f"sample_{i}.npz"), arr_0=seqs)
    return str(batch)


@pytest.fixture
def nested_batch_dir(tmp_path):
    """Create a nested batch directory with subdirectories per sample."""
    np.random.seed(42)
    batch = tmp_path / "nested_batch"
    batch.mkdir()
    for i in range(2):
        sub = batch / f"sample_{i}"
        sub.mkdir()
        for j in range(3):
            seqs = _random_onehot(15, DEEPSTARR_SEQ_LEN, fmt="NLA")
            np.savez(str(sub / f"run_{j}.npz"), arr_0=seqs)
    return str(batch)
