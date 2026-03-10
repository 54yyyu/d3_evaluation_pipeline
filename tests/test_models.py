"""
Tests for model architectures:
  - deepstarr.py (DeepSTARR, PL_DeepSTARR)
  - mpralegnet.py (LegNet, LitModel, TrainingConfig)
  - sei.py (Sei, NonStrandSpecific)

These tests verify forward pass shapes and basic model properties
without loading real checkpoints.
"""

import numpy as np
import torch
import pytest

from tests.conftest import DEEPSTARR_SEQ_LEN, LENTIMPRA_SEQ_LEN, SEI_SEQ_LEN


# ===================================================================
# DeepSTARR
# ===================================================================

class TestDeepSTARR:
    """Tests for the DeepSTARR model architecture."""

    def test_forward_output_shape(self):
        from deepstarr import DeepSTARR

        model = DeepSTARR(output_dim=2).eval()
        x = torch.randn(4, 4, DEEPSTARR_SEQ_LEN)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_forward_different_output_dim(self):
        from deepstarr import DeepSTARR

        model = DeepSTARR(output_dim=5).eval()
        x = torch.randn(2, 4, DEEPSTARR_SEQ_LEN)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)

    def test_batchnorm6_exists(self):
        """get_penultimate_embeddings hooks into model.batchnorm6 — must exist."""
        from deepstarr import DeepSTARR

        model = DeepSTARR(output_dim=2)
        layer_names = [name for name, _ in model.named_modules()]
        assert "model.batchnorm6" in layer_names

    def test_gradients_flow(self):
        """Verify gradients propagate through the model."""
        from deepstarr import DeepSTARR

        model = DeepSTARR(output_dim=2)
        x = torch.randn(2, 4, DEEPSTARR_SEQ_LEN, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_pl_deepstarr_forward(self):
        """PL_DeepSTARR wrapping should produce same output shape."""
        from deepstarr import PL_DeepSTARR

        model = PL_DeepSTARR().eval()
        x = torch.randn(3, 4, DEEPSTARR_SEQ_LEN)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 3


# ===================================================================
# MPRALegNet / LegNet
# ===================================================================

class TestMPRALegNet:
    """Tests for the MPRALegNet / LegNet model architecture."""

    def test_legnet_forward_shape(self):
        from mpralegnet import TrainingConfig

        cfg = TrainingConfig()
        model = cfg.get_model().eval()
        x = torch.randn(4, cfg.in_ch, LENTIMPRA_SEQ_LEN)
        with torch.no_grad():
            out = model(x)
        # LegNet outputs a single value per sequence
        assert out.shape == (4,) or out.shape == (4, 1)

    def test_training_config_defaults(self):
        from mpralegnet import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.stem_ch == 64
        assert cfg.epoch_num == 25
        assert cfg.in_ch == 4  # default (no reverse channel)

    def test_training_config_serialization(self, tmp_path):
        from mpralegnet import TrainingConfig

        cfg = TrainingConfig()
        json_path = str(tmp_path / "config.json")
        cfg.to_json(json_path)
        cfg2 = TrainingConfig.from_json(json_path)
        assert cfg.stem_ch == cfg2.stem_ch
        assert cfg.epoch_num == cfg2.epoch_num

    def test_litmodel_forward(self):
        from mpralegnet import LitModel, TrainingConfig

        cfg = TrainingConfig()
        model = LitModel(cfg).eval()
        x = torch.randn(3, cfg.in_ch, LENTIMPRA_SEQ_LEN)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 3

    def test_head_layer_2_exists(self):
        """get_penultimate_embeddings hooks into model.head.2 — must exist."""
        from mpralegnet import TrainingConfig

        cfg = TrainingConfig()
        model = cfg.get_model()
        layer_names = [name for name, _ in model.named_modules()]
        assert "head.2" in layer_names

    def test_seq2tensor_string_input(self):
        from mpralegnet import Seq2Tensor

        converter = Seq2Tensor()
        t = converter("ACGTACGT")
        assert t.shape == (4, 8)
        # A at position 0 should be [1, 0, 0, 0]
        assert t[0, 0] == 1.0

    def test_seq2tensor_handles_N(self):
        """N nucleotide should map to 0.25 per channel."""
        from mpralegnet import Seq2Tensor

        converter = Seq2Tensor()
        t = converter("N")
        assert t.shape == (4, 1)
        np.testing.assert_allclose(t[:, 0].numpy(), 0.25, atol=1e-6)


# ===================================================================
# SEI
# ===================================================================

class TestSei:
    """Tests for the SEI model architecture."""

    def test_sei_forward_shape(self):
        from sei import Sei

        # Use smaller feature count for testing speed
        model = Sei(sequence_length=SEI_SEQ_LEN, n_genomic_features=128).eval()
        x = torch.randn(2, 4, SEI_SEQ_LEN)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 128)

    def test_sei_output_bounded(self):
        """SEI uses sigmoid output — values must be in [0, 1]."""
        from sei import Sei

        model = Sei(sequence_length=SEI_SEQ_LEN, n_genomic_features=64).eval()
        x = torch.randn(2, 4, SEI_SEQ_LEN)
        with torch.no_grad():
            out = model(x)
        assert torch.all(out >= 0)
        assert torch.all(out <= 1)

    def test_non_strand_specific_mean(self):
        """NonStrandSpecific with mode='mean' should average forward and RC."""
        from sei import Sei, NonStrandSpecific

        base = Sei(sequence_length=SEI_SEQ_LEN, n_genomic_features=64).eval()
        model = NonStrandSpecific(base, mode="mean").eval()
        x = torch.randn(2, 4, SEI_SEQ_LEN)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 64)

    def test_non_strand_specific_max(self):
        """NonStrandSpecific with mode='max' should take element-wise max."""
        from sei import Sei, NonStrandSpecific

        base = Sei(sequence_length=SEI_SEQ_LEN, n_genomic_features=64).eval()
        model = NonStrandSpecific(base, mode="max").eval()
        x = torch.randn(2, 4, SEI_SEQ_LEN)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 64)

    def test_sei_input_validation(self):
        """SEI should handle the expected 4-channel input."""
        from sei import Sei

        model = Sei(sequence_length=SEI_SEQ_LEN, n_genomic_features=32).eval()
        # Wrong channel count should fail
        x_bad = torch.randn(1, 3, SEI_SEQ_LEN)
        with pytest.raises(Exception):
            model(x_bad)
