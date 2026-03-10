"""
Tests for utils/batch_helpers.py — batch sample discovery,
loading, CSV/H5 output writing, and concise metric extraction.
"""

import numpy as np
import pandas as pd
import h5py
import pytest
import os
from pathlib import Path


# =====================================================================
# discover_batch_samples
# =====================================================================

class TestDiscoverBatchSamples:
    def test_flat_structure_with_existing_csv(self, tmp_path):
        from utils.batch_helpers import discover_batch_samples
        # Create sample files
        np.savez(str(tmp_path / "sample1.npz"), arr_0=np.zeros((5, 10, 4)))
        np.savez(str(tmp_path / "sample2.npz"), arr_0=np.ones((5, 10, 4)))
        # Create CSV metadata
        df = pd.DataFrame([
            {'sample_name': 'sample1', 'file_path': 'sample1.npz'},
            {'sample_name': 'sample2', 'file_path': 'sample2.npz'},
        ])
        df.to_csv(str(tmp_path / "metadata.csv"), index=False)

        records = discover_batch_samples(str(tmp_path))
        assert len(records) == 2
        assert records[0]['sample_name'] == 'sample1'

    def test_creates_template_csv_and_exits(self, tmp_path):
        from utils.batch_helpers import discover_batch_samples
        # Create sample files but no CSV
        np.savez(str(tmp_path / "s1.npz"), arr_0=np.zeros((5, 10, 4)))
        np.savez(str(tmp_path / "s2.npz"), arr_0=np.ones((5, 10, 4)))

        with pytest.raises(SystemExit):
            discover_batch_samples(str(tmp_path))

        # Template should have been created
        assert (tmp_path / "metadata.csv").exists()
        df = pd.read_csv(str(tmp_path / "metadata.csv"))
        assert len(df) == 2

    def test_nested_structure_creates_template(self, tmp_path):
        from utils.batch_helpers import discover_batch_samples
        # Create nested structure
        sub1 = tmp_path / "exp1"
        sub1.mkdir()
        np.savez(str(sub1 / "data.npz"), arr_0=np.zeros((5, 10, 4)))
        sub2 = tmp_path / "exp2"
        sub2.mkdir()
        np.savez(str(sub2 / "data.npz"), arr_0=np.ones((5, 10, 4)))

        with pytest.raises(SystemExit):
            discover_batch_samples(str(tmp_path))

        df = pd.read_csv(str(tmp_path / "metadata.csv"))
        assert len(df) == 2
        # Names should include subfolder
        assert any("exp1" in name for name in df['sample_name'].values)

    def test_h5_files_discovered(self, tmp_path):
        from utils.batch_helpers import discover_batch_samples
        with h5py.File(str(tmp_path / "s1.h5"), 'w') as f:
            f.create_dataset('arr_0', data=np.zeros((5, 4, 10)))
        df = pd.DataFrame([{'sample_name': 's1', 'file_path': 's1.h5'}])
        df.to_csv(str(tmp_path / "metadata.csv"), index=False)

        records = discover_batch_samples(str(tmp_path))
        assert len(records) == 1

    def test_pt_files_discovered_in_template(self, tmp_path):
        from utils.batch_helpers import discover_batch_samples
        import torch
        torch.save(torch.randn(5, 4, 10), str(tmp_path / "s1.pt"))

        with pytest.raises(SystemExit):
            discover_batch_samples(str(tmp_path))

        df = pd.read_csv(str(tmp_path / "metadata.csv"))
        assert len(df) == 1
        assert 's1' in df['sample_name'].values[0]


# =====================================================================
# write_concise_csv
# =====================================================================

class TestWriteConciseCsv:
    def test_creates_new_csv(self, tmp_output_dir):
        from utils.batch_helpers import write_concise_csv
        metrics = {'mse': 0.05, 'r2': 0.95}
        write_concise_csv(tmp_output_dir, 'test_analysis', 'sample_A', metrics)
        csv_path = Path(tmp_output_dir) / "test_analysis.csv"
        assert csv_path.exists()
        df = pd.read_csv(str(csv_path), index_col=0)
        assert 'sample_A' in df.columns
        assert df.loc['mse', 'sample_A'] == pytest.approx(0.05)

    def test_appends_to_existing(self, tmp_output_dir):
        from utils.batch_helpers import write_concise_csv
        write_concise_csv(tmp_output_dir, 'test_analysis', 'sample_A', {'metric1': 1.0})
        write_concise_csv(tmp_output_dir, 'test_analysis', 'sample_B', {'metric1': 2.0})
        df = pd.read_csv(str(Path(tmp_output_dir) / "test_analysis.csv"), index_col=0)
        assert 'sample_A' in df.columns
        assert 'sample_B' in df.columns


# =====================================================================
# write_full_h5
# =====================================================================

class TestWriteFullH5:
    def test_creates_h5_with_group(self, tmp_output_dir):
        from utils.batch_helpers import write_full_h5
        results = {
            'metric_scalar': 0.5,
            'matrix': np.array([[1, 2], [3, 4]]),
            'label': 'test'
        }
        write_full_h5(tmp_output_dir, 'test_analysis', 'sample_X', results)
        h5_path = Path(tmp_output_dir) / "test_analysis.h5"
        assert h5_path.exists()
        with h5py.File(str(h5_path), 'r') as f:
            assert 'sample_X' in f
            assert 'matrix' in f['sample_X']
            np.testing.assert_array_equal(f['sample_X']['matrix'][()], [[1, 2], [3, 4]])

    def test_multiple_samples(self, tmp_output_dir):
        from utils.batch_helpers import write_full_h5
        write_full_h5(tmp_output_dir, 'analysis', 'sA', {'val': 1.0})
        write_full_h5(tmp_output_dir, 'analysis', 'sB', {'val': 2.0})
        with h5py.File(str(Path(tmp_output_dir) / "analysis.h5"), 'r') as f:
            assert 'sA' in f
            assert 'sB' in f


# =====================================================================
# get_concise_metrics
# =====================================================================

class TestGetConciseMetrics:
    def test_known_analysis_types(self):
        from utils.batch_helpers import get_concise_metrics

        # Test each analysis type returns expected keys
        results_cgf = {'conditional_generation_fidelity_mse': 0.1}
        assert get_concise_metrics('cond_gen_fidelity', results_cgf) == {'conditional_generation_fidelity_mse': 0.1}

        results_fd = {'frechet_distance': 5.0, 'mu1': np.zeros(3)}
        metrics_fd = get_concise_metrics('frechet_distance', results_fd)
        assert 'frechet_distance' in metrics_fd
        assert 'mu1' not in metrics_fd

        results_pds = {'predictive_distribution_shift_ks_statistic': 0.3, 'y_hat_test': np.zeros(10)}
        metrics_pds = get_concise_metrics('predictive_dist_shift', results_pds)
        assert 'predictive_distribution_shift_ks_statistic' in metrics_pds
        assert 'y_hat_test' not in metrics_pds

    def test_discriminability(self):
        from utils.batch_helpers import get_concise_metrics
        results = {'auroc': 0.85, 'n_samples': 100}
        metrics = get_concise_metrics('discriminability', results)
        assert metrics == {'auroc': 0.85}

    def test_percent_identity(self):
        from utils.batch_helpers import get_concise_metrics
        results = {
            'average_max_percent_identity_samples_vs_training': 0.7,
            'global_max_percent_identity_samples_vs_training': 0.9,
            'percent_identity_matrix_samples_vs_training': np.zeros((10, 10))
        }
        metrics = get_concise_metrics('percent_identity', results)
        assert 'average_max_percent_identity_samples_vs_training' in metrics
        assert 'percent_identity_matrix_samples_vs_training' not in metrics

    def test_unknown_analysis_returns_empty(self):
        from utils.batch_helpers import get_concise_metrics
        metrics = get_concise_metrics('nonexistent_analysis', {'a': 1})
        assert metrics == {}

    def test_missing_metric_warns(self):
        from utils.batch_helpers import get_concise_metrics
        # frechet_distance expects 'frechet_distance' key
        metrics = get_concise_metrics('frechet_distance', {'other_key': 1.0})
        assert 'frechet_distance' not in metrics

    def test_numpy_scalar_converted(self):
        from utils.batch_helpers import get_concise_metrics
        results = {'auroc': np.float64(0.9)}
        metrics = get_concise_metrics('discriminability', results)
        assert isinstance(metrics['auroc'], float)
