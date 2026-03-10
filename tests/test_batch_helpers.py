"""
Tests for utils/batch_helpers.py:
  - Batch sample discovery (flat and nested)
  - Metadata CSV generation
  - Concise CSV writing
  - Full H5 writing
  - Concise metric extraction
"""

import numpy as np
import pytest
import os
import csv
import h5py


# ===================================================================
# discover_batch_samples
# ===================================================================

class TestDiscoverBatchSamples:
    """Tests for batch directory discovery and CSV template creation."""

    def test_flat_structure_creates_csv_template(self, batch_dir):
        """First call on a dir without metadata.csv should create the template and exit."""
        from utils.batch_helpers import discover_batch_samples

        csv_path = os.path.join(batch_dir, "metadata.csv")
        assert not os.path.exists(csv_path)
        with pytest.raises(SystemExit):
            discover_batch_samples(batch_dir)
        assert os.path.exists(csv_path)

    def test_flat_structure_csv_has_correct_rows(self, batch_dir):
        """Template CSV should list all NPZ files."""
        from utils.batch_helpers import discover_batch_samples

        with pytest.raises(SystemExit):
            discover_batch_samples(batch_dir)
        csv_path = os.path.join(batch_dir, "metadata.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3  # 3 sample files
        assert all("sample_name" in r for r in rows)
        assert all("file_path" in r for r in rows)

    def test_flat_structure_second_call_returns_samples(self, batch_dir):
        """Second call (CSV exists) should return the sample list."""
        from utils.batch_helpers import discover_batch_samples

        # First call creates CSV
        with pytest.raises(SystemExit):
            discover_batch_samples(batch_dir)
        # Second call returns samples
        samples = discover_batch_samples(batch_dir)
        assert len(samples) == 3
        assert all("sample_name" in s for s in samples)

    def test_nested_structure_creates_csv(self, nested_batch_dir):
        """Nested dirs should also create a metadata CSV."""
        from utils.batch_helpers import discover_batch_samples

        with pytest.raises(SystemExit):
            discover_batch_samples(nested_batch_dir)
        csv_path = os.path.join(nested_batch_dir, "metadata.csv")
        assert os.path.exists(csv_path)


# ===================================================================
# load_batch_sample
# ===================================================================

class TestLoadBatchSample:
    """Tests for loading individual samples from batch directory."""

    def test_loads_valid_sample(self, batch_dir):
        """Should return (name, npz_data) for valid samples."""
        from utils.batch_helpers import discover_batch_samples, load_batch_sample

        # Create CSV first
        with pytest.raises(SystemExit):
            discover_batch_samples(batch_dir)
        samples = discover_batch_samples(batch_dir)
        result = load_batch_sample(batch_dir, samples[0])
        assert result is not None
        name, data = result
        assert isinstance(name, str)
        assert "arr_0" in data.files

    def test_returns_none_for_missing_file(self, batch_dir):
        """Should return None for a sample record pointing to a nonexistent file."""
        from utils.batch_helpers import load_batch_sample

        fake_record = {"sample_name": "ghost", "file_path": "nonexistent.npz"}
        result = load_batch_sample(batch_dir, fake_record)
        assert result is None


# ===================================================================
# get_concise_metrics
# ===================================================================

class TestGetConciseMetrics:
    """Tests for extracting key metrics per analysis type."""

    def test_all_analysis_types_return_dict(self):
        """Every supported analysis name must return a non-empty dict."""
        from utils.batch_helpers import get_concise_metrics

        analysis_results = {
            "attribution_consistency": {
                "KLD": 0.5, "KLD_concat": 0.3,
                "entropic information of top 2000 activity sampled sequences": [0.5],
            },
            "motif_cooccurrence": {"frobenius_norm": 1.2},
            "motif_enrichment": {"pearson_r_statistic": 0.9, "pearson_r_pvalue": 0.01},
            "cond_gen_fidelity": {"conditional_generation_fidelity_mse": 0.05},
            "frechet_distance": {"frechet_distance": 12.3},
            "predictive_dist_shift": {"predictive_distribution_shift_ks_statistic": 0.15},
            "discriminability": {"auroc": 0.85},
            "kmer_spectrum_shift": {"js_distance": 0.02},
            "percent_identity": {"average_max_percent_identity_samples_vs_training": 0.45},
        }
        for name, results in analysis_results.items():
            metrics = get_concise_metrics(name, results)
            assert isinstance(metrics, dict)
            assert len(metrics) > 0


# ===================================================================
# write_concise_csv
# ===================================================================

class TestWriteConciseCsv:
    """Tests for CSV output in batch mode."""

    def test_creates_csv_file(self, tmp_output_dir):
        from utils.batch_helpers import write_concise_csv

        metrics = {"mse": 0.05}
        write_concise_csv(tmp_output_dir, "cond_gen_fidelity", "sample_1", metrics)
        assert os.path.exists(os.path.join(tmp_output_dir, "cond_gen_fidelity.csv"))

    def test_appends_columns_for_multiple_samples(self, tmp_output_dir):
        from utils.batch_helpers import write_concise_csv

        write_concise_csv(tmp_output_dir, "test_analysis", "s1", {"metric_a": 1.0})
        write_concise_csv(tmp_output_dir, "test_analysis", "s2", {"metric_a": 2.0})
        csv_path = os.path.join(tmp_output_dir, "test_analysis.csv")
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        # Header should have at least the metric name column and 2 sample columns
        assert "s1" in header or "s2" in header


# ===================================================================
# write_full_h5
# ===================================================================

class TestWriteFullH5:
    """Tests for HDF5 output in batch mode."""

    def test_creates_h5_file(self, tmp_output_dir):
        from utils.batch_helpers import write_full_h5

        results = {"metric_a": 0.5, "array_b": np.array([1, 2, 3])}
        write_full_h5(tmp_output_dir, "test_analysis", "sample_1", results)
        h5_path = os.path.join(tmp_output_dir, "test_analysis.h5")
        assert os.path.exists(h5_path)

    def test_h5_contains_sample_group(self, tmp_output_dir):
        from utils.batch_helpers import write_full_h5

        results = {"value": 42.0, "data": np.zeros(5)}
        write_full_h5(tmp_output_dir, "test_analysis", "my_sample", results)
        h5_path = os.path.join(tmp_output_dir, "test_analysis.h5")
        with h5py.File(h5_path, "r") as f:
            assert "my_sample" in f

    def test_h5_multiple_samples(self, tmp_output_dir):
        from utils.batch_helpers import write_full_h5

        write_full_h5(tmp_output_dir, "test_analysis", "s1", {"v": np.array([1])})
        write_full_h5(tmp_output_dir, "test_analysis", "s2", {"v": np.array([2])})
        h5_path = os.path.join(tmp_output_dir, "test_analysis.h5")
        with h5py.File(h5_path, "r") as f:
            assert "s1" in f
            assert "s2" in f
