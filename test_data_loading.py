"""
Comprehensive tests for data loading refactoring.

Tests the new data loading functions with synthetic data to ensure:
1. Key resolution works correctly
2. File format detection works
3. Backward compatibility is preserved
4. Index-encoded sequences are converted properly
5. Shape handling is correct
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import h5py
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import (
    resolve_key_from_file,
    load_file_by_type,
    ensure_correct_shape,
    extract_sequences,
    is_index_encoded,
    index_to_onehot,
    KEY_PRIORITIES
)


class TestDataLoadingRefactoring:
    """Test suite for data loading refactoring."""

    def __init__(self):
        self.temp_dir = None
        self.test_results = []

    def setup(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp(prefix="d3_test_")
        print(f"Created temp directory: {self.temp_dir}")

    def teardown(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temp directory: {self.temp_dir}")

    def log_result(self, test_name, passed, message=""):
        """Log test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        self.test_results.append((test_name, passed, message))
        print(f"{status}: {test_name}")
        if message:
            print(f"  → {message}")

    def create_synthetic_sequences(self, n_samples=100, seq_len=249, onehot=True):
        """Create synthetic DNA sequences."""
        if onehot:
            # One-hot encoded: (N, L, 4)
            sequences = np.zeros((n_samples, seq_len, 4), dtype=np.float32)
            for i in range(n_samples):
                for j in range(seq_len):
                    base = np.random.randint(0, 4)
                    sequences[i, j, base] = 1.0
            return sequences
        else:
            # Index encoded: (N, L) with values 0-3
            return np.random.randint(0, 4, (n_samples, seq_len), dtype=np.int32)

    # ==================== Test 1: NPZ File Creation and Loading ====================

    def test_npz_default_keys(self):
        """Test NPZ file with default keys."""
        test_name = "NPZ file with default keys (arr_0)"

        try:
            # Create NPZ file with default key
            file_path = os.path.join(self.temp_dir, "samples_default.npz")
            samples = self.create_synthetic_sequences(n_samples=50, seq_len=249)
            np.savez(file_path, arr_0=samples)

            # Load using load_file_by_type
            data, key_used, file_type = load_file_by_type(file_path, 'samples', user_keys=None)

            # Verify
            assert file_type == 'npz', f"Expected 'npz', got '{file_type}'"
            assert isinstance(data, list), "NPZ samples should return a list"
            assert len(data) == 1, f"Expected 1 array, got {len(data)}"
            assert data[0].shape == samples.shape, f"Shape mismatch: {data[0].shape} vs {samples.shape}"

            self.log_result(test_name, True, f"Loaded NPZ with key '{key_used}'")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_npz_custom_keys(self):
        """Test NPZ file with custom user-specified keys."""
        test_name = "NPZ file with custom user-specified keys"

        try:
            # Create NPZ file with multiple keys
            file_path = os.path.join(self.temp_dir, "samples_custom.npz")
            samples1 = self.create_synthetic_sequences(n_samples=30)
            samples2 = self.create_synthetic_sequences(n_samples=40)
            np.savez(file_path, my_samples=samples1, other_data=samples2)

            # Load with user-specified key
            data, key_used, file_type = load_file_by_type(
                file_path, 'samples', user_keys=['my_samples']
            )

            # Verify
            assert key_used == 'my_samples', f"Expected 'my_samples', got '{key_used}'"
            assert data.shape == samples1.shape, "Should load the specified key"

            self.log_result(test_name, True, f"Successfully used custom key '{key_used}'")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    # ==================== Test 2: H5 File Creation and Loading ====================

    def test_h5_default_keys(self):
        """Test H5 file with default priority keys."""
        test_name = "H5 file with default priority keys"

        try:
            file_path = os.path.join(self.temp_dir, "samples_default.h5")
            samples = self.create_synthetic_sequences(n_samples=50)

            # Create H5 with 'arr_0' (highest priority)
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('arr_0', data=samples)
                f.create_dataset('other_key', data=np.zeros((10, 249, 4)))

            # Load without specifying key
            data, key_used, file_type = load_file_by_type(file_path, 'samples')

            # Verify
            assert file_type == 'h5', f"Expected 'h5', got '{file_type}'"
            assert key_used == 'arr_0', f"Expected 'arr_0', got '{key_used}'"
            assert data.shape == samples.shape, "Shape should match"

            self.log_result(test_name, True, f"Loaded H5 with priority key '{key_used}'")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_h5_priority_order(self):
        """Test H5 key priority order."""
        test_name = "H5 key priority order"

        try:
            file_path = os.path.join(self.temp_dir, "samples_priority.h5")

            # Create H5 with multiple keys - should pick 'samples' over 'x_synthetic'
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('x_synthetic', data=np.zeros((10, 249, 4)))
                f.create_dataset('samples', data=np.ones((20, 249, 4)))  # Higher priority

            data, key_used, file_type = load_file_by_type(file_path, 'samples')

            # Should pick 'samples' (second in priority) over 'x_synthetic' (third)
            assert key_used == 'samples', f"Expected 'samples', got '{key_used}'"
            assert data.shape[0] == 20, "Should load the 'samples' dataset"

            self.log_result(test_name, True, "Priority order respected")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_h5_lentimpra_priority(self):
        """Test H5 with lentimpra-specific keys."""
        test_name = "H5 with lentimpra priority (onehot_test)"

        try:
            file_path = os.path.join(self.temp_dir, "test_lentimpra.h5")

            # Create H5 with lentimpra keys
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('X_test', data=np.zeros((10, 230, 4)))
                f.create_dataset('onehot_test', data=np.ones((20, 230, 4)))  # Higher priority for lentimpra

            # Load with lentimpra model type
            data, key_used, file_type = load_file_by_type(
                file_path, 'test', user_keys=None, model_type='lentimpra'
            )

            # Should pick 'onehot_test' for lentimpra
            assert key_used == 'onehot_test', f"Expected 'onehot_test', got '{key_used}'"
            assert data.shape[0] == 20, "Should load the lentimpra-specific dataset"

            self.log_result(test_name, True, "Lentimpra priority keys work correctly")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    # ==================== Test 3: PyTorch File Loading ====================

    def test_pt_file_loading(self):
        """Test PyTorch .pt file loading."""
        test_name = "PyTorch .pt file loading"

        try:
            file_path = os.path.join(self.temp_dir, "samples.pt")
            samples = self.create_synthetic_sequences(n_samples=50)

            # Save as PyTorch tensor
            torch.save(torch.from_numpy(samples), file_path)

            # Load
            data, key_used, file_type = load_file_by_type(file_path, 'samples')

            # Verify
            assert file_type == 'pt', f"Expected 'pt', got '{file_type}'"
            assert key_used == 'tensor', f"Expected 'tensor', got '{key_used}'"
            assert isinstance(data, np.ndarray), "Should convert to numpy array"
            assert data.shape == samples.shape, "Shape should match"

            self.log_result(test_name, True, "PyTorch file loaded successfully")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_h5_disguised_as_pt(self):
        """Test H5 file with .h5 extension but actually PyTorch format."""
        test_name = "H5 file disguised as PyTorch"

        try:
            file_path = os.path.join(self.temp_dir, "samples_disguised.h5")
            samples = self.create_synthetic_sequences(n_samples=50)

            # Save PyTorch tensor with .h5 extension
            torch.save(torch.from_numpy(samples), file_path)

            # Should detect it's actually PyTorch
            data, key_used, file_type = load_file_by_type(file_path, 'samples')

            assert file_type == 'pt', "Should detect as PyTorch despite .h5 extension"
            assert data.shape == samples.shape, "Should load correctly"

            self.log_result(test_name, True, "Correctly detected PyTorch file with .h5 extension")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    # ==================== Test 4: Index-Encoded Sequences ====================

    def test_index_encoded_detection(self):
        """Test detection of index-encoded sequences."""
        test_name = "Index-encoded sequence detection"

        try:
            # Create index-encoded sequences
            index_seqs = self.create_synthetic_sequences(n_samples=50, onehot=False)
            onehot_seqs = self.create_synthetic_sequences(n_samples=50, onehot=True)

            # Test detection
            assert is_index_encoded(index_seqs), "Should detect index-encoded"
            assert not is_index_encoded(onehot_seqs), "Should not detect one-hot as index-encoded"

            self.log_result(test_name, True, "Index encoding detection works correctly")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_index_to_onehot_conversion(self):
        """Test conversion from index-encoded to one-hot."""
        test_name = "Index-to-onehot conversion"

        try:
            # Create index-encoded sequences
            index_seqs = np.array([[0, 1, 2, 3, 0], [3, 2, 1, 0, 1]], dtype=np.int32)

            # Convert
            onehot_seqs = index_to_onehot(index_seqs)

            # Verify shape
            assert onehot_seqs.shape == (2, 5, 4), f"Expected (2, 5, 4), got {onehot_seqs.shape}"

            # Verify correctness
            assert onehot_seqs[0, 0, 0] == 1, "First base should be A (index 0)"
            assert onehot_seqs[0, 1, 1] == 1, "Second base should be C (index 1)"
            assert onehot_seqs[0, 2, 2] == 1, "Third base should be G (index 2)"
            assert onehot_seqs[0, 3, 3] == 1, "Fourth base should be T (index 3)"

            self.log_result(test_name, True, "Index-to-onehot conversion correct")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_index_encoded_file_loading(self):
        """Test loading index-encoded sequences from file."""
        test_name = "Index-encoded file loading and conversion"

        try:
            file_path = os.path.join(self.temp_dir, "index_samples.npz")
            index_seqs = self.create_synthetic_sequences(n_samples=50, seq_len=249, onehot=False)

            # Save index-encoded
            np.savez(file_path, arr_0=index_seqs)

            # Extract using full pipeline
            dummy_data_file = os.path.join(self.temp_dir, "dummy_data.npz")
            test_data = self.create_synthetic_sequences(n_samples=10, seq_len=249)
            train_data = self.create_synthetic_sequences(n_samples=10, seq_len=249)
            np.savez(dummy_data_file, X_test=test_data.transpose(0, 2, 1),
                    X_train=train_data.transpose(0, 2, 1))

            x_test, x_synthetic, x_train = extract_sequences(
                file_path, dummy_data_file, model_type='deepstarr'
            )

            # Verify conversion happened
            assert x_synthetic.shape == (50, 4, 249), f"Expected (50, 4, 249), got {x_synthetic.shape}"
            assert x_synthetic.ndim == 3, "Should be 3D"
            assert x_synthetic.shape[1] == 4, "Channel dimension should be 4"

            self.log_result(test_name, True, "Index-encoded sequences converted automatically")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    # ==================== Test 5: Shape Handling ====================

    def test_ensure_correct_shape_already_correct(self):
        """Test ensure_correct_shape with already correct shape."""
        test_name = "Shape handling - already correct (N, 4, L)"

        try:
            data = np.random.rand(50, 4, 249).astype(np.float32)
            result = ensure_correct_shape(data)

            assert result.shape == (50, 4, 249), "Should maintain correct shape"
            assert np.array_equal(result, data), "Should not modify data"

            self.log_result(test_name, True, "Correct shape preserved")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_ensure_correct_shape_needs_transpose(self):
        """Test ensure_correct_shape with transposition needed."""
        test_name = "Shape handling - needs transpose (N, L, 4) → (N, 4, L)"

        try:
            data = np.random.rand(50, 249, 4).astype(np.float32)
            result = ensure_correct_shape(data)

            assert result.shape == (50, 4, 249), f"Expected (50, 4, 249), got {result.shape}"

            self.log_result(test_name, True, "Transposition applied correctly")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_ensure_correct_shape_index_encoded(self):
        """Test ensure_correct_shape with index-encoded input."""
        test_name = "Shape handling - index-encoded conversion"

        try:
            # Create index-encoded (N, L)
            data = np.random.randint(0, 4, (50, 249), dtype=np.int32)
            result = ensure_correct_shape(data)

            # Should convert to (N, 4, L)
            assert result.shape == (50, 4, 249), f"Expected (50, 4, 249), got {result.shape}"

            self.log_result(test_name, True, "Index-encoded converted and shaped correctly")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    # ==================== Test 6: Extract Sequences Full Pipeline ====================

    def test_extract_sequences_complete(self):
        """Test complete extract_sequences pipeline."""
        test_name = "Complete extract_sequences pipeline"

        try:
            # Create sample files
            samples_file = os.path.join(self.temp_dir, "complete_samples.npz")
            data_file = os.path.join(self.temp_dir, "complete_data.h5")

            # Create synthetic data
            samples = self.create_synthetic_sequences(100, 249)
            test_seqs = self.create_synthetic_sequences(50, 249)
            train_seqs = self.create_synthetic_sequences(200, 249)

            # Save files
            np.savez(samples_file, arr_0=samples)

            with h5py.File(data_file, 'w') as f:
                # Save in (N, 4, L) format
                f.create_dataset('X_test', data=test_seqs.transpose(0, 2, 1))
                f.create_dataset('X_train', data=train_seqs.transpose(0, 2, 1))

            # Extract
            x_test, x_synthetic, x_train = extract_sequences(
                samples_file, data_file, model_type='deepstarr'
            )

            # Verify all outputs
            assert x_test.shape == (50, 4, 249), f"Test shape: {x_test.shape}"
            assert x_synthetic.shape == (100, 4, 249), f"Synthetic shape: {x_synthetic.shape}"
            assert x_train.shape == (200, 4, 249), f"Train shape: {x_train.shape}"

            self.log_result(test_name, True, "Complete pipeline works correctly")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_extract_sequences_with_custom_keys(self):
        """Test extract_sequences with user-specified keys."""
        test_name = "Extract sequences with custom keys"

        try:
            samples_file = os.path.join(self.temp_dir, "custom_samples.h5")
            data_file = os.path.join(self.temp_dir, "custom_data.h5")

            samples = self.create_synthetic_sequences(100, 249)
            test_seqs = self.create_synthetic_sequences(50, 249)
            train_seqs = self.create_synthetic_sequences(200, 249)

            # Save with custom keys
            with h5py.File(samples_file, 'w') as f:
                f.create_dataset('my_samples', data=samples)

            with h5py.File(data_file, 'w') as f:
                f.create_dataset('my_test', data=test_seqs.transpose(0, 2, 1))
                f.create_dataset('my_train', data=train_seqs.transpose(0, 2, 1))

            # Extract with custom keys
            x_test, x_synthetic, x_train = extract_sequences(
                samples_file, data_file,
                samples_keys=['my_samples'],
                test_keys=['my_test'],
                train_keys=['my_train'],
                model_type='deepstarr'
            )

            # Verify
            assert x_test.shape == (50, 4, 249), "Test loaded with custom key"
            assert x_synthetic.shape == (100, 4, 249), "Samples loaded with custom key"
            assert x_train.shape == (200, 4, 249), "Train loaded with custom key"

            self.log_result(test_name, True, "Custom keys work correctly")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    # ==================== Test 7: Backward Compatibility ====================

    def test_backward_compatibility_defaults(self):
        """Test that default behavior is preserved."""
        test_name = "Backward compatibility - default behavior"

        try:
            # Create files using old expected key names
            samples_file = os.path.join(self.temp_dir, "old_samples.h5")
            data_file = os.path.join(self.temp_dir, "old_data.npz")

            samples = self.create_synthetic_sequences(100, 249)
            test_seqs = self.create_synthetic_sequences(50, 249)
            train_seqs = self.create_synthetic_sequences(200, 249)

            # Old-style keys
            with h5py.File(samples_file, 'w') as f:
                f.create_dataset('sequences_onehot', data=samples)

            np.savez(data_file,
                    x_test=test_seqs.transpose(0, 2, 1),
                    x_train=train_seqs.transpose(0, 2, 1))

            # Should load without specifying keys (backward compatible)
            x_test, x_synthetic, x_train = extract_sequences(
                samples_file, data_file, model_type='deepstarr'
            )

            assert x_test.shape[0] == 50, "Test data loaded with old keys"
            assert x_synthetic.shape[0] == 100, "Samples loaded with old keys"
            assert x_train.shape[0] == 200, "Train data loaded with old keys"

            self.log_result(test_name, True, "Backward compatibility maintained")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    # ==================== Test 8: Error Handling ====================

    def test_invalid_key_error(self):
        """Test that invalid user keys raise appropriate errors."""
        test_name = "Error handling - invalid user key"

        try:
            file_path = os.path.join(self.temp_dir, "error_test.npz")
            samples = self.create_synthetic_sequences(50)
            np.savez(file_path, arr_0=samples)

            # Try to load with non-existent key
            try:
                data, key_used, file_type = load_file_by_type(
                    file_path, 'samples', user_keys=['nonexistent_key']
                )
                self.log_result(test_name, False, "Should have raised KeyError")
            except KeyError as e:
                # Expected behavior
                assert 'nonexistent_key' in str(e), "Error should mention the key"
                self.log_result(test_name, True, "Correctly raised KeyError for invalid key")

        except Exception as e:
            self.log_result(test_name, False, f"Unexpected error: {str(e)}")

    # ==================== Run All Tests ====================

    def run_all_tests(self):
        """Run all test methods."""
        print("\n" + "="*70)
        print("D3 DATA LOADING REFACTORING TEST SUITE")
        print("="*70 + "\n")

        self.setup()

        try:
            # Run all test methods
            test_methods = [
                self.test_npz_default_keys,
                self.test_npz_custom_keys,
                self.test_h5_default_keys,
                self.test_h5_priority_order,
                self.test_h5_lentimpra_priority,
                self.test_pt_file_loading,
                self.test_h5_disguised_as_pt,
                self.test_index_encoded_detection,
                self.test_index_to_onehot_conversion,
                self.test_index_encoded_file_loading,
                self.test_ensure_correct_shape_already_correct,
                self.test_ensure_correct_shape_needs_transpose,
                self.test_ensure_correct_shape_index_encoded,
                self.test_extract_sequences_complete,
                self.test_extract_sequences_with_custom_keys,
                self.test_backward_compatibility_defaults,
                self.test_invalid_key_error,
            ]

            for test_method in test_methods:
                print()
                test_method()

        finally:
            self.teardown()

        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        total = len(self.test_results)
        passed = sum(1 for _, p, _ in self.test_results if p)
        failed = total - passed

        print(f"\nTotal tests: {total}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for name, passed, msg in self.test_results:
                if not passed:
                    print(f"  - {name}: {msg}")

        print("\n" + "="*70)

        if failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {failed} test(s) failed")
        print("="*70 + "\n")

        return failed == 0


def main():
    """Main entry point."""
    tester = TestDataLoadingRefactoring()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
