import numpy as np
from numpy import load
import h5py
import pandas as pd
import torch
import tqdm as tqdm_module
import os
import signal
import time

from deepstarr import *
from mpralegnet import LitModel
from sei import Sei, NonStrandSpecific
import re


class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(seconds):
    """Context manager for timing out operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Reset the alarm and handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator

def detect_data_format(file_path):
    """
    Detect the format of a data file.

    Args:
        file_path: Path to the data file

    Returns:
        str: 'npz', 'h5', or 'unknown'
    """
    if not os.path.exists(file_path):
        return 'unknown'

    if file_path.endswith('.npz'):
        return 'npz'
    elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
        return 'h5'
    else:
        # Try to detect based on content
        try:
            with h5py.File(file_path, 'r') as f:
                return 'h5'
        except:
            try:
                np.load(file_path)
                return 'npz'
            except:
                return 'unknown'

def print_data_file_info(file_path):
    """
    Print information about a data file's contents.

    Args:
        file_path: Path to the data file
    """
    format_type = detect_data_format(file_path)
    print(f"Data file format: {format_type}")

    if format_type == 'npz':
        data = np.load(file_path)
        print(f"Available keys: {list(data.files)}")
        for key in data.files[:5]:  # Show first 5 keys
            print(f"  {key}: {data[key].shape} {data[key].dtype}")
    elif format_type == 'h5':
        with h5py.File(file_path, 'r') as f:
            print(f"Available keys: {list(f.keys())}")
            for key in list(f.keys())[:5]:  # Show first 5 keys
                print(f"  {key}: {f[key].shape} {f[key].dtype}")

class EmbeddingExtractor:
    def __init__(self):
        self.embedding = None

    def hook(self, module, input, output):
        self.embedding = output.detach()

def extract_data(samples_file_path, data_file):
    """Extract data for deepstarr from either .h5 or .npz files."""
    # Load samples from .npz or .h5 file
    if samples_file_path.endswith('.npz'):
        data = load(samples_file_path)
        samples = []
        lst = data.files
        for item in lst:
            samples.append(data[item])
    elif samples_file_path.endswith('.h5') or samples_file_path.endswith('.hdf5'):
        with h5py.File(samples_file_path, 'r') as f:
            samples = []
            # Try common naming conventions for samples
            if 'arr_0' in f.keys():
                samples.append(f['arr_0'][()])
            elif 'samples' in f.keys():
                samples.append(f['samples'][()])
            elif 'x_synthetic' in f.keys():
                samples.append(f['x_synthetic'][()])
            elif 'synthetic_data' in f.keys():
                samples.append(f['synthetic_data'][()])
            else:
                # Take the first available key
                first_key = list(f.keys())[0]
                samples.append(f[first_key][()])
                print(f"Warning: Using key '{first_key}' for samples from H5 file")
    else:
        raise ValueError(f"Unsupported samples file format. Expected .npz or .h5/.hdf5, got: {samples_file_path}")

    # Load training/test data based on file format
    if data_file.endswith('.npz'):
        # Load from .npz file
        npz_data = load(data_file)

        # Try different naming conventions for test data
        if 'x_test' in npz_data.files:
            x_test = npz_data['x_test']
        elif 'X_test' in npz_data.files:
            x_test = npz_data['X_test']
        elif 'test_data' in npz_data.files:
            x_test = npz_data['test_data']
        else:
            raise KeyError(f"Could not find test data in .npz file. Available keys: {npz_data.files}")

        # Try different naming conventions for training data
        if 'x_train' in npz_data.files:
            x_train = npz_data['x_train']
        elif 'X_train' in npz_data.files:
            x_train = npz_data['X_train']
        elif 'train_data' in npz_data.files:
            x_train = npz_data['train_data']
        else:
            raise KeyError(f"Could not find training data in .npz file. Available keys: {npz_data.files}")

    else:
        # Load from .h5 file (existing functionality)
        with h5py.File(data_file, 'r') as f:
            # Access the data for the specific X_test key
            x_test = f['X_test'][()]
            x_train = f['X_train'][()]

    # Transpose samples to get shape (n, 4, seq_len)
    if samples[0].ndim == 3 and samples[0].shape[1] != 4:
        # Transpose from (n, seq_len, 4) to (n, 4, seq_len)
        x_synthetic = np.transpose(samples[0], (0, 2, 1))
    else:
        # Already in correct format
        x_synthetic = samples[0]

    return x_test, x_synthetic, x_train

def extract_lentimpra_data(samples_file_path, data_file):
    """Extract data for lentimpra with 230-length sequences from either .h5 or .npz files."""
    # Load samples from .npz or .h5 file
    if samples_file_path.endswith('.npz'):
        data = load(samples_file_path)
        samples = []
        lst = data.files
        for item in lst:
            samples.append(data[item])
    elif samples_file_path.endswith('.h5') or samples_file_path.endswith('.hdf5'):
        with h5py.File(samples_file_path, 'r') as f:
            samples = []
            # Try common naming conventions for samples
            if 'arr_0' in f.keys():
                samples.append(f['arr_0'][()])
            elif 'samples' in f.keys():
                samples.append(f['samples'][()])
            elif 'x_synthetic' in f.keys():
                samples.append(f['x_synthetic'][()])
            elif 'synthetic_data' in f.keys():
                samples.append(f['synthetic_data'][()])
            else:
                # Take the first available key
                first_key = list(f.keys())[0]
                samples.append(f[first_key][()])
                print(f"Warning: Using key '{first_key}' for samples from H5 file")
    else:
        raise ValueError(f"Unsupported samples file format. Expected .npz or .h5/.hdf5, got: {samples_file_path}")

    # Load training/test data based on file format
    if data_file.endswith('.npz'):
        # Load from .npz file
        npz_data = load(data_file)

        # Try different naming conventions for test data
        if 'x_test' in npz_data.files:
            x_test = npz_data['x_test']
        elif 'X_test' in npz_data.files:
            x_test = npz_data['X_test']
        elif 'test_data' in npz_data.files:
            x_test = npz_data['test_data']
        elif 'onehot_test' in npz_data.files:
            x_test = npz_data['onehot_test']
            if x_test.shape[-1] == 4:  # (n, 230, 4) format
                x_test = np.transpose(x_test, (0, 2, 1))  # Convert to (n, 4, 230)
        else:
            raise KeyError(f"Could not find test data in .npz file. Available keys: {npz_data.files}")

        # Try different naming conventions for training data
        if 'x_train' in npz_data.files:
            x_train = npz_data['x_train']
        elif 'X_train' in npz_data.files:
            x_train = npz_data['X_train']
        elif 'train_data' in npz_data.files:
            x_train = npz_data['train_data']
        elif 'onehot_train' in npz_data.files:
            x_train = npz_data['onehot_train']
            if x_train.shape[-1] == 4:  # (n, 230, 4) format
                x_train = np.transpose(x_train, (0, 2, 1))  # Convert to (n, 4, 230)
        else:
            raise KeyError(f"Could not find training data in .npz file. Available keys: {npz_data.files}")

    else:
        # Load from .h5 file (existing functionality)
        with h5py.File(data_file, 'r') as f:
            # Access the data for lentimpra format
            # Assuming similar structure but with onehot_test and onehot_train
            if 'onehot_test' in f.keys():
                x_test = f['onehot_test'][()]  # shape: (n, 230, 4)
                x_test = np.transpose(x_test, (0, 2, 1))  # Convert to (n, 4, 230)
            elif 'X_test' in f.keys():
                x_test = f['X_test'][()]
            else:
                raise KeyError("Could not find test data in lentimpra file")

            if 'onehot_train' in f.keys():
                x_train = f['onehot_train'][()]  # shape: (n, 230, 4)
                x_train = np.transpose(x_train, (0, 2, 1))  # Convert to (n, 4, 230)
            elif 'X_train' in f.keys():
                x_train = f['X_train'][()]
            else:
                raise KeyError("Could not find training data in lentimpra file")

    # Handle samples - they should be 230 length for lentimpra
    if samples[0].shape[-1] == 230:
        # Already correct length: (n, 4, 230)
        x_synthetic = samples[0]
    else:
        # If they come in different format, transpose: (n, 230, 4) -> (n, 4, 230)
        x_synthetic = np.transpose(samples[0], (0, 2, 1))

    return x_test, x_synthetic, x_train

def extract_sei_data(samples_file_path, data_file):
    """Extract data for SEI with 4096-length sequences from either .h5 or .npz files."""
    print(f"  Loading samples from: {samples_file_path}")

    # Load samples from .npz or .h5 file
    if samples_file_path.endswith('.npz'):
        print("    Loading NPZ samples...")
        data = load(samples_file_path)
        samples = []
        lst = data.files
        print(f"    Found keys: {lst}")
        for item in lst:
            samples.append(data[item])
            print(f"    Loaded {item}: {data[item].shape}")
    elif samples_file_path.endswith('.h5') or samples_file_path.endswith('.hdf5'):
        print("    Loading H5 samples...")
        with h5py.File(samples_file_path, 'r') as f:
            print(f"    Found keys: {list(f.keys())}")
            samples = []
            # Try common naming conventions for samples
            if 'arr_0' in f.keys():
                samples.append(f['arr_0'][()])
                print(f"    Loaded arr_0: {f['arr_0'][()].shape}")
            elif 'samples' in f.keys():
                samples.append(f['samples'][()])
                print(f"    Loaded samples: {f['samples'][()].shape}")
            elif 'x_synthetic' in f.keys():
                samples.append(f['x_synthetic'][()])
                print(f"    Loaded x_synthetic: {f['x_synthetic'][()].shape}")
            elif 'synthetic_data' in f.keys():
                samples.append(f['synthetic_data'][()])
                print(f"    Loaded synthetic_data: {f['synthetic_data'][()].shape}")
            else:
                # Take the first available key
                first_key = list(f.keys())[0]
                samples.append(f[first_key][()])
                print(f"    Warning: Using key '{first_key}' for samples from H5 file: {f[first_key][()].shape}")
    else:
        raise ValueError(f"Unsupported samples file format. Expected .npz or .h5/.hdf5, got: {samples_file_path}")

    print(f"  Loading training/test data from: {data_file}")

    # Load training/test data based on file format
    if data_file.endswith('.npz'):
        print("    Loading NPZ training/test data...")
        # Load from .npz file
        npz_data = load(data_file)
        print(f"    Found keys: {npz_data.files}")

        # Check for promoter dataset format (train/valid/test splits)
        if 'train' in npz_data.files and 'test' in npz_data.files:
            print("    Detected promoter dataset format")
            # Promoter dataset format: each split has shape (N, seq_len, 6)
            # where [:, :, :4] are sequences and [:, :, 4:5] is activity
            test_data = npz_data['test']
            train_data = npz_data['train']
            print(f"    Test data shape: {test_data.shape}")
            print(f"    Train data shape: {train_data.shape}")

            # Extract sequences (first 4 channels) and transpose to (N, 4, seq_len)
            x_test = test_data[:, :, :4].transpose(0, 2, 1)
            x_train = train_data[:, :, :4].transpose(0, 2, 1)
            print(f"    Extracted test sequences: {x_test.shape}")
            print(f"    Extracted train sequences: {x_train.shape}")
        else:
            print("    Using standard dataset format")
            # Try standard naming conventions for other formats
            if 'x_test' in npz_data.files:
                x_test = npz_data['x_test']
            elif 'X_test' in npz_data.files:
                x_test = npz_data['X_test']
            elif 'test_data' in npz_data.files:
                x_test = npz_data['test_data']
            else:
                raise KeyError(f"Could not find test data in .npz file. Available keys: {npz_data.files}")

            # Try different naming conventions for training data
            if 'x_train' in npz_data.files:
                x_train = npz_data['x_train']
            elif 'X_train' in npz_data.files:
                x_train = npz_data['X_train']
            elif 'train_data' in npz_data.files:
                x_train = npz_data['train_data']
            else:
                raise KeyError(f"Could not find training data in .npz file. Available keys: {npz_data.files}")

    else:
        print("    Loading H5 training/test data...")
        # Load from .h5 file
        with h5py.File(data_file, 'r') as f:
            print(f"    Found keys: {list(f.keys())}")
            # Check for promoter dataset format
            if 'train' in f.keys() and 'test' in f.keys():
                print("    Detected promoter dataset format")
                # Promoter dataset format
                test_data = f['test'][()]
                train_data = f['train'][()]
                print(f"    Test data shape: {test_data.shape}")
                print(f"    Train data shape: {train_data.shape}")

                # Extract sequences and transpose
                x_test = test_data[:, :, :4].transpose(0, 2, 1)
                x_train = train_data[:, :, :4].transpose(0, 2, 1)
                print(f"    Extracted test sequences: {x_test.shape}")
                print(f"    Extracted train sequences: {x_train.shape}")
            else:
                print("    Using standard dataset format")
                # Standard format
                if 'X_test' in f.keys():
                    x_test = f['X_test'][()]
                elif 'x_test' in f.keys():
                    x_test = f['x_test'][()]
                else:
                    raise KeyError("Could not find test data in SEI file")

                if 'X_train' in f.keys():
                    x_train = f['X_train'][()]
                elif 'x_train' in f.keys():
                    x_train = f['x_train'][()]
                else:
                    raise KeyError("Could not find training data in SEI file")

    # Handle samples - ensure proper format for SEI (sequences should be padded to 4096)
    if samples[0].ndim == 3:
        if samples[0].shape[-1] == 6:
            # Promoter format: (n, seq_len, 6) - extract first 4 channels and transpose
            x_synthetic = samples[0][:, :, :4].transpose(0, 2, 1)
        elif samples[0].shape[-1] == 4:
            # Standard format: (n, seq_len, 4) - transpose to (n, 4, seq_len)
            x_synthetic = samples[0].transpose(0, 2, 1)
        elif samples[0].shape[1] == 4:
            # Already in (n, 4, seq_len) format
            x_synthetic = samples[0]
        else:
            # Try transpose anyway
            x_synthetic = np.transpose(samples[0], (0, 2, 1))
    else:
        # Already in correct format or 2D
        x_synthetic = samples[0]

    # Ensure sequences are padded to 4096 for SEI
    current_seq_len = x_synthetic.shape[-1]
    if current_seq_len < 4096:
        # Pad sequences to 4096 length with uniform background (0.25 for each nucleotide)
        pad_size = 4096 - current_seq_len
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left

        # Create padding with uniform background
        padding_shape = (x_synthetic.shape[0], 4, pad_left)
        left_pad = np.full(padding_shape, 0.25)
        padding_shape = (x_synthetic.shape[0], 4, pad_right)
        right_pad = np.full(padding_shape, 0.25)

        # Concatenate padding
        x_synthetic = np.concatenate([left_pad, x_synthetic, right_pad], axis=-1)
    elif current_seq_len > 4096:
        # Truncate if longer than 4096
        center_start = (current_seq_len - 4096) // 2
        x_synthetic = x_synthetic[:, :, center_start:center_start + 4096]

    # Apply same padding logic to test and train data
    for data_name, data_array in [('x_test', x_test), ('x_train', x_train)]:
        current_seq_len = data_array.shape[-1]
        if current_seq_len < 4096:
            # Pad sequences to 4096 length
            pad_size = 4096 - current_seq_len
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left

            # Create padding with uniform background
            padding_shape = (data_array.shape[0], 4, pad_left)
            left_pad = np.full(padding_shape, 0.25)
            padding_shape = (data_array.shape[0], 4, pad_right)
            right_pad = np.full(padding_shape, 0.25)

            # Apply padding
            data_array = np.concatenate([left_pad, data_array, right_pad], axis=-1)
        elif current_seq_len > 4096:
            # Truncate if longer than 4096
            center_start = (current_seq_len - 4096) // 2
            data_array = data_array[:, :, center_start:center_start + 4096]

        # Update the variable
        if data_name == 'x_test':
            x_test = data_array
        else:
            x_train = data_array

    return x_test, x_synthetic, x_train

def numpy_to_tensor(array):
    return torch.from_numpy(array).float()

def load_deepstarr(oracle_path):
    """Load DeepSTARR model from checkpoint."""
    #load model
    ckpt_aug_path = oracle_path
    deepstarr = PL_DeepSTARR.load_from_checkpoint(ckpt_aug_path).eval()

    return deepstarr

def load_mpralegnet(oracle_path):
    """Load MPRALegNet model from checkpoint."""
    try:
        from mpralegnet import load_model
        # Use the proper load_model function that handles tr_cfg correctly
        model, config = load_model(oracle_path)
        return model
    except Exception as e:
        if "lightning" in str(e).lower():
            raise ImportError(
                "PyTorch Lightning is required to load MPRALegNet models. "
                "Please install with: pip install lightning or pytorch-lightning"
            ) from e
        elif "tr_cfg" in str(e).lower():
            # Try with default config if tr_cfg is missing
            try:
                from mpralegnet import get_default_config
                default_config = get_default_config()
                model = LitModel.load_from_checkpoint(oracle_path, tr_cfg=default_config).eval()
                return model
            except Exception as inner_e:
                raise RuntimeError(
                    f"Failed to load MPRALegNet model. Original error: {e}. "
                    f"Tried with default config but got: {inner_e}"
                ) from e
        raise

def upgrade_state_dict(state_dict, prefixes=["encoder.sentence_encoder.", "encoder.", "module."]):
    """Removes prefixes from state dict keys for SEI model loading."""
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict

def load_sei_model(oracle_path):
    """Load SEI model from checkpoint."""
    import torch
    import numpy as np

    try:
        print(f"✓ Creating SEI model architecture...")
        # Create SEI model with proper architecture
        sei_model = Sei(4096, 21907)  # 4096 seq length, 21907 features
        oracle = NonStrandSpecific(sei_model)
        print(f"✓ SEI model architecture created")

        # Load checkpoint if provided
        if oracle_path and oracle_path != 'null':
            print(f"✓ Loading SEI checkpoint from: {oracle_path}")

            # Check if file exists and is readable
            if not os.path.exists(oracle_path):
                raise FileNotFoundError(f"Checkpoint file not found: {oracle_path}")

            # Try to get file size to verify it's not corrupted
            try:
                file_size = os.path.getsize(oracle_path)
                print(f"✓ Checkpoint file size: {file_size / (1024*1024):.1f} MB")
            except Exception as e:
                print(f"Warning: Could not get file size: {e}")

            print(f"✓ Attempting to load checkpoint...")
            checkpoint = None

            try:
                # First try: weights_only=False (for trusted checkpoints)
                print("  Trying loading with weights_only=False...")
                checkpoint = torch.load(oracle_path, map_location='cpu', weights_only=False)
                print("✓ Successfully loaded checkpoint with weights_only=False")
            except Exception as first_error:
                print(f"  weights_only=False loading failed: {first_error}")
                try:
                    # Second try: Use safe globals context manager for numpy objects
                    print("  Trying loading with safe globals for numpy...")
                    with torch.serialization.safe_globals([
                        np.core.multiarray.scalar,
                        np.core.multiarray._reconstruct,
                        np.ndarray,
                        np.dtype,
                        np.int64,
                        np.float32,
                        np.float64,
                        np.bool_
                    ]):
                        checkpoint = torch.load(oracle_path, map_location='cpu', weights_only=True)
                    print("✓ Successfully loaded checkpoint with safe globals")
                except Exception as second_error:
                    print(f"  Safe globals loading failed: {second_error}")
                    try:
                        # Third try: Legacy method for older PyTorch
                        print("  Trying legacy loading method...")
                        checkpoint = torch.load(oracle_path, map_location=torch.device('cpu'), weights_only=False)
                        print("✓ Successfully loaded checkpoint with legacy method")
                    except Exception as third_error:
                        # If all fail, raise a comprehensive error
                        raise RuntimeError(
                            f"Failed to load checkpoint from {oracle_path}. "
                            f"This may be due to:\n"
                            f"1. Corrupted checkpoint file\n"
                            f"2. Incompatible PyTorch version\n"
                            f"3. File system issues on cluster\n"
                            f"Errors:\n"
                            f"  weights_only=False: {first_error}\n"
                            f"  safe_globals: {second_error}\n"
                            f"  legacy: {third_error}"
                        )

            print(f"✓ Checkpoint loaded, processing state dict...")

            # Extract state dict - follow the working example pattern
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✓ Found 'state_dict' key in checkpoint")
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("✓ Found 'model_state_dict' key in checkpoint")
            elif isinstance(checkpoint, dict):
                # If checkpoint is just the state dict itself
                state_dict = checkpoint
                print("✓ Using checkpoint dict directly as state dict")
            else:
                raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

            print(f"✓ State dict has {len(state_dict)} keys")
            print(f"✓ Sample checkpoint keys: {list(state_dict.keys())[:3]}")

            # Follow the working example: upgrade_state_dict with module prefix removal
            print(f"✓ Cleaning up state dict keys using working example pattern...")
            state_dict = upgrade_state_dict(state_dict, prefixes=['module.'])

            print(f"✓ Sample cleaned keys: {list(state_dict.keys())[:3]}")

            print(f"✓ Loading state dict into model...")
            # Load state dict with strict=False to handle missing/extra keys (like the working example)
            missing_keys, unexpected_keys = oracle.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"⚠ Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"⚠ Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

            print("✓ Successfully loaded state dict into model")

        # Ensure model is in eval mode
        print(f"✓ Setting model to eval mode...")
        oracle.eval()
        print("✓ SEI model loaded and set to eval mode")

        # Test a small forward pass to ensure model works
        print(f"✓ Testing model with dummy input...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 4, 4096)
            dummy_output = oracle(dummy_input)
            print(f"✓ Model test successful - output shape: {dummy_output.shape}")

        return oracle

    except Exception as e:
        print(f"✗ Error in load_sei_model: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load SEI oracle model: {e}")

def load_oracle_model(oracle_path, model_type='deepstarr'):
    """Load oracle model based on model type."""
    if model_type.lower() == 'deepstarr':
        return load_deepstarr(oracle_path)
    elif model_type.lower() in ['mpralegnet', 'lentimpra']:
        return load_mpralegnet(oracle_path)
    elif model_type.lower() == 'sei':
        return load_sei_model(oracle_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_predictions(x_test_tensor, x_synthetic_tensor, oracle_model, model_type='deepstarr'):
    """Load predictions from oracle model (works with DeepSTARR, MPRALegNet, and SEI)."""
    # Ensure tensors are on the same device as the model
    device = next(oracle_model.parameters()).device
    x_test_tensor = x_test_tensor.to(device)
    x_synthetic_tensor = x_synthetic_tensor.to(device)

    #run model predictions
    y_hat_test = oracle_model(x_test_tensor)
    y_hat_syn = oracle_model(x_synthetic_tensor)

    # For SEI model, filter for specific features (e.g., H3K4me3)
    if model_type.lower() == 'sei':
        # Take mean across all chromatin features as a proxy
        # In practice, you might want to filter for specific features
        y_hat_test = y_hat_test.mean(dim=1, keepdim=True)
        y_hat_syn = y_hat_syn.mean(dim=1, keepdim=True)

    #returns numpy arrays of oracle predictions from samples and x test
    return y_hat_test.detach().cpu().numpy(), y_hat_syn.detach().cpu().numpy()

def load_predictions_batched(x_test_tensor, x_synthetic_tensor, oracle_model, model_type='deepstarr', batch_size=32):
    """Load predictions from oracle model with batching for GPU memory efficiency."""
    # Ensure tensors are on the same device as the model
    device = next(oracle_model.parameters()).device

    def process_batched(tensor, batch_size):
        """Process tensor in batches to avoid GPU memory issues."""
        predictions = []
        tensor = tensor.to(device)

        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                batch = tensor[i:i+batch_size]
                batch_pred = oracle_model(batch)

                # For SEI model, filter for specific features
                if model_type.lower() == 'sei':
                    batch_pred = batch_pred.mean(dim=1, keepdim=True)

                predictions.append(batch_pred.detach().cpu())

                # Clear GPU cache after each batch
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return torch.cat(predictions, dim=0).numpy()

    print(f"Processing {len(x_test_tensor)} test sequences in batches of {batch_size}...")
    y_hat_test = process_batched(x_test_tensor, batch_size)

    print(f"Processing {len(x_synthetic_tensor)} synthetic sequences in batches of {batch_size}...")
    y_hat_syn = process_batched(x_synthetic_tensor, batch_size)

    return y_hat_test, y_hat_syn

def load_predictions_deepstarr(x_test_tensor, x_synthetic_tensor, deepstarr):
    """Legacy function - use load_predictions instead."""
    return load_predictions(x_test_tensor, x_synthetic_tensor, deepstarr)


extractor = EmbeddingExtractor()
def get_penultimate_embeddings(model, x, model_type='deepstarr'):
    """Get penultimate embeddings from model (works with DeepSTARR, MPRALegNet, and SEI)."""
    # Ensure tensor is on the same device as the model
    device = next(model.parameters()).device
    x = x.to(device)

    # Find the penultimate layer based on model type
    if model_type.lower() == 'deepstarr':
        target_layer = 'model.batchnorm6'
    elif model_type.lower() in ['mpralegnet', 'lentimpra']:
        # For MPRALegNet, hook into the last layer before output
        target_layer = 'model.head.2'  # This is the activation before final linear layer
    elif model_type.lower() == 'sei':
        # For SEI, hook into the spline transformation layer
        target_layer = 'model.spline_tr.1'  # BSplineTransformation layer
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Find the penultimate layer
    for name, module in model.named_modules():
        if name == target_layer:
            handle = module.register_forward_hook(extractor.hook)
            break
    else:
        raise ValueError(f"Could not find '{target_layer}' layer in {model_type} model")

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Remove the hook
    handle.remove()

    return extractor.embedding

#preparing data to put into kmer_statistics function
def put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor):
    return x_test_tensor.detach().cpu().numpy().transpose(0,2,1), x_synthetic_tensor.detach().cpu().numpy().transpose(0,2,1)

def write_to_h5(filename, data_dict):
    """
    Write multiple columns of data to an HDF5 file.
    
    :param filename: Name of the HDF5 file to create
    :param data_dict: Dictionary where keys are column names and values are data arrays
    """
    with h5py.File(filename, 'w') as hf:
        for column_name, data in data_dict.items():
            hf.create_dataset(column_name, data=data)


#converting a one hot encoded sequence into ACGT
def one_hot_to_seq(
    X,
    dna_dict = {
        0: "A",
        1: "C",
        2: "G",
        3: "T"
      }
    ):
    # convert one hot to A,C,G,T
    seq_list = []

    for index in tqdm_module.tqdm(range(len(X)), desc="Converting sequences to text"): #for loop is what actually converts a list of one-hot encoded sequences into ACGT

        seq = X[index]

        seq_list += ["".join([dna_dict[np.where(i)[0][0]] for i in seq])]

    return seq_list


#create a fasta file given a sequence and a path w the file name
def create_fasta_file(sequence_list, path):
    '''
    sequence_list is the input sequences to put into the fasta file
    path is the output filepath
    '''
    output_path = path
    output_file = open(output_path, 'w')
    for i in tqdm_module.tqdm(range(len(sequence_list)), desc="Writing FASTA file"):
        identifier_line = '>Seq' + str(i) + '\n'
        output_file.write(identifier_line)
        sequence_line = sequence_list[i]
        output_file.write(sequence_line + '\n')
