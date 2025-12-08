import numpy as np
from numpy import load
import h5py
import pandas as pd
import torch
import tqdm as tqdm_module
import os

from deepstarr import *
from mpralegnet import LitModel


def is_index_encoded(sequences):
    """
    Detect if sequences are index-encoded (0123=ACGT) or one-hot encoded.

    Args:
        sequences: numpy array of sequences

    Returns:
        bool: True if index-encoded, False if one-hot encoded
    """
    # Index-encoded sequences are 2D (N, L) with integer values 0-3
    # One-hot encoded sequences are 3D (N, L, 4) or (N, 4, L)

    if sequences.ndim == 2:
        # 2D array is likely index-encoded
        # Check if values are integers in range [0, 3]
        if np.issubdtype(sequences.dtype, np.integer):
            unique_vals = np.unique(sequences)
            if len(unique_vals) <= 4 and np.all((unique_vals >= 0) & (unique_vals <= 3)):
                return True
        # Also check if float values are effectively integers 0-3
        elif np.issubdtype(sequences.dtype, np.floating):
            if np.all(sequences == sequences.astype(int)):
                int_seqs = sequences.astype(int)
                unique_vals = np.unique(int_seqs)
                if len(unique_vals) <= 4 and np.all((unique_vals >= 0) & (unique_vals <= 3)):
                    return True

    return False


def index_to_onehot(sequences, num_classes=4):
    """
    Convert index-encoded sequences to one-hot encoding.

    Args:
        sequences: numpy array of shape (N, L) with values 0-3 representing ACGT
        num_classes: number of classes (default 4 for DNA)

    Returns:
        numpy array of shape (N, L, 4) with one-hot encoding
    """
    # Ensure sequences are integers
    sequences = sequences.astype(int)

    # Get shape
    N, L = sequences.shape

    # Create one-hot encoding
    one_hot = np.zeros((N, L, num_classes), dtype=np.float32)

    # Fill in the one-hot encoding
    for i in range(N):
        for j in range(L):
            if 0 <= sequences[i, j] < num_classes:
                one_hot[i, j, sequences[i, j]] = 1.0

    return one_hot


KEY_PRIORITIES = {
    'samples': {
        'h5': ['arr_0', 'samples', 'x_synthetic', 'synthetic_data',
               'shuffled_sequences', 'sequences_onehot'],
        'npz': 'all_keys'
    },
    'test': {
        'h5': ['X_test', 'x_test', 'test_data'],
        'h5_lentimpra': ['onehot_test', 'X_test', 'x_test', 'test_data'],
        'npz': ['X_test', 'x_test', 'test_data'],
        'npz_lentimpra': ['onehot_test', 'X_test', 'x_test', 'test_data']
    },
    'train': {
        'h5': ['X_train', 'x_train', 'train_data'],
        'h5_lentimpra': ['onehot_train', 'X_train', 'x_train', 'train_data'],
        'npz': ['X_train', 'x_train', 'train_data'],
        'npz_lentimpra': ['onehot_train', 'X_train', 'x_train', 'train_data']
    }
}


def resolve_key_from_file(file_handle, file_type, data_category, user_keys=None, model_type='deepstarr'):
    """
    Resolve and load data from a file using priority-based key matching.

    Args:
        file_handle: Open file handle (h5py.File or np.lib.npyio.NpzFile)
        file_type: 'npz' or 'h5'
        data_category: 'samples', 'test', or 'train'
        user_keys: User-specified keys via --samples-key (overrides defaults)
        model_type: Model type affects priority for lentimpra

    Returns:
        Tuple of (data_array, key_used) or (data_list, keys_used) for NPZ samples
    """
    available_keys = list(file_handle.keys()) if file_type == 'h5' else file_handle.files

    if user_keys:
        for key in user_keys:
            if key in available_keys:
                data = file_handle[key][()] if file_type == 'h5' else file_handle[key]
                return data, key
        raise KeyError(f"None of the specified keys {user_keys} found in {file_type} file. Available: {available_keys}")

    priority_key = f"{file_type}_{model_type}" if model_type in ['lentimpra', 'mpralegnet', 'multi-oracle'] else file_type
    priorities = KEY_PRIORITIES.get(data_category, {}).get(priority_key, KEY_PRIORITIES[data_category].get(file_type, []))

    if priorities == 'all_keys' and data_category == 'samples':
        data_list = [file_handle[key] for key in available_keys]
        return data_list, available_keys

    for key in priorities:
        if key in available_keys:
            data = file_handle[key][()] if file_type == 'h5' else file_handle[key]
            return data, key

    if data_category == 'test':
        fallback_key = available_keys[0]
    elif data_category == 'train':
        fallback_key = available_keys[1] if len(available_keys) > 1 else available_keys[0]
    else:
        fallback_key = available_keys[0]

    print(f"Warning: Using key '{fallback_key}' for {data_category} data from {file_type.upper()} file")
    data = file_handle[fallback_key][()] if file_type == 'h5' else file_handle[fallback_key]
    return data, fallback_key


def load_file_by_type(file_path, data_category, user_keys=None, model_type='deepstarr'):
    """
    Load data from NPZ, H5, or PT file.

    Args:
        file_path: Path to the data file
        data_category: 'samples', 'test', or 'train'
        user_keys: User-specified keys (overrides defaults)
        model_type: Model type affects priority for lentimpra

    Returns:
        Tuple of (data_array, key_used, file_type)
    """
    if file_path.endswith(('.pt', '.pth')):
        tensor_data = torch.load(file_path, map_location='cpu')
        if isinstance(tensor_data, torch.Tensor):
            return tensor_data.numpy(), 'tensor', 'pt'
        else:
            raise ValueError(f"Expected tensor in .pt file, got {type(tensor_data)}")

    elif file_path.endswith(('.h5', '.hdf5')):
        try:
            tensor_data = torch.load(file_path, map_location='cpu')
            if isinstance(tensor_data, torch.Tensor):
                return tensor_data.numpy(), 'tensor', 'pt'
        except:
            pass

        with h5py.File(file_path, 'r') as f:
            data, key_used = resolve_key_from_file(f, 'h5', data_category, user_keys, model_type)
        return data, key_used, 'h5'

    elif file_path.endswith('.npz'):
        npz_data = load(file_path)
        data, key_used = resolve_key_from_file(npz_data, 'npz', data_category, user_keys, model_type)
        return data, key_used, 'npz'

    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def ensure_correct_shape(data, expected_channels=4, data_name="data"):
    """
    Ensure data is in (N, 4, L) format.

    Handles:
    - Index-encoded (N, L) → one-hot (N, L, 4) → transpose (N, 4, L)
    - One-hot (N, L, 4) → transpose → (N, 4, L)
    - Already correct (N, 4, L) → no change

    Args:
        data: Input data array
        expected_channels: Expected number of channels (default 4 for DNA)
        data_name: Name for logging

    Returns:
        Data in (N, 4, L) format
    """
    if is_index_encoded(data):
        print(f"Detected index-encoded {data_name} (0123=ACGT). Converting to one-hot encoding...")
        data = index_to_onehot(data)
        print(f"Converted to one-hot with shape: {data.shape}")

    if data.ndim != 3:
        raise ValueError(f"Expected 3D array for {data_name}, got {data.ndim}D with shape {data.shape}")

    if data.shape[1] == expected_channels:
        return data
    elif data.shape[-1] == expected_channels:
        return np.transpose(data, (0, 2, 1))
    else:
        raise ValueError(f"Unexpected shape for {data_name}: {data.shape}. Expected channel dimension to be {expected_channels}")


def extract_sequences(samples_file_path, data_file_path, samples_keys=None, test_keys=None, train_keys=None, model_type='deepstarr'):
    """
    Extract test, synthetic, and training sequences from files.

    Handles all file formats (NPZ, H5, PT) and model types.
    Automatically detects and converts index-encoded sequences.
    Transposes to correct shape (N, 4, L).

    Args:
        samples_file_path: Path to synthetic samples file
        data_file_path: Path to test/train data file
        samples_keys: User-specified keys for samples (via --samples-key)
        test_keys: User-specified keys for test data
        train_keys: User-specified keys for train data
        model_type: 'deepstarr', 'mpralegnet', 'lentimpra', or 'multi-oracle'

    Returns:
        Tuple of (x_test, x_synthetic, x_train) in shape (N, 4, L)
    """
    samples_data, _, _ = load_file_by_type(samples_file_path, 'samples', samples_keys, model_type)

    if isinstance(samples_data, list):
        x_synthetic = ensure_correct_shape(samples_data[0], data_name="samples")
    else:
        x_synthetic = ensure_correct_shape(samples_data, data_name="samples")

    x_test, _, _ = load_file_by_type(data_file_path, 'test', test_keys, model_type)
    if x_test.ndim == 3 and x_test.shape[-1] == 4 and x_test.shape[1] != 4:
        x_test = np.transpose(x_test, (0, 2, 1))

    x_train, _, _ = load_file_by_type(data_file_path, 'train', train_keys, model_type)
    if x_train.ndim == 3 and x_train.shape[-1] == 4 and x_train.shape[1] != 4:
        x_train = np.transpose(x_train, (0, 2, 1))

    return x_test, x_synthetic, x_train


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

def load_oracle_model(oracle_path, model_type='deepstarr'):
    """Load oracle model based on model type."""
    if model_type.lower() == 'deepstarr':
        return load_deepstarr(oracle_path)
    elif model_type.lower() in ['mpralegnet', 'lentimpra']:
        return load_mpralegnet(oracle_path)
    elif model_type.lower() == 'multi-oracle':
        raise ValueError("Use load_multi_oracle_models() for multi-oracle setup")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_multi_oracle_models(model_path1, model_path2, model_path3):
    """
    Load three MPRALegNet oracle models for multi-oracle setup.

    Args:
        model_path1: Path to first oracle model
        model_path2: Path to second oracle model
        model_path3: Path to third oracle model

    Returns:
        Tuple of (model1, model2, model3)
    """
    print("Loading multi-oracle models (3 MPRALegNet models)...")
    model1 = load_mpralegnet(model_path1)
    print(f"✓ Loaded oracle model 1 from {model_path1}")

    model2 = load_mpralegnet(model_path2)
    print(f"✓ Loaded oracle model 2 from {model_path2}")

    model3 = load_mpralegnet(model_path3)
    print(f"✓ Loaded oracle model 3 from {model_path3}")

    return model1, model2, model3

def load_predictions(x_test_tensor, x_synthetic_tensor, oracle_model):
    """Load predictions from oracle model (works with both DeepSTARR and MPRALegNet).

    Runs inference in batches to avoid oversized tensors that can trigger 32-bit
    indexing limitations on some CUDA ops. Only per-batch tensors are moved to
    the device; results are collected on CPU.
    """
    # Ensure float32 dtype
    x_test_tensor = x_test_tensor.float()
    x_synthetic_tensor = x_synthetic_tensor.float()

    # Determine device from model
    device = next(oracle_model.parameters()).device

    # Choose a conservative batch size; allow override via env
    try:
        default_bs = int(os.environ.get("D3_INFER_BATCH_SIZE", "1024"))
    except ValueError:
        default_bs = 1024

    batch_size = max(1, default_bs)

    oracle_model.eval()

    y_test_chunks = []
    y_syn_chunks = []

    with torch.no_grad():
        # Iterate x_test in batches
        num_test = x_test_tensor.shape[0]
        for start in range(0, num_test, batch_size):
            end = min(start + batch_size, num_test)
            batch = x_test_tensor[start:end].to(device, non_blocking=True)
            preds = oracle_model(batch)
            y_test_chunks.append(preds.detach().cpu())

        # Iterate x_synthetic in batches
        num_syn = x_synthetic_tensor.shape[0]
        for start in range(0, num_syn, batch_size):
            end = min(start + batch_size, num_syn)
            batch = x_synthetic_tensor[start:end].to(device, non_blocking=True)
            preds = oracle_model(batch)
            y_syn_chunks.append(preds.detach().cpu())

    y_hat_test = torch.cat(y_test_chunks, dim=0).numpy() if y_test_chunks else np.empty((0,))
    y_hat_syn = torch.cat(y_syn_chunks, dim=0).numpy() if y_syn_chunks else np.empty((0,))

    return y_hat_test, y_hat_syn

def load_multi_oracle_predictions(x_test_tensor, x_synthetic_tensor, oracle_models):
    """
    Load predictions from three oracle models for multi-oracle setup.

    Args:
        x_test_tensor: Test sequences tensor (n, 4, 230)
        x_synthetic_tensor: Synthetic sequences tensor (n, 4, 230)
        oracle_models: Tuple of (model1, model2, model3)

    Returns:
        Tuple of (y_hat_test, y_hat_syn) where each is shape (n, 3)
        - y_hat_test[:, 0] = predictions from model1
        - y_hat_test[:, 1] = predictions from model2
        - y_hat_test[:, 2] = predictions from model3
    """
    model1, model2, model3 = oracle_models

    # Get device from first model (assuming all on same device)
    device = next(model1.parameters()).device
    x_test_tensor = x_test_tensor.to(device)
    x_synthetic_tensor = x_synthetic_tensor.to(device)

    # Get predictions from each model - each returns (n, 1) for MPRALegNet
    y_hat_test_1 = model1(x_test_tensor).detach().cpu().numpy().reshape(-1, 1)  # (n, 1)
    y_hat_test_2 = model2(x_test_tensor).detach().cpu().numpy().reshape(-1, 1)  # (n, 1)
    y_hat_test_3 = model3(x_test_tensor).detach().cpu().numpy().reshape(-1, 1)  # (n, 1)

    y_hat_syn_1 = model1(x_synthetic_tensor).detach().cpu().numpy().reshape(-1, 1)  # (n, 1)
    y_hat_syn_2 = model2(x_synthetic_tensor).detach().cpu().numpy().reshape(-1, 1)  # (n, 1)
    y_hat_syn_3 = model3(x_synthetic_tensor).detach().cpu().numpy().reshape(-1, 1)  # (n, 1)

    # Concatenate to create (n, 3) arrays
    y_hat_test = np.concatenate([y_hat_test_1, y_hat_test_2, y_hat_test_3], axis=1)  # (n, 3)
    y_hat_syn = np.concatenate([y_hat_syn_1, y_hat_syn_2, y_hat_syn_3], axis=1)      # (n, 3)

    return y_hat_test, y_hat_syn

def load_predictions_deepstarr(x_test_tensor, x_synthetic_tensor, deepstarr):
    """Legacy function - use load_predictions instead."""
    return load_predictions(x_test_tensor, x_synthetic_tensor, deepstarr)


extractor = EmbeddingExtractor()
def get_penultimate_embeddings(model, x, model_type='deepstarr'):
    """Get penultimate embeddings from model (works with both DeepSTARR and MPRALegNet)."""
    # Ensure tensor is on the same device as the model
    device = next(model.parameters()).device
    x = x.to(device)

    # Find the penultimate layer based on model type
    if model_type.lower() == 'deepstarr':
        target_layer = 'model.batchnorm6'
    elif model_type.lower() in ['mpralegnet', 'lentimpra']:
        # For MPRALegNet, hook into the last layer before output
        target_layer = 'model.head.2'  # This is the activation before final linear layer
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

def get_multi_oracle_embeddings(oracle_models, x, model_type='lentimpra'):
    """
    Get penultimate embeddings from three oracle models and concatenate them.

    Args:
        oracle_models: Tuple of (model1, model2, model3)
        x: Input tensor (n, 4, 230)
        model_type: Model type (should be 'lentimpra' for multi-oracle)

    Returns:
        Concatenated embeddings tensor (n, embedding_dim * 3)
    """
    model1, model2, model3 = oracle_models

    # Get embeddings from each model
    embedding1 = get_penultimate_embeddings(model1, x, model_type)  # (n, embedding_dim)
    embedding2 = get_penultimate_embeddings(model2, x, model_type)  # (n, embedding_dim)
    embedding3 = get_penultimate_embeddings(model3, x, model_type)  # (n, embedding_dim)

    # Concatenate embeddings along feature dimension
    concatenated_embeddings = torch.cat([embedding1, embedding2, embedding3], dim=1)  # (n, embedding_dim * 3)

    return concatenated_embeddings

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
