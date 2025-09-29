import numpy as np
from numpy import load
import h5py
import pandas as pd
import torch
import tqdm as tqdm_module
import os

from deepstarr import *
from mpralegnet import LitModel


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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

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
