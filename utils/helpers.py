import numpy as np
from numpy import load
import h5py
import pandas as pd
import torch
import lightning as L
import tqdm as tqdm_module

from pytorch_lightning import LightningModule
from deepstarr import *
from mpralegnet import LitModel


class EmbeddingExtractor:
    def __init__(self):
        self.embedding = None

    def hook(self, module, input, output):
        self.embedding = output.detach()

def extract_data(samples_file_path, data_file):
    """Extract data for deepstarr (legacy function)."""
    #load samples from .npz file
    data = load(samples_file_path)
    samples = []
    lst = data.files
    for item in lst:
        samples.append(data[item])

    #load in data
    with h5py.File(data_file, 'r') as f:
        # Access the data for the specific X_test key
        x_test = f['X_test'][()]
        x_train = f['X_train'][()]

    #transpose samples to get shape (41186, 4, 249)
    x_synthetic = np.transpose(samples[0], (0, 2, 1))

    return x_test, x_synthetic, x_train

def extract_lentimpra_data(samples_file_path, lentimpra_data):
    """Extract data for lentimpra with 230-length sequences."""
    #load samples from .npz file
    data = load(samples_file_path)
    samples = []
    lst = data.files
    for item in lst:
        samples.append(data[item])

    #load in data
    with h5py.File(lentimpra_data, 'r') as f:
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
    #load model
    ckpt_path = oracle_path
    mpralegnet = LitModel.load_from_checkpoint(ckpt_path).eval()

    return mpralegnet

def load_oracle_model(oracle_path, model_type='deepstarr'):
    """Load oracle model based on model type."""
    if model_type.lower() == 'deepstarr':
        return load_deepstarr(oracle_path)
    elif model_type.lower() in ['mpralegnet', 'lentimpra']:
        return load_mpralegnet(oracle_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_predictions(x_test_tensor, x_synthetic_tensor, oracle_model):
    """Load predictions from oracle model (works with both DeepSTARR and MPRALegNet)."""
    # Ensure tensors are on the same device as the model
    device = next(oracle_model.parameters()).device
    x_test_tensor = x_test_tensor.to(device)
    x_synthetic_tensor = x_synthetic_tensor.to(device)

    #run model predictions
    y_hat_test = oracle_model(x_test_tensor)
    y_hat_syn = oracle_model(x_synthetic_tensor)

    #returns numpy arrays of oracle predictions from samples and x test
    return y_hat_test.detach().cpu().numpy(), y_hat_syn.detach().cpu().numpy()

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
