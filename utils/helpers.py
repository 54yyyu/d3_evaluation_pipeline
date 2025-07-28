import numpy as np
from numpy import load
import h5py
import pandas as pd
import torch
import lightning as L
from tqdm import tqdm

from pytorch_lightning import LightningModule
from deepstarr import *


class EmbeddingExtractor:
    def __init__(self):
        self.embedding = None

    def hook(self, module, input, output):
        self.embedding = output.detach()

def extract_data(samples_file_path, deepSTARR_data):
   #load samples from .npz file
    data = load(samples_file_path)
    samples = []
    lst = data.files
    for item in lst:
        samples.append(data[item])

    #load in data
    with h5py.File(deepSTARR_data, 'r') as f:
        # Access the data for the specific X_test key
        x_test = f['X_test'][()]
        x_train = f['X_train'][()]

    #transpose samples to get shape (41186, 4, 249)
    x_synthetic = np.transpose(samples[0], (0, 2, 1))

    return x_test, x_synthetic, x_train

def numpy_to_tensor(array):
    return torch.from_numpy(array).float()

def load_deepstarr(oracle_path):
    
    #load model
    ckpt_aug_path = oracle_path
    deepstarr = PL_DeepSTARR.load_from_checkpoint(ckpt_aug_path).eval()

    return deepstarr

def load_predictions(x_test_tensor, x_synthetic_tensor, deepstarr):

    #run model predictions
    y_hat_test = deepstarr(x_test_tensor)
    y_hat_syn = deepstarr(x_synthetic_tensor)

    #returns numpy arrays of deepstarr predictions from samples and x test
    return y_hat_test.detach().numpy(), y_hat_syn.detach().numpy()


extractor = EmbeddingExtractor()
def get_penultimate_embeddings(model, x):
    # Find the penultimate layer
    for name, module in model.named_modules():
        if name == 'model.batchnorm6':
            handle = module.register_forward_hook(extractor.hook)
            break
    else:
        raise ValueError("Could not find 'model.batchnorm6' layer")

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Remove the hook
    handle.remove()

    return extractor.embedding

#preparing data to put into kmer_statistics function
def put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor):
    return x_test_tensor.detach().numpy().transpose(0,2,1), x_synthetic_tensor.detach().numpy().transpose(0,2,1)

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

    for index in tqdm(range(len(X))): #for loop is what actually converts a list of one-hot encoded sequences into ACGT

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
    for i in range(len(sequence_list)):
        identifier_line = '>Seq' + str(i) + '\n'
        output_file.write(identifier_line)
        sequence_line = sequence_list[i]
        output_file.write(sequence_line + '\n')
