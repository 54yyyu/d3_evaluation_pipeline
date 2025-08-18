#!/usr/bin/env python3
"""
Discriminability analysis using binary classification.

This module implements a binary classifier to distinguish between real and synthetic sequences.
The classifier uses the same DeepSTARR architecture but with a binary classification head.
The AUROC score measures how well the classifier can discriminate between real and synthetic sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
import h5py
import os
import random
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import the base DeepSTARR architecture
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepstarr import DeepSTARR

def prep_data_for_classification(x_test_tensor, x_synthetic_tensor):
    """Prepare data for discriminability classification."""
    x_train = np.vstack([x_test_tensor.detach().cpu().numpy(), x_synthetic_tensor.detach().cpu().numpy()])
    y_train = np.vstack([np.ones((x_test_tensor.shape[0],1)), np.zeros((x_synthetic_tensor.shape[0],1))])
    x_train = np.transpose(x_train, (0, 2, 1)) 

    #write x_train and y_train into dict to create .h5 file
    data_dict = {
        'x_train': x_train,
        'y_train': y_train,
    }

    return data_dict


class BinaryDeepSTARR(nn.Module):
    """
    Binary classification version of DeepSTARR for discriminability analysis.
    
    Uses the same CNN architecture as DeepSTARR but with a single output neuron
    for binary classification (real vs synthetic sequences).
    """
    
    def __init__(self, d=256,
                 conv1_filters=None, learn_conv1_filters=True,
                 conv2_filters=None, learn_conv2_filters=True,
                 conv3_filters=None, learn_conv3_filters=True,
                 conv4_filters=None, learn_conv4_filters=True):
        super().__init__()
        
        # Use the base DeepSTARR architecture with single output
        self.deepstarr = DeepSTARR(
            output_dim=1,  # Single output for binary classification
            d=d,
            conv1_filters=conv1_filters,
            learn_conv1_filters=learn_conv1_filters,
            conv2_filters=conv2_filters,
            learn_conv2_filters=learn_conv2_filters,
            conv3_filters=conv3_filters,
            learn_conv3_filters=learn_conv3_filters,
            conv4_filters=conv4_filters,
            learn_conv4_filters=learn_conv4_filters
        )
        
    def forward(self, x):
        # Get output from DeepSTARR and apply sigmoid for binary classification
        logits = self.deepstarr(x)
        return logits  # Return logits (use BCEWithLogitsLoss for numerical stability)


class PL_BinaryDeepSTARR(pl.LightningModule):
    """
    PyTorch Lightning module for binary DeepSTARR classifier.
    
    This adapts the original PL_DeepSTARR for binary classification tasks.
    """
    
    def __init__(self,
                 batch_size=128,
                 train_max_epochs=100,
                 patience=10,
                 min_delta=0.001,
                 lr=0.002,
                 weight_decay=1e-6,
                 min_lr=0.0,
                 lr_patience=10,
                 decay_factor=0.1,
                 input_h5_file='Discriminatability.h5'):
        super().__init__()
        
        # Store hyperparameters
        self.batch_size = batch_size
        self.train_max_epochs = train_max_epochs
        self.patience = patience
        self.lr = lr
        self.min_delta = min_delta
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.lr_patience = lr_patience
        self.decay_factor = decay_factor
        self.input_h5_file = input_h5_file
        
        # Initialize binary classifier
        self.model = BinaryDeepSTARR(output_dim=1)
        self.name = 'BinaryDeepSTARR'
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load discriminability data from H5 file."""
        try:
            with h5py.File(self.input_h5_file, 'r') as data:
                # Load training data
                x_train = np.array(data['x_train'])  # Shape: (N, A, L)
                y_train = np.array(data['y_train'])  # Shape: (N, 1)
                
                # Convert to tensors
                self.X_train = torch.tensor(x_train, dtype=torch.float32)
                self.y_train = torch.tensor(y_train, dtype=torch.float32)
                
                # Create train/validation split (80/20)
                n_samples = len(self.X_train)
                n_train = int(0.8 * n_samples)
                
                # Shuffle indices
                indices = torch.randperm(n_samples)
                train_indices = indices[:n_train]
                val_indices = indices[n_train:]
                
                # Split data
                self.X_train_split = self.X_train[train_indices]
                self.y_train_split = self.y_train[train_indices]
                self.X_valid = self.X_train[val_indices]
                self.y_valid = self.y_train[val_indices]
                
                # Also keep full dataset for testing
                self.X_test = self.X_train
                self.y_test = self.y_train
                
        except Exception as e:
            raise RuntimeError(f"Failed to load discriminability data from {self.input_h5_file}: {e}")
    
    def training_step(self, batch, batch_idx):
        """Training step with binary cross entropy loss."""
        inputs, labels = batch
        logits = self.model(inputs)
        
        # Use BCEWithLogitsLoss for numerical stability
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits.squeeze(), labels.squeeze())
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with AUROC metric."""
        inputs, labels = batch
        logits = self.model(inputs)
        
        # Calculate loss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits.squeeze(), labels.squeeze())
        
        # Calculate AUROC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        try:
            auroc = roc_auc_score(labels_np, probs)
            self.log("val_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        except ValueError:
            # Handle case where all labels are the same class
            pass
            
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        """Test step with AUROC metric."""
        inputs, labels = batch
        logits = self.model(inputs)
        
        # Calculate loss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits.squeeze(), labels.squeeze())
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss, "logits": logits, "labels": labels}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=self.lr_patience, 
            min_lr=self.min_lr, 
            factor=self.decay_factor
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def predict_custom(self, X, keepgrad=False):
        """Custom prediction method similar to original DeepSTARR."""
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)
        preds = torch.empty(0)
        
        if keepgrad:
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()
            
        for x in tqdm(dataloader, total=len(dataloader), desc="Predicting"):
            pred = self.model(x)
            if not keepgrad:
                pred = pred.detach().cpu()
            preds = torch.cat((preds, pred), axis=0)
            
        return preds


def train_discriminability_classifier(h5_file='Discriminatability.h5', 
                                    output_dir='results',
                                    verbose=True,
                                    seed=42):
    """
    Train a binary classifier to distinguish between real and synthetic sequences.
    
    Args:
        h5_file (str): Path to the discriminability data H5 file
        output_dir (str): Directory to save results and checkpoints
        verbose (bool): Whether to print verbose output
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Results containing AUROC score and other metrics
    """
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    if verbose:
        print(f"Training discriminability classifier using data from {h5_file}")
    
    # Check if data file exists
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"Discriminability data file not found: {h5_file}")
    
    # Initialize model
    model = PL_BinaryDeepSTARR(input_h5_file=h5_file)
    
    if verbose:
        print(f"Loaded {len(model.X_train)} total samples")
        print(f"Training samples: {len(model.X_train_split)}")
        print(f"Validation samples: {len(model.X_valid)}")
        print(f"Input shape: {model.X_train_split[0].shape}")
    
    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        list(zip(model.X_train_split, model.y_train_split)), 
        batch_size=model.batch_size, 
        shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        list(zip(model.X_valid, model.y_valid)), 
        batch_size=model.batch_size, 
        shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        list(zip(model.X_test, model.y_test)), 
        batch_size=model.batch_size, 
        shuffle=False
    )
    
    # Setup logging and checkpointing
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "lightning_logs_discriminability")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)
    
    # Checkpoint callback
    ckpt_filename = "discriminability_classifier"
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath=output_dir,
        filename=ckpt_filename
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=model.min_delta,
        patience=model.patience,
        verbose=False,
        mode='min'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=model.train_max_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic=True
    )
    
    if verbose:
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"Training on {device}")
    
    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    
    # Load best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    if verbose:
        print(f"Loading best model from: {best_model_path}")
    
    # Test the model
    trainer.test(model, test_dataloader)
    
    # Calculate final AUROC on test set
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Convert to probabilities and calculate AUROC
    all_probs = torch.sigmoid(all_logits).numpy()
    all_labels_np = all_labels.numpy()
    
    try:
        final_auroc = roc_auc_score(all_labels_np, all_probs)
    except ValueError as e:
        # Handle edge case where all labels are the same
        final_auroc = 0.5
        if verbose:
            print(f"Warning: Could not calculate AUROC: {e}")
    
    results = {
        'auroc': final_auroc,
        'n_samples': len(model.X_test),
        'n_real': int(torch.sum(model.y_test == 1).item()),
        'n_synthetic': int(torch.sum(model.y_test == 0).item()),
        'best_model_path': best_model_path,
        'checkpoint_dir': output_dir
    }
    
    if verbose:
        print(f"\n=== Discriminability Results ===")
        print(f"AUROC: {final_auroc:.4f}")
        print(f"Total samples: {results['n_samples']}")
        print(f"Real sequences: {results['n_real']}")
        print(f"Synthetic sequences: {results['n_synthetic']}")
    
    return results


def run_discriminability_analysis(output_dir, h5_file='Discriminatability.h5', verbose=True):
    """
    Run complete discriminability analysis.
    
    This function trains a binary classifier and returns the AUROC score
    measuring how well the classifier can distinguish between real and synthetic sequences.
    
    Args:
        output_dir (str): Directory to save results
        h5_file (str): Path to discriminability data file  
        verbose (bool): Whether to print verbose output
        
    Returns:
        dict: Analysis results including AUROC score
    """
    
    if verbose:
        print("=== Running Discriminability Analysis ===")
        print("Training binary classifier to distinguish real vs synthetic sequences...")
    
    try:
        # Train classifier and get results
        results = train_discriminability_classifier(
            h5_file=h5_file,
            output_dir=output_dir,
            verbose=verbose
        )
        
        # Save results
        results_file = os.path.join(output_dir, 'discriminability_results.json')
        import json
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, (np.integer, np.floating)):
                    json_results[key] = value.item()
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)
        
        if verbose:
            print(f"Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        if verbose:
            import traceback
            print(f"Discriminability analysis failed: {e}")
            traceback.print_exc()
        raise