"""
Discriminability Classifier for distinguishing synthetic from test sequences.

This module contains a binary classifier based on the DeepSTARR architecture
that is trained to distinguish between synthetic and test sequences. The AUROC
score provides a measure of discriminability - lower scores indicate better
generator performance (harder to distinguish synthetic from real sequences).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import random
import h5py
import os
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from pytorch_lightning import loggers as pl_loggers
import tqdm
from typing import Any, Dict, Optional, Tuple
from .deepstarr import DeepSTARR


class DiscriminabilityClassifier(nn.Module):
    """Binary classifier based on DeepSTARR architecture.
    
    This model uses the same convolutional backbone as DeepSTARR but with
    a single sigmoid output for binary classification (synthetic vs test).
    """
    
    def __init__(self, d=256,
                 conv1_filters=None, learn_conv1_filters=True,
                 conv2_filters=None, learn_conv2_filters=True,
                 conv3_filters=None, learn_conv3_filters=True,
                 conv4_filters=None, learn_conv4_filters=True):
        super().__init__()
        
        if d != 256:
            print("NB: number of first-layer convolutional filters in original DeepSTARR model is 256; current number of first-layer convolutional filters is not set to 256")
        
        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        
        self.init_conv1_filters = conv1_filters
        self.init_conv2_filters = conv2_filters
        self.init_conv3_filters = conv3_filters
        self.init_conv4_filters = conv4_filters
        
        assert (not (conv1_filters is None and not learn_conv1_filters)), "initial conv1_filters cannot be set to None while learn_conv1_filters is set to False"
        assert (not (conv2_filters is None and not learn_conv2_filters)), "initial conv2_filters cannot be set to None while learn_conv2_filters is set to False"
        assert (not (conv3_filters is None and not learn_conv3_filters)), "initial conv3_filters cannot be set to None while learn_conv3_filters is set to False"
        assert (not (conv4_filters is None and not learn_conv4_filters)), "initial conv4_filters cannot be set to None while learn_conv4_filters is set to False"
        
        # Layer 1 (convolutional), constituent parts
        if conv1_filters is not None:
            if learn_conv1_filters:
                self.conv1_filters = nn.Parameter(torch.Tensor(conv1_filters))
            else:
                self.register_buffer("conv1_filters", torch.Tensor(conv1_filters))
        else:
            self.conv1_filters = nn.Parameter(torch.zeros(d, 4, 7))
            nn.init.kaiming_normal_(self.conv1_filters)
        self.batchnorm1 = nn.BatchNorm1d(d)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)
        
        # Layer 2 (convolutional), constituent parts
        if conv2_filters is not None:
            if learn_conv2_filters:
                self.conv2_filters = nn.Parameter(torch.Tensor(conv2_filters))
            else:
                self.register_buffer("conv2_filters", torch.Tensor(conv2_filters))
        else:
            self.conv2_filters = nn.Parameter(torch.zeros(60, d, 3))
            nn.init.kaiming_normal_(self.conv2_filters)
        self.batchnorm2 = nn.BatchNorm1d(60)
        self.maxpool2 = nn.MaxPool1d(2)
        
        # Layer 3 (convolutional), constituent parts
        if conv3_filters is not None:
            if learn_conv3_filters:
                self.conv3_filters = nn.Parameter(torch.Tensor(conv3_filters))
            else:
                self.register_buffer("conv3_filters", torch.Tensor(conv3_filters))
        else:
            self.conv3_filters = nn.Parameter(torch.zeros(60, 60, 5))
            nn.init.kaiming_normal_(self.conv3_filters)
        self.batchnorm3 = nn.BatchNorm1d(60)
        self.maxpool3 = nn.MaxPool1d(2)
        
        # Layer 4 (convolutional), constituent parts
        if conv4_filters is not None:
            if learn_conv4_filters:
                self.conv4_filters = nn.Parameter(torch.Tensor(conv4_filters))
            else:
                self.register_buffer("conv4_filters", torch.Tensor(conv4_filters))
        else:
            self.conv4_filters = nn.Parameter(torch.zeros(120, 60, 3))
            nn.init.kaiming_normal_(self.conv4_filters)
        self.batchnorm4 = nn.BatchNorm1d(120)
        self.maxpool4 = nn.MaxPool1d(2)
        
        # Layer 5 (fully connected), constituent parts
        self.fc5 = nn.LazyLinear(256, bias=True)
        self.batchnorm5 = nn.BatchNorm1d(256)
        
        # Layer 6 (fully connected), constituent parts
        self.fc6 = nn.Linear(256, 256, bias=True)
        self.batchnorm6 = nn.BatchNorm1d(256)
        
        # Output layer (binary classification)
        self.fc7 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Layer 1
        cnn = torch.conv1d(x, self.conv1_filters, stride=1, padding="same")
        cnn = self.batchnorm1(cnn)
        cnn = self.activation1(cnn)
        cnn = self.maxpool1(cnn)
        
        # Layer 2
        cnn = torch.conv1d(cnn, self.conv2_filters, stride=1, padding="same")
        cnn = self.batchnorm2(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool2(cnn)
        
        # Layer 3
        cnn = torch.conv1d(cnn, self.conv3_filters, stride=1, padding="same")
        cnn = self.batchnorm3(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool3(cnn)
        
        # Layer 4
        cnn = torch.conv1d(cnn, self.conv4_filters, stride=1, padding="same")
        cnn = self.batchnorm4(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool4(cnn)
        
        # Layer 5
        cnn = self.flatten(cnn)
        cnn = self.fc5(cnn)
        cnn = self.batchnorm5(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)
        
        # Layer 6
        cnn = self.fc6(cnn)
        cnn = self.batchnorm6(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)
        
        # Output layer (sigmoid for binary classification)
        logits = self.fc7(cnn)
        y_pred = self.sigmoid(logits)
        
        return y_pred.squeeze(-1)  # Remove last dimension to get (batch_size,) shape


class PL_DiscriminabilityClassifier(pl.LightningModule):
    """PyTorch Lightning wrapper for discriminability classifier.
    
    This class handles training, validation, and testing of the binary classifier
    used to measure discriminability between synthetic and test sequences.
    """
    
    def __init__(self,
                 batch_size: int = 128,
                 train_max_epochs: int = 50,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 lr: float = 0.002,
                 weight_decay: float = 1e-6,
                 min_lr: float = 0.0,
                 lr_patience: int = 10,
                 decay_factor: float = 0.1,
                 pos_weight: Optional[float] = None):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = DiscriminabilityClassifier()
        self.name = 'DiscriminabilityClassifier'
        
        # Training configuration
        self.batch_size = batch_size
        self.train_max_epochs = train_max_epochs
        self.patience = patience
        self.lr = lr
        self.min_delta = min_delta
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.lr_patience = lr_patience
        self.decay_factor = decay_factor
        
        # Loss function setup
        if pos_weight is not None:
            self.loss_fn = nn.BCELoss(weight=torch.tensor([pos_weight]))
        else:
            self.loss_fn = nn.BCELoss()
        
        # Storage for predictions during validation/test
        self.validation_outputs = []
        self.test_outputs = []
        
    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        self.model.train()
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels.float())
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, 
                prog_bar=True, logger=True, sync_dist=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        self.model.eval()
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels.float())
        
        # Store outputs for AUROC calculation
        self.validation_outputs.append({
            'predictions': outputs.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, 
                prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate and log AUROC at end of validation epoch."""
        if len(self.validation_outputs) > 0:
            # Concatenate all predictions and labels
            all_preds = torch.cat([x['predictions'] for x in self.validation_outputs])
            all_labels = torch.cat([x['labels'] for x in self.validation_outputs])
            
            # Calculate AUROC
            auroc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
            
            self.log("val_auroc", auroc, on_epoch=True, prog_bar=True, logger=True)
            
            # Clear outputs for next epoch
            self.validation_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """Test step for one batch."""
        self.model.eval()
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels.float())
        
        # Store outputs for final AUROC calculation
        self.test_outputs.append({
            'predictions': outputs.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, 
                prog_bar=True, logger=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """Calculate final test AUROC."""
        if len(self.test_outputs) > 0:
            # Concatenate all predictions and labels
            all_preds = torch.cat([x['predictions'] for x in self.test_outputs])
            all_labels = torch.cat([x['labels'] for x in self.test_outputs])
            
            # Calculate AUROC
            auroc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
            
            self.log("test_auroc", auroc, on_epoch=True, prog_bar=True, logger=True)
            
            # Store the final AUROC for retrieval
            self.final_auroc = auroc
            
            # Clear outputs
            self.test_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=self.lr_patience, 
            min_lr=self.min_lr, 
            factor=self.decay_factor
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "val_loss"
            }
        }
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def predict_proba(self, X, batch_size=None):
        """Get prediction probabilities for input sequences."""
        if batch_size is None:
            batch_size = self.batch_size
            
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            X, batch_size=batch_size, shuffle=False
        )
        preds = []
        
        with torch.no_grad():
            for x in tqdm.tqdm(dataloader, total=len(dataloader)):
                if torch.cuda.is_available():
                    x = x.cuda()
                pred = self.model(x)
                preds.append(pred.detach().cpu().numpy())
                
        return np.concatenate(preds, axis=0)


def prepare_discriminability_data(x_synthetic: np.ndarray, 
                                x_test: np.ndarray,
                                validation_split: float = 0.2,
                                random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for discriminability training.
    
    Args:
        x_synthetic: Synthetic sequences (N_syn, L, A) or (N_syn, A, L)
        x_test: Test sequences (N_test, L, A) or (N_test, A, L)
        validation_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    np.random.seed(random_seed)
    
    # Ensure correct format (N, A, L) for DeepSTARR
    if x_synthetic.shape[1] != 4 and x_synthetic.shape[2] == 4:
        x_synthetic = np.transpose(x_synthetic, (0, 2, 1))
    if x_test.shape[1] != 4 and x_test.shape[2] == 4:
        x_test = np.transpose(x_test, (0, 2, 1))
    
    # Create labels: synthetic = 1, test = 0
    y_synthetic = np.ones(len(x_synthetic))
    y_test = np.zeros(len(x_test))
    
    # Combine data
    X_combined = np.concatenate([x_synthetic, x_test], axis=0)
    y_combined = np.concatenate([y_synthetic, y_test], axis=0)
    
    # Shuffle data
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    # Split into train/validation
    n_val = int(len(X_combined) * validation_split)
    n_train = len(X_combined) - n_val
    
    X_train = X_combined[:n_train]
    y_train = y_combined[:n_train]
    X_val = X_combined[n_train:]
    y_val = y_combined[n_train:]
    
    return X_train, y_train, X_val, y_val


def save_discriminability_data(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              output_path: str) -> None:
    """
    Save discriminability training data to H5 file for future use.
    
    Args:
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        output_path: Path to save H5 file
    """
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_val', data=X_val)
        f.create_dataset('y_val', data=y_val)
        
        # Add metadata
        f.attrs['synthetic_label'] = 1
        f.attrs['test_label'] = 0
        f.attrs['n_synthetic'] = np.sum(y_train == 1) + np.sum(y_val == 1)
        f.attrs['n_test'] = np.sum(y_train == 0) + np.sum(y_val == 0)
        f.attrs['validation_split'] = len(X_val) / (len(X_train) + len(X_val))


def train_discriminability_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                                    X_val: np.ndarray, y_val: np.ndarray,
                                    config: Dict[str, Any]) -> Tuple[PL_DiscriminabilityClassifier, float]:
    """
    Train discriminability classifier and return trained model with AUROC.
    
    Args:
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences  
        y_val: Validation labels
        config: Configuration dictionary
        
    Returns:
        Tuple of (trained_model, final_auroc)
    """
    # Set random seeds
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Calculate positive class weight for imbalanced data
    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)
    pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    # Initialize model
    model = PL_DiscriminabilityClassifier(
        batch_size=config.get('batch_size', 128),
        train_max_epochs=config.get('train_max_epochs', 50),
        patience=config.get('patience', 10),
        lr=config.get('lr', 0.002),
        pos_weight=pos_weight
    )
    
    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model.batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=model.batch_size, shuffle=False
    )
    
    # Setup trainer
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=model.min_delta,
            patience=model.patience,
            verbose=False,
            mode='min'
        ),
        pl.callbacks.ModelCheckpoint(
            monitor='val_auroc',
            mode='max',
            save_top_k=1,
            save_weights_only=True,
            dirpath="./",
            filename="discriminability_classifier_best"
        )
    ]
    
    trainer = pl.Trainer(
        max_epochs=model.train_max_epochs,
        callbacks=callbacks,
        logger=False,  # Disable logging for cleaner output
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator='auto',
        devices=1
    )
    
    # Train model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    # Test on validation set to get final AUROC
    trainer.test(model, dataloaders=val_dataloader)
    
    # Get final AUROC
    final_auroc = getattr(model, 'final_auroc', 0.5)
    
    return model, final_auroc