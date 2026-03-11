"""
Training script for Autoencoder model
Trains only on BENIGN traffic for anomaly detection
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
import time
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config import *
from utils import (
    setup_logging, EarlyStopping, PerformanceMonitor, 
    create_data_loaders, save_model, MetricsCalculator, count_parameters, safe_json_dump,
    plot_confusion_matrix
)
from models import ModelFactory, ReconstructionLoss
from data_preprocessing import load_processed_data

# =============================================================================
# AUTOENCODER TRAINING CONFIGURATION
# =============================================================================
class AutoencoderTrainingConfig:
    """Autoencoder training configuration"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = TRAINING_CONFIG['use_amp'] and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Model configuration
        self.model_config = AUTOENCODER_CONFIG
        
        # Training parameters
        self.batch_size = TRAINING_CONFIG['batch_size']
        self.learning_rate = TRAINING_CONFIG['learning_rate'] * 0.1  # Lower LR for autoencoder
        self.weight_decay = TRAINING_CONFIG['weight_decay']
        self.epochs = TRAINING_CONFIG['epochs']
        self.patience = TRAINING_CONFIG['patience']
        self.min_delta = TRAINING_CONFIG['min_delta']
        
        # Create model
        self.model = ModelFactory.create_model('autoencoder', self.model_config, self.device)
        
        # Loss function
        self.criterion = ReconstructionLoss(loss_type='mse')
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=TRAINING_CONFIG['patience'], min_lr=1e-6
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            min_delta=self.min_delta
        )
        
        # Performance monitor
        self.monitor = PerformanceMonitor()
        
        self.logger = setup_logging()
        self.logger.info("Autoencoder training configuration initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Use AMP: {self.use_amp}")
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")
    
    def prepare_benign_data(self, X_train, y_train):
        """Prepare BENIGN data for autoencoder training"""
        # Filter BENIGN traffic (class 0)
        benign_mask = y_train == 0
        X_benign = X_train[benign_mask]
        
        self.logger.info(f"BENIGN samples: {len(X_benign):,} out of {len(X_train):,} total samples")
        
        # Split BENIGN data into train/validation
        from config import CV_CONFIG
        
        X_benign_train, X_benign_val = train_test_split(
            X_benign, test_size=CV_CONFIG['test_size'], random_state=CV_CONFIG['random_state'], shuffle=True
        )
        
        self.logger.info(f"BENIGN train: {len(X_benign_train):,}, BENIGN val: {len(X_benign_val):,}")
        
        return X_benign_train, X_benign_val
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with progress bar"""
        self.model.train()
        total_loss = 0
        total_reconstruction_error = 0
        
        # Create progress bar for consistency with Lightning models
        from tqdm import tqdm
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False, 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    reconstructed = self.model(data)
                    loss = self.criterion(reconstructed, data)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstructed = self.model(data)
                loss = self.criterion(reconstructed, data)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate reconstruction error
            with torch.no_grad():
                reconstruction_error = torch.mean((reconstructed - data) ** 2, dim=1)
                total_reconstruction_error += torch.mean(reconstruction_error).item()
            
            # Update progress bar with current metrics (simplified to match Lightning style)
            if batch_idx % 50 == 0:  # Update less frequently to match Lightning
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Recon Error': f'{torch.mean(reconstruction_error).item():.6f}'
                })
            
            # Log every 100 batches for detailed logging
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {loss.item():.6f}, Reconstruction Error: {torch.mean(reconstruction_error).item():.6f}'
                )
        
        avg_loss = total_loss / len(train_loader)
        avg_reconstruction_error = total_reconstruction_error / len(train_loader)
        
        return avg_loss, avg_reconstruction_error
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch with progress bar"""
        self.model.eval()
        total_loss = 0
        total_reconstruction_error = 0
        all_reconstruction_errors = []
        
        # Create progress bar for validation consistency (matching Lightning style)
        from tqdm import tqdm
        progress_bar = tqdm(val_loader, desc=f'Val Epoch {epoch}', leave=False,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        with torch.no_grad():
            for data, _ in progress_bar:
                data = data.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        reconstructed = self.model(data)
                        loss = self.criterion(reconstructed, data)
                else:
                    reconstructed = self.model(data)
                    loss = self.criterion(reconstructed, data)
                
                total_loss += loss.item()
                
                # Calculate reconstruction error for each sample
                reconstruction_error = torch.mean((reconstructed - data) ** 2, dim=1)
                total_reconstruction_error += torch.mean(reconstruction_error).item()
                all_reconstruction_errors.extend(reconstruction_error.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Val Loss': f'{loss.item():.6f}',
                    'Val Recon Error': f'{torch.mean(reconstruction_error).item():.6f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_reconstruction_error = total_reconstruction_error / len(val_loader)
        
        return avg_loss, avg_reconstruction_error, all_reconstruction_errors
    
    def train_autoencoder(self, X_benign_train, X_benign_val):
        """Train the autoencoder on BENIGN data"""
        self.logger.info("Training autoencoder on BENIGN traffic...")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X_benign_train, np.zeros(len(X_benign_train)),  # Dummy labels
            X_benign_val, np.zeros(len(X_benign_val)),      # Dummy labels
            batch_size=self.batch_size,
            num_workers=DATALOADER_CONFIG['num_workers']
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        best_threshold = None
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_recon_error = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_recon_error, val_recon_errors = self.validate_epoch(val_loader, epoch)
            
            # Update learning rate
            self.lr_scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update performance monitor
            self.monitor.update(train_loss, val_loss, 0, 0, current_lr)  # No accuracy for autoencoder
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                
                # Calculate threshold based on 95th percentile of reconstruction errors
                best_threshold = np.percentile(val_recon_errors, ANOMALY_THRESHOLD_PERCENTILE)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            epoch_time = time.time() - start_time
            self.logger.info(
                f'Epoch {epoch}: '
                f'Train Loss: {train_loss:.6f}, Train Recon Error: {train_recon_error:.6f}, '
                f'Val Loss: {val_loss:.6f}, Val Recon Error: {val_recon_error:.6f}, '
                f'LR: {current_lr:.6f}, Time: {epoch_time:.2f}s'
            )
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.logger.info(f"Autoencoder training completed. Best threshold: {best_threshold:.6f}")
        
        return best_threshold
    
    def evaluate_anomaly_detection(self, X_test, y_test, threshold):
        """Evaluate autoencoder for anomaly detection"""
        self.logger.info("Evaluating autoencoder for anomaly detection...")
        
        # Create test loader
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.FloatTensor(X_test), 
                torch.LongTensor(y_test)
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=DATALOADER_CONFIG['num_workers']
        )
        
        self.model.eval()
        all_reconstruction_errors = []
        all_predictions = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        reconstructed = self.model(data)
                else:
                    reconstructed = self.model(data)
                
                # Calculate reconstruction error for each sample
                reconstruction_error = torch.mean((reconstructed - data) ** 2, dim=1)
                all_reconstruction_errors.extend(reconstruction_error.cpu().numpy())
                
                # Predict anomalies (1 for anomaly, 0 for normal)
                predictions = (reconstruction_error.cpu().numpy() >= threshold).astype(int)
                all_predictions.extend(predictions)
        
        # Convert to numpy arrays
        reconstruction_errors = np.array(all_reconstruction_errors)
        predictions = np.array(all_predictions)
        
        # Create binary labels (0 for BENIGN, 1 for attacks)
        binary_labels = (y_test != 0).astype(int)
        
        # Calculate metrics
        metrics_calc = MetricsCalculator()
        anomaly_metrics = metrics_calc.calculate_anomaly_metrics(
            binary_labels, reconstruction_errors, threshold
        )
        
        self.logger.info(f"Anomaly Detection Results:")
        self.logger.info(f"  Accuracy: {anomaly_metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {anomaly_metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {anomaly_metrics['recall']:.4f}")
        self.logger.info(f"  F1-Score: {anomaly_metrics['f1_score']:.4f}")
        self.logger.info(f"  ROC-AUC: {anomaly_metrics['roc_auc']:.4f}")
        self.logger.info(f"  PR-AUC: {anomaly_metrics['pr_auc']:.4f}")
        
        return {
            'reconstruction_errors': reconstruction_errors,
            'predictions': predictions,
            'binary_labels': binary_labels,
            'threshold': threshold,
            'metrics': anomaly_metrics
        }

def train_autoencoder():
    """Main autoencoder training function"""
    logger = setup_logging()
    logger.info("Starting autoencoder training...")
    
    try:
        # Load processed data
        logger.info("Loading processed data...")
        X_train, X_test, y_train, y_test, scaler, label_encoder, label_mapping = load_processed_data()
        
        # Initialize training configuration
        config = AutoencoderTrainingConfig()
        
        # Prepare BENIGN data
        X_benign_train, X_benign_val = config.prepare_benign_data(X_train, y_train)
        
        # Train autoencoder
        threshold = config.train_autoencoder(X_benign_train, X_benign_val)
        
        # Evaluate on test set
        evaluation_results = config.evaluate_anomaly_detection(X_test, y_test, threshold)
        
        # Save model
        model_path = MODELS_PATH / "autoencoder" / "best_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_model(
            config.model,
            model_path,
            0,  # No epoch for autoencoder
            evaluation_results['metrics']['accuracy'],
            evaluation_results['metrics']
        )
        
        logger.info(f"Autoencoder model saved to {model_path}")
        
        # Save threshold
        threshold_path = MODELS_PATH / "autoencoder" / "threshold.json"
        safe_json_dump({'threshold': threshold}, threshold_path)
        
        logger.info(f"Threshold saved to {threshold_path}")
        
        # Save evaluation results
        results_path = RESULTS_PATH / "autoencoder_results.json"
        safe_json_dump(evaluation_results, results_path)
        
        logger.info(f"Autoencoder results saved to {results_path}")
        
        # Plot reconstruction error distribution
        plot_reconstruction_errors(evaluation_results)
        
        logger.info("Autoencoder training completed successfully!")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error in autoencoder training: {str(e)}")
        raise

def plot_reconstruction_errors(evaluation_results):
    """Plot reconstruction error distribution"""
    logger = setup_logging()
    logger.info("Creating reconstruction error plots...")
    
    reconstruction_errors = evaluation_results['reconstruction_errors']
    binary_labels = evaluation_results['binary_labels']
    threshold = evaluation_results['threshold']
    
    # Create plots directory
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Reconstruction error distribution
    plt.figure(figsize=(12, 8))
    
    # Separate errors by class
    benign_errors = reconstruction_errors[binary_labels == 0]
    attack_errors = reconstruction_errors[binary_labels == 1]
    
    plt.hist(benign_errors, bins=50, alpha=0.7, label='BENIGN', color='blue')
    plt.hist(attack_errors, bins=50, alpha=0.7, label='ATTACKS', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.4f}')
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(PLOTS_PATH / 'reconstruction_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    # Plot 2: ROC Curve
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(binary_labels, reconstruction_errors)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Anomaly Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(PLOTS_PATH / 'roc_curve_autoencoder.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    # Plot 3: Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(binary_labels, reconstruction_errors)
    pr_auc = average_precision_score(binary_labels, reconstruction_errors)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Anomaly Detection')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(PLOTS_PATH / 'pr_curve_autoencoder.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    # Plot 4: Confusion Matrix (Autoencoder)
    try:
        cm = np.array(evaluation_results['metrics'].get('confusion_matrix'))
        if cm is not None and cm.size > 0:
            class_names = ['BENIGN', 'ATTACK']
            plot_confusion_matrix(
                cm,
                class_names,
                title='Confusion Matrix - Autoencoder',
                save_path=PLOTS_PATH / 'confusion_matrix_autoencoder.png'
            )
    except Exception as e:
        logger.warning(f"Failed to create autoencoder confusion matrix plot: {e}")
    
    logger.info(f"Plots saved to {PLOTS_PATH}")

if __name__ == "__main__":
    # Run autoencoder training
    print("Starting Autoencoder Training for Anomaly Detection...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Use AMP: {TRAINING_CONFIG['use_amp'] and torch.cuda.is_available()}")
    
    # Train autoencoder
    results = train_autoencoder()
    
    print("Autoencoder training completed successfully!")
    print(f"Results saved to {RESULTS_PATH}")
    print(f"Plots saved to {PLOTS_PATH}")
