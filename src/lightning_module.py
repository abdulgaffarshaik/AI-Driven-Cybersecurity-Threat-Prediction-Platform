"""
PyTorch Lightning module for AI-based Intrusion Detection System
Matches the style and functionality of the pneumonia project
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, List, Optional, Tuple
import numpy as np
try:
    from torchmetrics.classification import (
        MulticlassAccuracy,
        MulticlassF1Score,
        MulticlassAUROC,
    )
    TORCHMETRICS_AVAILABLE = True
except Exception:
    TORCHMETRICS_AVAILABLE = False

# Optional hardware monitoring imports
try:
    import psutil
    import GPUtil
    HARDWARE_MONITORING_AVAILABLE = True
except ImportError:
    HARDWARE_MONITORING_AVAILABLE = False

from config import *
from models import ModelFactory, FocalLoss
from utils import MetricsCalculator, count_parameters


class IDSLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for IDS training with advanced features
    """
    
    def __init__(
        self,
        model_type: str = 'cnn_lstm',
        model_config: Dict = None,
        training_config: Dict = None,
        num_classes: int = 15,
        learning_rate: float = None,
        weight_decay: float = None
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Import config for defaults
        from config import TRAINING_CONFIG
        
        # Configuration
        self.model_type = model_type
        self.model_config = model_config or (CNN_LSTM_CONFIG if model_type == 'cnn_lstm' else DNN_CONFIG)
        self.training_config = training_config or TRAINING_CONFIG
        self.num_classes = num_classes
        self.learning_rate = learning_rate or TRAINING_CONFIG['learning_rate']
        self.weight_decay = weight_decay or TRAINING_CONFIG['weight_decay']
        
        # Create model - EXACT SAME AS PNEUMONIA PROJECT
        self.model = ModelFactory.create_model(model_type, self.model_config)
        
        # Loss function configuration
        self.use_focal_loss = True  # Always use focal loss for IDS class imbalance
        self.focal_alpha = 1.0  # Will be set dynamically based on class weights
        self.focal_gamma = 2.0
        
        # Class weights for handling imbalanced dataset
        # Register as buffer so it's moved to the correct device automatically
        self.register_buffer('class_weights', torch.tensor(list(CLASS_WEIGHTS.values()), dtype=torch.float32))
        self._criterion = None  # Will be initialized in training_step
        
        # Metrics
        # Prefer GPU-friendly torchmetrics when available to lower CPU usage
        if TORCHMETRICS_AVAILABLE:
            self.tm_train_acc = MulticlassAccuracy(num_classes=num_classes)
            self.tm_train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
            self.tm_train_auroc = MulticlassAUROC(num_classes=num_classes, average='macro')

            self.tm_val_acc = MulticlassAccuracy(num_classes=num_classes)
            self.tm_val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
            self.tm_val_auroc = MulticlassAUROC(num_classes=num_classes, average='macro')

            self.tm_test_acc = MulticlassAccuracy(num_classes=num_classes)
            self.tm_test_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
            self.tm_test_auroc = MulticlassAUROC(num_classes=num_classes, average='macro')
            # Ensure external code that expects these attributes does not fail
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
        else:
            self.train_metrics = MetricsCalculator(num_classes=num_classes)
            self.val_metrics = MetricsCalculator(num_classes=num_classes)
            self.test_metrics = MetricsCalculator(num_classes=num_classes)
        
        # Model summary
        self.model_summary = {
            'model_type': model_type,
            'total_parameters': count_parameters(self.model),
            'model_config': self.model_config
        }
        
        # Hardware monitoring
        self.monitor_hardware = HARDWARE_MONITORING_AVAILABLE
    
    def on_train_start(self):
        """Called when training starts - Monitor GPU memory"""
        # Log device information (only numeric values can be logged)
        if torch.cuda.is_available():
            self.log('gpu_memory_total', torch.cuda.get_device_properties(0).total_memory / 1024**3)
            # REMOVED: Print statement that breaks progress bar
            
            # REMOVED: GPU monitoring that breaks progress bar
        
    def forward(self, x):
        return self.model(x)
    
    def _initialize_criterion(self, device):
        """Initialize loss function based on configuration"""
        if self._criterion is None:
            if self.use_focal_loss:
                # Use class weights for focal loss alpha
                alpha_weights = self.class_weights.to(device)
                # Normalize weights to sum to 1
                alpha_weights = alpha_weights / alpha_weights.sum()
                self._criterion = FocalLoss(alpha=alpha_weights, gamma=self.focal_gamma)
            else:
                class_weights = self.class_weights.to(device)
                self._criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def _check_hardware_safety(self):
        """Monitor hardware usage and warn if unsafe"""
        if not self.monitor_hardware:
            return
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.log('warning_cpu_high', cpu_percent, prog_bar=True)
            
            # Check GPU usage if available
            if torch.cuda.is_available():
                gpu = GPUtil.getGPUs()[0]
                gpu_memory_percent = gpu.memoryUtil * 100
                gpu_temp = gpu.temperature
                
                if gpu_memory_percent > 90:
                    self.log('warning_gpu_memory_high', gpu_memory_percent, prog_bar=True)
                
                if gpu_temp > 80:
                    self.log('warning_gpu_temp_high', gpu_temp, prog_bar=True)
                    
        except Exception as e:
            # Silently continue if monitoring fails
            pass
    
    def training_step(self, batch, batch_idx):
        """SAFE Training step - Monitor GPU memory and handle device placement properly"""
        # REMOVED: All monitoring that breaks progress bar
        
        features, targets = batch
        
        # REMOVED: Manual CUDA movement - Lightning handles this automatically
        # if torch.cuda.is_available():
        #     features = features.cuda()
        #     targets = targets.cuda()
        
        # Initialize criterion with proper device
        self._initialize_criterion(features.device)
        
        # Forward pass
        outputs = self(features)
        loss = self._criterion(outputs, targets)
        
        # Metrics update (GPU via torchmetrics if available)
        with torch.no_grad():
            predictions = torch.argmax(outputs.detach(), dim=1)
            probabilities = F.softmax(outputs.detach(), dim=1)
            if TORCHMETRICS_AVAILABLE:
                self.tm_train_acc.update(predictions, targets)
                self.tm_train_f1.update(predictions, targets)
                self.tm_train_auroc.update(probabilities, targets)
            else:
                # Fallback to CPU metrics calculator
                if batch_idx % 20 == 0:
                    self.train_metrics.update(predictions, targets, probabilities)
        
        # Log loss on every step (EXACTLY like Pneumonia project)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """End of training epoch"""
        if TORCHMETRICS_AVAILABLE:
            acc = self.tm_train_acc.compute()
            f1 = self.tm_train_f1.compute()
            try:
                auroc = self.tm_train_auroc.compute()
            except Exception:
                auroc = torch.tensor(0.0, device=self.device)
            self.log('train_accuracy', acc, on_epoch=True, prog_bar=True)
            self.log('train_f1_macro', f1, on_epoch=True, prog_bar=True)
            self.log('train_roc_auc_macro', auroc, on_epoch=True, prog_bar=False)
            self.tm_train_acc.reset(); self.tm_train_f1.reset(); self.tm_train_auroc.reset()
        else:
            metrics = self.train_metrics.compute()
            for name, value in metrics.items():
                if isinstance(value, dict):
                    for class_name, class_value in value.items():
                        if isinstance(class_value, (int, float, np.number)):
                            self.log(f'train_{name}_{class_name}', float(class_value), on_epoch=True, prog_bar=False)
                elif isinstance(value, (int, float, np.number)):
                    self.log(f'train_{name}', float(value), on_epoch=True, prog_bar=True)
            self.train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        """OPTIMIZED Validation step - No manual CUDA calls"""
        features, targets = batch
        
        # Initialize criterion with proper device
        self._initialize_criterion(features.device)
        
        # Forward pass with no gradients (CRITICAL for speed)
        with torch.no_grad():
            outputs = self(features)
            loss = self._criterion(outputs, targets)
            predictions = torch.argmax(outputs, dim=1)
            probabilities = F.softmax(outputs, dim=1)
            if TORCHMETRICS_AVAILABLE:
                self.tm_val_acc.update(predictions, targets)
                self.tm_val_f1.update(predictions, targets)
                try:
                    self.tm_val_auroc.update(probabilities, targets)
                except Exception:
                    pass
            else:
                self.val_metrics.update(predictions, targets, probabilities)
        
        # Log only loss during validation step (metrics computed at epoch end for speed)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """End of validation epoch"""
        if TORCHMETRICS_AVAILABLE:
            acc = self.tm_val_acc.compute()
            f1 = self.tm_val_f1.compute()
            self.log('val_accuracy', acc, on_epoch=True, prog_bar=True)
            self.log('val_f1_macro', f1, on_epoch=True, prog_bar=True)
            try:
                auroc = self.tm_val_auroc.compute()
                self.log('val_roc_auc_macro', auroc, on_epoch=True, prog_bar=False)
            except Exception:
                pass
            self.tm_val_acc.reset(); self.tm_val_f1.reset(); self.tm_val_auroc.reset()
        else:
            metrics = self.val_metrics.compute()
            for name, value in metrics.items():
                if isinstance(value, dict):
                    for class_name, class_value in value.items():
                        if isinstance(class_value, (int, float, np.number)):
                            self.log(f'val_{name}_{class_name}', float(class_value), on_epoch=True, prog_bar=False)
                elif isinstance(value, (int, float, np.number)):
                    self.log(f'val_{name}', float(value), on_epoch=True, prog_bar=(name == 'f1'))
            self.val_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        features, targets = batch
        
        # Move data to GPU explicitly (same as Pneumonia project)
        if torch.cuda.is_available():
            features = features.cuda()
            targets = targets.cuda()
        
        # Initialize criterion with proper device
        self._initialize_criterion(features.device)
        
        # Forward pass
        outputs = self(features)
        loss = self._criterion(outputs, targets)
        
        # Get predictions
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        # Update metrics (torchmetrics on GPU if available)
        if TORCHMETRICS_AVAILABLE:
            self.tm_test_acc.update(predictions, targets)
            self.tm_test_f1.update(predictions, targets)
            try:
                self.tm_test_auroc.update(probabilities, targets)
            except Exception:
                pass
        else:
            self.test_metrics.update(predictions, targets, probabilities)
        
        # Log loss
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """End of test epoch"""
        if TORCHMETRICS_AVAILABLE:
            acc = self.tm_test_acc.compute()
            f1 = self.tm_test_f1.compute()
            self.log('test_accuracy', acc, on_epoch=True, prog_bar=True)
            self.log('test_f1_macro', f1, on_epoch=True, prog_bar=True)
            try:
                auroc = self.tm_test_auroc.compute()
                self.log('test_roc_auc_macro', auroc, on_epoch=True, prog_bar=False)
            except Exception:
                pass
            # Prepare a summary dict for printing compatibility
            metrics = {
                'accuracy': float(acc.detach().cpu()),
                'f1_macro': float(f1.detach().cpu()),
            }
            self.tm_test_acc.reset(); self.tm_test_f1.reset(); self.tm_test_auroc.reset()
            self._print_test_results(metrics)
        else:
            metrics = self.test_metrics.compute()
            for name, value in metrics.items():
                if isinstance(value, dict):
                    for class_name, class_value in value.items():
                        if isinstance(class_value, (int, float, np.number)):
                            self.log(f'test_{name}_{class_name}', float(class_value), on_epoch=True, prog_bar=False)
                elif isinstance(value, (int, float, np.number)):
                    self.log(f'test_{name}', float(value), on_epoch=True, prog_bar=True)
            self._print_test_results(metrics)
            self.test_metrics.reset()
    
    def _print_test_results(self, metrics: Dict[str, float]):
        """Print detailed test results"""
        print("\n" + "="*50)
        print(f"TEST RESULTS - {self.model_type.upper()}")
        print("="*50)
        for name, value in metrics.items():
            print(f"{name.upper()}: {value:.4f}")
        print("="*50)
    
    def configure_optimizers(self):
        """OPTIMIZED Configure optimizer and scheduler for RTX 3050"""
        # Use AdamW with fused=False to allow gradient clipping
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=False  # Disable fused to allow gradient clipping
        )
        
        # Configure scheduler based on config
        scheduler_type = self.training_config.get('scheduler', 0)
        
        if scheduler_type == 0:  # onecycle
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * 3,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy='cos',
                div_factor=10,
                final_div_factor=100
            )
            interval = "step"
        elif scheduler_type == 1:  # cosine
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_config.get('epochs', 20),
                eta_min=self.learning_rate * 0.01
            )
            interval = "epoch"
        elif scheduler_type == 2:  # reduce
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=self.training_config.get('patience', 6),
                min_lr=self.learning_rate * 0.001
            )
            interval = "epoch"
        else:
            # Default to OneCycleLR
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * 3,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy='cos',
                div_factor=10,
                final_div_factor=100
            )
            interval = "step"
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": 1
            }
        }
    
    def get_predictions(self, dataloader) -> Tuple[List, List, List]:
        """
        Get predictions from the model
        
        Args:
            dataloader: DataLoader to get predictions for
        
        Returns:
            Tuple of (predictions, targets, probabilities)
        """
        self.eval()
        predictions = []
        targets = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                features, batch_targets = batch
                features = features.to(self.device)
                
                outputs = self(features)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return predictions, targets, probabilities


def create_trainer(
    model_type: str,
    training_config: Dict,
    hardware_config: Dict = None,
    logging_config: Dict = None,
    callbacks: Optional[List] = None
) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer with safety features
    
    Args:
        model_type: Type of model ('cnn_lstm' or 'dnn')
        training_config: Training configuration
        hardware_config: Hardware configuration
        logging_config: Logging configuration
        callbacks: Additional callbacks
    
    Returns:
        PyTorch Lightning trainer
    """
    # Default callbacks
    if callbacks is None:
        callbacks = []
    
    # Model checkpoint callback
    checkpoint_monitor = 'val_accuracy'
    checkpoint_mode = 'max'
    
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        dirpath=f'logs/ids_training/{model_type}/checkpoints',
        filename=f'{model_type}-{{epoch:02d}}-{{val_accuracy:.3f}}',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    # Early stopping callback
    early_stopping_patience = training_config.get('patience', TRAINING_CONFIG['patience'])
    early_stopping_callback = EarlyStopping(
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        patience=early_stopping_patience,
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    callbacks.extend([checkpoint_callback, early_stopping_callback, lr_monitor])
    
    # RTX 3050 optimized configuration
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1
    strategy = 'auto'
    
    # Use 16-bit precision for RTX 3050 Tensor Cores
    precision = 16 if training_config.get('use_amp', True) and torch.cuda.is_available() else 32
    
    # Safety limits
    max_epochs = training_config.get('epochs', TRAINING_CONFIG['epochs'])
    gradient_clip_val = training_config.get('gradient_clip_norm', 1.0)
    accumulate_grad_batches = 1
    
    # Create OPTIMIZED trainer for RTX 3050 - 4x Performance Improvement
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir='logs',
            name='ids_training',
            version=model_type
        ),
        # Performance optimizations (EXACTLY matching Pneumonia project)
        log_every_n_steps=10,  # EXACTLY like Pneumonia project
        enable_progress_bar=True,
        enable_model_summary=True,  # EXACTLY like Pneumonia project
        enable_checkpointing=True,
        deterministic=False,  # For speed
        benchmark=False,  # Reduce CPU spikes from kernel benchmarking on Windows
        
        # Validation optimizations
        num_sanity_val_steps=0,  # Skip sanity validation
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        limit_val_batches=1.0,  # Use full validation set
        
        # Memory optimizations
        sync_batchnorm=False,
        
        # Additional RTX 3050 optimizations
        profiler=None,  # Disable profiling for speed
        detect_anomaly=False,  # Disable anomaly detection
    )
    
    return trainer


def train_model(
    model: IDSLightningModule,
    train_loader,
    val_loader,
    trainer: pl.Trainer
) -> pl.Trainer:
    """
    Train the model with safety monitoring
    
    Args:
        model: Lightning module
        train_loader: Training data loader
        val_loader: Validation data loader
        trainer: PyTorch Lightning trainer
    
    Returns:
        Trained trainer
    """
    try:
        trainer.fit(model, train_loader, val_loader)
        return trainer
    except Exception as e:
        print(f"Training failed: {e}")
        raise e


def test_model(
    model: IDSLightningModule,
    test_loader,
    trainer: pl.Trainer
) -> Dict:
    """
    Test the model
    
    Args:
        model: Lightning module
        test_loader: Test data loader
        trainer: PyTorch Lightning trainer
    
    Returns:
        Test results dictionary
    """
    try:
        results = trainer.test(model, test_loader)
        return results[0] if results else {}
    except Exception as e:
        print(f"Testing failed: {e}")
        return {}