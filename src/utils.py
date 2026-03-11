"""
Utility functions for AI-based Intrusion Detection System
Includes metrics, logging, data quality fixes, and helper functions
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config imports are done locally in functions to avoid circular imports
from config import LOGS_PATH, LOGGING_CONFIG, NUM_CLASSES, CONSTANT_COLUMNS, PLACEHOLDER_COLUMNS, INFINITE_COLUMNS, get_device_info

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging():
    """Setup logging configuration - CONSOLE ONLY"""
    # Configure logging - CONSOLE ONLY, no file logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.StreamHandler()  # Only console output - no file logging
        ]
    )
    
    return logging.getLogger(__name__)

# =============================================================================
# DATA QUALITY FUNCTIONS
# =============================================================================
def fix_data_quality(df):
    """
    Fix data quality issues identified in CIC-IDS-2017 dataset
    """
    logger = logging.getLogger(__name__)
    original_shape = df.shape
    
    # Remove constant columns
    constant_cols = [col for col in CONSTANT_COLUMNS if col in df.columns]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        logger.info(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
    
    # Handle -1 placeholder values
    for col in PLACEHOLDER_COLUMNS:
        if col in df.columns:
            # Replace -1 with median value for that column
            median_val = df[df[col] != -1][col].median()
            df[col] = df[col].replace(-1, median_val)
    
    # Handle infinite values
    for col in INFINITE_COLUMNS:
        if col in df.columns:
            # Replace infinite values with max finite value
            finite_mask = np.isfinite(df[col])
            if not finite_mask.all():
                max_finite = df[finite_mask][col].max()
                df[col] = df[col].replace([np.inf, -np.inf], max_finite)
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        # Fill missing values with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        missing_after = df.isnull().sum().sum()
    
    logger.info(f"Data quality fix: {original_shape} -> {df.shape}")
    return df

def get_class_weights(y):
    """
    Calculate class weights for imbalanced dataset
    """
    from collections import Counter
    class_counts = Counter(y)
    total_samples = len(y)
    
    weights = {}
    for class_id, count in class_counts.items():
        weights[class_id] = total_samples / (len(class_counts) * count)
    
    return weights

# =============================================================================
# METRICS AND EVALUATION
# =============================================================================
class MetricsCalculator:
    """Comprehensive metrics calculator for IDS evaluation"""
    
    def __init__(self, num_classes=None, class_names=None):
        self.num_classes = num_classes or NUM_CLASSES
        self.class_names = class_names or [f"Class_{i}" for i in range(self.num_classes)]
        
        # For Lightning compatibility
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def calculate_classification_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['precision_per_class'] = dict(zip(self.class_names, precision_per_class))
        metrics['recall_per_class'] = dict(zip(self.class_names, recall_per_class))
        metrics['f1_per_class'] = dict(zip(self.class_names, f1_per_class))
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # ROC-AUC and PR-AUC
        if y_prob is not None:
            try:
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
                metrics['roc_auc_micro'] = roc_auc_score(y_true, y_prob, average='micro', multi_class='ovr')
                metrics['pr_auc_macro'] = average_precision_score(y_true, y_prob, average='macro')
                metrics['pr_auc_micro'] = average_precision_score(y_true, y_prob, average='micro')
            except:
                metrics['roc_auc_macro'] = 0.0
                metrics['roc_auc_micro'] = 0.0
                metrics['pr_auc_macro'] = 0.0
                metrics['pr_auc_micro'] = 0.0
        
        # Detection Rate and False Alarm Rate
        cm = metrics['confusion_matrix']
        tn = cm[0, 0]  # True Negatives (BENIGN correctly classified)
        fp = cm[0, 1:].sum()  # False Positives (BENIGN misclassified as attacks)
        fn = cm[1:, 0].sum()  # False Negatives (Attacks misclassified as BENIGN)
        tp = cm[1:, 1:].sum()  # True Positives (Attacks correctly classified)
        
        metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return metrics
    
    def update(self, predictions, targets, probabilities):
        """Update metrics with new batch data (Lightning compatibility)"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
            
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
        self.probabilities.extend(probabilities)
    
    def compute(self):
        """Compute metrics from accumulated data (Lightning compatibility)"""
        if not self.predictions:
            return {}
            
        y_pred = np.array(self.predictions)
        y_true = np.array(self.targets)
        y_prob = np.array(self.probabilities)
        
        return self.calculate_classification_metrics(y_true, y_pred, y_prob)
    
    def reset(self):
        """Reset accumulated data (Lightning compatibility)"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def calculate_anomaly_metrics(self, y_true, y_scores, threshold):
        """Calculate anomaly detection metrics"""
        y_pred = (y_scores >= threshold).astype(int)
        
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC and PR-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        except:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid display

def plot_roc_curve(y_true, y_prob, class_names, save_path=None):
    """Plot ROC curves for multi-class classification"""
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        if i < y_prob.shape[1]:
            fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc_score(y_true == i, y_prob[:, i]):.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid display

def plot_precision_recall_curve(y_true, y_prob, class_names, save_path=None):
    """Plot Precision-Recall curves for multi-class classification"""
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve for each class
    for i, class_name in enumerate(class_names):
        if i < y_prob.shape[1]:
            precision, recall, _ = precision_recall_curve(y_true == i, y_prob[:, i])
            plt.plot(recall, precision, label=f'{class_name} (AP = {average_precision_score(y_true == i, y_prob[:, i]):.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid display

# =============================================================================
# MODEL UTILITIES
# =============================================================================
def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

def save_model(model, path, epoch, loss, metrics=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)

def load_model(model, path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Extract model state dict if it's wrapped in a Lightning module
    if 'model.' in str(list(state_dict.keys())[0]):
        # Remove 'model.' prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Remove class_weights if present (not part of model state)
    if 'class_weights' in state_dict:
        del state_dict['class_weights']
    
    model.load_state_dict(state_dict)
    return checkpoint

# =============================================================================
# TRAINING UTILITIES
# =============================================================================
class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=None, min_delta=None, restore_best_weights=True):
        from config import TRAINING_CONFIG
        
        # Use config defaults if not provided
        self.patience = patience or TRAINING_CONFIG['patience']
        self.min_delta = min_delta or TRAINING_CONFIG['min_delta']
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

class LearningRateScheduler:
    """Learning rate scheduler"""
    def __init__(self, optimizer, mode='min', factor=0.5, patience=None, min_lr=1e-6):
        from config import TRAINING_CONFIG
        
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience or TRAINING_CONFIG['patience']
        self.min_lr = min_lr
        self.best = None
        self.num_bad_epochs = 0
        
    def step(self, metrics):
        if self.best is None:
            self.best = metrics
        elif self.mode == 'min' and metrics < self.best:
            self.best = metrics
            self.num_bad_epochs = 0
        elif self.mode == 'max' and metrics > self.best:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
            self.num_bad_epochs = 0

def monitor_gpu_memory():
    """Monitor GPU memory usage and warn if unsafe"""
    if not torch.cuda.is_available():
        return None
    
    # Get GPU memory info
    memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
    
    usage_percent = (memory_reserved / memory_total) * 100
    
    gpu_info = {
        'allocated_mb': memory_allocated,
        'reserved_mb': memory_reserved,
        'total_mb': memory_total,
        'usage_percent': usage_percent,
        'free_mb': memory_total - memory_reserved
    }
    
    # Warn if usage is too high
    if usage_percent > 90:
        print(f"⚠️ CRITICAL: GPU memory usage at {usage_percent:.1f}%")
        print("Consider reducing batch size immediately")
    elif usage_percent > 80:
        print(f"⚠️ WARNING: GPU memory usage at {usage_percent:.1f}%")
        print("Consider reducing batch size")
    elif usage_percent > 70:
        print(f"⚠️ CAUTION: GPU memory usage at {usage_percent:.1f}%")
        print("Monitor closely")
    else:
        print(f"✅ GPU memory usage safe: {usage_percent:.1f}%")
    
    return gpu_info

def safe_gpu_cleanup():
    """Safely clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("🧹 GPU memory cleaned up")

# =============================================================================
# DATA UTILITIES
# =============================================================================
def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=None, num_workers=None):
    """Create OPTIMIZED PyTorch data loaders for RTX 3050 - 4x Performance Improvement"""
    from torch.utils.data import DataLoader, TensorDataset
    from config import DATALOADER_CONFIG
    
    # Use config defaults if not provided
    if batch_size is None:
        batch_size = DATALOADER_CONFIG['batch_size']
    if num_workers is None:
        num_workers = DATALOADER_CONFIG['num_workers']
    
    # Convert to tensors with pinned memory for faster GPU transfer
    X_train_tensor = torch.FloatTensor(X_train).pin_memory()
    y_train_tensor = torch.LongTensor(y_train).pin_memory()
    X_val_tensor = torch.FloatTensor(X_val).pin_memory()
    y_val_tensor = torch.LongTensor(y_val).pin_memory()
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # OPTIMIZED loader settings for RTX 3050
    # Only include multiprocessing-related keys when workers > 0 (Windows-safe)
    loader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': DATALOADER_CONFIG['pin_memory'],
    }
    if num_workers and num_workers > 0:
        loader_kwargs.update({
            'prefetch_factor': DATALOADER_CONFIG['prefetch_factor'],
            'multiprocessing_context': DATALOADER_CONFIG['multiprocessing_context']
        })
    
    # Create OPTIMIZED data loaders
    train_loader = DataLoader(
        train_dataset, 
        shuffle=DATALOADER_CONFIG['shuffle'], 
        num_workers=num_workers,
        drop_last=DATALOADER_CONFIG['drop_last'],
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,  # Always False for validation
        num_workers=num_workers,  # Use same workers as training (consistent)
        drop_last=False,  # Don't drop last batch for validation
        batch_size=batch_size,  # Use same batch size as training
        **{k: v for k, v in loader_kwargs.items() if k != 'batch_size'}
    )
    
    return train_loader, val_loader

# =============================================================================
# JSON SERIALIZATION UTILITIES
# =============================================================================
def convert_numpy_types_for_json(obj):
    """
    Comprehensive function to convert all numpy types to JSON-serializable types
    Use this function before any json.dump() call
    """
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, 'dtype'):  # Catch any other numpy types
        try:
            return float(obj)
        except:
            return str(obj)
    elif isinstance(obj, (int, float, str, bool, list, tuple)) or obj is None:
        return obj
    else:
        return str(obj)

def safe_json_dump(data, filepath, indent=2):
    """
    Safely dump data to JSON file with automatic numpy type conversion
    """
    serializable_data = convert_numpy_types_for_json(data)
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=indent)

# =============================================================================
# FILE UTILITIES
# =============================================================================
def save_metrics(metrics, filepath):
    """Save metrics to JSON file with comprehensive numpy type conversion"""
    safe_json_dump(metrics, filepath)

def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def validate_features(features: np.ndarray) -> np.ndarray:
    """
    Validate and clean extracted features
    """
    # Replace NaN values with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Replace infinite values with large finite values
    features = np.where(np.isinf(features), np.finfo(np.float64).max, features)
    
    return features

def normalize_features(features: np.ndarray, scaler) -> np.ndarray:
    """
    Normalize features using the trained scaler
    """
    try:
        # Reshape for scaler (expects 2D array)
        features_2d = features.reshape(1, -1)
        normalized = scaler.transform(features_2d)
        return normalized.flatten()
    except Exception as e:
        logging.getLogger(__name__).error(f"Error normalizing features: {str(e)}")
        return features

def normalize_features(features: np.ndarray, scaler) -> np.ndarray:
    """
    Normalize features using the trained scaler
    """
    try:
        # Reshape for scaler (expects 2D array)
        features_2d = features.reshape(1, -1)
        normalized = scaler.transform(features_2d)
        return normalized.flatten()
    except Exception as e:
        logging.getLogger(__name__).error(f"Error normalizing features: {str(e)}")
        return features

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================
class PerformanceMonitor:
    """Monitor training performance"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def update(self, train_loss, val_loss, train_acc, val_acc, lr):
        """Update performance metrics"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        axes[1, 1].plot(loss_diff)
        axes[1, 1].set_title('Train-Validation Loss Difference')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to avoid display

def verify_device_placement(model, expected_device='cuda'):
    """Verify that a model is actually on the expected device"""
    try:
        device = next(model.parameters()).device
        if expected_device == 'cuda' and device.type == 'cuda':
            return True, f"GPU:{device.index}"
        elif expected_device == 'cpu' and device.type == 'cpu':
            return True, "CPU"
        else:
            return False, f"Unexpected: {device}"
    except Exception as e:
        return False, f"Error: {e}"

def cleanup_gpu_memory():
    """Clean up GPU memory and force garbage collection"""
    try:
        import torch
        import gc
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            print("GPU memory cleaned up")
        
        gc.collect()
        print("System memory cleaned up")
        
    except Exception as e:
        print(f"Memory cleanup warning: {e}")

def check_gpu_memory():
    """Check current GPU memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Total: {total:.2f}GB")
            return allocated, cached, total
        else:
            print("No GPU available for memory check")
            return 0, 0, 0
    except Exception as e:
        print(f"GPU memory check failed: {e}")
        return 0, 0, 0

if __name__ == "__main__":
    # Test configuration
    logger = setup_logging()
    logger.info("Utils module loaded successfully")
    logger.info(f"Device: {get_device_info()}")
