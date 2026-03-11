"""
Configuration file for AI-based Intrusion Detection System
Optimized for RTX 3050 with PyTorch + CUDA
Uses YAML configuration file for easy management
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Any

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(__file__).parent.parent / config_path
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Convert scientific notation strings to floats
    if 'training' in config and 'weight_decay' in config['training']:
        if isinstance(config['training']['weight_decay'], str):
            config['training']['weight_decay'] = float(config['training']['weight_decay'])
    
    return config

# Load configuration
CONFIG = load_config()

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "CIC-IDS-2017" / "MachineLearningCVE"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "train"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "test"
FEATURES_PATH = PROJECT_ROOT / "data" / "features"
MODELS_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"
LOGS_PATH = RESULTS_PATH / "logs"
# Save plots in the project root `plots/` directory as requested
PLOTS_PATH = PROJECT_ROOT / "plots"
METRICS_PATH = RESULTS_PATH / "metrics"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# CIC-IDS-2017 CSV files
PROCESSED_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

# Attack type mappings (based on our EDA analysis)
ATTACK_TYPES = {
    'BENIGN': 0,
    'DDoS': 1,
    'PortScan': 2,
    'DoS Hulk': 3,
    'DoS GoldenEye': 4,
    'DoS slowloris': 5,
    'DoS Slowhttptest': 6,
    'FTP-Patator': 7,
    'SSH-Patator': 8,
    'Web Attack – Brute Force': 9,
    'Web Attack – XSS': 10,
    'Web Attack – Sql Injection': 11,
    'Infiltration': 12,
    'Bot': 13,
    'Heartbleed': 14
}

# Reverse mapping for inference
ID_TO_ATTACK = {v: k for k, v in ATTACK_TYPES.items()}

# Class weights from YAML
CLASS_WEIGHTS = CONFIG['class_weights']

# Data quality issues to handle (from our EDA)
CONSTANT_COLUMNS = [
    ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', 
    ' RST Flag Count', ' CWE Flag Count', ' ECE Flag Count',
    'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate',
    ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
]

# Features that commonly have -1 values (placeholders)
PLACEHOLDER_COLUMNS = [
    'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' Flow IAT Min'
]

# Features that may have infinite values
INFINITE_COLUMNS = [
    'Flow Bytes/s', ' Flow Packets/s'
]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Input features (66 after removing constant columns)
INPUT_FEATURES = 66
NUM_CLASSES = len(ATTACK_TYPES)

# Model configurations from YAML
CNN_LSTM_CONFIG = CONFIG['models']['cnn_lstm']
DNN_CONFIG = CONFIG['models']['dnn']
AUTOENCODER_CONFIG = CONFIG['models']['autoencoder']

# =============================================================================
# TRAINING CONFIGURATION (Optimized for RTX 3050)
# =============================================================================
# GPU Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_AVAILABLE = torch.cuda.is_available()

# Training configuration from YAML
TRAINING_CONFIG = CONFIG['training']

# DataLoader configuration from YAML
DATALOADER_CONFIG = CONFIG['dataloader']

# Cross-validation configuration from YAML
CV_CONFIG = CONFIG['cross_validation']

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
EVALUATION_METRICS = CONFIG['evaluation']['metrics']
ANOMALY_THRESHOLD_PERCENTILE = CONFIG['evaluation']['anomaly_threshold_percentile']  # 95th percentile of reconstruction error

# =============================================================================
# REAL-TIME INFERENCE CONFIGURATION
# =============================================================================
INFERENCE_CONFIG = CONFIG['inference']

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOGGING_CONFIG = CONFIG['logging']
# No file logging - console only

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_device_info():
    """Get GPU information for RTX 3050"""
    if CUDA_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB"
    else:
        return "CPU only"

def get_optimal_batch_size():
    """Get optimal batch size for RTX 3050"""
    if CUDA_AVAILABLE:
        # RTX 3050 has 4GB VRAM, start with 512 and adjust
        return 512
    else:
        return 64

def reload_config(config_path: str = "config.yaml"):
    """Reload configuration from YAML file"""
    global CONFIG, TRAINING_CONFIG, DATALOADER_CONFIG, CV_CONFIG
    global EVALUATION_METRICS, ANOMALY_THRESHOLD_PERCENTILE, INFERENCE_CONFIG
    global LOGGING_CONFIG, CNN_LSTM_CONFIG, DNN_CONFIG, AUTOENCODER_CONFIG, CLASS_WEIGHTS
    
    CONFIG = load_config(config_path)
    TRAINING_CONFIG = CONFIG['training']
    DATALOADER_CONFIG = CONFIG['dataloader']
    CV_CONFIG = CONFIG['cross_validation']
    EVALUATION_METRICS = CONFIG['evaluation']['metrics']
    ANOMALY_THRESHOLD_PERCENTILE = CONFIG['evaluation']['anomaly_threshold_percentile']
    INFERENCE_CONFIG = CONFIG['inference']
    LOGGING_CONFIG = CONFIG['logging']
    # No file logging - console only
    CNN_LSTM_CONFIG = CONFIG['models']['cnn_lstm']
    DNN_CONFIG = CONFIG['models']['dnn']
    AUTOENCODER_CONFIG = CONFIG['models']['autoencoder']
    CLASS_WEIGHTS = CONFIG['class_weights']
    
    print(f"Configuration reloaded from {config_path}")

def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("AI-BASED INTRUSION DETECTION SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"Device: {get_device_info()}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Input Features: {INPUT_FEATURES}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"Use AMP: {TRAINING_CONFIG['use_amp']}")
    print(f"CV Folds: {CV_CONFIG['n_folds']}")
    print(f"Config Source: config.yaml")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
