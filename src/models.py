"""
Neural network models for AI-based Intrusion Detection System
Optimized for RTX 3050 with PyTorch + CUDA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
import math

from config import *

# =============================================================================
# CNN-LSTM MODEL FOR INTRUSION DETECTION
# =============================================================================
class CNNLSTM_IDS(nn.Module):
    """
    CNN-LSTM architecture for network intrusion detection
    Optimized for CIC-IDS-2017 dataset with 78 features
    """
    
    def __init__(self, config: Dict = None):
        super(CNNLSTM_IDS, self).__init__()
        
        # Use default config if none provided
        if config is None:
            config = CNN_LSTM_CONFIG
        
        self.config = config
        self.input_size = config['input_size']
        self.num_classes = config['output_size']
        
        # CNN layers for spatial feature extraction
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # Treat features as 1D signal
        
        for i, (filters, kernel_size) in enumerate(zip(config['cnn_filters'], config['cnn_kernel_sizes'])):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.Dropout(config['cnn_dropout'])
                )
            )
            in_channels = filters
        
        # Calculate LSTM input size after CNN
        # Assuming input is reshaped to (batch_size, 1, input_size)
        lstm_input_size = config['cnn_filters'][-1]
        
        # LSTM layers for temporal patterns
        # Ensure all config values are properly converted to correct types
        num_layers = int(config['lstm_num_layers'])
        dropout_rate = float(config['lstm_dropout'])
        hidden_size = int(config['lstm_hidden_size'])
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Calculate LSTM output size (bidirectional)
        lstm_output_size = hidden_size * 2
        
        # Fully connected layers
        fc_hidden_size = int(config['fc_hidden_size'])
        fc_dropout = float(config['fc_dropout'])
        
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, fc_hidden_size // 2),
            nn.BatchNorm1d(fc_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size // 2, self.num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize model weights"""
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
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape for CNN: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Reshape for LSTM: (batch_size, seq_len, features)
        # Use the last dimension as sequence length
        x = x.transpose(1, 2)  # (batch_size, input_size, filters)
        
        # Apply LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output from LSTM
        lstm_out = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size * 2)
        
        # Apply fully connected layers
        output = self.fc_layers(lstm_out)
        
        return output
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability (placeholder)"""
        # This would implement attention mechanism for model interpretability
        pass

# =============================================================================
# DEEP NEURAL NETWORK (DNN) MODEL
# =============================================================================
class DNN_IDS(nn.Module):
    """
    Deep Neural Network for intrusion detection
    Baseline classifier with fully connected layers
    """
    
    def __init__(self, config: Dict = None):
        super(DNN_IDS, self).__init__()
        
        if config is None:
            config = DNN_CONFIG
        
        self.config = config
        self.input_size = config['input_size']
        self.num_classes = config['output_size']
        
        # Build fully connected layers
        layers = []
        prev_size = self.input_size
        
        for i, (hidden_size, dropout_rate) in enumerate(zip(config['hidden_sizes'], config['dropout_rates'])):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.network(x)

# =============================================================================
# AUTOENCODER MODEL FOR ANOMALY DETECTION
# =============================================================================
class Autoencoder_IDS(nn.Module):
    """
    Autoencoder for anomaly detection
    Trained only on BENIGN traffic to learn normal patterns
    """
    
    def __init__(self, config: Dict = None):
        super(Autoencoder_IDS, self).__init__()
        
        if config is None:
            config = AUTOENCODER_CONFIG
        
        self.config = config
        self.input_size = config['input_size']
        
        # Encoder layers
        encoder_layers = []
        prev_size = self.input_size
        
        for i, hidden_size in enumerate(config['encoder_sizes']):
            encoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                self._get_activation(config['activation']),
                nn.Dropout(config['dropout_rate'])
            ])
            prev_size = hidden_size
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        prev_size = config['decoder_sizes'][0]
        
        for i, hidden_size in enumerate(config['decoder_sizes'][1:], 1):
            decoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                self._get_activation(config['activation']),
                nn.Dropout(config['dropout_rate'])
            ])
            prev_size = hidden_size
        
        # Final output layer (no activation for reconstruction)
        decoder_layers.append(nn.Linear(prev_size, self.input_size))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Reconstructed tensor of shape (batch_size, input_size)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input to latent representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to output"""
        return self.decoder(z)
    
    def _get_activation(self, activation_type):
        """Get activation function by type"""
        if activation_type == 0 or activation_type == 'relu':
            return nn.ReLU()
        elif activation_type == 1 or activation_type == 'tanh':
            return nn.Tanh()
        elif activation_type == 2 or activation_type == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()  # Default to ReLU
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error"""
        reconstructed = self.forward(x)
        return F.mse_loss(reconstructed, x, reduction='none').mean(dim=1)

# =============================================================================
# ENSEMBLE MODEL
# =============================================================================
class Ensemble_IDS(nn.Module):
    """
    Ensemble model combining CNN-LSTM and DNN predictions
    """
    
    def __init__(self, cnn_lstm_model, dnn_model, ensemble_type='weighted_average'):
        super(Ensemble_IDS, self).__init__()
        
        self.cnn_lstm = cnn_lstm_model
        self.dnn = dnn_model
        self.ensemble_type = ensemble_type
        
        # Learnable weights for ensemble
        if ensemble_type == 'learnable_weights':
            self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:
            self.weights = None
    
    def forward(self, x):
        """
        Forward pass with ensemble prediction
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Ensemble prediction of shape (batch_size, num_classes)
        """
        # Get predictions from both models
        cnn_lstm_pred = F.softmax(self.cnn_lstm(x), dim=1)
        dnn_pred = F.softmax(self.dnn(x), dim=1)
        
        if self.ensemble_type == 'majority_voting':
            # Simple average
            ensemble_pred = (cnn_lstm_pred + dnn_pred) / 2
        elif self.ensemble_type == 'weighted_average':
            # Weighted average (equal weights)
            ensemble_pred = 0.5 * cnn_lstm_pred + 0.5 * dnn_pred
        elif self.ensemble_type == 'learnable_weights':
            # Learnable weights
            weights = F.softmax(self.weights, dim=0)
            ensemble_pred = weights[0] * cnn_lstm_pred + weights[1] * dnn_pred
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
        
        return ensemble_pred

# =============================================================================
# MODEL FACTORY
# =============================================================================
class ModelFactory:
    """Factory class for creating models"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict = None, device: torch.device = None):
        """
        Create a model instance
        
        Args:
            model_type: Type of model ('cnn_lstm', 'dnn', 'autoencoder', 'ensemble')
            config: Model configuration dictionary
            device: Device to place model on
        
        Returns:
            Model instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'cnn_lstm':
            model = CNNLSTM_IDS(config)
        elif model_type == 'dnn':
            model = DNN_IDS(config)
        elif model_type == 'autoencoder':
            model = Autoencoder_IDS(config)
        elif model_type == 'ensemble':
            # For ensemble, we need the individual models
            cnn_lstm = CNNLSTM_IDS(config.get('cnn_lstm_config'))
            dnn = DNN_IDS(config.get('dnn_config'))
            model = Ensemble_IDS(cnn_lstm, dnn, config.get('ensemble_type', 'weighted_average'))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(device)
        return model
    
    @staticmethod
    def get_model_info(model):
        """Get model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': model.__class__.__name__
        }

# =============================================================================
# MODEL UTILITIES
# =============================================================================
def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def model_summary(model, input_size):
    """Print model summary"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size(model):.2f} MB")
    print(f"Input size: {input_size}")
    
    # Test forward pass
    try:
        dummy_input = torch.randn(1, input_size)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            model = model.cuda()
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")

# =============================================================================
# CUSTOM LOSS FUNCTIONS
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Handle alpha weights properly
        if isinstance(self.alpha, torch.Tensor):
            # Index alpha by targets to get the correct weight for each sample
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            # Scalar alpha
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for autoencoder
    """
    def __init__(self, loss_type='mse'):
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, reconstructed, original):
        if self.loss_type == 'mse':
            return F.mse_loss(reconstructed, original)
        elif self.loss_type == 'mae':
            return F.l1_loss(reconstructed, original)
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(reconstructed, original)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test CNN-LSTM model
    print("\n=== CNN-LSTM Model ===")
    cnn_lstm = ModelFactory.create_model('cnn_lstm', device=device)
    model_summary(cnn_lstm, INPUT_FEATURES)
    
    # Test DNN model
    print("\n=== DNN Model ===")
    dnn = ModelFactory.create_model('dnn', device=device)
    model_summary(dnn, INPUT_FEATURES)
    
    # Test Autoencoder model
    print("\n=== Autoencoder Model ===")
    autoencoder = ModelFactory.create_model('autoencoder', device=device)
    model_summary(autoencoder, INPUT_FEATURES)
    
    print("\nAll models created successfully!")
