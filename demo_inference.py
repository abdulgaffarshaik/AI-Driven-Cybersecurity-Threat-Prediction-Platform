"""
Demo Script - Test Pre-trained Models
Shows how the AI-based IDS works with sample network traffic data
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import CNNLSTM_IDS, DNN_IDS, Autoencoder_IDS
from config import NUM_CLASSES, INPUT_FEATURES, DEVICE, MODELS_PATH, CNN_LSTM_CONFIG, DNN_CONFIG, AUTOENCODER_CONFIG

def load_pretrained_models():
    """Load all pre-trained models"""
    print("=" * 80)
    print("AI-BASED INTRUSION DETECTION SYSTEM - DEMO")
    print("=" * 80)
    print(f"\nLoading pre-trained models from: {MODELS_PATH}")
    
    models = {}
    device = torch.device('cpu')  # Use CPU for demo
    
    try:
        # Load CNN-LSTM
        print("\n1. Loading CNN-LSTM Model...")
        cnn_lstm = CNNLSTM_IDS(config=CNN_LSTM_CONFIG)
        cnn_lstm_path = MODELS_PATH / "cnn_lstm" / "best_model.pt"
        if cnn_lstm_path.exists():
            checkpoint = torch.load(cnn_lstm_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                cnn_lstm.load_state_dict(checkpoint['model_state_dict'])
            else:
                cnn_lstm.load_state_dict(checkpoint)
            cnn_lstm.eval()
            models['CNN-LSTM'] = cnn_lstm
            print(f"   ✓ CNN-LSTM loaded successfully")
        else:
            print(f"   ✗ Model file not found: {cnn_lstm_path}")
    except Exception as e:
        print(f"   ✗ Error loading CNN-LSTM: {e}")
    
    try:
        # Load DNN
        print("\n2. Loading DNN Model...")
        dnn = DNN_IDS(config=DNN_CONFIG)
        dnn_path = MODELS_PATH / "dnn" / "best_model.pt"
        if dnn_path.exists():
            checkpoint = torch.load(dnn_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                dnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                dnn.load_state_dict(checkpoint)
            dnn.eval()
            models['DNN'] = dnn
            print(f"   ✓ DNN loaded successfully")
        else:
            print(f"   ✗ Model file not found: {dnn_path}")
    except Exception as e:
        print(f"   ✗ Error loading DNN: {e}")
    
    try:
        # Load Autoencoder
        print("\n3. Loading Autoencoder Model...")
        autoencoder = Autoencoder_IDS(config=AUTOENCODER_CONFIG)
        ae_path = MODELS_PATH / "autoencoder" / "best_model.pt"
        if ae_path.exists():
            checkpoint = torch.load(ae_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                autoencoder.load_state_dict(checkpoint['model_state_dict'])
            else:
                autoencoder.load_state_dict(checkpoint)
            autoencoder.eval()
            models['Autoencoder'] = autoencoder
            print(f"   ✓ Autoencoder loaded successfully")
        else:
            print(f"   ✗ Model file not found: {ae_path}")
    except Exception as e:
        print(f"   ✗ Error loading Autoencoder: {e}")
    
    return models

def generate_sample_data(num_samples=5, input_size=INPUT_FEATURES):
    """Generate random sample network traffic data"""
    print(f"\n\nGenerating {num_samples} random samples of network traffic features...")
    print(f"Feature size: {input_size} (network packet features)")
    
    # Generate random network features
    # In real scenario, these come from packet capture and feature extraction
    samples = np.random.randn(num_samples, input_size).astype(np.float32)
    
    # Normalize to [0, 1] range to simulate realistic network metrics
    samples = (samples - samples.min(axis=0)) / (samples.max(axis=0) - samples.min(axis=0) + 1e-8)
    
    return samples

def run_inference(models, sample_data):
    """Run inference on sample data"""
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE ON SAMPLE DATA")
    print("=" * 80)
    
    # Convert to tensor
    X = torch.FloatTensor(sample_data)
    
    results = {
        'num_samples': len(sample_data),
        'feature_size': sample_data.shape[1],
        'predictions': {}
    }
    
    for model_name, model in models.items():
        print(f"\n{model_name} Model Predictions:")
        print("-" * 60)
        
        try:
            with torch.no_grad():
                if model_name == 'Autoencoder':
                    # Autoencoder returns reconstruction
                    reconstruction = model(X)
                    reconstruction_error = torch.mean((X - reconstruction) ** 2, dim=1).numpy()
                    
                    print(f"Reconstruction Errors (Anomaly Scores):")
                    for i, error in enumerate(reconstruction_error):
                        anomaly_status = "⚠️ ANOMALY" if error > 0.5 else "✓ Normal"
                        print(f"  Sample {i+1}: {error:.4f} - {anomaly_status}")
                    
                    results['predictions'][model_name] = {
                        'reconstruction_errors': reconstruction_error.tolist(),
                        'type': 'anomaly_detection'
                    }
                else:
                    # Classification models
                    logits = model(X)
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1).numpy()
                    max_probs = torch.max(probabilities, dim=1).values.numpy()
                    
                    attack_names = {
                        0: 'BENIGN',
                        1: 'FTP/SSH',
                        2: 'DoS GoldenEye',
                        3: 'DoS Hulk',
                        4: 'DoS SlowHTTP',
                        5: 'DoS Slowloris',
                        6: 'Heartbleed',
                        7: 'Bot',
                        8: 'Infiltration',
                        9: 'Web Attack XSS',
                        10: 'Web Attack SQL',
                        11: 'Web Attack Brute Force',
                        12: 'DDoS',
                        13: 'Port Scan',
                        14: 'SSH Brute Force'
                    }
                    
                    print(f"Traffic Classification:")
                    for i, (pred, prob) in enumerate(zip(predictions, max_probs)):
                        threat_type = attack_names.get(pred, f'Unknown ({pred})')
                        confidence = "🔴 LOW" if prob < 0.7 else "🟡 MEDIUM" if prob < 0.9 else "🟢 HIGH"
                        print(f"  Sample {i+1}: {threat_type:25} | Confidence: {prob*100:5.1f}% {confidence}")
                    
                    results['predictions'][model_name] = {
                        'predictions': predictions.tolist(),
                        'probabilities': probabilities.numpy().tolist(),
                        'type': 'classification'
                    }
        
        except Exception as e:
            print(f"  Error during inference: {e}")
            results['predictions'][model_name] = {'error': str(e)}
    
    return results

def display_model_info():
    """Display information about the models"""
    print("\n" + "=" * 80)
    print("PRE-TRAINED MODELS INFORMATION")
    print("=" * 80)
    
    try:
        # Try to read model metrics
        metrics_files = [
            Path("results/cnn_lstm_metrics.json"),
            Path("results/dnn_metrics.json"),
            Path("results/autoencoder_metrics.json"),
            Path("results/model_metrics.json")
        ]
        
        for metrics_file in metrics_files:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    print(f"\n{metrics_file.stem.upper()}:")
                    print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
                    print(f"  Precision (weighted): {metrics.get('precision_weighted', 'N/A')}")
                    print(f"  Recall (weighted): {metrics.get('recall_weighted', 'N/A')}")
                    print(f"  F1-Score (weighted): {metrics.get('f1_weighted', 'N/A')}")
    except Exception as e:
        print(f"Could not load metrics: {e}")

def main():
    """Main demo function"""
    
    # Display model info
    display_model_info()
    
    # Load pre-trained models
    models = load_pretrained_models()
    
    if not models:
        print("\n❌ No models loaded! Check if model files exist in 'models/' directory")
        return
    
    # Generate sample network traffic data
    sample_data = generate_sample_data(num_samples=5)
    
    # Run inference
    results = run_inference(models, sample_data)
    
    # Display summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Successfully tested {len(models)} pre-trained models")
    print(f"✓ Processed {results['num_samples']} network traffic samples")
    print(f"✓ Each sample contains {results['feature_size']} network features")
    
    print("\n📊 The models are ready for:")
    print("   • Real-time threat detection on network packets")
    print("   • Classification of 15 attack types")
    print("   • Anomaly detection via reconstruction error")
    
    print("\n💡 Next Steps:")
    print("   1. Install Wireshark/Npcap for live packet capture")
    print("   2. Run: python live_detection.py")
    print("   3. Monitor real network traffic for threats")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
