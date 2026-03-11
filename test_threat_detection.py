"""
Live Detection System - Synthetic Data Test
Simulates network traffic detection without requiring real packets
"""

import torch
import numpy as np
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import CNNLSTM_IDS, DNN_IDS, Autoencoder_IDS
from config import CNN_LSTM_CONFIG, DNN_CONFIG, AUTOENCODER_CONFIG, NUM_CLASSES

class SyntheticNetworkDemoIDS:
    """Simulate real-time threat detection on synthetic network data"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.models = self._load_models()
        self.attack_names = {
            0: '🟢 BENIGN',
            1: '🔴 FTP/SSH Brute Force',
            2: '🔴 DoS GoldenEye',
            3: '🔴 DoS Hulk',
            4: '🔴 DoS SlowHTTP',
            5: '🔴 DoS Slowloris',
            6: '🔴 Heartbleed',
            7: '🔴 Bot Activity',
            8: '🔴 Infiltration',
            9: '🔴 XSS Web Attack',
            10: '🔴 SQL Injection',
            11: '🔴 Brute Force Web',
            12: '🔴 DDoS Attack',
            13: '🔴 Port Scan',
            14: '🔴 SSH Brute Force'
        }
        self.packet_count = 0
        self.threats_detected = 0
    
    def _load_models(self):
        """Load all three models"""
        print("Loading AI models...")
        models = {}
        
        try:
            models['CNN-LSTM'] = CNNLSTM_IDS(config=CNN_LSTM_CONFIG).eval()
            models['DNN'] = DNN_IDS(config=DNN_CONFIG).eval()
            models['Autoencoder'] = Autoencoder_IDS(config=AUTOENCODER_CONFIG).eval()
            print("✓ All models loaded successfully\n")
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            exit(1)
        
        return models
    
    def generate_network_sample(self, is_malicious=False):
        """Generate synthetic network traffic sample"""
        sample = np.random.randn(66).astype(np.float32)
        
        if is_malicious:
            # Make it look like an attack pattern
            sample[:10] *= 3  # Spike in first 10 features
            sample[20:30] *= 2  # Spike in middle features
        
        # Normalize
        sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
        return sample
    
    def run_simulation(self, num_packets=20, attack_probability=0.3):
        """Simulate live packet processing"""
        print("=" * 90)
        print("🛡️  LIVE THREAT DETECTION SYSTEM - SYNTHETIC DATA SIMULATION")
        print("=" * 90)
        print(f"\nSimulating {num_packets} network packets...")
        print(f"Attack probability: {attack_probability*100:.0f}%\n")
        print("-" * 90)
        
        for packet_id in range(1, num_packets + 1):
            # Decide if this packet is malicious
            is_malicious = np.random.random() < attack_probability
            
            # Generate sample
            sample = self.generate_network_sample(is_malicious=is_malicious)
            sample_tensor = torch.FloatTensor(sample).unsqueeze(0)
            
            # Run inference
            predictions = {}
            threat_level = "🟢 LOW"
            threat_detected = False
            
            with torch.no_grad():
                # CNN-LSTM
                cnn_logits = self.models['CNN-LSTM'](sample_tensor)
                cnn_pred = torch.argmax(cnn_logits, dim=1).item()
                cnn_prob = torch.softmax(cnn_logits, dim=1).max().item()
                predictions['CNN-LSTM'] = (cnn_pred, cnn_prob)
                
                # DNN
                dnn_logits = self.models['DNN'](sample_tensor)
                dnn_pred = torch.argmax(dnn_logits, dim=1).item()
                dnn_prob = torch.softmax(dnn_logits, dim=1).max().item()
                predictions['DNN'] = (dnn_pred, dnn_prob)
                
                # Autoencoder (anomaly detection)
                ae_output = self.models['Autoencoder'](sample_tensor)
                ae_error = torch.mean((sample_tensor - ae_output) ** 2).item()
                
                # Determine threat
                if dnn_pred != 0 or cnn_pred != 0:
                    threat_detected = True
                    if cnn_prob > 0.9 or dnn_prob > 0.9:
                        threat_level = "🔴 HIGH"
                    elif cnn_prob > 0.7 or dnn_prob > 0.7:
                        threat_level = "🟠 MEDIUM"
                    
                    if threat_detected:
                        self.threats_detected += 1
            
            # Display result
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n[{packet_id:2d}] {timestamp} | Packet #{self.packet_count + 1}")
            print(f"     Status: {threat_level}")
            
            if threat_detected:
                print(f"     🚨 THREAT DETECTED!")
                print(f"     CNN-LSTM: {self.attack_names[cnn_pred]} ({cnn_prob*100:.1f}%)")
                print(f"     DNN: {self.attack_names[dnn_pred]} ({dnn_prob*100:.1f}%)")
            else:
                print(f"     ✓ Normal traffic")
            
            print(f"     Anomaly Score: {ae_error:.4f}")
            
            self.packet_count += 1
            
            # Simulate real-time delay
            time.sleep(0.5)
        
        # Summary
        print("\n" + "=" * 90)
        print("📊 SIMULATION SUMMARY")
        print("=" * 90)
        print(f"Total Packets Analyzed: {self.packet_count}")
        print(f"Threats Detected: {self.threats_detected}")
        print(f"Detection Rate: {(self.threats_detected/self.packet_count)*100:.1f}%")
        print(f"Normal Traffic: {self.packet_count - self.threats_detected}")
        
        print("\n✓ Simulation completed successfully!")
        print("\n💡 Next Steps:")
        print("   1. Run live_detection.py --dashboard for web interface")
        print("   2. Select your Wi-Fi interface to monitor real traffic")
        print("   3. Install Wireshark/Npcap for better packet capture")
        print("=" * 90)

if __name__ == "__main__":
    try:
        sim = SyntheticNetworkDemoIDS()
        
        # Run simulation with default parameters
        print()
        sim.run_simulation(num_packets=15, attack_probability=0.4)
        
    except KeyboardInterrupt:
        print("\n\n⛔ Simulation stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
