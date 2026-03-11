"""
Real-time Inference Engine for Live Intrusion Detection
Loads trained models and provides real-time threat detection
"""

import torch
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading
import queue
from collections import deque

from config import *
from models import CNNLSTM_IDS, DNN_IDS, Autoencoder_IDS
from utils import load_model, safe_json_dump
from feature_engineering import RealTimeFeatureExtractor

class ModelLoader:
    """Load and manage trained models for inference"""
    
    def __init__(self, models_dir: str = "models"):
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.label_mapping = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_all_models(self):
        """Load all trained models"""
        self.logger.info("Loading trained models...")
        
        # Load preprocessors
        self._load_preprocessors()
        
        # Load CNN-LSTM model
        self._load_cnn_lstm()
        
        # Load DNN model
        self._load_dnn()
        
        # Load Autoencoder model
        self._load_autoencoder()
        
        self.logger.info("All models loaded successfully")
        
    def _load_preprocessors(self):
        """Load preprocessing components"""
        try:
            import joblib
            
            # Load scaler
            scaler_path = self.models_dir.parent / "data" / "processed" / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info("Scaler loaded successfully")
            
            # Load label encoder
            label_encoder_path = self.models_dir.parent / "data" / "processed" / "label_encoder.pkl"
            if label_encoder_path.exists():
                self.label_encoder = joblib.load(label_encoder_path)
                self.logger.info("Label encoder loaded successfully")
            
            # Load label mapping
            label_mapping_path = self.models_dir.parent / "data" / "processed" / "label_mapping.pkl"
            if label_mapping_path.exists():
                self.label_mapping = joblib.load(label_mapping_path)
                self.logger.info("Label mapping loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Error loading preprocessors: {e}")
            raise
    
    def _load_cnn_lstm(self):
        """Load CNN-LSTM model"""
        try:
            model_path = self.models_dir / "cnn_lstm" / "best_model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"CNN-LSTM model not found: {model_path}")
            
            # Load model state - use weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create model
            model = CNNLSTM_IDS(config=CNN_LSTM_CONFIG)
            
            # Load state dict - handle both formats
            state_dict = checkpoint['model_state_dict']
            
            # Remove 'model.' prefix if present (from PyTorch Lightning)
            if any(key.startswith('model.') for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # Remove 'model.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            # Remove non-model keys (like class_weights)
            model_keys = set(model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            state_dict = filtered_state_dict
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.models['cnn_lstm'] = model
            self.logger.info("CNN-LSTM model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading CNN-LSTM model: {e}")
            raise
    
    def _load_dnn(self):
        """Load DNN model"""
        try:
            model_path = self.models_dir / "dnn" / "best_model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"DNN model not found: {model_path}")
            
            # Load model state - use weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create model
            model = DNN_IDS(config=DNN_CONFIG)
            
            # Load state dict - handle both formats
            state_dict = checkpoint['model_state_dict']
            
            # Remove 'model.' prefix if present (from PyTorch Lightning)
            if any(key.startswith('model.') for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # Remove 'model.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            # Remove non-model keys (like class_weights)
            model_keys = set(model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            state_dict = filtered_state_dict
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.models['dnn'] = model
            self.logger.info("DNN model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading DNN model: {e}")
            raise
    
    def _load_autoencoder(self):
        """Load Autoencoder model"""
        try:
            model_path = self.models_dir / "autoencoder" / "best_model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Autoencoder model not found: {model_path}")
            
            # Load model state - use weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create model
            model = Autoencoder_IDS(config=AUTOENCODER_CONFIG)
            
            # Load state dict - handle both formats
            state_dict = checkpoint['model_state_dict']
            
            # Remove 'model.' prefix if present (from PyTorch Lightning)
            if any(key.startswith('model.') for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # Remove 'model.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            # Remove non-model keys (like class_weights)
            model_keys = set(model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            state_dict = filtered_state_dict
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.models['autoencoder'] = model
            self.logger.info("Autoencoder model loaded successfully")
            
            # Load threshold
            threshold_path = self.models_dir / "autoencoder" / "threshold.json"
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                    self.models['autoencoder_threshold'] = threshold_data['threshold']
                    self.logger.info(f"Autoencoder threshold loaded: {self.models['autoencoder_threshold']}")
            
        except Exception as e:
            self.logger.error(f"Error loading Autoencoder model: {e}")
            raise

class RealTimeInferenceEngine:
    """Real-time inference engine for live threat detection"""
    
    def __init__(self, models_dir: str = "models"):
        self.logger = logging.getLogger(__name__)
        self.model_loader = ModelLoader(models_dir)
        self.feature_extractor = RealTimeFeatureExtractor()
        
        # Load models
        self.model_loader.load_all_models()
        
        # Prepare feature alignment from 78 -> expected (typically 66)
        self.expected_num_features = None
        self._feature_keep_indices = None
        try:
            if self.model_loader.scaler is not None and hasattr(self.model_loader.scaler, 'n_features_in_'):
                self.expected_num_features = int(self.model_loader.scaler.n_features_in_)
            else:
                self.expected_num_features = INPUT_FEATURES
        except Exception:
            self.expected_num_features = INPUT_FEATURES
        
        # Compute indices to drop constant columns when going from 78 to 66
        try:
            feature_names_78 = []
            if hasattr(self.feature_extractor, 'feature_extractor') and hasattr(self.feature_extractor.feature_extractor, 'feature_names'):
                feature_names_78 = list(self.feature_extractor.feature_extractor.feature_names)
            if feature_names_78 and self.expected_num_features == 66 and len(feature_names_78) == 78:
                def _norm(n):
                    return ' '.join(str(n).strip().split())
                normalized_feature_names = [_norm(n) for n in feature_names_78]
                normalized_constant = set(_norm(n) for n in CONSTANT_COLUMNS)
                keep_indices = [i for i, n in enumerate(normalized_feature_names) if n not in normalized_constant]
                if len(keep_indices) == 66:
                    self._feature_keep_indices = keep_indices
                else:
                    self.logger.warning(f"Feature keep-indices computed {len(keep_indices)} items, expected 66. Fallback to truncate.")
        except Exception as e:
            self.logger.warning(f"Could not prepare feature alignment: {e}")
        
        # Inference configuration
        self.confidence_threshold = INFERENCE_CONFIG['confidence_threshold']
        self.anomaly_threshold = INFERENCE_CONFIG['anomaly_threshold']
        self.log_predictions = INFERENCE_CONFIG['log_predictions']
        self.save_predictions = INFERENCE_CONFIG['save_predictions']
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'threats_detected': 0,
            'anomalies_detected': 0,
            'start_time': datetime.now(),
            'last_prediction_time': None,
            'prediction_times': deque(maxlen=100)
        }
        # Rolling window of reconstruction errors for dynamic thresholding
        self._recon_error_window = deque(maxlen=500)
        
        # Prediction queue for batch processing
        self.prediction_queue = queue.Queue(maxsize=100)
        self.prediction_thread = None
        self.is_running = False
        
        # Results storage
        self.recent_predictions = deque(maxlen=1000)

    def start(self):
        """Start the inference engine processing thread."""
        self.logger.info("Starting real-time inference engine...")
        self.is_running = True
        self.prediction_thread = threading.Thread(target=self._process_predictions, daemon=True)
        self.prediction_thread.start()
        self.logger.info("Inference engine started successfully")

    def stop(self):
        """Stop the inference engine processing thread."""
        self.logger.info("Stopping inference engine...")
        self.is_running = False
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=5)
        self.logger.info("Inference engine stopped")

    def _align_features(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Align a 78-length feature vector to the expected input size.
        If longer than expected, truncate; if shorter, pad with zeros.
        When going 78->66, drop constant columns if known.
        """
        try:
            if self.expected_num_features is None:
                return features
            n = features.shape[0]
            if n == self.expected_num_features:
                return features
            if n == 78 and self.expected_num_features == 66:
                if self._feature_keep_indices is not None and len(self._feature_keep_indices) == 66:
                    return features[self._feature_keep_indices]
                return features[:66]
            if n > self.expected_num_features:
                return features[:self.expected_num_features]
            if n < self.expected_num_features:
                return np.pad(features, (0, self.expected_num_features - n))
            return features
        except Exception as e:
            self.logger.error(f"Feature alignment failed: {e}")
            return None

    def process_pcap(self, pcap_path: str, max_packets: int = 1000) -> Dict:
        """Offline analysis of a pcap file. Returns aggregate results.

        Tries PyShark first for broad protocol support; falls back to Scapy.
        Limits processing to max_packets to keep latency bounded.
        """
        results = {
            'file': str(pcap_path),
            'total_packets': 0,
            'processed': 0,
            'threats': 0,
            'anomalies': 0,
            'predictions': []  # keep only a sample for UI
        }
        try:
            count = 0
            # Helper to convert scapy pkt -> dict (borrowed from live_capture)
            def scapy_to_dict(pkt):
                from scapy.layers.inet import IP, TCP, UDP
                from scapy.packet import Raw
                d = {
                    'timestamp': time.time(),
                    'src_ip': '0.0.0.0',
                    'dst_ip': '0.0.0.0',
                    'src_port': 0,
                    'dst_port': 0,
                    'protocol': 0,
                    'packet_length': len(pkt),
                    'flags': 0,
                    'payload': b''
                }
                if pkt.haslayer(IP):
                    ip = pkt[IP]
                    d['src_ip'] = ip.src
                    d['dst_ip'] = ip.dst
                    d['protocol'] = ip.proto
                    if pkt.haslayer(TCP):
                        t = pkt[TCP]
                        d['src_port'] = t.sport
                        d['dst_port'] = t.dport
                        d['flags'] = int(t.flags)
                    elif pkt.haslayer(UDP):
                        u = pkt[UDP]
                        d['src_port'] = u.sport
                        d['dst_port'] = u.dport
                    if pkt.haslayer(Raw):
                        d['payload'] = bytes(pkt[Raw].load)
                return d

            # Fast path: use Scapy with lazy reading, limit packet count
            from scapy.all import PcapReader
            try:
                with PcapReader(pcap_path) as pr:
                    for pkt in pr:
                        if count >= max_packets:
                            break
                        results['total_packets'] += 1
                        try:
                            pkt_dict = scapy_to_dict(pkt)
                            features = self.feature_extractor.process_packet(pkt_dict)
                            if features is None:
                                continue
                            pred = self._perform_inference({'features': features, 'packet': pkt_dict, 'timestamp': datetime.now()})
                            if pred:
                                results['processed'] += 1
                                if pred['prediction']['is_threat']:
                                    results['threats'] += 1
                                if pred['anomaly_detection']['is_anomaly']:
                                    results['anomalies'] += 1
                                if len(results['predictions']) < 100:
                                    results['predictions'].append(pred)
                                count += 1
                        except Exception:
                            continue
            except Exception:
                # As a last resort, try full rdpcap (could be memory heavy)
                from scapy.all import rdpcap
                pkts = rdpcap(pcap_path)
                for pkt in pkts[:max_packets]:
                    results['total_packets'] += 1
                    try:
                        pkt_dict = scapy_to_dict(pkt)
                        features = self.feature_extractor.process_packet(pkt_dict)
                        if features is None:
                            continue
                        pred = self._perform_inference({'features': features, 'packet': pkt_dict, 'timestamp': datetime.now()})
                        if pred:
                            results['processed'] += 1
                            if pred['prediction']['is_threat']:
                                results['threats'] += 1
                            if pred['anomaly_detection']['is_anomaly']:
                                results['anomalies'] += 1
                            if len(results['predictions']) < 100:
                                results['predictions'].append(pred)
                    except Exception:
                        continue
            return results
        except Exception as e:
            self.logger.error(f"PCAP processing failed: {e}")
            return results
    
    def process_packet(self, packet: Dict) -> Optional[Dict]:
        """Process a single packet and return prediction"""
        try:
            # Extract features
            features = self.feature_extractor.process_packet(packet)
            if features is None:
                return None
            
            # Build prediction payload
            prediction_data = {
                'features': features,
                'packet': packet,
                'timestamp': datetime.now()
            }
            
            # Best-effort queue for async processing/logging
            try:
                self.prediction_queue.put(prediction_data, timeout=0.1)
            except queue.Full:
                self.logger.warning("Prediction queue full, dropping packet")
            
            # Perform inference synchronously for immediate result to callers
            result = self._perform_inference(prediction_data)
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
            return None
    
    def _process_predictions(self):
        """Process predictions from queue"""
        while self.is_running:
            try:
                # Get prediction data from queue
                prediction_data = self.prediction_queue.get(timeout=1)
                
                # Perform inference
                result = self._perform_inference(prediction_data)
                
                if result:
                    # Store result
                    self.recent_predictions.append(result)
                    
                    # Update statistics
                    self.stats['total_predictions'] += 1
                    self.stats['last_prediction_time'] = datetime.now()
                    
                    if result['prediction']['is_threat']:
                        self.stats['threats_detected'] += 1
                    
                    if result['anomaly_detection']['is_anomaly']:
                        self.stats['anomalies_detected'] += 1
                    
                    # Log prediction if enabled
                    if self.log_predictions:
                        self._log_prediction(result)
                 
                self.prediction_queue.task_done()
                 
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing prediction: {e}")
    
    def _perform_inference(self, prediction_data: Dict) -> Optional[Dict]:
        """Perform inference on features"""
        try:
            features = prediction_data['features']
            packet = prediction_data['packet']
            timestamp = prediction_data['timestamp']
            
            # Validate and align features
            if features is None:
                return None
            features = np.asarray(features, dtype=np.float64).flatten()
            features = self._align_features(features)
            if features is None:
                return None
            
            # Preprocess features
            if self.model_loader.scaler:
                features_scaled = self.model_loader.scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled).to(self.model_loader.device)
            
            # Get predictions from all models
            predictions = {}
            
            # CNN-LSTM prediction
            with torch.no_grad():
                cnn_lstm_output = self.model_loader.models['cnn_lstm'](features_tensor)
                cnn_lstm_probs = torch.softmax(cnn_lstm_output, dim=1)
                predictions['cnn_lstm'] = {
                    'probabilities': cnn_lstm_probs.cpu().numpy()[0],
                    'prediction': torch.argmax(cnn_lstm_probs, dim=1).cpu().numpy()[0]
                }
            
            # DNN prediction
            with torch.no_grad():
                dnn_output = self.model_loader.models['dnn'](features_tensor)
                dnn_probs = torch.softmax(dnn_output, dim=1)
                predictions['dnn'] = {
                    'probabilities': dnn_probs.cpu().numpy()[0],
                    'prediction': torch.argmax(dnn_probs, dim=1).cpu().numpy()[0]
                }
            
            # Ensemble prediction
            ensemble_probs = 0.5 * predictions['cnn_lstm']['probabilities'] + 0.5 * predictions['dnn']['probabilities']
            ensemble_prediction = np.argmax(ensemble_probs)
            ensemble_confidence = np.max(ensemble_probs)
            
            # Anomaly detection
            is_anomaly = False
            reconstruction_error = 0.0
            
            if 'autoencoder' in self.model_loader.models:
                with torch.no_grad():
                    reconstructed = self.model_loader.models['autoencoder'](features_tensor)
                    reconstruction_error = torch.mean((features_tensor - reconstructed) ** 2).cpu().numpy()
                    # Record for dynamic thresholding
                    try:
                        self._recon_error_window.append(float(reconstruction_error))
                    except Exception:
                        pass
                    # Base threshold from training
                    base_threshold = float(self.model_loader.models.get('autoencoder_threshold', 0))
                    # Dynamic adjustment: use rolling 99th percentile if we have enough data
                    adjusted_threshold = base_threshold * 1.5
                    try:
                        if len(self._recon_error_window) >= 50:
                            arr = np.fromiter(self._recon_error_window, dtype=float)
                            p99 = float(np.percentile(arr, 99))
                            # Use the max of base-adjusted threshold and 99th pct to avoid oversensitivity
                            adjusted_threshold = max(adjusted_threshold, p99)
                    except Exception:
                        pass
                    is_anomaly = reconstruction_error > adjusted_threshold
            
            # Determine if it's a threat (only high-confidence non-BENIGN predictions)
            is_threat = ensemble_prediction != 0 and ensemble_confidence > self.confidence_threshold
            is_high_confidence_threat = is_threat
            
            # Get class name
            class_name = "BENIGN"
            if self.model_loader.label_mapping and ensemble_prediction in self.model_loader.label_mapping:
                class_name = self.model_loader.label_mapping[ensemble_prediction]
            
            # Debug logging for threshold analysis
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
            
            if self._debug_count <= 5:  # Log first 5 packets for debugging
                self.logger.info(f"DEBUG: pred={ensemble_prediction}, conf={ensemble_confidence:.3f}, "
                               f"recon_error={reconstruction_error:.3f}, threshold={adjusted_threshold:.3f}, "
                               f"is_threat={is_threat}, is_anomaly={is_anomaly}")
            
            # Create result
            result = {
                'timestamp': timestamp.isoformat(),
                'packet_info': {
                    'src_ip': packet.get('src_ip', 'unknown'),
                    'dst_ip': packet.get('dst_ip', 'unknown'),
                    'src_port': int(packet.get('src_port', 0)),
                    'dst_port': int(packet.get('dst_port', 0)),
                    'protocol': int(packet.get('protocol', 0))
                },
                'prediction': {
                    'class_id': int(ensemble_prediction),
                    'class_name': class_name,
                    'confidence': float(ensemble_confidence),
                    'is_threat': bool(is_threat),
                    'is_high_confidence_threat': bool(is_high_confidence_threat)
                },
                'anomaly_detection': {
                    'is_anomaly': bool(is_anomaly),
                    'reconstruction_error': float(reconstruction_error),
                    'threshold': float(adjusted_threshold)
                },
                'model_predictions': {
                    'cnn_lstm': {
                        'class_id': int(predictions['cnn_lstm']['prediction']),
                        'confidence': float(np.max(predictions['cnn_lstm']['probabilities']))
                    },
                    'dnn': {
                        'class_id': int(predictions['dnn']['prediction']),
                        'confidence': float(np.max(predictions['dnn']['probabilities']))
                    }
                },
                'features_shape': list(features.shape)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in inference: {e}")
            return None
    
    def _log_prediction(self, result: Dict):
        """Log prediction result"""
        packet_info = result['packet_info']
        prediction = result['prediction']
        anomaly = result['anomaly_detection']
        
        log_msg = (
            f"PREDICTION: {packet_info['src_ip']}:{packet_info['src_port']} -> "
            f"{packet_info['dst_ip']}:{packet_info['dst_port']} | "
            f"Class: {prediction['class_name']} | "
            f"Confidence: {prediction['confidence']:.3f} | "
            f"Threat: {prediction['is_threat']} | "
            f"Anomaly: {anomaly['is_anomaly']}"
        )
        
        if prediction['is_threat']:
            self.logger.warning(f"🚨 THREAT DETECTED: {log_msg}")
        else:
            self.logger.info(log_msg)
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """Get recent predictions"""
        return list(self.recent_predictions)[-limit:]
    
    def get_threats(self, limit: int = 50) -> List[Dict]:
        """Get recent threat predictions"""
        threats = [p for p in self.recent_predictions if p['prediction']['is_threat']]
        return threats[-limit:]
    
    def get_anomalies(self, limit: int = 50) -> List[Dict]:
        """Get recent anomaly predictions"""
        anomalies = [p for p in self.recent_predictions if p['anomaly_detection']['is_anomaly']]
        return anomalies[-limit:]
    
    def get_stats(self) -> Dict:
        """Get inference statistics"""
        stats = self.stats.copy()
        
        # Calculate prediction rate
        if stats['start_time']:
            uptime = (datetime.now() - stats['start_time']).total_seconds()
            if uptime > 0:
                stats['predictions_per_second'] = stats['total_predictions'] / uptime
        
        # Calculate threat rate
        if stats['total_predictions'] > 0:
            stats['threat_rate'] = stats['threats_detected'] / stats['total_predictions']
            stats['anomaly_rate'] = stats['anomalies_detected'] / stats['total_predictions']
        
        return stats
    
    def save_predictions_to_file(self, filename: str = None):
        """Save recent predictions to file"""
        if not self.save_predictions:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.json"
        
        predictions_data = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(self.recent_predictions),
            'predictions': list(self.recent_predictions)
        }
        
        try:
            safe_json_dump(predictions_data, Path(filename))
            self.logger.info(f"Predictions saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving predictions: {e}")

# Test function
def test_inference_engine():
    """Test the inference engine"""
    print("Testing inference engine...")
    
    # Create mock packet data
    mock_packet = {
        'timestamp': time.time(),
        'src_ip': '192.168.1.100',
        'dst_ip': '192.168.1.1',
        'src_port': 12345,
        'dst_port': 80,
        'protocol': 6,  # TCP
        'packet_length': 1024,
        'flags': 2,  # SYN
        'payload': b'GET / HTTP/1.1\r\nHost: example.com\r\n\r\n'
    }
    
    try:
        # Initialize engine
        engine = RealTimeInferenceEngine()
        engine.start()
        
        print("Engine started, processing mock packet...")
        
        # Process packet
        result = engine.process_packet(mock_packet)
        
        if result:
            print("Prediction result:")
            print(f"  Class: {result['prediction']['class_name']}")
            print(f"  Confidence: {result['prediction']['confidence']:.3f}")
            print(f"  Is Threat: {result['prediction']['is_threat']}")
            print(f"  Is Anomaly: {result['anomaly_detection']['is_anomaly']}")
        
        # Wait a bit for processing
        time.sleep(2)
        
        # Get stats
        stats = engine.get_stats()
        print(f"Stats: {stats}")
        
    except Exception as e:
        print(f"Error testing inference engine: {e}")
    finally:
        engine.stop()

if __name__ == "__main__":
    test_inference_engine()
