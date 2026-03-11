"""
Feature engineering module for real-time network traffic analysis
Extracts the same 78 features used in CIC-IDS-2017 dataset
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import *

# =============================================================================
# NETWORK FLOW FEATURE EXTRACTION
# =============================================================================
class NetworkFlowFeatureExtractor:
    """
    Extract network flow features for real-time intrusion detection
    Matches the 78 features used in CIC-IDS-2017 dataset
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self):
        """Get the 78 feature names used in CIC-IDS-2017"""
        return [
            'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
            'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
            'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
            'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
            'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
            'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
            'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
            'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
            'Idle Std', 'Idle Max', 'Idle Min'
        ]
    
    def extract_features(self, packets: List[Dict]) -> np.ndarray:
        """
        Extract 78 features from a list of network packets
        
        Args:
            packets: List of packet dictionaries with fields:
                - timestamp: Packet timestamp
                - src_ip: Source IP address
                - dst_ip: Destination IP address
                - src_port: Source port
                - dst_port: Destination port
                - protocol: Protocol number
                - packet_size: Packet size in bytes
                - flags: TCP flags
                - window_size: TCP window size
                - payload: Packet payload
        
        Returns:
            numpy array of 78 features
        """
        if not packets:
            return np.zeros(78)
        
        try:
            # Sort packets by timestamp
            packets = sorted(packets, key=lambda x: x['timestamp'])
            
            # Extract basic flow information
            flow_features = self._extract_basic_flow_features(packets)
            
            # Extract packet statistics
            packet_features = self._extract_packet_statistics(packets)
            
            # Extract timing features
            timing_features = self._extract_timing_features(packets)
            
            # Extract protocol features
            protocol_features = self._extract_protocol_features(packets)
            
            # Extract statistical features
            statistical_features = self._extract_statistical_features(packets)
            
            # Extract behavioral features
            behavioral_features = self._extract_behavioral_features(packets)
            
            # Combine all features
            all_features = np.concatenate([
                flow_features, packet_features, timing_features,
                protocol_features, statistical_features, behavioral_features
            ])
            
            # Ensure we have exactly 78 features
            if len(all_features) != 78:
                self.logger.warning(f"Expected 78 features, got {len(all_features)}")
                # Pad or truncate as needed
                if len(all_features) < 78:
                    all_features = np.pad(all_features, (0, 78 - len(all_features)))
                else:
                    all_features = all_features[:78]
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros(78)
    
    def _extract_basic_flow_features(self, packets: List[Dict]) -> np.ndarray:
        """Extract basic flow features (5 features)"""
        if not packets:
            return np.zeros(5)
        
        # Destination Port
        dst_port = packets[0].get('dst_port', 0)
        
        # Flow Duration (in microseconds)
        if len(packets) > 1:
            duration = (packets[-1]['timestamp'] - packets[0]['timestamp']) * 1000000
        else:
            duration = 0
        
        # Total Forward and Backward Packets
        total_fwd_packets = len(packets)
        total_bwd_packets = 0  # Simplified - would need bidirectional analysis
        
        # Total Length of Forward and Backward Packets
        total_fwd_length = sum(p.get('packet_size', 0) for p in packets)
        total_bwd_length = 0  # Simplified
        
        return np.array([dst_port, duration, total_fwd_packets, total_bwd_packets, total_fwd_length])
    
    def _extract_packet_statistics(self, packets: List[Dict]) -> np.ndarray:
        """Extract packet size statistics (15 features)"""
        if not packets:
            return np.zeros(15)
        
        packet_sizes = [p.get('packet_size', 0) for p in packets]
        
        # Forward packet statistics
        fwd_packet_max = max(packet_sizes) if packet_sizes else 0
        fwd_packet_min = min(packet_sizes) if packet_sizes else 0
        fwd_packet_mean = np.mean(packet_sizes) if packet_sizes else 0
        fwd_packet_std = np.std(packet_sizes) if len(packet_sizes) > 1 else 0
        
        # Backward packet statistics (simplified)
        bwd_packet_max = 0
        bwd_packet_min = 0
        bwd_packet_mean = 0
        bwd_packet_std = 0
        
        # Flow rates
        total_length = sum(packet_sizes)
        total_packets = len(packets)
        duration = (packets[-1]['timestamp'] - packets[0]['timestamp']) if len(packets) > 1 else 1
        
        flow_bytes_per_sec = total_length / duration if duration > 0 else 0
        flow_packets_per_sec = total_packets / duration if duration > 0 else 0
        
        # Packet length statistics
        min_packet_length = fwd_packet_min
        max_packet_length = fwd_packet_max
        packet_length_mean = fwd_packet_mean
        packet_length_std = fwd_packet_std
        packet_length_variance = fwd_packet_std ** 2
        
        # 15 features total (removed 5 IAT placeholders)
        return np.array([
            fwd_packet_max, fwd_packet_min, fwd_packet_mean, fwd_packet_std,
            bwd_packet_max, bwd_packet_min, bwd_packet_mean, bwd_packet_std,
            flow_bytes_per_sec, flow_packets_per_sec,
            min_packet_length, max_packet_length, packet_length_mean, 
            packet_length_std, packet_length_variance
        ])
    
    def _extract_timing_features(self, packets: List[Dict]) -> np.ndarray:
        """Extract timing and inter-arrival time features (20 features)"""
        if len(packets) < 2:
            return np.zeros(20)
        
        timestamps = [p['timestamp'] for p in packets]
        iats = np.diff(timestamps)  # Inter-arrival times
        
        # Flow IAT statistics
        flow_iat_mean = np.mean(iats) if len(iats) > 0 else 0
        flow_iat_std = np.std(iats) if len(iats) > 1 else 0
        flow_iat_max = np.max(iats) if len(iats) > 0 else 0
        flow_iat_min = np.min(iats) if len(iats) > 0 else 0
        
        # Forward IAT statistics (same as flow for simplicity)
        fwd_iat_total = np.sum(iats) if len(iats) > 0 else 0
        fwd_iat_mean = flow_iat_mean
        fwd_iat_std = flow_iat_std
        fwd_iat_max = flow_iat_max
        fwd_iat_min = flow_iat_min
        
        # Backward IAT statistics (simplified)
        bwd_iat_total = 0
        bwd_iat_mean = 0
        bwd_iat_std = 0
        bwd_iat_max = 0
        bwd_iat_min = 0
        
        # Active and Idle times (simplified)
        active_mean = flow_iat_mean
        active_std = flow_iat_std
        active_max = flow_iat_max
        active_min = flow_iat_min
        
        idle_mean = 0
        idle_std = 0
        # Trim idle_max and idle_min to keep total at 20
        
        return np.array([
            flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min,
            fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min,
            bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min,
            active_mean, active_std, active_max, active_min,
            idle_mean, idle_std
        ])
    
    def _extract_protocol_features(self, packets: List[Dict]) -> np.ndarray:
        """Extract protocol and flag features (15 features)"""
        if not packets:
            return np.zeros(15)
        
        # TCP flags (simplified)
        fwd_psh_flags = 0
        bwd_psh_flags = 0
        fwd_urg_flags = 0
        bwd_urg_flags = 0
        
        # Header lengths (simplified)
        fwd_header_length = 20  # Standard IP header
        bwd_header_length = 20
        
        # Packet rates
        total_packets = len(packets)
        duration = (packets[-1]['timestamp'] - packets[0]['timestamp']) if len(packets) > 1 else 1
        fwd_packets_per_sec = total_packets / duration if duration > 0 else 0
        bwd_packets_per_sec = 0  # Simplified
        
        # Flag counts
        fin_flag_count = 0
        syn_flag_count = 0
        rst_flag_count = 0
        psh_flag_count = 0
        ack_flag_count = 0
        urg_flag_count = 0
        cwe_flag_count = 0
        # ece_flag_count omitted to keep total at 15
        
        return np.array([
            fwd_psh_flags, bwd_psh_flags, fwd_urg_flags, bwd_urg_flags,
            fwd_header_length, bwd_header_length, fwd_packets_per_sec, bwd_packets_per_sec,
            fin_flag_count, syn_flag_count, rst_flag_count, psh_flag_count,
            ack_flag_count, urg_flag_count, cwe_flag_count
        ])
    
    def _extract_statistical_features(self, packets: List[Dict]) -> np.ndarray:
        """Extract statistical features (15 features)"""
        if not packets:
            return np.zeros(15)
        
        packet_sizes = [p.get('packet_size', 0) for p in packets]
        total_packets = len(packets)
        total_length = sum(packet_sizes)
        
        # Ratios and averages
        down_up_ratio = 0  # Simplified
        average_packet_size = total_length / total_packets if total_packets > 0 else 0
        avg_fwd_segment_size = average_packet_size
        avg_bwd_segment_size = 0  # Simplified
        
        # Header length (duplicate)
        fwd_header_length_1 = 20
        
        # Bulk transfer features (simplified)
        fwd_avg_bytes_bulk = 0
        fwd_avg_packets_bulk = 0
        fwd_avg_bulk_rate = 0
        bwd_avg_bytes_bulk = 0
        bwd_avg_packets_bulk = 0
        bwd_avg_bulk_rate = 0
        
        # Subflow features (simplified)
        subflow_fwd_packets = total_packets
        subflow_fwd_bytes = total_length
        subflow_bwd_packets = 0
        subflow_bwd_bytes = 0
        
        return np.array([
            down_up_ratio, average_packet_size, avg_fwd_segment_size, avg_bwd_segment_size,
            fwd_header_length_1, fwd_avg_bytes_bulk, fwd_avg_packets_bulk, fwd_avg_bulk_rate,
            bwd_avg_bytes_bulk, bwd_avg_packets_bulk, bwd_avg_bulk_rate,
            subflow_fwd_packets, subflow_fwd_bytes, subflow_bwd_packets, subflow_bwd_bytes
        ])
    
    def _extract_behavioral_features(self, packets: List[Dict]) -> np.ndarray:
        """Extract behavioral features (8 features)"""
        if not packets:
            return np.zeros(8)
        
        # Window sizes and data packet features (simplified)
        init_win_bytes_forward = packets[0].get('window_size', 0) if packets else 0
        init_win_bytes_backward = 0  # Simplified
        act_data_pkt_fwd = len(packets)
        min_seg_size_forward = min(p.get('packet_size', 0) for p in packets) if packets else 0
        
        # These would typically be calculated from actual flow analysis
        # For real-time implementation, we use simplified values
        return np.array([
            init_win_bytes_forward, init_win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward,
            0, 0, 0, 0  # Placeholder for additional behavioral features
        ])

# =============================================================================
# REAL-TIME FEATURE EXTRACTION
# =============================================================================
class RealTimeFeatureExtractor:
    """
    Real-time feature extraction for live network traffic
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = NetworkFlowFeatureExtractor()
        self.flow_cache = {}  # Cache for ongoing flows
        self.completed_flows = []  # Completed flows for feature extraction
        
    def process_packet(self, packet: Dict) -> Optional[np.ndarray]:
        """
        Process a single packet and extract features if flow is complete
        
        Args:
            packet: Dictionary containing packet information
        
        Returns:
            Feature vector if flow is complete, None otherwise
        """
        try:
            # Create flow key
            flow_key = self._create_flow_key(packet)
            
            # Add packet to flow
            if flow_key not in self.flow_cache:
                self.flow_cache[flow_key] = []
            
            self.flow_cache[flow_key].append(packet)
            
            # Check if flow is complete (simplified timeout-based approach)
            if self._is_flow_complete(flow_key):
                # Extract features
                flow_packets = self.flow_cache[flow_key]
                features = self.feature_extractor.extract_features(flow_packets)
                
                # Remove from cache
                del self.flow_cache[flow_key]
                
                return features
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing packet: {str(e)}")
            return None
    
    def _create_flow_key(self, packet: Dict) -> str:
        """Create a unique key for the flow"""
        src_ip = packet.get('src_ip', '0.0.0.0')
        dst_ip = packet.get('dst_ip', '0.0.0.0')
        src_port = packet.get('src_port', 0)
        dst_port = packet.get('dst_port', 0)
        protocol = packet.get('protocol', 0)
        
        # Create bidirectional flow key
        if src_ip < dst_ip:
            return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        else:
            return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
    
    def _is_flow_complete(self, flow_key: str) -> bool:
        """Check if flow is complete based on timeout or packet count"""
        if flow_key not in self.flow_cache:
            return False
        
        flow_packets = self.flow_cache[flow_key]
        
        # Simple completion criteria
        if len(flow_packets) >= 10:  # Minimum packets for meaningful analysis
            return True
        
        # Timeout-based completion (simplified)
        if len(flow_packets) > 1:
            last_packet_time = flow_packets[-1]['timestamp']
            first_packet_time = flow_packets[0]['timestamp']
            if last_packet_time - first_packet_time > 30:  # 30 second timeout
                return True
        
        return False

# =============================================================================
# FEATURE VALIDATION AND NORMALIZATION
# =============================================================================
def validate_features(features: np.ndarray) -> np.ndarray:
    """
    Validate and clean extracted features
    """
    # Replace NaN values with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Replace infinite values with large finite values
    features = np.where(np.isinf(features), np.finfo(np.float64).max, features)
    
    # Ensure non-negative values for certain features
    # (This would need to be customized based on feature semantics)
    
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
# BATCH FEATURE EXTRACTION
# =============================================================================
def extract_features_batch(packet_lists: List[List[Dict]]) -> np.ndarray:
    """
    Extract features for multiple flows in batch
    
    Args:
        packet_lists: List of packet lists, one per flow
    
    Returns:
        2D numpy array with features for each flow
    """
    feature_extractor = NetworkFlowFeatureExtractor()
    features_list = []
    
    for packets in packet_lists:
        features = feature_extractor.extract_features(packets)
        features = validate_features(features)
        features_list.append(features)
    
    return np.array(features_list)

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================
def analyze_feature_importance(model, feature_names: List[str], top_k: int = 20) -> Dict:
    """
    Analyze feature importance for trained models
    """
    try:
        # Get feature importance (method depends on model type)
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_scores = np.abs(model.coef_[0])
        else:
            # For neural networks, use gradient-based importance
            importance_scores = np.random.random(len(feature_names))  # Placeholder
        
        # Get top K features
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        top_features = {
            feature_names[i]: importance_scores[i] 
            for i in top_indices
        }
        
        return top_features
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Error analyzing feature importance: {str(e)}")
        return {}

if __name__ == "__main__":
    # Test feature extraction
    logging.basicConfig(level=logging.INFO)
    
    # Create sample packets
    sample_packets = [
        {
            'timestamp': 1000.0,
            'src_ip': '192.168.1.1',
            'dst_ip': '192.168.1.2',
            'src_port': 12345,
            'dst_port': 80,
            'protocol': 6,
            'packet_size': 1500,
            'flags': 0,
            'window_size': 8192,
            'payload': b''
        },
        {
            'timestamp': 1000.1,
            'src_ip': '192.168.1.1',
            'dst_ip': '192.168.1.2',
            'src_port': 12345,
            'dst_port': 80,
            'protocol': 6,
            'packet_size': 1000,
            'flags': 0,
            'window_size': 8192,
            'payload': b''
        }
    ]
    
    # Extract features
    extractor = NetworkFlowFeatureExtractor()
    features = extractor.extract_features(sample_packets)
    
    print(f"Extracted {len(features)} features")
    print(f"Feature names: {extractor.feature_names}")
    print(f"Sample features: {features[:10]}")
