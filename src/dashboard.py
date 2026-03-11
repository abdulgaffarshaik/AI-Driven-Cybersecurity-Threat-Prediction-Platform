"""
Real-time Dashboard for Live Intrusion Detection System
Provides web-based monitoring of packet capture and threat detection
"""

import time
import threading
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import queue
from collections import deque, defaultdict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not available. Install with: pip install flask flask-socketio")

try:
    import plotly.graph_objs as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")

from live_capture import PacketCaptureManager
from inference import RealTimeInferenceEngine

class DashboardData:
    """Manage dashboard data and statistics"""
    
    def __init__(self, max_history: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_history = max_history
        
        # Real-time data
        self.packet_stats = {
            'total_captured': 0,
            'total_processed': 0,
            'capture_rate': 0.0,
            'processing_rate': 0.0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        self.threat_stats = {
            'total_threats': 0,
            'total_anomalies': 0,
            'threat_rate': 0.0,
            'anomaly_rate': 0.0,
            'last_threat_time': None,
            'threats_by_class': defaultdict(int)
        }
        
        # Historical data for charts
        self.packet_history = deque(maxlen=max_history)
        self.threat_history = deque(maxlen=max_history)
        self.anomaly_history = deque(maxlen=max_history)
        self.classification_history = deque(maxlen=max_history)
        
        # Recent events
        self.recent_events = deque(maxlen=100)
        self.recent_threats = deque(maxlen=50)
        self.recent_anomalies = deque(maxlen=50)
        
        # Network interfaces
        self.interfaces = []
        self.active_interface = None
        
        # System status
        self.system_status = {
            'capture_running': False,
            'inference_running': False,
            'models_loaded': False,
            'gpu_available': False,
            'last_update': datetime.now()
        }
    
    def update_packet_stats(self, stats: Dict):
        """Update packet capture statistics"""
        self.packet_stats.update(stats)
        # Keep totals in sync with capture stats for immediate UI accuracy
        try:
            if 'packets_captured' in stats:
                self.packet_stats['total_captured'] = int(stats.get('packets_captured', 0))
            if 'packets_processed' in stats:
                self.packet_stats['total_processed'] = int(stats.get('packets_processed', 0))
        except Exception:
            pass
        
        # Add to history
        self.packet_history.append({
            'timestamp': datetime.now(),
            'captured': stats.get('packets_captured', 0),
            'processed': stats.get('packets_processed', 0),
            'errors': stats.get('errors', 0)
        })
    
    def update_threat_stats(self, stats: Dict):
        """Update threat detection statistics"""
        self.threat_stats.update(stats)
        
        # Add to history
        self.threat_history.append({
            'timestamp': datetime.now(),
            'threats': stats.get('threats_detected', 0),
            'anomalies': stats.get('anomalies_detected', 0)
        })
    
    def add_prediction(self, prediction: Dict):
        """Add new prediction result"""
        timestamp = datetime.now()
        
        # Increment counters (since we have a prediction, a packet was processed)
        # Do NOT increment captured here; rely on capture stats to avoid drift
        self.packet_stats['packets_processed'] += 1
        # Keep total_processed in sync for UI from processed
        self.packet_stats['total_processed'] = int(self.packet_stats.get('packets_processed', 0))
        
        # Debug logging for counter consistency
        if hasattr(self, '_debug_prediction_count'):
            self._debug_prediction_count += 1
        else:
            self._debug_prediction_count = 1
        
        if self._debug_prediction_count <= 3:  # Log first 3 predictions
            self.logger.info(f"DEBUG: Added prediction #{self._debug_prediction_count} - "
                           f"Packets processed: {self.packet_stats['packets_processed']}, "
                           f"Threats: {self.threat_stats['total_threats']}, "
                           f"Anomalies: {self.threat_stats['total_anomalies']}")
        
        # Update threat stats
        if prediction['prediction']['is_threat']:
            self.threat_stats['total_threats'] += 1
            self.threat_stats['last_threat_time'] = timestamp
            self.threat_stats['threats_by_class'][prediction['prediction']['class_name']] += 1
            
            # Add to recent threats
            self.recent_threats.append({
                'timestamp': timestamp,
                'prediction': prediction
            })
        
        if prediction['anomaly_detection']['is_anomaly']:
            self.threat_stats['total_anomalies'] += 1
            
            # Add to recent anomalies
            self.recent_anomalies.append({
                'timestamp': timestamp,
                'prediction': prediction
            })
        
        # Add to recent events
        self.recent_events.append({
            'timestamp': timestamp,
            'type': 'threat' if prediction['prediction']['is_threat'] else 'normal',
            'prediction': prediction
        })
        
        # Add to classification history
        self.classification_history.append({
            'timestamp': timestamp,
            'class_name': prediction['prediction']['class_name'],
            'confidence': prediction['prediction']['confidence'],
            'is_threat': prediction['prediction']['is_threat']
        })
    
    def update_interfaces(self, interfaces: List[Dict]):
        """Update available network interfaces"""
        self.interfaces = interfaces
    
    def update_system_status(self, status: Dict):
        """Update system status"""
        self.system_status.update(status)
        self.system_status['last_update'] = datetime.now()
    
    def get_dashboard_data(self) -> Dict:
        """Get complete dashboard data"""
        # Convert datetime objects to ISO format strings and ensure JSON serializable types
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, bool):
                return True if obj else False  # Convert to JSON-compatible boolean
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            elif isinstance(obj, deque):
                return [convert_datetime(item) for item in list(obj)]
            elif isinstance(obj, defaultdict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            else:
                return obj
        
        # Convert all data structures
        ps_copy = self.packet_stats.copy()
        # Consistency guards for UI: processed should not exceed captured
        try:
            tc = int(ps_copy.get('total_captured', 0) or 0)
            tp = int(ps_copy.get('total_processed', 0) or 0)
            pc = int(ps_copy.get('packets_captured', tc) or 0)
            pp = int(ps_copy.get('packets_processed', tp) or 0)
            # Use latest snapshot from capture if totals missing
            if tc == 0 and pc > 0:
                tc = pc
            if tp == 0 and pp > 0:
                tp = pp
            # Clamp processed to not exceed captured
            tp = min(tp, tc) if tc > 0 else tp
            pp = min(pp, pc) if pc > 0 else pp
            ps_copy['total_captured'] = max(tc, 0)
            ps_copy['total_processed'] = max(tp, 0)
            ps_copy['packets_captured'] = max(pc, 0)
            ps_copy['packets_processed'] = max(pp, 0)
        except Exception:
            pass
        packet_stats = convert_datetime(ps_copy)
        
        # Convert threat_stats (ensure defaultdict -> dict)
        ts_copy = self.threat_stats.copy()
        try:
            ts_copy['threats_by_class'] = dict(ts_copy.get('threats_by_class', {}))
        except Exception:
            pass
        threat_stats = convert_datetime(ts_copy)
        
        # Convert system_status
        system_status = convert_datetime(self.system_status.copy())
        
        # Convert interfaces
        interfaces = convert_datetime(self.interfaces.copy() if isinstance(self.interfaces, list) else list(self.interfaces))
        
        return {
            'packet_stats': packet_stats,
            'threat_stats': threat_stats,
            'system_status': system_status,
            'interfaces': interfaces,
            'active_interface': self.active_interface,
            'recent_events': convert_datetime(list(self.recent_events)[-20:]),
            'recent_threats': convert_datetime(list(self.recent_threats)[-10:]),
            'recent_anomalies': convert_datetime(list(self.recent_anomalies)[-10:]),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_chart_data(self) -> Dict:
        """Get data for charts"""
        # Convert datetime objects to ISO format strings and ensure JSON serializable types
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, bool):
                return True if obj else False
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            elif isinstance(obj, deque):
                return [convert_datetime(item) for item in list(obj)]
            elif isinstance(obj, defaultdict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            else:
                return obj
        
        now = datetime.now()
        time_window = timedelta(minutes=10)  # Last 10 minutes
        
        # Filter recent data
        recent_packets = [p for p in self.packet_history if now - p['timestamp'] <= time_window]
        recent_threats = [t for t in self.threat_history if now - t['timestamp'] <= time_window]
        recent_classifications = [c for c in self.classification_history if now - c['timestamp'] <= time_window]
        
        return convert_datetime({
            'packet_timeline': {
                'timestamps': [p['timestamp'] for p in recent_packets],
                'captured': [p['captured'] for p in recent_packets],
                'processed': [p['processed'] for p in recent_packets],
                'errors': [p['errors'] for p in recent_packets]
            },
            'threat_timeline': {
                'timestamps': [t['timestamp'] for t in recent_threats],
                'threats': [t['threats'] for t in recent_threats],
                'anomalies': [t['anomalies'] for t in recent_threats]
            },
            'classification_distribution': self._get_classification_distribution(recent_classifications),
            'threat_distribution': dict(self.threat_stats['threats_by_class'])
        })
    
    def _get_classification_distribution(self, classifications: List[Dict]) -> Dict:
        """Get classification distribution"""
        distribution = defaultdict(int)
        for c in classifications:
            distribution[c['class_name']] += 1
        return dict(distribution)
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, bool):
            return True if obj else False
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, deque):
            return [self._make_json_serializable(item) for item in list(obj)]
        elif isinstance(obj, defaultdict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        else:
            return obj

class LiveDetectionDashboard:
    """Main dashboard class for live intrusion detection"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        
        # Initialize components
        self.capture_manager = PacketCaptureManager()
        self.inference_engine = None
        self.dashboard_data = DashboardData()
        
        # Flask app
        if FLASK_AVAILABLE:
            import os
            template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
            self.app = Flask(__name__, template_folder=template_path)
            self.app.config['SECRET_KEY'] = 'ids_dashboard_secret_key'
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
            self._setup_routes()
        else:
            self.app = None
            self.socketio = None
        
        # Update thread
        self.update_thread = None
        self.is_running = False
        
        # Prediction callback
        self.prediction_callback = None
    
    def _setup_routes(self):
        """Setup Flask routes"""
        if not self.app:
            return
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/data')
        def get_data():
            return jsonify(self.dashboard_data.get_dashboard_data())
        
        @self.app.route('/api/charts')
        def get_charts():
            return jsonify(self.dashboard_data.get_chart_data())
        
        @self.app.route('/api/interfaces')
        def get_interfaces():
            interfaces = self.dashboard_data.interfaces
            self.logger.info(f"API returning {len(interfaces)} interfaces")
            return jsonify(interfaces)
        
        @self.app.route('/api/start_capture', methods=['POST'])
        def start_capture():
            data = request.get_json()
            interface = data.get('interface')
            
            try:
                if self.inference_engine is None:
                    self.inference_engine = RealTimeInferenceEngine()
                    self.inference_engine.start()
                    self.dashboard_data.update_system_status({'inference_running': True, 'models_loaded': True})
                
                # Start capture
                self.capture_manager.start_capture(interface=interface, callback=self._prediction_callback)
                self.dashboard_data.active_interface = interface
                # Verify capture actually running
                running = interface in self.capture_manager.captures and self.capture_manager.captures[interface].is_capturing
                self.dashboard_data.update_system_status({'capture_running': bool(running)})
                if running:
                    return jsonify({'success': True, 'message': f'Capture started on {interface}'})
                else:
                    return jsonify({'success': False, 'message': 'Capture did not start. Try running as Administrator and ensure Npcap is installed.'})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/stop_capture', methods=['POST'])
        def stop_capture():
            try:
                self.capture_manager.stop_all_captures()
                self.dashboard_data.update_system_status({'capture_running': False})
                return jsonify({'success': True, 'message': 'Capture stopped'})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/recent_threats')
        def get_recent_threats():
            return jsonify(self.dashboard_data._make_json_serializable(list(self.dashboard_data.recent_threats)[-20:]))
        
        @self.app.route('/api/recent_anomalies')
        def get_recent_anomalies():
            return jsonify(self.dashboard_data._make_json_serializable(list(self.dashboard_data.recent_anomalies)[-20:]))

        @self.app.route('/api/upload_pcap', methods=['POST'])
        def upload_pcap():
            try:
                if 'file' not in request.files:
                    return jsonify({'success': False, 'message': 'No file uploaded'}), 400
                file = request.files['file']
                if not file.filename.lower().endswith(('.pcap', '.pcapng')):
                    return jsonify({'success': False, 'message': 'Please upload a .pcap or .pcapng file'}), 400
                upload_dir = Path('results') / 'uploads'
                upload_dir.mkdir(parents=True, exist_ok=True)
                save_path = upload_dir / file.filename
                file.save(save_path)

                # Ensure inference engine is ready
                if self.inference_engine is None:
                    self.inference_engine = RealTimeInferenceEngine()
                    self.inference_engine.start()
                    self.dashboard_data.update_system_status({'inference_running': True, 'models_loaded': True})

                results = self.inference_engine.process_pcap(str(save_path))

                # Make results JSON-safe and light for the UI
                def to_safe(pred):
                    try:
                        ts = pred.get('timestamp')
                        ts = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                        p = pred.get('prediction', {})
                        a = pred.get('anomaly_detection', {})
                        info = pred.get('packet_info', {})
                        return {
                            'timestamp': ts,
                            'class_name': p.get('class_name'),
                            'confidence': float(p.get('confidence', 0.0)),
                            'is_threat': bool(p.get('is_threat', False)),
                            'is_anomaly': bool(a.get('is_anomaly', False)),
                            'src': f"{info.get('src_ip','')}:{info.get('src_port',0)}",
                            'dst': f"{info.get('dst_ip','')}:{info.get('dst_port',0)}"
                        }
                    except Exception:
                        return {}

                safe = {
                    'file': results.get('file'),
                    'total_packets': int(results.get('total_packets', 0)),
                    'processed': int(results.get('processed', 0)),
                    'threats': int(results.get('threats', 0)),
                    'anomalies': int(results.get('anomalies', 0)),
                    'predictions': [to_safe(p) for p in (results.get('predictions') or [])][:50]
                }
                return jsonify({'success': True, 'results': safe})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info('Client connected to dashboard')
            emit('status', {'message': 'Connected to live detection dashboard'})
            # Send interfaces immediately when client connects
            emit('interfaces_loaded', self.dashboard_data.interfaces)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info('Client disconnected from dashboard')
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, bool):
            return True if obj else False
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, deque):
            return [self._make_json_serializable(item) for item in list(obj)]
        elif isinstance(obj, defaultdict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    def _prediction_callback(self, features, packet):
        """Callback for processing predictions"""
        if self.inference_engine:
            prediction = self.inference_engine.process_packet(packet)
            if prediction:
                self.dashboard_data.add_prediction(prediction)
                
                # Emit real-time update
                if self.socketio:
                    # Convert prediction to JSON-serializable format
                    serializable_prediction = self._make_json_serializable(prediction)
                    self.socketio.emit('new_prediction', serializable_prediction)
    
    def start(self):
        """Start the dashboard"""
        if not FLASK_AVAILABLE:
            self.logger.error("Flask not available. Cannot start dashboard.")
            return
        
        self.logger.info(f"Starting dashboard on {self.host}:{self.port}")
        
        # Update interfaces
        interfaces = self.capture_manager.get_available_interfaces()
        self.logger.info(f"Found {len(interfaces)} interfaces: {[iface['name'] for iface in interfaces]}")
        self.dashboard_data.update_interfaces(interfaces)
        
        # Start update thread
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Start Flask app
        try:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {e}")
    
    def stop(self):
        """Stop the dashboard"""
        self.logger.info("Stopping dashboard...")
        self.is_running = False
        
        # Stop capture and inference
        self.capture_manager.stop_all_captures()
        if self.inference_engine:
            self.inference_engine.stop()
        
        # Stop update thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
    
    def _update_loop(self):
        """Update loop for dashboard data"""
        while self.is_running:
            try:
                # Update packet stats
                capture_stats = self.capture_manager.get_capture_stats()
                if capture_stats:
                    for interface, stats in capture_stats.items():
                        self.dashboard_data.update_packet_stats(stats)
                
                # Update threat stats
                if self.inference_engine:
                    inference_stats = self.inference_engine.get_stats()
                    self.dashboard_data.update_threat_stats(inference_stats)
                
                # Update system status
                gpu_available = torch.cuda.is_available() if TORCH_AVAILABLE else False
                self.dashboard_data.update_system_status({
                    'gpu_available': gpu_available,
                    'capture_running': len(self.capture_manager.captures) > 0,
                    'inference_running': self.inference_engine is not None
                })
                
                # Emit updates
                if self.socketio:
                    self.socketio.emit('stats_update', self.dashboard_data.get_dashboard_data())
                    self.socketio.emit('charts_update', self.dashboard_data.get_chart_data())
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(5)
    
    def get_status(self) -> Dict:
        """Get current dashboard status"""
        return {
            'running': self.is_running,
            'host': self.host,
            'port': self.port,
            'interfaces': len(self.dashboard_data.interfaces),
            'capture_active': len(self.capture_manager.captures) > 0,
            'inference_active': self.inference_engine is not None
        }

# Create HTML template for dashboard
def create_dashboard_template():
    """Create HTML template for the dashboard"""
    template_dir = Path("templates")
    template_dir.mkdir(exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Intrusion Detection System</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .threat-card {
            background: #ffebee;
            border-left: 4px solid #f44336;
        }
        .normal-card {
            background: #e8f5e8;
            border-left: 4px solid #4caf50;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .controls {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .btn-danger {
            background: #f44336;
        }
        .btn-danger:hover {
            background: #d32f2f;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-running {
            background: #4caf50;
        }
        .status-stopped {
            background: #f44336;
        }
        .events-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        .event-item {
            padding: 10px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .event-threat {
            background: #ffebee;
        }
        .event-normal {
            background: #e8f5e8;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Live Intrusion Detection System</h1>
        <p>Real-time Network Traffic Monitoring & Threat Detection</p>
    </div>

    <div class="controls">
        <h3>System Controls</h3>
        <div>
            <label>Network Interface:</label>
            <select id="interfaceSelect">
                <option value="">Select Interface...</option>
            </select>
            <button class="btn" onclick="startCapture()">Start Capture</button>
            <button class="btn btn-danger" onclick="stopCapture()">Stop Capture</button>
        </div>
        <div style="margin-top: 10px;">
            <span>Status: </span>
            <span class="status-indicator" id="captureStatus"></span>
            <span id="captureStatusText">Stopped</span>
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value" id="packetsCaptured">0</div>
            <div class="stat-label">Packets Captured</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="packetsProcessed">0</div>
            <div class="stat-label">Packets Processed</div>
        </div>
        <div class="stat-card threat-card">
            <div class="stat-value" id="threatsDetected">0</div>
            <div class="stat-label">Threats Detected</div>
        </div>
        <div class="stat-card threat-card">
            <div class="stat-value" id="anomaliesDetected">0</div>
            <div class="stat-label">Anomalies Detected</div>
        </div>
    </div>

    <div class="chart-container">
        <h3>Packet Capture Timeline</h3>
        <div id="packetChart" style="height: 300px;"></div>
    </div>

    <div class="chart-container">
        <h3>Threat Detection Timeline</h3>
        <div id="threatChart" style="height: 300px;"></div>
    </div>

    <div class="events-container">
        <h3>Recent Events</h3>
        <div id="eventsList"></div>
    </div>

    <script>
        const socket = io();
        let packetChart, threatChart;

        // Initialize charts
        function initCharts() {
            packetChart = Plotly.newPlot('packetChart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Packets Captured'
            }, {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Packets Processed'
            }], {
                title: 'Packet Capture Rate',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Packets' }
            });

            threatChart = Plotly.newPlot('threatChart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Threats',
                line: { color: 'red' }
            }, {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Anomalies',
                line: { color: 'orange' }
            }], {
                title: 'Threat Detection Rate',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Count' }
            });
        }

        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to dashboard');
            loadData();
        });

        socket.on('interfaces_loaded', function(interfaces) {
            console.log('Interfaces loaded:', interfaces);
            updateInterfaceDropdown(interfaces);
        });

        socket.on('stats_update', function(data) {
            updateStats(data);
        });

        socket.on('charts_update', function(data) {
            updateCharts(data);
        });

        socket.on('new_prediction', function(prediction) {
            addEvent(prediction);
        });

        // Update interface dropdown
        function updateInterfaceDropdown(interfaces) {
            const select = document.getElementById('interfaceSelect');
            select.innerHTML = '<option value="">Select Interface...</option>';
            interfaces.forEach(iface => {
                const option = document.createElement('option');
                option.value = iface.name;
                const ip = iface.addresses && iface.addresses[0] ? iface.addresses[0].ip : 'No IP';
                option.textContent = `${iface.name} (${ip})`;
                select.appendChild(option);
            });
        }

        // Load interfaces (fallback)
        function loadInterfaces() {
            fetch('/api/interfaces')
                .then(response => response.json())
                .then(data => {
                    updateInterfaceDropdown(data);
                })
                .catch(error => {
                    console.error('Error loading interfaces:', error);
                });
        }

        // Load initial data
        function loadData() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    updateStats(data);
                });
        }

        // Update statistics
        function updateStats(data) {
            document.getElementById('packetsCaptured').textContent = data.packet_stats.total_captured || 0;
            document.getElementById('packetsProcessed').textContent = data.packet_stats.total_processed || 0;
            document.getElementById('threatsDetected').textContent = data.threat_stats.total_threats || 0;
            document.getElementById('anomaliesDetected').textContent = data.threat_stats.total_anomalies || 0;

            // Update status
            const statusIndicator = document.getElementById('captureStatus');
            const statusText = document.getElementById('captureStatusText');
            if (data.system_status.capture_running) {
                statusIndicator.className = 'status-indicator status-running';
                statusText.textContent = 'Running';
            } else {
                statusIndicator.className = 'status-indicator status-stopped';
                statusText.textContent = 'Stopped';
            }
        }

        // Update charts
        function updateCharts(data) {
            if (data.packet_timeline) {
                Plotly.update('packetChart', {
                    x: [data.packet_timeline.timestamps, data.packet_timeline.timestamps],
                    y: [data.packet_timeline.captured, data.packet_timeline.processed]
                });
            }

            if (data.threat_timeline) {
                Plotly.update('threatChart', {
                    x: [data.threat_timeline.timestamps, data.threat_timeline.timestamps],
                    y: [data.threat_timeline.threats, data.threat_timeline.anomalies]
                });
            }
        }

        // Add event to list
        function addEvent(prediction) {
            const eventsList = document.getElementById('eventsList');
            const eventDiv = document.createElement('div');
            eventDiv.className = `event-item ${prediction.prediction.is_threat ? 'event-threat' : 'event-normal'}`;
            
            const time = new Date(prediction.timestamp).toLocaleTimeString();
            const packetInfo = prediction.packet_info;
            const predictionInfo = prediction.prediction;
            
            eventDiv.innerHTML = `
                <div>
                    <strong>${time}</strong> - 
                    ${packetInfo.src_ip}:${packetInfo.src_port} → ${packetInfo.dst_ip}:${packetInfo.dst_port}
                    <br>
                    <small>Class: ${predictionInfo.class_name} | Confidence: ${(predictionInfo.confidence * 100).toFixed(1)}%</small>
                </div>
                <div>
                    ${prediction.prediction.is_threat ? '🚨 THREAT' : '✅ Normal'}
                </div>
            `;
            
            eventsList.insertBefore(eventDiv, eventsList.firstChild);
            
            // Keep only last 50 events
            while (eventsList.children.length > 50) {
                eventsList.removeChild(eventsList.lastChild);
            }
        }

        // Control functions
        function startCapture() {
            const interface = document.getElementById('interfaceSelect').value;
            if (!interface) {
                alert('Please select a network interface');
                return;
            }

            fetch('/api/start_capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ interface: interface })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log('Capture started:', data.message);
                } else {
                    alert('Error starting capture: ' + data.message);
                }
            });
        }

        function stopCapture() {
            fetch('/api/stop_capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log('Capture stopped:', data.message);
                } else {
                    alert('Error stopping capture: ' + data.message);
                }
            });
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            
            // Load interfaces on page load as fallback
            loadInterfaces();
            
            // Auto-refresh data every 5 seconds
            setInterval(loadData, 5000);
        });
    </script>
</body>
</html>
    """
    
    template_path = template_dir / "dashboard.html"
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard template created at: {template_path}")

# Test function
def test_dashboard():
    """Test the dashboard"""
    print("Testing dashboard...")
    
    if not FLASK_AVAILABLE:
        print("Flask not available. Cannot test dashboard.")
        return
    
    # Create template
    create_dashboard_template()
    
    # Create dashboard
    dashboard = LiveDetectionDashboard(host='127.0.0.1', port=5000)
    
    try:
        print("Starting dashboard on http://127.0.0.1:5000")
        dashboard.start()
    except KeyboardInterrupt:
        print("Stopping dashboard...")
        dashboard.stop()

if __name__ == "__main__":
    test_dashboard()
