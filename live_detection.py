"""
Live Intrusion Detection System - Main Entry Point
Integrates packet capture, feature extraction, and real-time threat detection
"""

import sys
import time
import logging
import argparse
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.live_capture import PacketCaptureManager
from src.inference import RealTimeInferenceEngine
from src.dashboard import LiveDetectionDashboard
from src.config import get_device_info
from src.utils import setup_logging

class LiveDetectionSystem:
    """Main live detection system coordinator"""
    
    def __init__(self, config: dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Components
        self.capture_manager = PacketCaptureManager()
        self.inference_engine = None
        self.dashboard = None
        
        # State
        self.is_running = False
        self.capture_interface = None
        self.dashboard_enabled = config.get('dashboard_enabled', False) if config else False
        
        # Statistics
        self.stats = {
            'start_time': None,
            'packets_captured': 0,
            'packets_processed': 0,
            'threats_detected': 0,
            'anomalies_detected': 0,
            'errors': 0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self):
        """Initialize the detection system"""
        self.logger.info("Initializing Live Intrusion Detection System...")
        
        try:
            # Initialize inference engine
            self.logger.info("Loading AI models...")
            self.inference_engine = RealTimeInferenceEngine()
            self.inference_engine.start()
            self.logger.info("AI models loaded successfully")
            
            # Initialize dashboard if enabled
            if self.dashboard_enabled:
                self.logger.info("Starting web dashboard...")
                self.dashboard = LiveDetectionDashboard(
                    host=self.config.get('dashboard_host', '0.0.0.0'),
                    port=self.config.get('dashboard_port', 5000)
                )
                # Start dashboard in separate thread
                dashboard_thread = threading.Thread(target=self.dashboard.start, daemon=True)
                dashboard_thread.start()
                self.logger.info(f"Dashboard started on http://{self.config.get('dashboard_host', '0.0.0.0')}:{self.config.get('dashboard_port', 5000)}")
            
            self.logger.info("System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            return False
    
    def start_capture(self, interface: str = None):
        """Start packet capture on specified interface"""
        if self.is_running:
            self.logger.warning("System already running")
            return False
        
        try:
            # Get available interfaces
            interfaces = self.capture_manager.get_available_interfaces()
            if not interfaces:
                self.logger.error("No network interfaces available")
                return False
            
            # Select interface
            if interface is None:
                interface = interfaces[0]['name']
                self.logger.info(f"Auto-selected interface: {interface}")
            else:
                # Validate interface
                if not any(iface['name'] == interface for iface in interfaces):
                    self.logger.error(f"Interface '{interface}' not found")
                    self.logger.info(f"Available interfaces: {[iface['name'] for iface in interfaces]}")
                    return False
            
            # Start capture
            self.logger.info(f"Starting packet capture on interface: {interface}")
            self.capture_manager.start_capture(
                interface=interface,
                callback=self._packet_callback
            )
            
            self.capture_interface = interface
            self.is_running = True
            self.stats['start_time'] = datetime.now()
            
            self.logger.info("Packet capture started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting capture: {e}")
            return False
    
    def stop_capture(self):
        """Stop packet capture"""
        if not self.is_running:
            self.logger.warning("System not running")
            return
        
        self.logger.info("Stopping packet capture...")
        
        try:
            self.capture_manager.stop_all_captures()
            self.is_running = False
            self.capture_interface = None
            
            self.logger.info("Packet capture stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping capture: {e}")
    
    def _packet_callback(self, features, packet):
        """Callback for processing captured packets"""
        try:
            self.stats['packets_captured'] += 1
            
            # Process with inference engine
            if self.inference_engine:
                prediction = self.inference_engine.process_packet(packet)
                if prediction:
                    # NOTE: packets_processed is incremented in dashboard.add_prediction()
                    # NOTE: threats_detected and anomalies_detected are also incremented there
                    # Avoid double counting - only log here
                    
                    # Log threat/anomaly
                    if prediction['prediction']['is_threat']:
                        self._log_threat(prediction)
                    
                    if prediction['anomaly_detection']['is_anomaly']:
                        self._log_anomaly(prediction)
                    
                    # Log prediction
                    self._log_prediction(prediction)
            
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
            self.stats['errors'] += 1
    
    def _log_threat(self, prediction):
        """Log threat detection"""
        packet_info = prediction['packet_info']
        prediction_info = prediction['prediction']
        
        self.logger.warning(
            f"🚨 THREAT DETECTED: {packet_info['src_ip']}:{packet_info['src_port']} -> "
            f"{packet_info['dst_ip']}:{packet_info['dst_port']} | "
            f"Class: {prediction_info['class_name']} | "
            f"Confidence: {prediction_info['confidence']:.3f}"
        )
    
    def _log_anomaly(self, prediction):
        """Log anomaly detection"""
        packet_info = prediction['packet_info']
        anomaly_info = prediction['anomaly_detection']
        
        self.logger.warning(
            f"⚠️ ANOMALY DETECTED: {packet_info['src_ip']}:{packet_info['src_port']} -> "
            f"{packet_info['dst_ip']}:{packet_info['dst_port']} | "
            f"Reconstruction Error: {anomaly_info['reconstruction_error']:.6f} | "
            f"Threshold: {anomaly_info['threshold']:.6f}"
        )
    
    def _log_prediction(self, prediction):
        """Log prediction result"""
        packet_info = prediction['packet_info']
        prediction_info = prediction['prediction']
        
        if prediction_info['is_threat']:
            return  # Already logged as threat
        
        self.logger.info(
            f"📊 PREDICTION: {packet_info['src_ip']}:{packet_info['src_port']} -> "
            f"{packet_info['dst_ip']}:{packet_info['dst_port']} | "
            f"Class: {prediction_info['class_name']} | "
            f"Confidence: {prediction_info['confidence']:.3f}"
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def shutdown(self):
        """Shutdown the system"""
        self.logger.info("Shutting down Live Intrusion Detection System...")
        
        # Stop capture
        if self.is_running:
            self.stop_capture()
        
        # Stop inference engine
        if self.inference_engine:
            self.inference_engine.stop()
        
        # Stop dashboard
        if self.dashboard:
            self.dashboard.stop()
        
        # Print final statistics
        self._print_final_stats()
        
        self.logger.info("System shutdown complete")
    
    def _print_final_stats(self):
        """Print final statistics"""
        if self.stats['start_time']:
            uptime = datetime.now() - self.stats['start_time']
            self.logger.info("=" * 60)
            self.logger.info("FINAL STATISTICS")
            self.logger.info("=" * 60)
            self.logger.info(f"Uptime: {uptime}")
            self.logger.info(f"Packets Captured: {self.stats['packets_captured']}")
            self.logger.info(f"Packets Processed: {self.stats['packets_processed']}")
            self.logger.info(f"Threats Detected: {self.stats['threats_detected']}")
            self.logger.info(f"Anomalies Detected: {self.stats['anomalies_detected']}")
            self.logger.info(f"Errors: {self.stats['errors']}")
            
            if self.stats['packets_captured'] > 0:
                threat_rate = (self.stats['threats_detected'] / self.stats['packets_captured']) * 100
                anomaly_rate = (self.stats['anomalies_detected'] / self.stats['packets_captured']) * 100
                self.logger.info(f"Threat Rate: {threat_rate:.2f}%")
                self.logger.info(f"Anomaly Rate: {anomaly_rate:.2f}%")
            
            self.logger.info("=" * 60)
    
    def get_status(self):
        """Get current system status"""
        return {
            'running': self.is_running,
            'interface': self.capture_interface,
            'dashboard_enabled': self.dashboard_enabled,
            'stats': self.stats.copy(),
            'inference_engine_active': self.inference_engine is not None,
            'dashboard_active': self.dashboard is not None
        }
    
    def start_dashboard(self):
        """Start the web dashboard"""
        if not self.dashboard_enabled:
            self.logger.error("Dashboard not enabled in config")
            return False
        
        try:
            from src.dashboard import LiveDetectionDashboard
            
            self.dashboard = LiveDetectionDashboard(
                host=self.config.get('dashboard_host', '0.0.0.0'),
                port=self.config.get('dashboard_port', 5000)
            )
            
            self.logger.info("Starting web dashboard...")
            print("🌐 Starting web dashboard...")
            print(f"📱 Dashboard available at: http://{self.config.get('dashboard_host', '0.0.0.0')}:{self.config.get('dashboard_port', 5000)}")
            print("🔄 Press Ctrl+C to stop")
            
            # Start dashboard in a separate thread
            self.dashboard_thread = threading.Thread(target=self.dashboard.start, daemon=True)
            self.dashboard_thread.start()
            
            # Wait a moment for dashboard to start
            time.sleep(2)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            return False

    def run_interactive(self):
        """Run in interactive mode"""
        print("=" * 80)
        print("🛡️ LIVE INTRUSION DETECTION SYSTEM")
        print("=" * 80)
        print(f"Device: {get_device_info()}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
        print("=" * 80)
        
        # Initialize system
        if not self.initialize():
            print("❌ Failed to initialize system")
            return
        
        # Get available interfaces
        interfaces = self.capture_manager.get_available_interfaces()
        if not interfaces:
            print("❌ No network interfaces available")
            return
        
        print("\n📡 Available Network Interfaces:")
        for i, iface in enumerate(interfaces):
            ip_addr = iface['addresses'][0]['ip'] if iface['addresses'] else 'No IP'
            print(f"  {i+1}. {iface['name']} ({ip_addr})")
        
        # Select interface
        while True:
            try:
                choice = input(f"\nSelect interface (1-{len(interfaces)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(interfaces):
                    selected_interface = interfaces[choice_idx]['name']
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number or 'q'.")
        
        # Ask about dashboard
        dashboard_choice = input("\nEnable web dashboard? (y/n): ").strip().lower()
        if dashboard_choice == 'y':
            self.dashboard_enabled = True
            print("✅ Dashboard will be available at http://localhost:5000")
        
        # Start capture
        print(f"\n🚀 Starting packet capture on {selected_interface}...")
        if not self.start_capture(selected_interface):
            print("❌ Failed to start packet capture")
            return
        
        print("✅ System running! Press Ctrl+C to stop.")
        
        try:
            # Main loop
            while self.is_running:
                time.sleep(1)
                
                # Print periodic stats
                if self.stats['packets_captured'] > 0 and self.stats['packets_captured'] % 100 == 0:
                    print(f"📊 Processed {self.stats['packets_captured']} packets, "
                          f"{self.stats['threats_detected']} threats, "
                          f"{self.stats['anomalies_detected']} anomalies")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping system...")
        finally:
            self.shutdown()

def main():
    """Main entry point"""
    print("Starting live detection system...")
    
    parser = argparse.ArgumentParser(description='Live Intrusion Detection System')
    parser.add_argument('--interface', '-i', help='Network interface to capture on')
    parser.add_argument('--dashboard', '-d', action='store_true', help='Enable web dashboard')
    parser.add_argument('--dashboard-host', default='0.0.0.0', help='Dashboard host (default: 0.0.0.0)')
    parser.add_argument('--dashboard-port', type=int, default=5000, help='Dashboard port (default: 5000)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create system
    config = {
        'dashboard_enabled': args.dashboard,
        'dashboard_host': args.dashboard_host,
        'dashboard_port': args.dashboard_port
    }
    
    system = LiveDetectionSystem(config)
    
    try:
        if args.interface or args.dashboard:
            # Command line mode
            print("Initializing system...")
            if not system.initialize():
                print("❌ Failed to initialize system")
                sys.exit(1)
            
            if args.dashboard:
                # In dashboard mode, start only the web UI and let user start/stop capture
                # from the browser so status stays in sync.
                if args.interface:
                    print("ℹ️ Dashboard mode: ignoring --interface. Start capture from the web UI.")
                print("🌐 Starting web dashboard...")
                print(f"📱 Dashboard available at: http://{args.dashboard_host}:{args.dashboard_port}")
                if system.start_dashboard():
                    print("✅ Dashboard started successfully")
                    print("Press Ctrl+C to stop...")
                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        pass
                else:
                    print("❌ Failed to start dashboard")
                    sys.exit(1)
            else:
                if args.interface:
                    if not system.start_capture(args.interface):
                        print("❌ Failed to start packet capture")
                        sys.exit(1)
                    print(f"✅ System running on {args.interface}")
                    print("Press Ctrl+C to stop...")
                    try:
                        while system.is_running:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        pass
                else:
                    print("System initialized. Use --interface to start capture or run with --dashboard.")
                    print("Press Ctrl+C to exit...")
                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        pass
        else:
            # Interactive mode
            system.run_interactive()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()
