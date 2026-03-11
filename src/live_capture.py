"""
Live Network Packet Capture Module
Integrates with Wireshark/Npcap for real-time packet capture on Windows
"""

import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Callable
from pathlib import Path
import subprocess
import psutil
import json
from datetime import datetime

try:
    import pyshark
    PYSHARK_AVAILABLE = True
except ImportError:
    PYSHARK_AVAILABLE = False
    print("Warning: pyshark not available. Install with: pip install pyshark")

try:
    from scapy.all import *
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.l2 import Ether
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: scapy not available. Install with: pip install scapy")

from config import *
from feature_engineering import RealTimeFeatureExtractor

def _get_windows_wifi_ssids() -> dict:
    """Return a mapping of interface name -> connected SSID on Windows.

    Uses `netsh wlan show interfaces` which is available on Windows. If parsing
    fails, returns an empty mapping. This helps the UI show Wi‑Fi SSIDs like
    "Amrita-5G-809" next to the interface (typically named "Wi-Fi").
    """
    try:
        import platform, subprocess, re
        if platform.system() != 'Windows':
            return {}
        result = subprocess.run(
            ['netsh', 'wlan', 'show', 'interfaces'],
            capture_output=True, text=True, check=False
        )
        text = result.stdout or ''
        # Split by blocks starting with "Name"
        blocks = re.split(r"\r?\n\s*\r?\n", text)
        mapping = {}
        name_re = re.compile(r"^\s*Name\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
        ssid_re = re.compile(r"^\s*SSID\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
        state_re = re.compile(r"^\s*State\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
        for blk in blocks:
            name_m = name_re.search(blk)
            ssid_m = ssid_re.search(blk)
            state_m = state_re.search(blk)
            if name_m and ssid_m and state_m and 'connected' in state_m.group(1).lower():
                mapping[name_m.group(1).strip()] = ssid_m.group(1).strip()
        return mapping
    except Exception:
        return {}

class NetworkInterfaceDetector:
    """Detect available network interfaces for packet capture"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_available_interfaces(self) -> List[Dict]:
        """Get list of available network interfaces with IPs and SSIDs (Windows).

        Priority order:
        1) psutil (names, IP addresses, up/down)
        2) Add Wi‑Fi SSID from netsh (Windows only)
        Falls back to Scapy list if psutil unavailable.
        """
        try:
            interfaces: List[Dict] = []
            wifi_ssids = _get_windows_wifi_ssids()
            try:
                import psutil as _ps
                stats = _ps.net_if_stats()
                addrs = _ps.net_if_addrs()
                for name, s in stats.items():
                    # Only include interfaces that are up
                    if not getattr(s, 'isup', False):
                        continue
                    ip_entries = []
                    for a in addrs.get(name, []):
                        # AF_INET only
                        if getattr(a, 'family', None) == getattr(__import__('socket'), 'AF_INET'):
                            ip_entries.append({'ip': a.address, 'netmask': a.netmask, 'broadcast': a.broadcast})
                    interfaces.append({
                        'name': name,
                        'description': 'Wi-Fi' if 'wi' in name.lower() else 'Network Interface',
                        'addresses': ip_entries,
                        'ssid': wifi_ssids.get(name)
                    })
            except Exception:
                pass
            # Fallback to scapy list if nothing found
            if not interfaces and SCAPY_AVAILABLE:
                try:
                    for name in get_if_list():
                        ip = None
                        try:
                            ip = get_if_addr(name)
                        except Exception:
                            ip = None
                        interfaces.append({
                            'name': name,
                            'description': 'Network Interface',
                            'addresses': [{'ip': ip or '0.0.0.0', 'netmask': None, 'broadcast': None}],
                            'ssid': wifi_ssids.get(name)
                        })
                except Exception:
                    pass
            if not interfaces:
                self.logger.warning("No active interfaces detected")
            else:
                self.logger.info(f"Detected interfaces: {[ (i['name'], i.get('ssid')) for i in interfaces ]}")
            return interfaces
        except Exception as e:
            self.logger.error(f"Interface detection error: {e}")
            return []
    
    def get_default_interface(self) -> Optional[str]:
        """Get the default network interface"""
        interfaces = self.get_available_interfaces()
        if not interfaces:
            return None
            
        # Prefer interfaces with active connections
        for interface in interfaces:
            if any(addr['ip'] != '127.0.0.1' for addr in interface['addresses']):
                return interface['name']
                
        return interfaces[0]['name'] if interfaces else None

class LivePacketCapture:
    """Live packet capture using pyshark and scapy"""
    
    def __init__(self, interface: str = None, callback: Callable = None):
        self.logger = logging.getLogger(__name__)
        self.interface = interface or self._get_default_interface()
        self.callback = callback
        self.is_capturing = False
        self.capture_thread = None
        self.packet_queue = queue.Queue(maxsize=1000)
        self.stats = {
            'packets_captured': 0,
            'packets_processed': 0,
            'errors': 0,
            'start_time': None,
            'last_packet_time': None
        }
        
        # Feature extractor for real-time processing
        self.feature_extractor = RealTimeFeatureExtractor()
        self._resolved_iface = None  # scapy/tshark-friendly name

    def _resolve_interface_for_scapy(self, name: str, addresses: List[Dict]) -> Optional[str]:
        """Resolve a human-friendly name (e.g., "Wi-Fi") to a Scapy/Npcap capture name on Windows.

        Strategy:
        - Try exact name first.
        - On Windows, map via get_windows_if_list() by matching friendly name or IPv4.
        - Prefer returning Npcap GUID form: "\\Device\\NPF_{GUID}" when available.
        - Fallback to first up interface containing the substring.
        """
        try:
            if not SCAPY_AVAILABLE:
                return name
            # Try direct use first
            try:
                get_if_addr(name)
                return name
            except Exception:
                pass
            ipv4s = set([a.get('ip') for a in (addresses or []) if a.get('ip')])
            try:
                from scapy.arch.windows import get_windows_if_list
                win_ifs = get_windows_if_list()
                # Helper to build Npcap GUID name
                def npf_name(wi):
                    gid = wi.get('guid') or wi.get('netid') or wi.get('index')
                    return f"\\Device\\NPF_{gid}" if gid else None
                # Exact friendly/description match → prefer GUID name
                for wi in win_ifs:
                    if wi.get('name') == name or wi.get('description') == name:
                        return npf_name(wi) or wi.get('name') or wi.get('description')
                # Match by IP
                for wi in win_ifs:
                    wi_ips = set(wi.get('ips') or [])
                    if wi_ips & ipv4s:
                        return npf_name(wi) or wi.get('name') or wi.get('description')
                # Substring match
                for wi in win_ifs:
                    if name.lower() in (wi.get('name','') + wi.get('description','')).lower():
                        return npf_name(wi) or wi.get('name') or wi.get('description')
            except Exception:
                pass
        except Exception:
            pass
        return name
        
    def _get_default_interface(self) -> str:
        """Get default network interface"""
        detector = NetworkInterfaceDetector()
        interface = detector.get_default_interface()
        if not interface:
            raise RuntimeError("No network interfaces available for packet capture")
        return interface
    
    def start_capture(self):
        """Start live packet capture"""
        if self.is_capturing:
            self.logger.warning("Capture already running")
            return
            
        self.logger.info(f"Starting packet capture on interface: {self.interface}")
        try:
            # Resolve to scapy-friendly name for reliability
            # Use the interface list to get addresses to aid resolution
            try:
                det = NetworkInterfaceDetector()
                available = det.get_available_interfaces()
                match = next((i for i in available if i['name'] == self.interface), None)
                self._resolved_iface = self._resolve_interface_for_scapy(self.interface, match.get('addresses') if match else [])
            except Exception:
                self._resolved_iface = self.interface
            self.logger.info(f"Resolved capture interface: {self._resolved_iface}")
        except Exception as e:
            self.logger.warning(f"Interface resolution warning: {e}")
            self._resolved_iface = self.interface
        self.is_capturing = True
        self.stats['start_time'] = datetime.now()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_packets, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_packets, daemon=True)
        self.process_thread.start()
        
    def stop_capture(self):
        """Stop live packet capture"""
        self.logger.info("Stopping packet capture...")
        self.is_capturing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
            
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=5)
            
    def _capture_packets(self):
        """Capture packets using Scapy only (robust on Windows with Npcap)."""
        try:
            if SCAPY_AVAILABLE:
                self._capture_with_scapy()
            else:
                raise RuntimeError("Scapy not available for packet capture")
        except Exception as e:
            self.logger.error(f"Error in packet capture: {e}")
            self.stats['errors'] += 1
    
    def _capture_with_scapy(self):
        """Capture packets using scapy (fallback)"""
        try:
            self.logger.info(f"Scapy capture started on {self.interface}")
            iface = self._resolved_iface or self.interface
            self.logger.info(f"Scapy using iface: {iface}")
            
            def packet_handler(packet):
                if not self.is_capturing:
                    return
                
                try:
                    packet_data = self._scapy_to_packet_dict(packet)
                    if packet_data:
                        self.packet_queue.put(packet_data, timeout=1)
                        self.stats['packets_captured'] += 1
                        self.stats['last_packet_time'] = datetime.now()
                
                except queue.Full:
                    self.logger.warning("Packet queue full, dropping packet")
                except Exception as e:
                    self.logger.error(f"Error processing scapy packet: {e}")
                    self.stats['errors'] += 1
            
            # Start sniffing
            sniff(iface=iface, prn=packet_handler, stop_filter=lambda x: not self.is_capturing, store=False)
            
        except Exception as e:
            self.logger.error(f"Scapy capture error: {e}")
            self.stats['errors'] += 1
    
    def _pyshark_to_packet_dict(self, packet) -> Optional[Dict]:
        """Convert pyshark packet to our packet format"""
        try:
            packet_dict = {
                'timestamp': float(packet.sniff_timestamp),
                'src_ip': '0.0.0.0',
                'dst_ip': '0.0.0.0',
                'src_port': 0,
                'dst_port': 0,
                'protocol': 0,
                'packet_length': int(packet.length) if hasattr(packet, 'length') else 0,
                'flags': 0,
                'payload': b''
            }
            
            # Extract IP layer
            if hasattr(packet, 'ip'):
                packet_dict['src_ip'] = packet.ip.src
                packet_dict['dst_ip'] = packet.ip.dst
                packet_dict['protocol'] = int(packet.ip.proto) if hasattr(packet.ip, 'proto') else 0
            
            # Extract TCP layer
            if hasattr(packet, 'tcp'):
                packet_dict['src_port'] = int(packet.tcp.srcport)
                packet_dict['dst_port'] = int(packet.tcp.dstport)
                packet_dict['flags'] = int(packet.tcp.flags) if hasattr(packet.tcp, 'flags') else 0
            
            # Extract UDP layer
            elif hasattr(packet, 'udp'):
                packet_dict['src_port'] = int(packet.udp.srcport)
                packet_dict['dst_port'] = int(packet.udp.dstport)
            
            # Extract payload
            if hasattr(packet, 'data'):
                packet_dict['payload'] = bytes(packet.data.data) if hasattr(packet.data, 'data') else b''
            
            return packet_dict
            
        except Exception as e:
            self.logger.error(f"Error converting pyshark packet: {e}")
            return None
    
    def _scapy_to_packet_dict(self, packet) -> Optional[Dict]:
        """Convert scapy packet to our packet format"""
        try:
            packet_dict = {
                'timestamp': time.time(),
                'src_ip': '0.0.0.0',
                'dst_ip': '0.0.0.0',
                'src_port': 0,
                'dst_port': 0,
                'protocol': 0,
                'packet_length': len(packet),
                'flags': 0,
                'payload': b''
            }
            
            # Extract IP layer
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                packet_dict['src_ip'] = ip_layer.src
                packet_dict['dst_ip'] = ip_layer.dst
                packet_dict['protocol'] = ip_layer.proto
                
                # Extract TCP layer
                if packet.haslayer(TCP):
                    tcp_layer = packet[TCP]
                    packet_dict['src_port'] = tcp_layer.sport
                    packet_dict['dst_port'] = tcp_layer.dport
                    packet_dict['flags'] = tcp_layer.flags
                
                # Extract UDP layer
                elif packet.haslayer(UDP):
                    udp_layer = packet[UDP]
                    packet_dict['src_port'] = udp_layer.sport
                    packet_dict['dst_port'] = udp_layer.dport
                
                # Extract payload
                if packet.haslayer(Raw):
                    packet_dict['payload'] = bytes(packet[Raw].load)
            
            return packet_dict
            
        except Exception as e:
            self.logger.error(f"Error converting scapy packet: {e}")
            return None
    
    def _process_packets(self):
        """Process captured packets and extract features"""
        while self.is_capturing:
            try:
                # Get packet from queue
                packet = self.packet_queue.get(timeout=1)
                
                # Extract features using real-time feature extractor
                features = self.feature_extractor.process_packet(packet)
                
                if features is not None:
                    # Call callback with features
                    if self.callback:
                        self.callback(features, packet)
                    
                    self.stats['packets_processed'] += 1
                
                self.packet_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing packet: {e}")
                self.stats['errors'] += 1
    
    def get_stats(self) -> Dict:
        """Get capture statistics"""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['uptime_seconds'] = (datetime.now() - stats['start_time']).total_seconds()
        return stats

class PacketCaptureManager:
    """Manager for live packet capture with multiple interfaces"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.captures = {}
        self.interface_detector = NetworkInterfaceDetector()
        
    def start_capture(self, interface: str = None, callback: Callable = None) -> str:
        """Start packet capture on specified interface"""
        if interface is None:
            interface = self.interface_detector.get_default_interface()
            
        if not interface:
            raise RuntimeError("No network interface available")
            
        if interface in self.captures:
            self.logger.warning(f"Capture already running on {interface}")
            return interface
            
        # Create new capture
        capture = LivePacketCapture(interface=interface, callback=callback)
        capture.start_capture()
        
        self.captures[interface] = capture
        self.logger.info(f"Started capture on interface: {interface}")
        
        return interface
    
    def stop_capture(self, interface: str):
        """Stop packet capture on specified interface"""
        if interface not in self.captures:
            self.logger.warning(f"No capture running on {interface}")
            return
            
        self.captures[interface].stop_capture()
        del self.captures[interface]
        self.logger.info(f"Stopped capture on interface: {interface}")
    
    def stop_all_captures(self):
        """Stop all running captures"""
        for interface in list(self.captures.keys()):
            self.stop_capture(interface)
    
    def get_capture_stats(self, interface: str = None) -> Dict:
        """Get statistics for capture"""
        if interface:
            if interface in self.captures:
                return self.captures[interface].get_stats()
            return {}
        
        # Return stats for all captures
        all_stats = {}
        for iface, capture in self.captures.items():
            all_stats[iface] = capture.get_stats()
        return all_stats
    
    def get_available_interfaces(self) -> List[Dict]:
        """Get available network interfaces"""
        return self.interface_detector.get_available_interfaces()

# Test function
def test_interface_detection():
    """Test interface detection"""
    print("Testing interface detection...")
    
    detector = NetworkInterfaceDetector()
    interfaces = detector.get_available_interfaces()
    
    print(f"Found {len(interfaces)} interfaces:")
    for i, iface in enumerate(interfaces):
        print(f"  {i+1}. {iface['name']} - {iface['description']}")
        for addr in iface['addresses']:
            print(f"     IP: {addr['ip']}")
    
    return interfaces

def test_packet_capture():
    """Test packet capture functionality"""
    print("Testing packet capture...")
    
    def packet_callback(features, packet):
        print(f"Processed packet: {packet['src_ip']}:{packet['src_port']} -> {packet['dst_ip']}:{packet['dst_port']}")
        print(f"Features shape: {features.shape}")
    
    manager = PacketCaptureManager()
    
    # Get available interfaces
    interfaces = manager.get_available_interfaces()
    print(f"Available interfaces: {[iface['name'] for iface in interfaces]}")
    
    # Start capture
    interface = manager.start_capture(callback=packet_callback)
    print(f"Started capture on: {interface}")
    
    try:
        # Run for 10 seconds
        time.sleep(10)
    except KeyboardInterrupt:
        print("Stopping capture...")
    finally:
        manager.stop_all_captures()
        stats = manager.get_capture_stats()
        print(f"Final stats: {stats}")

if __name__ == "__main__":
    test_packet_capture()
