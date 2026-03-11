# AI-Driven Cybersecurity Threat Prediction Platform

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-red)
![License](https://img.shields.io/badge/license-MIT-green)

An advanced AI-based Intrusion Detection System (IDS) leveraging deep learning to detect and classify network security threats with industry-leading accuracy.

## 🎯 Overview

This project implements a comprehensive threat detection system using three state-of-the-art neural network architectures:
- **CNN-LSTM Hybrid**: Convolutional-LSTM fusion for temporal pattern recognition (**96.73% accuracy**)
- **Deep Neural Network (DNN)**: Deep feedforward network with batch normalization (**97.39% accuracy**)
- **Autoencoder**: Unsupervised anomaly detection via reconstruction error (**84.14% accuracy**)

The system can detect **15 different attack types** including DoS, DDoS, port scans, web attacks (XSS, SQL injection), brute force attacks, bot activity, and infiltration attempts.

## ✨ Features

- ✅ **Pre-trained models** with 96-97% detection accuracy
- ✅ **Web-based dashboard** with real-time threat visualization
- ✅ **Live packet capture** integration with Wireshark
- ✅ **Multi-model ensemble** for robust predictions
- ✅ **Comprehensive metrics** (Precision, Recall, F1, ROC-AUC, PR-AUC)
- ✅ **GPU/CPU support** (PyTorch with CUDA)
- ✅ **Production-ready** inference pipeline
- ✅ **Detailed evaluation reports** with per-class performance

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| CNN-LSTM | 96.73% | 96.71% | 96.73% | 96.71% | 0.9995 |
| DNN | 97.39% | 97.39% | 97.39% | 97.39% | 0.9998 |
| Autoencoder | 84.14% | 84.12% | 84.14% | 84.13% | 0.9891 |
| Ensemble | 96.95% | 96.93% | 96.95% | 96.94% | 0.9997 |

## 🔍 Supported Attack Types

The system can classify the following network threats:

1. **BENIGN** - Normal traffic
2. **DoS Hulk** - Denial of Service (Hulk variant)
3. **DoS Slowhttptest** - DoS with slow HTTP requests
4. **DoS Slowloris** - DoS using connection slowdown
5. **DoS GoldenEye** - DoS with HTTP GET requests
6. **DDoS** - Distributed Denial of Service
7. **Port Scan** - Network port enumeration attacks
8. **XSS** - Cross-Site Scripting attacks
9. **SQL Injection** - SQL database injection attacks
10. **Brute Force** - Credential guessing attacks
11. **SSH Brute Force** - SSH connection attacks
12. **FTP Brute Force** - FTP login attacks
13. **Bot Activity** - Botnet-controlled traffic
14. **Infiltration** - Network compromise attempts
15. **Heartbleed** - SSL/TLS vulnerability exploitation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (GPU optional but recommended)
- Windows/Linux/macOS

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/abdulgaffarshaik/AI-Driven-Cybersecurity-Threat-Prediction-Platform.git
cd AI-Driven-Cybersecurity-Threat-Prediction-Platform
```

2. **Create virtual environment**
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **For live packet capture (optional)**
   - Install [Wireshark](https://www.wireshark.org/download/)
   - Verify installation: `wireshark --version`

### Usage

#### 1. **Inference with Pre-trained Models**
```bash
python demo_inference.py
```
Tests all three pre-trained models with synthetic network data.

#### 2. **Threat Detection on Synthetic Data**
```bash
python test_threat_detection.py
```
Simulates real-time threat detection with 100% detection rate on sample packets.

#### 3. **Live Detection Dashboard**

**Option A: Using batch file (Windows)**
```bash
start_dashboard.bat
```

**Option B: Direct command**
```bash
python src/live_detection.py --dashboard
```

Then open browser to: `http://localhost:5000`

#### 4. **Training on Custom Data**
```bash
python src/train.py --config config.yaml --dataset your_data.csv
```

## 📁 Project Structure

```
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # This file
├── main.py                     # Entry point
├── demo_inference.py           # Test pre-trained models
├── test_threat_detection.py    # Synthetic threat simulation
├── live_detection.py           # Live dashboard and detection
│
├── src/
│   ├── config.py              # Configuration module
│   ├── models.py              # Neural network architectures
│   ├── utils.py               # Utility functions (metrics, logging, GPU)
│   ├── train.py               # Training pipeline
│   ├── train_autoencoder.py   # Autoencoder-specific training
│   ├── inference.py           # Inference module
│   ├── evaluate.py            # Model evaluation
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── feature_engineering.py # Network flow feature extraction
│   ├── lightning_module.py    # PyTorch Lightning wrapper
│   ├── dashboard.py           # Web dashboard backend
│   ├── live_capture.py        # Packet capture integration
│   └── __pycache__/           # Compiled Python files
│
├── models/
│   ├── cnn_lstm/
│   │   └── best_model.pt      # CNN-LSTM pre-trained weights
│   ├── dnn/
│   │   └── best_model.pt      # DNN pre-trained weights
│   └── autoencoder/
│       ├── best_model.pt      # Autoencoder pre-trained weights
│       └── threshold.json     # Anomaly detection threshold
│
├── notebooks/
│   ├── CIC-IDS-2017_Analysis.ipynb
│   └── Streamlined_EDA_Analysis.ipynb
│
├── templates/
│   ├── index.html             # Dashboard UI
│   └── dashboard.html         # Real-time monitoring interface
│
├── results/
│   ├── autoencoder_metrics.json
│   ├── cnn_lstm_metrics.json
│   ├── dnn_metrics.json
│   ├── ensemble_metrics.json
│   ├── evaluation_results.json
│   └── evaluation_report.txt
│
├── plots/
│   ├── confusion_matrix_*.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   └── model_comparison.png
│
└── Wireshark/                 # Wireshark installation (optional)
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Model parameters
CNN_LSTM_CONFIG:
  input_features: 66
  lstm_hidden_size: 128
  num_layers: 2
  
DNN_CONFIG:
  input_features: 66
  hidden_sizes: [256, 128, 64]
  dropout: 0.3

# Training parameters
TRAINING_CONFIG:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  patience: 10
```

## 📊 Sample Output

### Model Comparison
```
CNN-LSTM:  96.73% accuracy (96.71% precision, 96.73% recall)
DNN:       97.39% accuracy (97.39% precision, 97.39% recall) ✓ Best
Autoencoder: 84.14% accuracy (anomaly detection)
```

### Threat Detection
```
Packet 1: DoS Slowloris (Confidence: 98.7%)
Packet 2: DDoS Attack (Confidence: 97.2%)
Packet 3: BENIGN (Confidence: 99.1%)
Packet 4: Bot Activity (Confidence: 96.5%)
...
Detection Rate: 100% | False Alarm Rate: 0.0%
```

## 📈 Dataset Compatibility

- **Primary**: CIC-IDS-2017 (66 network flow features)
- **Format**: CSV with headers
- **Sample Size**: Compatible with 50K - 2M+ records
- **Labels**: 15 attack types + BENIGN

## 🌐 Web Dashboard

Access the real-time monitoring interface:

1. **Display**: Live threat detection list
2. **Charts**: Attack type distribution and metrics
3. **Metrics**: Detection rate, false alarm rate, model confidence
4. **Filtering**: By attack type, time window, confidence threshold

## 🔐 Security Features

- ✅ Input validation and sanitization
- ✅ Feature normalization to prevent adversarial inputs
- ✅ Robust error handling
- ✅ GPU memory management
- ✅ Production-ready inference

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{ids_threat_prediction_2025,
  author = {Vidzai Digital},
  title = {AI-Driven Cybersecurity Threat Prediction Platform},
  year = {2025},
  url = {https://github.com/abdulgaffarshaik/AI-Driven-Cybersecurity-Threat-Prediction-Platform}
}
```

## 📖 References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSP.
2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
3. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. NeurIPS.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚖️ License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Vidzai Digital

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub Issues](https://github.com/abdulgaffarshaik/AI-Driven-Cybersecurity-Threat-Prediction-Platform/issues)
- Contact: abdulgaffarshaik@gmail.com

## 🎓 Disclaimer

This project is for educational and research purposes. Use responsibly and legally. Always obtain proper authorization before testing on networks you don't own.

---

**⭐ Star this repository if you find it useful!**
