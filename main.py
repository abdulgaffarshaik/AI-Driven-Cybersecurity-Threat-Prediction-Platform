"""
Main execution script for AI-based Intrusion Detection System
Provides a command-line interface for all system operations
"""

import os
import sys
import logging
from pathlib import Path
import torch
import warnings
warnings.filterwarnings('ignore')

# Add src to path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


warnings.filterwarnings('ignore')

from src.config import print_config, get_device_info
from src.data_preprocessing import run_preprocessing_pipeline
from src.train import train_models_lightning
from src.train_autoencoder import train_autoencoder
from src.evaluate import evaluate_all_models
from src.utils import verify_device_placement, cleanup_gpu_memory, check_gpu_memory

def setup_logging():
    """Setup logging configuration - CONSOLE ONLY"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Only console output - no file logging
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Run the complete pipeline end-to-end every time."""
    try:
        # Setup logging
        logger = setup_logging()

        # Banner
        print("=" * 80)
        print("AI-BASED INTRUSION DETECTION SYSTEM")
        print("=" * 80)
        print(f"Device: {get_device_info()}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
        print("=" * 80)

        logger.info("Running complete AI-based IDS pipeline...")
        logger.info("=" * 60)

        # Step 1: Preprocessing
        logger.info("Step 1/4: Data Preprocessing...")
        run_preprocessing_pipeline()
        logger.info("Data preprocessing completed!")
        cleanup_gpu_memory()

        # Step 2: Train classification models
        logger.info("Step 2/4: Training Classification Models...")
        train_models_lightning()
        logger.info("Classification models training completed!")
        cleanup_gpu_memory()

        # Step 3: Train autoencoder
        logger.info("Step 3/4: Training Autoencoder...")
        train_autoencoder()
        logger.info("Autoencoder training completed!")
        cleanup_gpu_memory()

        # Step 4: Evaluate all models
        logger.info("Step 4/5: Evaluating Models...")
        evaluate_all_models()
        logger.info("Model evaluation completed!")
        cleanup_gpu_memory()

        # Step 5: Setup live detection system
        logger.info("Step 5/5: Setting up Live Detection System...")
        logger.info("Live detection system is ready!")
        logger.info("Run 'python live_detection.py' to start live packet capture and threat detection")
        logger.info("Run 'python live_detection.py --dashboard' to start with web interface")

        logger.info("=" * 60)
        logger.info("Complete AI-based IDS pipeline finished successfully!")
        logger.info("Check the 'results/' folder for outputs and visualizations")
        logger.info("Use 'python live_detection.py' for live network monitoring")
    except KeyboardInterrupt:
        print("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
