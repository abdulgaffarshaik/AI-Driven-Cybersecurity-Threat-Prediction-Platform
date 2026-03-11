"""
Lightning-based training script for IDS models
Matches the pneumonia project style exactly
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import gc
from rich.console import Console
from loguru import logger
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils import (
    setup_logging, create_data_loaders, save_model, 
    MetricsCalculator, get_class_weights, count_parameters,
    cleanup_gpu_memory, check_gpu_memory
)
from data_preprocessing import load_processed_data, create_cv_splits
from lightning_module import IDSLightningModule, create_trainer, train_model, test_model


def check_gpu():
    """Check GPU availability and print info - matches pneumonia project"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_compute_capability = torch.cuda.get_device_capability(0)
        
        logger.info(f"GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"Compute Capability: {gpu_compute_capability[0]}.{gpu_compute_capability[1]}")
        
        # Check if Tensor Cores are available and can be used
        tensor_cores_available = gpu_compute_capability[0] >= 7
        if tensor_cores_available:
            try:
                torch.set_float32_matmul_precision('high')
                if torch.get_float32_matmul_precision() == 'high':
                    logger.info("✅ Tensor Cores enabled and configured for optimal performance")
                else:
                    logger.info("⚠️ Tensor Cores available but not configured (PyTorch version limitation)")
            except Exception as e:
                logger.info(f"⚠️ Tensor Cores available but configuration failed: {e}")
        else:
            logger.info("⚠️ Tensor Cores not available (requires compute capability 7.0+)")
        
        # Check current device and verify CUDA is working
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device}")
        
        # Test GPU computation to verify it's working
        try:
            test_tensor = torch.randn(10, 10).cuda()
            test_result = torch.mm(test_tensor, test_tensor)
            del test_tensor, test_result
            torch.cuda.empty_cache()
            logger.info("✅ GPU computation test successful")
        except Exception as e:
            logger.warning(f"⚠️ GPU computation test failed: {e}")
        
        return True
    else:
        logger.warning("❌ No GPU available, using CPU")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "models", 
        "results",
        "plots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")


def verify_device_placement(model, expected_device='cuda'):
    """Verify that a model is actually on the expected device"""
    try:
        device = next(model.parameters()).device
        if expected_device == 'cuda' and device.type == 'cuda':
            return True, f"GPU:{device.index}"
        elif expected_device == 'cpu' and device.type == 'cpu':
            return True, "CPU"
        else:
            return False, f"Unexpected: {device}"
    except Exception as e:
        return False, f"Error: {e}"


def train_kfold_models(model_type: str, cv_splits: List, class_weights: Dict) -> List[IDSLightningModule]:
    """
    Train models using cross-validation - matches pneumonia project style
    
    Args:
        model_type: Type of model ('cnn_lstm' or 'dnn')
        cv_splits: Cross-validation splits
        class_weights: Class weights for imbalanced dataset
    
    Returns:
        List of trained models (one per fold)
    """
    trained_models = []
    
    for fold_data in cv_splits:
        fold_name = f"Fold {fold_data['fold']}"
        logger.info(f"Training {fold_name}...")
        
        # Create model for this fold
        model = IDSLightningModule(
            model_type=model_type,
            model_config=CNN_LSTM_CONFIG if model_type == 'cnn_lstm' else DNN_CONFIG,
            training_config=TRAINING_CONFIG,
            num_classes=NUM_CLASSES,
            learning_rate=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # Create trainer
        trainer = create_trainer(
            model_type=model_type,
            training_config=TRAINING_CONFIG,
            hardware_config={'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu', 'devices': 1},
            logging_config={},
            callbacks=[]
        )
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            fold_data['X_train'], fold_data['y_train'], 
            fold_data['X_val'], fold_data['y_val'],
            batch_size=TRAINING_CONFIG['batch_size'],
            num_workers=DATALOADER_CONFIG['num_workers']
        )
        
        # Train model
        try:
            # Log device before training
            if torch.cuda.is_available():
                logger.info(f"🚀 {fold_name} training starting on GPU: {torch.cuda.get_device_name()}")
            else:
                logger.info(f"🖥️ {fold_name} training starting on CPU")
                
            trainer.fit(model, train_loader, val_loader)
            
            # Log device after training
            if torch.cuda.is_available():
                logger.info(f"✅ {fold_name} training completed on GPU: {torch.cuda.get_device_name()}")
            else:
                logger.info(f"✅ {fold_name} training completed on CPU")
                
            # Load best checkpoint weights and capture best validation accuracy for this fold
            best_path = None
            best_score = None
            try:
                for cb in getattr(trainer, 'checkpoint_callbacks', []):
                    if hasattr(cb, 'best_model_path') and cb.best_model_path:
                        best_path = cb.best_model_path
                    if hasattr(cb, 'best_model_score') and cb.best_model_score is not None:
                        try:
                            best_score = float(cb.best_model_score)
                        except Exception:
                            try:
                                best_score = cb.best_model_score.item()
                            except Exception:
                                best_score = None
                if best_path:
                    best_model = IDSLightningModule.load_from_checkpoint(best_path)
                    model = best_model
                    logger.info(f"Loaded best checkpoint for {fold_name}: {best_path}")
                    # If score missing, try parse from filename ...val_accuracy=0.606.ckpt
                    if best_score is None:
                        try:
                            import re
                            m = re.search(r"val_accuracy=([0-9]*\.[0-9]+)", best_path)
                            if m:
                                best_score = float(m.group(1))
                        except Exception:
                            best_score = None
                # Attach for later selection among folds
                try:
                    setattr(model, 'best_val_accuracy', best_score if best_score is not None else 0.0)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Could not load best checkpoint for {fold_name}: {e}")

            # Move model to CPU to free GPU memory between folds
            try:
                model.cpu()
            except Exception:
                pass
            trained_models.append(model)
            logger.info(f"{fold_name} training completed successfully")
        except Exception as e:
            logger.error(f"Training failed for {fold_name}: {e}")
            raise e
        finally:
            # Enhanced cleanup to prevent handshake timeout issues
            try:
                # Properly teardown distributed components if present
                if hasattr(trainer, 'strategy') and hasattr(trainer.strategy, 'teardown'):
                    trainer.strategy.teardown()
                
                # Clear any remaining processes
                if hasattr(trainer, 'accelerator') and hasattr(trainer.accelerator, 'teardown'):
                    trainer.accelerator.teardown()
                    
                # Delete trainer object
                del trainer
            except Exception as e:
                logger.warning(f"Trainer cleanup warning: {e}")
                pass
            
            # GPU memory cleanup
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"CUDA cleanup warning: {e}")
                    pass
            
            # Force garbage collection
            gc.collect()
            
            # Small delay to ensure proper cleanup between folds
            time.sleep(0.5)
    
    return trained_models


def evaluate_ensemble_on_test(trained_models: List[IDSLightningModule], test_loader, model_type: str) -> Dict:
    """
    Evaluate ensemble of trained models on test set - matches pneumonia project style
    
    Args:
        trained_models: List of trained models
        test_loader: Test data loader
        model_type: Type of model
    
    Returns:
        Test results dictionary
    """
    logger.info("Evaluating ensemble on test set...")
    
    # Get predictions from all models
    all_predictions = []
    all_probabilities = []
    
    # Move ALL models to GPU once at the start
    logger.info("Moving all models to GPU for batch prediction...")
    try:
        if torch.cuda.is_available():
            for i, model in enumerate(trained_models):
                model = model.to('cuda')
                is_on_gpu, device_info = verify_device_placement(model, 'cuda')
                if is_on_gpu:
                    logger.info(f"✅ {model_type.upper()} Fold {i+1} moved to {device_info}")
                else:
                    logger.warning(f"⚠️ {model_type.upper()} Fold {i+1} device issue: {device_info}")
            logger.info(f"✅ All models moved to GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("🖥️ No GPU available, models staying on CPU")
    except Exception as e:
        logger.warning(f"Failed to move models to GPU: {e}")
    
    # Get predictions from all models while they're on GPU
    for i, model in enumerate(trained_models):
        current_device = next(model.parameters()).device
        device_name = "GPU" if current_device.type == 'cuda' else "CPU"
        logger.info(f"Getting predictions from {model_type.upper()} Fold {i+1}/{len(trained_models)} on {device_name}")
        
        try:
            predictions, targets, probabilities = model.get_predictions(test_loader)
        except Exception as e:
            logger.warning(f"Failed to get predictions from {model_type.upper()} Fold {i+1}: {e}")
            continue
            
        if i == 0:
            all_targets = targets
        all_predictions.append(predictions)
        all_probabilities.append(probabilities)
    
    # Move ALL models back to CPU once at the end
    logger.info("Moving all models back to CPU...")
    try:
        for i, model in enumerate(trained_models):
            model = model.to('cpu')
            is_on_cpu, device_info = verify_device_placement(model, 'cpu')
            if is_on_cpu:
                logger.info(f"✅ {model_type.upper()} Fold {i+1} moved to {device_info}")
            else:
                logger.warning(f"⚠️ {model_type.upper()} Fold {i+1} device issue: {device_info}")
        logger.info("✅ All models moved back to CPU")
    except Exception as e:
        logger.warning(f"Failed to move models to CPU: {e}")
    
    # Clean up GPU memory once after all predictions
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("🧹 GPU memory cleaned up")
        except Exception as e:
            logger.warning(f"GPU cleanup warning: {e}")
    else:
        logger.info("🖥️ No GPU cleanup needed (CPU only)")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    # Ensemble across folds/models with weights derived from each model's performance
    try:
        per_model_weights = []
        for i in range(len(trained_models)):
            fold_metrics_calc = MetricsCalculator(num_classes=NUM_CLASSES)
            fold_metrics_calc.update(
                torch.tensor(all_predictions[i]),
                torch.tensor(all_targets),
                torch.tensor(all_probabilities[i])
            )
            m = fold_metrics_calc.compute()
            # Use raw metrics for consistency
            w = 0.0
            for k in ['precision', 'recall', 'f1', 'auc_roc']:
                if k in m:
                    w += m[k]
            per_model_weights.append(w)
        per_model_weights = np.array(per_model_weights)
        if np.all(per_model_weights == 0) or not np.isfinite(per_model_weights).all():
            per_model_weights = np.ones(len(trained_models))
        
        # Normalize fold weights to sum to 1
        per_model_weights = np.array(per_model_weights)
        if np.sum(per_model_weights) > 0:
            per_model_weights = per_model_weights / np.sum(per_model_weights)
        else:
            per_model_weights = np.ones(len(trained_models)) / len(trained_models)
        
        # Weighted average over models
        ensemble_probabilities = np.sum(all_probabilities * per_model_weights[:, None, None], axis=0)
        logger.info(f"Fold model weights used for ensembling: {per_model_weights.tolist()}")
        model_level_weights = per_model_weights.tolist()
    except Exception as e:
        logger.warning(f"Model-level weighting failed, falling back to mean: {e}")
        ensemble_probabilities = np.mean(all_probabilities, axis=0)
        model_level_weights = [1.0 / len(trained_models)] * len(trained_models)
    ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
    
    # Calculate ensemble metrics
    ensemble_metrics_calc = MetricsCalculator(num_classes=NUM_CLASSES)
    ensemble_metrics_calc.update(
        torch.tensor(ensemble_predictions),
        torch.tensor(all_targets),
        torch.tensor(ensemble_probabilities)
    )
    ensemble_metrics = ensemble_metrics_calc.compute()
    
    # Store results
    results = {
        'ensemble_metrics': ensemble_metrics,
        'predictions': ensemble_predictions.tolist(),
        'targets': all_targets.tolist(),
        'probabilities': ensemble_probabilities.tolist(),
        'fold_model_weights': model_level_weights,
        'model_type': model_type
    }
    
    return results


def train_models_lightning():
    """Main training function with Lightning - matches pneumonia project style"""
    console = Console()
    
    # Setup - EXACT SAME AS PNEUMONIA PROJECT
    setup_logging()
    create_directories()
    
    console.print("[bold blue]AI-Based Intrusion Detection System[/bold blue]")
    console.print("=" * 60)
    console.print("[bold green]Features:[/bold green]")
    console.print(f"• {CV_CONFIG['n_folds']}-Fold Cross-Validation")
    console.print("• CNN-LSTM + DNN + Autoencoder + Ensemble")
    console.print("• GPU Acceleration (RTX 3050)")
    console.print("• Automatic Mixed Precision")
    console.print("• Focal Loss for Class Imbalance")
    console.print("=" * 60)
    
    # Check GPU
    use_gpu = check_gpu()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    logger.info("Starting model training...")
    
    try:
        # Load processed data
        logger.info("Loading processed data...")
        X_train, X_test, y_train, y_test, scaler, label_encoder, label_mapping = load_processed_data()
        
        # Create cross-validation splits
        cv_splits = create_cv_splits(X_train, y_train)
        
        # Get class weights
        class_weights = get_class_weights(y_train)
        
        # Train CNN-LSTM model
        console.print("\n[bold green]Training CNN-LSTM model...[/bold green]")
        logger.info("Training CNN-LSTM model...")
        
        cnn_lstm_models = train_kfold_models('cnn_lstm', cv_splits, class_weights)
        
        # Save best CNN-LSTM model (use captured best_val_accuracy)
        best_cnn_lstm_model = max(cnn_lstm_models, key=lambda x: getattr(x, 'best_val_accuracy', 0.0))
        cnn_lstm_model_path = MODELS_PATH / "cnn_lstm" / "best_model.pt"
        cnn_lstm_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': best_cnn_lstm_model.state_dict(),
            'model_config': CNN_LSTM_CONFIG,
            'training_config': TRAINING_CONFIG,
            'num_classes': NUM_CLASSES
        }, cnn_lstm_model_path)
        
        logger.info(f"CNN-LSTM model saved to {cnn_lstm_model_path}")
        
        # Train DNN model
        console.print("\n[bold green]Training DNN model...[/bold green]")
        logger.info("Training DNN model...")
        
        dnn_models = train_kfold_models('dnn', cv_splits, class_weights)
        
        # Save best DNN model (use captured best_val_accuracy)
        best_dnn_model = max(dnn_models, key=lambda x: getattr(x, 'best_val_accuracy', 0.0))
        dnn_model_path = MODELS_PATH / "dnn" / "best_model.pt"
        dnn_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': best_dnn_model.state_dict(),
            'model_config': DNN_CONFIG,
            'training_config': TRAINING_CONFIG,
            'num_classes': NUM_CLASSES
        }, dnn_model_path)
        
        logger.info(f"DNN model saved to {dnn_model_path}")
        
        # Evaluate models on test set
        logger.info("Evaluating models on test set...")
        
        # Create test loader
        test_loader, _ = create_data_loaders(
            X_test, y_test, X_test, y_test,  # Dummy validation data
            batch_size=TRAINING_CONFIG['batch_size'],
            num_workers=DATALOADER_CONFIG['num_workers']
        )
        
        # Evaluate CNN-LSTM ensemble
        cnn_lstm_results = evaluate_ensemble_on_test(cnn_lstm_models, test_loader, 'cnn_lstm')
        
        # Evaluate DNN ensemble
        dnn_results = evaluate_ensemble_on_test(dnn_models, test_loader, 'dnn')
        
        # Print final results
        console.print("\n[bold green]FINAL RESULTS[/bold green]")
        console.print("=" * 50)
        console.print("CNN-LSTM Results:")
        for metric, value in cnn_lstm_results['ensemble_metrics'].items():
            if isinstance(value, (int, float)):
                console.print(f"  {metric.upper()}: {value:.4f}")
            else:
                console.print(f"  {metric.upper()}: {value}")
        console.print("\n" + "=" * 50)
        console.print("CNN-LSTM TRAINING COMPLETE")
        console.print("=" * 50)
        
        console.print("\nDNN Results:")
        for metric, value in dnn_results['ensemble_metrics'].items():
            if isinstance(value, (int, float)):
                console.print(f"  {metric.upper()}: {value:.4f}")
            else:
                console.print(f"  {metric.upper()}: {value}")
        console.print("\n" + "=" * 50)
        console.print("DNN TRAINING COMPLETE")
        console.print("=" * 50)
        
        console.print("\n[bold green]ALL TRAINING COMPLETED SUCCESSFULLY![/bold green]")
        console.print("Check the results/ directory for detailed outputs.")
        
        # Print device usage summary
        console.print("\n[bold blue]DEVICE USAGE SUMMARY[/bold blue]")
        console.print("=" * 40)
        if use_gpu:
            console.print("🚀 GPU Training: All k-fold training runs on GPU")
            console.print("🚀 GPU Evaluation: All models moved to GPU for batch prediction")
            console.print("🖥️ CPU Storage: Models stored on CPU between operations")
            console.print("🧹 GPU Memory: Cleaned after each fold and final evaluation")
            console.print("ℹ️ Note: PyTorch Lightning automatically manages device placement")
        else:
            console.print("🖥️ CPU Training: All operations run on CPU")
            console.print("⚠️ Performance: CPU training will be significantly slower")
        console.print("=" * 40)
        
        return {
            'cnn_lstm_results': cnn_lstm_results,
            'dnn_results': dnn_results,
            'class_weights': class_weights,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise


if __name__ == "__main__":
    train_models_lightning()
