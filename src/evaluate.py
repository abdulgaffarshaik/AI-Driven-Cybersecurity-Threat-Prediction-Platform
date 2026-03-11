"""
Comprehensive evaluation script for all trained models
Generates detailed metrics, plots, and performance analysis
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from utils import (
    setup_logging, MetricsCalculator, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve, save_metrics
)
from models import ModelFactory
from data_preprocessing import load_processed_data

# =============================================================================
# MODEL EVALUATION CLASS
# =============================================================================
class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_calc = MetricsCalculator()
        
        # Load test data
        self.X_train, self.X_test, self.y_train, self.y_test, self.scaler, self.label_encoder, self.label_mapping = load_processed_data()
        
        # Get class names
        self.class_names = [ID_TO_ATTACK[i] for i in range(len(ATTACK_TYPES))]
        
        self.logger.info(f"Model evaluator initialized")
        self.logger.info(f"Test data shape: {self.X_test.shape}")
        self.logger.info(f"Number of classes: {len(self.class_names)}")
        self.logger.info(f"Class names: {self.class_names}")
    
    def load_model(self, model_type, model_path):
        """Load trained model"""
        try:
            if model_type == 'autoencoder':
                model = ModelFactory.create_model('autoencoder', device=self.device)
            elif model_type == 'cnn_lstm':
                model = ModelFactory.create_model('cnn_lstm', device=self.device)
            elif model_type == 'dnn':
                model = ModelFactory.create_model('dnn', device=self.device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load checkpoint with proper handling
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.logger.info(f"Loaded checkpoint with 'model_state_dict' key")
            else:
                state_dict = checkpoint
                self.logger.info(f"Loaded checkpoint without 'model_state_dict' key")
            
            # Extract model state dict if it's wrapped in a Lightning module
            first_key = list(state_dict.keys())[0]
            self.logger.info(f"First key in state_dict: {first_key}")
            
            # Check if any key has 'model.' prefix
            has_model_prefix = any('model.' in key for key in state_dict.keys())
            
            if has_model_prefix:
                self.logger.info("Removing 'model.' prefix from state dict keys")
                # Remove 'model.' prefix from all keys
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # Remove 'model.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            # Remove class_weights if present (not part of model state)
            if 'class_weights' in state_dict:
                del state_dict['class_weights']
            
            model.load_state_dict(state_dict)
            model.eval()
            
            self.logger.info(f"Loaded {model_type} model from {model_path}")
            return model, checkpoint
            
        except Exception as e:
            self.logger.error(f"Error loading {model_type} model: {str(e)}")
            return None, None
    
    def evaluate_classification_model(self, model, model_name):
        """Evaluate classification model (CNN-LSTM or DNN)"""
        self.logger.info(f"Evaluating {model_name}...")
        
        # Process in batches to avoid GPU memory issues
        batch_size = 1024  # Process 1K samples at a time for 4GB GPU
        n_samples = len(self.X_test)
        predictions = []
        probabilities = []
        
        self.logger.info(f"Processing {n_samples} samples in batches of {batch_size}")
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_data = torch.FloatTensor(self.X_test[i:end_idx]).to(self.device)
                
                # Get predictions for this batch
                outputs = model(batch_data)
                batch_predictions = outputs.argmax(dim=1).cpu().numpy()
                batch_probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                
                predictions.extend(batch_predictions)
                probabilities.extend(batch_probabilities)
                
                # Clear GPU memory aggressively
                del batch_data, outputs, batch_predictions, batch_probabilities
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                if (i // batch_size + 1) % 10 == 0:
                    self.logger.info(f"Processed {end_idx}/{n_samples} samples")
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_classification_metrics(
            self.y_test, predictions, probabilities
        )
        
        # Add model-specific information
        metrics['model_name'] = model_name
        metrics['model_type'] = 'classification'
        metrics['test_samples'] = len(self.y_test)
        metrics['timestamp'] = datetime.now().isoformat()
        
        self.logger.info(f"{model_name} evaluation completed")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        self.logger.info(f"F1-Score (Micro): {metrics['f1_micro']:.4f}")
        self.logger.info(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
        
        return metrics, predictions, probabilities
    
    def evaluate_autoencoder(self, model, threshold):
        """Evaluate autoencoder for anomaly detection"""
        self.logger.info("Evaluating autoencoder...")
        
        # Process in batches to avoid GPU memory issues
        batch_size = 1024  # Process 1K samples at a time for 4GB GPU
        n_samples = len(self.X_test)
        reconstruction_errors = []
        
        self.logger.info(f"Processing {n_samples} samples in batches of {batch_size}")
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_data = torch.FloatTensor(self.X_test[i:end_idx]).to(self.device)
                
                # Get reconstruction errors for this batch
                reconstructed = model(batch_data)
                batch_errors = torch.mean((reconstructed - batch_data) ** 2, dim=1).cpu().numpy()
                reconstruction_errors.extend(batch_errors)
                
                # Clear GPU memory aggressively
                del batch_data, reconstructed, batch_errors
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                if (i // batch_size + 1) % 10 == 0:
                    self.logger.info(f"Processed {end_idx}/{n_samples} samples")
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Create binary labels (0 for BENIGN, 1 for attacks)
        binary_labels = (self.y_test != 0).astype(int)
        
        # Predict anomalies
        predictions = (reconstruction_errors >= threshold).astype(int)
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_anomaly_metrics(
            binary_labels, reconstruction_errors, threshold
        )
        
        # Add model-specific information
        metrics['model_name'] = 'Autoencoder'
        metrics['model_type'] = 'anomaly_detection'
        metrics['test_samples'] = len(self.y_test)
        metrics['threshold'] = threshold
        metrics['timestamp'] = datetime.now().isoformat()
        
        self.logger.info("Autoencoder evaluation completed")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        self.logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics, predictions, reconstruction_errors
    
    def create_ensemble_predictions(self, cnn_lstm_probs, dnn_probs):
        """Create ensemble predictions"""
        self.logger.info("Creating ensemble predictions...")
        
        # Convert to numpy arrays if they're lists
        if isinstance(cnn_lstm_probs, list):
            cnn_lstm_probs = np.array(cnn_lstm_probs)
        if isinstance(dnn_probs, list):
            dnn_probs = np.array(dnn_probs)
        
        # Weighted average ensemble
        ensemble_probs = 0.5 * cnn_lstm_probs + 0.5 * dnn_probs
        ensemble_predictions = ensemble_probs.argmax(axis=1)
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_classification_metrics(
            self.y_test, ensemble_predictions, ensemble_probs
        )
        
        # Add model-specific information
        metrics['model_name'] = 'Ensemble (CNN-LSTM + DNN)'
        metrics['model_type'] = 'ensemble'
        metrics['test_samples'] = len(self.y_test)
        metrics['timestamp'] = datetime.now().isoformat()
        
        self.logger.info("Ensemble evaluation completed")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        
        return metrics, ensemble_predictions, ensemble_probs
    
    def generate_plots(self, results):
        """Generate comprehensive evaluation plots"""
        self.logger.info("Generating evaluation plots...")
        
        # Create plots directory in root
        root_plots_path = Path("plots")
        root_plots_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Model comparison
        self._plot_model_comparison(results)
        
        # Plot 2: Confusion matrices
        self._plot_confusion_matrices(results)
        
        # Plot 3: ROC curves
        self._plot_roc_curves(results)
        
        # Plot 4: Precision-Recall curves
        self._plot_pr_curves(results)
        
        # Plot 5: Per-class performance
        self._plot_per_class_performance(results)
        
        self.logger.info(f"Plots saved to plots/")
        
        # Save model metrics
        self._save_model_metrics(results)
    
    def _plot_model_comparison(self, results):
        """Plot model performance comparison"""
        models = []
        accuracies = []
        f1_scores = []
        
        for model_name, result in results.items():
            if result['model_type'] in ['classification', 'ensemble']:
                models.append(model_name)
                accuracies.append(result['accuracy'])
                f1_scores.append(result['f1_macro'])
        
        # Check if we have data to plot
        if not models:
            self.logger.warning("No classification models found for comparison plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(models, accuracies, color=['blue', 'green', 'orange'])
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # F1-Score comparison
        bars2 = ax2.bar(models, f1_scores, color=['blue', 'green', 'orange'])
        ax2.set_title('Model F1-Score (Macro) Comparison')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(Path("plots") / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to avoid display
    
    def _plot_confusion_matrices(self, results):
        """Plot separate confusion matrices for each model"""
        classification_models = {k: v for k, v in results.items() 
                               if v['model_type'] in ['classification', 'ensemble']}
        
        if not classification_models:
            return
        
        # Create separate confusion matrix for each model
        for model_name, result in classification_models.items():
            cm = result['confusion_matrix']
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title(f'{model_name} Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save individual confusion matrix
            filename = f'confusion_matrix_{model_name.lower().replace(" ", "_").replace("-", "_")}.png'
            plt.savefig(Path("plots") / filename, dpi=300, bbox_inches='tight')
            plt.close()  # Close instead of show to avoid display
    
    def _plot_roc_curves(self, results):
        """Plot ROC curves for classification models"""
        classification_models = {k: v for k, v in results.items() 
                               if v['model_type'] in ['classification', 'ensemble']}
        
        if not classification_models:
            return
        
        plt.figure(figsize=(12, 8))
        
        for model_name, result in classification_models.items():
            if 'probabilities' in result:
                y_prob = result['probabilities']
                y_true = self.y_test
                
                # Calculate macro-average ROC
                y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
                fpr, tpr, _ = roc_curve(y_true_bin.ravel(), np.array(y_prob).ravel())
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(Path("plots") / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to avoid display
    
    def _plot_pr_curves(self, results):
        """Plot Precision-Recall curves for classification models"""
        classification_models = {k: v for k, v in results.items() 
                               if v['model_type'] in ['classification', 'ensemble']}
        
        if not classification_models:
            return
        
        plt.figure(figsize=(12, 8))
        
        for model_name, result in classification_models.items():
            if 'probabilities' in result:
                y_prob = result['probabilities']
                y_true = self.y_test
                
                # Calculate macro-average PR
                y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
                precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), np.array(y_prob).ravel())
                pr_auc = average_precision_score(y_true_bin.ravel(), np.array(y_prob).ravel())
                
                plt.plot(recall, precision, label=f'{model_name} (AP = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(Path("plots") / 'pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to avoid display
    
    def _plot_per_class_performance(self, results):
        """Plot per-class performance metrics"""
        classification_models = {k: v for k, v in results.items() 
                               if v['model_type'] in ['classification', 'ensemble']}
        
        if not classification_models:
            return
        
        # Extract per-class metrics
        metrics_data = []
        for model_name, result in classification_models.items():
            for class_name, f1 in result['f1_per_class'].items():
                metrics_data.append({
                    'Model': model_name,
                    'Class': class_name,
                    'F1-Score': f1
                })
        
        df = pd.DataFrame(metrics_data)
        
        # Create pivot table for heatmap
        pivot_df = df.pivot(index='Class', columns='Model', values='F1-Score')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'F1-Score'})
        plt.title('Per-Class F1-Score Heatmap')
        plt.xlabel('Model')
        plt.ylabel('Attack Class')
        plt.tick_params(axis='x', rotation=45)
        plt.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig(Path("plots") / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to avoid display
    
    def _save_model_metrics(self, results):
        """Save model performance metrics to results folder"""
        self.logger.info("Saving model metrics...")
        
        # Create metrics summary
        metrics_summary = {}
        for model_name, result in results.items():
            if result['model_type'] in ['classification', 'ensemble']:
                metrics_summary[model_name] = {
                    'accuracy': result['accuracy'],
                    'precision_macro': result['precision_macro'],
                    'recall_macro': result['recall_macro'],
                    'f1_macro': result['f1_macro'],
                    'f1_micro': result['f1_micro'],
                    'roc_auc_macro': result['roc_auc_macro'],
                    'roc_auc_micro': result['roc_auc_micro'],
                    'precision_weighted': result['precision_weighted'],
                    'recall_weighted': result['recall_weighted'],
                    'f1_weighted': result['f1_weighted'],
                    'detection_rate': result['detection_rate'],
                    'false_alarm_rate': result['false_alarm_rate']
                }
            elif result['model_type'] == 'anomaly_detection':
                metrics_summary[model_name] = {
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                    'roc_auc': result['roc_auc'],
                    'pr_auc': result['pr_auc']
                }
        
        # Save to results folder
        from utils import safe_json_dump
        safe_json_dump(metrics_summary, Path("results") / "model_metrics.json")
        self.logger.info("Model metrics saved to results/model_metrics.json")
        
        # Also save individual model results (without large arrays)
        for model_name, result in results.items():
            # Create a simplified version without large arrays
            simplified_result = {}
            for key, value in result.items():
                if key in ['predictions', 'probabilities', 'reconstruction_errors']:
                    # Skip large arrays for individual files
                    continue
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    simplified_result[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (list, np.ndarray)) and len(sub_value) > 100:
                            # Skip large arrays
                            continue
                        simplified_result[key][sub_key] = sub_value
                else:
                    simplified_result[key] = value
            
            filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_metrics.json"
            safe_json_dump(simplified_result, Path("results") / filename)
            self.logger.info(f"{model_name} metrics saved to results/{filename}")
    
    def generate_detailed_report(self, results):
        """Generate detailed evaluation report"""
        self.logger.info("Generating detailed evaluation report...")
        
        # Convert numpy arrays to Python types for JSON serialization
        unique_classes, counts = np.unique(self.y_test, return_counts=True)
        class_distribution = {int(k): int(v) for k, v in zip(unique_classes, counts)}
        
        report = {
            'evaluation_summary': {
                'timestamp': datetime.now().isoformat(),
                'test_samples': len(self.y_test),
                'num_classes': len(self.class_names),
                'class_distribution': class_distribution
            },
            'model_results': results,
            'recommendations': self._generate_recommendations(results)
        }
        
        # Save detailed report
        report_path = RESULTS_PATH / "detailed_evaluation_report.json"
        from utils import safe_json_dump
        safe_json_dump(report, report_path)
        
        self.logger.info(f"Detailed report saved to {report_path}")
        
        # Generate text report
        self._generate_text_report(report)
        
        return report
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Find best performing model
        classification_models = {k: v for k, v in results.items() 
                               if v['model_type'] in ['classification', 'ensemble']}
        
        if classification_models:
            best_model = max(classification_models.items(), key=lambda x: x[1]['f1_macro'])
            recommendations.append(f"Best performing model: {best_model[0]} (F1-Score: {best_model[1]['f1_macro']:.4f})")
        
        # Check for class imbalance issues
        for model_name, result in classification_models.items():
            if 'f1_per_class' in result:
                f1_scores = list(result['f1_per_class'].values())
                min_f1 = min(f1_scores)
                if min_f1 < 0.5:
                    recommendations.append(f"{model_name}: Poor performance on some classes (min F1: {min_f1:.3f}). Consider class balancing.")
        
        # Check for overfitting
        for model_name, result in classification_models.items():
            if result['accuracy'] > 0.99:
                recommendations.append(f"{model_name}: Very high accuracy ({result['accuracy']:.3f}) - possible overfitting.")
        
        return recommendations
    
    def _generate_text_report(self, report):
        """Generate human-readable text report"""
        report_path = RESULTS_PATH / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AI-BASED INTRUSION DETECTION SYSTEM - EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation Date: {report['evaluation_summary']['timestamp']}\n")
            f.write(f"Test Samples: {report['evaluation_summary']['test_samples']:,}\n")
            f.write(f"Number of Classes: {report['evaluation_summary']['num_classes']}\n\n")
            
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            for class_id, count in report['evaluation_summary']['class_distribution'].items():
                class_name = ID_TO_ATTACK.get(class_id, f"Class_{class_id}")
                percentage = (count / report['evaluation_summary']['test_samples']) * 100
                f.write(f"{class_name}: {count:,} ({percentage:.2f}%)\n")
            f.write("\n")
            
            f.write("MODEL PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            for model_name, result in report['model_results'].items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Type: {result['model_type']}\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                
                if result['model_type'] in ['classification', 'ensemble']:
                    f.write(f"  Precision (Macro): {result['precision_macro']:.4f}\n")
                    f.write(f"  Recall (Macro): {result['recall_macro']:.4f}\n")
                    f.write(f"  F1-Score (Macro): {result['f1_macro']:.4f}\n")
                    f.write(f"  ROC-AUC (Macro): {result['roc_auc_macro']:.4f}\n")
                elif result['model_type'] == 'anomaly_detection':
                    f.write(f"  Precision: {result['precision']:.4f}\n")
                    f.write(f"  Recall: {result['recall']:.4f}\n")
                    f.write(f"  F1-Score: {result['f1_score']:.4f}\n")
                    f.write(f"  ROC-AUC: {result['roc_auc']:.4f}\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Text report saved to {report_path}")

def evaluate_all_models():
    """Main evaluation function"""
    logger = setup_logging()
    logger.info("Starting comprehensive model evaluation...")
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()
        results = {}
        
        # Evaluate CNN-LSTM model
        cnn_lstm_path = MODELS_PATH / "cnn_lstm" / "best_model.pt"
        if cnn_lstm_path.exists():
            model, checkpoint = evaluator.load_model('cnn_lstm', cnn_lstm_path)
            if model is not None:
                # Clear GPU memory before evaluation
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                metrics, predictions, probabilities = evaluator.evaluate_classification_model(model, 'CNN-LSTM')
                results['CNN-LSTM'] = {
                    **metrics,
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist()
                }
                # Clear GPU memory after evaluation
                del model
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            logger.warning("CNN-LSTM model not found")
        
        # Evaluate DNN model
        dnn_path = MODELS_PATH / "dnn" / "best_model.pt"
        if dnn_path.exists():
            model, checkpoint = evaluator.load_model('dnn', dnn_path)
            if model is not None:
                # Clear GPU memory before evaluation
                torch.cuda.empty_cache()
                metrics, predictions, probabilities = evaluator.evaluate_classification_model(model, 'DNN')
                results['DNN'] = {
                    **metrics,
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist()
                }
                # Clear GPU memory after evaluation
                del model
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            logger.warning("DNN model not found")
        
        # Create ensemble predictions
        if 'CNN-LSTM' in results and 'DNN' in results:
            ensemble_metrics, ensemble_predictions, ensemble_probabilities = evaluator.create_ensemble_predictions(
                np.array(results['CNN-LSTM']['probabilities']),
                np.array(results['DNN']['probabilities'])
            )
            results['Ensemble'] = {
                **ensemble_metrics,
                'predictions': ensemble_predictions.tolist(),
                'probabilities': ensemble_probabilities.tolist()
            }
        
        # Evaluate Autoencoder
        autoencoder_path = MODELS_PATH / "autoencoder" / "best_model.pt"
        threshold_path = MODELS_PATH / "autoencoder" / "threshold.json"
        
        if autoencoder_path.exists() and threshold_path.exists():
            model, checkpoint = evaluator.load_model('autoencoder', autoencoder_path)
            if model is not None:
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                    threshold = threshold_data['threshold']
                
                # Clear GPU memory before evaluation
                torch.cuda.empty_cache()
                metrics, predictions, reconstruction_errors = evaluator.evaluate_autoencoder(model, threshold)
                results['Autoencoder'] = {
                    **metrics,
                    'predictions': predictions.tolist(),
                    'reconstruction_errors': reconstruction_errors.tolist()
                }
                # Clear GPU memory after evaluation
                del model
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            logger.warning("Autoencoder model or threshold not found")
        
        # Generate plots
        evaluator.generate_plots(results)
        
        # Generate detailed report
        report = evaluator.generate_detailed_report(results)
        
        # Save results (without large arrays)
        results_path = RESULTS_PATH / "evaluation_results.json"
        from utils import safe_json_dump
        
        # Create simplified results without large arrays
        simplified_results = {}
        for model_name, result in results.items():
            simplified_result = {}
            for key, value in result.items():
                if key in ['predictions', 'probabilities', 'reconstruction_errors']:
                    # Skip large arrays
                    continue
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    simplified_result[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (list, np.ndarray)) and len(sub_value) > 100:
                            # Skip large arrays
                            continue
                        simplified_result[key][sub_key] = sub_value
                else:
                    simplified_result[key] = value
            simplified_results[model_name] = simplified_result
        
        safe_json_dump(simplified_results, results_path)
        
        logger.info(f"Evaluation results saved to {results_path}")
        logger.info("Comprehensive evaluation completed successfully!")
        logger.info("Plots saved to plots/ folder")
        logger.info("Metrics saved to results/ folder")
        
        return results, report
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    # Run comprehensive evaluation
    print("Starting Comprehensive Model Evaluation...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Evaluate all models
    results, report = evaluate_all_models()
    
    print("Evaluation completed successfully!")
    print(f"Results saved to {RESULTS_PATH}")
    print(f"Plots saved to {PLOTS_PATH}")
    print(f"Detailed report: {RESULTS_PATH / 'detailed_evaluation_report.json'}")
    print(f"Text report: {RESULTS_PATH / 'evaluation_report.txt'}")
