"""
Data preprocessing module for CIC-IDS-2017 dataset
Handles data loading, quality fixes, and train/test splitting
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *
from utils import fix_data_quality, setup_logging

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_cic_ids2017_data():
    """
    Load and concatenate all CIC-IDS-2017 CSV files
    """
    logger = setup_logging()
    logger.info("Loading CIC-IDS-2017 dataset...")
    
    dataframes = []
    total_records = 0
    
    for file in PROCESSED_FILES:
        file_path = DATA_PATH / file
        if file_path.exists():
            logger.info(f"Loading {file}...")
            df = pd.read_csv(file_path)
            dataframes.append(df)
            total_records += len(df)
            logger.info(f"  - Records: {len(df):,}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    if not dataframes:
        raise FileNotFoundError("No CSV files found in the data directory")
    
    # Concatenate all dataframes
    logger.info("Concatenating all datasets...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    logger.info(f"Total records loaded: {total_records:,}")
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

def inspect_dataset(df):
    """
    Inspect dataset and print comprehensive information
    """
    logger = setup_logging()
    
    print("=" * 80)
    print("CIC-IDS-2017 DATASET INSPECTION")
    print("=" * 80)
    
    # Basic information
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    
    # Column information
    print("COLUMN INFORMATION:")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Feature Columns: {len(df.columns) - 1}")  # Excluding label column
    print()
    
    # Data types
    print("DATA TYPES:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    print()
    
    # Missing values
    print("MISSING VALUES:")
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    print(f"Total missing values: {total_missing:,}")
    if total_missing > 0:
        print("Missing values by column:")
        for col, missing in missing_values[missing_values > 0].items():
            percentage = (missing / len(df)) * 100
            print(f"  {col}: {missing:,} ({percentage:.2f}%)")
    print()
    
    # Label distribution
    label_col = df.columns[-1]  # Last column is the label
    print("LABEL DISTRIBUTION:")
    label_counts = df[label_col].value_counts()
    total_samples = len(df)
    
    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {label}: {count:,} ({percentage:.2f}%)")
    print()
    
    # Data quality issues
    print("DATA QUALITY ISSUES:")
    
    # Check for -1 values
    minus_one_count = (df == -1).sum().sum()
    print(f"  -1 placeholder values: {minus_one_count:,}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = 0
    for col in numeric_cols:
        inf_count += np.isinf(df[col]).sum()
    print(f"  Infinite values: {inf_count:,}")
    
    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_cols.append(col)
    print(f"  Constant columns: {len(constant_cols)}")
    if constant_cols:
        print(f"    {constant_cols}")
    
    print("=" * 80)

def preprocess_data(df):
    """
    Preprocess the dataset for machine learning
    """
    logger = setup_logging()
    logger.info("Starting data preprocessing...")
    
    # Store original shape
    original_shape = df.shape
    logger.info(f"Original dataset shape: {original_shape}")
    
    # Fix data quality issues
    logger.info("Fixing data quality issues...")
    df = fix_data_quality(df)
    
    # Separate features and labels
    label_col = df.columns[-1]
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    # Encode labels
    logger.info("Encoding labels...")
    
    # Clean the label column BEFORE encoding to avoid Unicode issues
    y_clean = y.astype(str).str.replace('◆', '').str.replace('•', '').str.replace('★', '').str.replace('☆', '').str.strip()
    
    # Also clean the original DataFrame to avoid any future issues
    df[label_col] = y_clean
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_clean)
    
    # Store label mapping (now with clean labels)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # Clean label mapping for logging to avoid UnicodeEncodeError
    clean_label_mapping = {}
    for label, encoded in label_mapping.items():
        clean_label = str(label).encode('ascii', 'replace').decode('ascii')
        clean_label_mapping[clean_label] = str(encoded)
    logger.info(f"Label mapping: {clean_label_mapping}")
    
    # Check for any remaining issues
    logger.info("Final data quality check...")
    
    # Check for any remaining missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Warning: {missing_count} missing values still present")
    
    # Check for infinite values
    inf_count = 0
    for col in X.columns:
        inf_count += np.isinf(X[col]).sum()
    if inf_count > 0:
        logger.warning(f"Warning: {inf_count} infinite values still present")
    
    logger.info("Data preprocessing completed successfully!")
    
    return X, y_encoded, label_encoder, label_mapping

def split_data(X, y, test_size=None, random_state=None, stratify=True):
    """
    Split data into train and test sets with stratification
    """
    from config import CV_CONFIG
    
    # Use config defaults if not provided
    if test_size is None:
        test_size = CV_CONFIG['test_size']
    if random_state is None:
        random_state = CV_CONFIG['random_state']
    
    logger = setup_logging()
    logger.info(f"Splitting data: {test_size*100:.0f}% test, {(1-test_size)*100:.0f}% train")
    
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        logger.info("Stratified split applied to preserve class distribution")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            shuffle=True
        )
        logger.info("Random split applied")
    
    logger.info(f"Data split: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, scaler_type='standard'):
    """
    Scale features using StandardScaler or MinMaxScaler
    """
    logger = setup_logging()
    logger.info(f"Scaling features using {scaler_type} scaler...")
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    # Fit scaler on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Feature scaling completed")
    logger.info(f"Training data shape: {X_train_scaled.shape}")
    logger.info(f"Test data shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler

def create_cv_splits(X_train, y_train, n_folds=None, random_state=None):
    """
    Create cross-validation splits for training
    """
    from config import CV_CONFIG
    
    # Use config defaults if not provided
    if n_folds is None:
        n_folds = CV_CONFIG['n_folds']
    if random_state is None:
        random_state = CV_CONFIG['random_state']
    
    logger = setup_logging()
    logger.info(f"Creating {n_folds}-fold cross-validation splits...")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_splits = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        cv_splits.append({
            'fold': fold + 1,
            'X_train': X_fold_train,
            'X_val': X_fold_val,
            'y_train': y_fold_train,
            'y_val': y_fold_val
        })
        
        logger.info(f"Fold {fold + 1}: Train={len(train_idx):,}, Val={len(val_idx):,}")
    
    return cv_splits

def save_processed_data(X_train, X_test, y_train, y_test, scaler, label_encoder, label_mapping):
    """
    Save processed data and preprocessors
    """
    logger = setup_logging()
    logger.info("Saving processed data...")
    
    # Create directories
    TRAIN_DATA_PATH.mkdir(parents=True, exist_ok=True)
    TEST_DATA_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    np.save(TRAIN_DATA_PATH / "X_train.npy", X_train)
    np.save(TRAIN_DATA_PATH / "y_train.npy", y_train)
    logger.info(f"Training data saved to {TRAIN_DATA_PATH}")
    
    # Save test data
    np.save(TEST_DATA_PATH / "X_test.npy", X_test)
    np.save(TEST_DATA_PATH / "y_test.npy", y_test)
    logger.info(f"Test data saved to {TEST_DATA_PATH}")
    
    # Save preprocessors
    joblib.dump(scaler, PROCESSED_DATA_PATH / "scaler.pkl")
    joblib.dump(label_encoder, PROCESSED_DATA_PATH / "label_encoder.pkl")
    joblib.dump(label_mapping, PROCESSED_DATA_PATH / "label_mapping.pkl")
    logger.info(f"Preprocessors saved to {PROCESSED_DATA_PATH}")
    
    # Save metadata
    metadata = {
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'y_train_shape': y_train.shape,
        'y_test_shape': y_test.shape,
        'num_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'scaler_type': type(scaler).__name__
    }
    
    from utils import safe_json_dump
    safe_json_dump(metadata, PROCESSED_DATA_PATH / "metadata.json")
    
    logger.info("Processed data saved successfully!")

def load_processed_data():
    """
    Load previously processed data
    """
    logger = setup_logging()
    # Load training data
    X_train = np.load(TRAIN_DATA_PATH / "X_train.npy")
    y_train = np.load(TRAIN_DATA_PATH / "y_train.npy")
    
    # Load test data
    X_test = np.load(TEST_DATA_PATH / "X_test.npy")
    y_test = np.load(TEST_DATA_PATH / "y_test.npy")
    
    logger.info(f"Data loaded: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")
    
    # Load preprocessors
    scaler = joblib.load(PROCESSED_DATA_PATH / "scaler.pkl")
    label_encoder = joblib.load(PROCESSED_DATA_PATH / "label_encoder.pkl")
    label_mapping = joblib.load(PROCESSED_DATA_PATH / "label_mapping.pkl")
    logger.info("Preprocessors loaded")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder, label_mapping

def get_class_weights(y_train):
    """
    Calculate class weights for imbalanced dataset
    """
    logger = setup_logging()
    logger.info("Calculating class weights...")
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    logger.info(f"Class weights calculated for {len(class_weight_dict)} classes")
    
    return class_weight_dict

# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================
def run_preprocessing_pipeline():
    """
    Run the complete data preprocessing pipeline
    """
    logger = setup_logging()
    logger.info("Starting CIC-IDS-2017 preprocessing pipeline...")
    
    try:
        # Load and preprocess data
        logger.info("Loading CIC-IDS-2017 data...")
        df = load_cic_ids2017_data()
        
        logger.info("Preprocessing data...")
        X, y, label_encoder, label_mapping = preprocess_data(df)
        
        logger.info("Splitting and scaling data...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        logger.info("Calculating class weights...")
        class_weights = get_class_weights(y_train)
        
        logger.info("Saving processed data...")
        save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, 
                           scaler, label_encoder, label_mapping)
        
        logger.info("Creating cross-validation splits...")
        cv_splits = create_cv_splits(X_train_scaled, y_train)
        
        logger.info("Preprocessing pipeline completed successfully!")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'label_mapping': label_mapping,
            'class_weights': class_weights,
            'cv_splits': cv_splits
        }
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Run preprocessing pipeline
    results = run_preprocessing_pipeline()
    print("Preprocessing completed successfully!")
    print(f"Training data shape: {results['X_train'].shape}")
    print(f"Test data shape: {results['X_test'].shape}")
    print(f"Number of classes: {len(results['label_encoder'].classes_)}")
