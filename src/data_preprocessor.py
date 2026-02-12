"""
Data preprocessing module for cluster performance ML project.
Handles loading, cleaning, and preprocessing of cluster performance data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
import os
import logging
from typing import Tuple, List, Dict, Any

class ClusterDataPreprocessor:
    """
    Preprocessor for cluster performance data.
    Handles feature engineering, encoding, and data splitting.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self.label_encoders = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using default settings.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not found."""
        return {
            'model': {'test_size': 0.2, 'random_state': 42},
            'features': {
                'metadata_patterns': [],
                'metric_patterns': []
            }
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            self.logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def identify_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify feature and target columns from the dataframe.
        This method automatically detects columns based on naming patterns.
        """
        self.logger.info("Identifying feature and target columns...")
        
        # Get all columns
        all_columns = df.columns.tolist()
        
        # Get patterns from config
        metadata_patterns = self.config.get('features', {}).get('metadata_patterns', [
            'clusterType', 'controlPlaneArch', 'infraNodesCount', 'infraNodesType',
            'ipsecMode', 'jobConfig.', 'k8sVersion', 'masterNodesCount', 'masterNodesType',
            'ocpMajorVersion', 'ocpVersion', 'passed', 'platform', 'publish', 'region',
            'sdnType', 'totalNodes', 'workerArch', 'workerNodesCount', 'workerNodesType',
            'masterNodevCPU', 'workerNodevCPU', 'masterNodeMemory',
            'workerNodeMemory'
        ])
        
        metric_patterns = self.config.get('features', {}).get('metric_patterns', [
            'cpu-', 'max-cpu-', 'memory-', 'max-memory-', 'podLatency'
        ])
        
        # Identify feature columns
        feature_columns = []
        for col in all_columns:
            if any(pattern in col for pattern in metadata_patterns):
                feature_columns.append(col)
        
        # Identify target columns
        target_columns = []
        for col in all_columns:
            if any(pattern in col for pattern in metric_patterns):
                target_columns.append(col)
        
        # Remove overlapping columns (prioritize features)
        target_columns = [col for col in target_columns if col not in feature_columns]
        
        self.logger.info(f"Identified {len(feature_columns)} feature columns")
        self.logger.info(f"Identified {len(target_columns)} target columns")
        
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        
        return feature_columns, target_columns
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by handling missing values and outliers."""
        self.logger.info("Cleaning data...")
        
        # Make a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Handle missing values
        self.logger.info(f"Missing values before cleaning: {df_clean.isnull().sum().sum()}")
        
        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        self.logger.info(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
        
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        self.logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col in self.feature_columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        unique_values = set(df_encoded[col].astype(str))
                        known_values = set(le.classes_)
                        new_values = unique_values - known_values
                        
                        if new_values:
                            self.logger.warning(f"New categories found in {col}: {new_values}")
                            # For simplicity, replace with most frequent category
                            most_frequent = df_encoded[col].mode()[0]
                            df_encoded[col] = df_encoded[col].astype(str).replace(list(new_values), most_frequent)
                        
                        df_encoded[col] = le.transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True, method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features."""
        self.logger.info(f"Scaling features using {method} scaling...")
        
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        X_scaled = X.copy()
        numerical_columns = X_scaled.select_dtypes(include=[np.number]).columns
        
        for col in numerical_columns:
            if fit:
                scaler = scaler_class()
                X_scaled[col] = scaler.fit_transform(X_scaled[[col]])
                self.scalers[col] = scaler
            else:
                if col in self.scalers:
                    X_scaled[col] = self.scalers[col].transform(X_scaled[[col]])
        
        return X_scaled
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders and scalers (True for training data)
            
        Returns:
            Tuple of (features, targets)
        """
        self.logger.info("Starting complete preprocessing pipeline...")
        
        # Identify columns if not already done
        if not self.feature_columns or not self.target_columns:
            self.identify_columns(df)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean, fit=fit)
        
        # Separate features and targets
        X = df_encoded[self.feature_columns].copy()
        y = df_encoded[self.target_columns].copy()
        
        # Scale features
        X_scaled = self.scale_features(X, fit=fit)
        
        # Handle target scaling if needed (for regression)
        if fit:
            y_scaler = StandardScaler()
            y_scaled = pd.DataFrame(
                y_scaler.fit_transform(y),
                columns=y.columns,
                index=y.index
            )
            self.scalers['targets'] = y_scaler
        else:
            if 'targets' in self.scalers:
                y_scaled = pd.DataFrame(
                    self.scalers['targets'].transform(y),
                    columns=y.columns,
                    index=y.index
                )
            else:
                y_scaled = y
        
        self.logger.info("Preprocessing completed successfully")
        return X_scaled, y_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        test_size = self.config.get('model', {}).get('test_size', 0.2)
        random_state = self.config.get('model', {}).get('random_state', 42)
        
        self.logger.info(f"Splitting data with test_size={test_size}")
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def save_preprocessed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: pd.DataFrame, y_test: pd.DataFrame, 
                              output_dir: str = "data/processed/"):
        """Save preprocessed data to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training data
        train_data = pd.concat([X_train, y_train], axis=1)
        train_data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
        
        # Save test data
        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
        
        # Save feature and target column names
        metadata = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        
        with open(os.path.join(output_dir, "metadata.yaml"), 'w') as f:
            yaml.dump(metadata, f)
        
        self.logger.info(f"Preprocessed data saved to {output_dir}")
    
    def get_feature_importance_data(self) -> Dict[str, List[str]]:
        """Get feature and target column information for model training."""
        return {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
    
    def save_preprocessor_state(self, output_dir: str = "data/processed/"):
        """Save preprocessor state (encoders, scalers, column names) to disk."""
        import joblib
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save label encoders
        encoders_path = os.path.join(output_dir, "label_encoders.joblib")
        joblib.dump(self.label_encoders, encoders_path)
        self.logger.info(f"Label encoders saved to {encoders_path}")
        
        # Save scalers
        scalers_path = os.path.join(output_dir, "scalers.joblib")
        joblib.dump(self.scalers, scalers_path)
        self.logger.info(f"Scalers saved to {scalers_path}")
        
        # Save column information
        columns_metadata = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        columns_path = os.path.join(output_dir, "columns_metadata.yaml")
        with open(columns_path, 'w') as f:
            yaml.safe_dump(columns_metadata, f)
        self.logger.info(f"Column metadata saved to {columns_path}")
    
    def load_preprocessor_state(self, input_dir: str = "data/processed/"):
        """Load preprocessor state (encoders, scalers, column names) from disk."""
        import joblib
        
        try:
            # Load label encoders
            encoders_path = os.path.join(input_dir, "label_encoders.joblib")
            self.label_encoders = joblib.load(encoders_path)
            self.logger.info(f"Label encoders loaded from {encoders_path}")
            
            # Load scalers
            scalers_path = os.path.join(input_dir, "scalers.joblib")
            self.scalers = joblib.load(scalers_path)
            self.logger.info(f"Scalers loaded from {scalers_path}")
            
            # Load column information
            columns_path = os.path.join(input_dir, "columns_metadata.yaml")
            with open(columns_path, 'r') as f:
                columns_metadata = yaml.safe_load(f)
            self.feature_columns = columns_metadata.get('feature_columns', [])
            self.target_columns = columns_metadata.get('target_columns', [])
            self.logger.info(f"Column metadata loaded from {columns_path}")
            
        except FileNotFoundError as e:
            self.logger.error(f"Preprocessor state file not found: {str(e)}")
            raise ValueError("Preprocessor state not found. Please train the model first.")