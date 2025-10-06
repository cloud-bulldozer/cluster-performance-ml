"""
Multi-output regression model for cluster performance prediction.
Handles training, evaluation, and prediction of multiple performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                           r2_score, explained_variance_score)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import yaml
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MultiOutputClusterModel:
    """
    Multi-output regression model for cluster performance prediction.
    Supports various algorithms including Random Forest, XGBoost, LightGBM, and CatBoost.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the model with configuration."""
        self.config = self._load_config(config_path)
        self.models = {}
        self.model_scores = {}
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
            self.logger.warning(f"Config file {config_path} not found.")
            return {}
    
    def _get_model(self, model_config: Dict[str, Any]):
        """Get model instance based on configuration."""
        model_type = model_config.get('type')
        params = model_config.get('params', {})
        
        if model_type == 'RandomForestRegressor':
            return RandomForestRegressor(**params)
        elif model_type == 'XGBRegressor':
            return xgb.XGBRegressor(**params)
        elif model_type == 'LGBMRegressor':
            return lgb.LGBMRegressor(**params)
        elif model_type == 'CatBoostRegressor':
            return CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Train multiple models for multi-output regression."""
        self.logger.info("Starting model training...")
        
        self.feature_columns = X_train.columns.tolist()
        self.target_columns = y_train.columns.tolist()
        
        models_config = self.config.get('models', [])
        
        for model_config in models_config:
            model_name = model_config.get('name')
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Get base model
                base_model = self._get_model(model_config)
                
                # Wrap in MultiOutputRegressor for multi-target regression
                multi_model = MultiOutputRegressor(base_model)
                
                # Train the model
                multi_model.fit(X_train, y_train)
                
                # Store the trained model
                self.models[model_name] = multi_model
                
                self.logger.info(f"{model_name} training completed")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models on test data."""
        self.logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred)
                evaluation_results[model_name] = metrics
                
                self.logger.info(f"{model_name} evaluation completed")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        self.model_scores = evaluation_results
        return evaluation_results
    
    def _calculate_metrics(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics for multi-output prediction."""
        metrics = {}
        
        # Convert predictions to DataFrame for easier handling
        y_pred_df = pd.DataFrame(y_pred, columns=y_true.columns, index=y_true.index)
        
        # Overall metrics (average across all targets)
        metrics['overall_mse'] = mean_squared_error(y_true, y_pred)
        metrics['overall_rmse'] = np.sqrt(metrics['overall_mse'])
        metrics['overall_mae'] = mean_absolute_error(y_true, y_pred)
        metrics['overall_r2'] = r2_score(y_true, y_pred)
        metrics['overall_explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # Per-target metrics (top 10 most important targets)
        target_importance = []
        for col in y_true.columns:
            r2 = r2_score(y_true[col], y_pred_df[col])
            target_importance.append((col, abs(r2)))
        
        # Sort by importance
        target_importance.sort(key=lambda x: x[1], reverse=True)
        top_targets = [col for col, _ in target_importance]
        
        for col in top_targets:
            col_clean = col.replace('-', '_').replace('/', '_')  # Clean column name
            metrics[f'{col_clean}_r2'] = r2_score(y_true[col], y_pred_df[col])
            metrics[f'{col_clean}_mse'] = mean_squared_error(y_true[col], y_pred_df[col])
            metrics[f'{col_clean}_mae'] = mean_absolute_error(y_true[col], y_pred_df[col])
        
        return metrics
    
    def cross_validate_models(self, X: pd.DataFrame, y: pd.DataFrame, cv: int = 5) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation for all models."""
        self.logger.info(f"Performing {cv}-fold cross-validation...")
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Cross-validating {model_name}...")
            
            try:
                # Perform cross-validation for multiple scoring metrics
                scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
                cv_scores = {}
                
                for metric in scoring_metrics:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                    cv_scores[f'{metric}_mean'] = scores.mean()
                    cv_scores[f'{metric}_std'] = scores.std()
                
                cv_results[model_name] = cv_scores
                self.logger.info(f"{model_name} cross-validation completed")
                
            except Exception as e:
                self.logger.error(f"Error in cross-validation for {model_name}: {str(e)}")
                continue
        
        return cv_results
    
    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Get feature importance from the first estimator (they should be similar across outputs)
        if hasattr(model.estimators_[0], 'feature_importances_'):
            importance = model.estimators_[0].feature_importances_
        else:
            self.logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """Make predictions using specified model or best performing model."""
        if model_name is None:
            # Use the best performing model based on overall R²
            if not self.model_scores:
                raise ValueError("No model scores available. Train and evaluate models first.")
            
            best_model = max(self.model_scores.keys(), 
                           key=lambda x: self.model_scores[x].get('overall_r2', -np.inf))
            model_name = best_model
            self.logger.info(f"Using best model: {model_name}")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        
        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions, columns=self.target_columns, index=X.index)
        
        return pred_df
    
    def save_models(self, output_dir: str = "models/"):
        """Save all trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
            joblib.dump(model, model_path)
            self.logger.info(f"Model {model_name} saved to {model_path}")
        
        # Save model metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'model_scores': self.model_scores
        }
        
        metadata_path = os.path.join(output_dir, "model_metadata.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        
        self.logger.info(f"Model metadata saved to {metadata_path}")
    
    def load_models(self, input_dir: str = "models/"):
        """Load trained models from disk."""
        # Load metadata
        metadata_path = os.path.join(input_dir, "model_metadata.yaml")
        try:
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            self.feature_columns = metadata.get('feature_columns', [])
            self.target_columns = metadata.get('target_columns', [])
            self.model_scores = metadata.get('model_scores', {})
            
        except FileNotFoundError:
            self.logger.warning(f"Metadata file {metadata_path} not found")
        
        # Load models
        for file in os.listdir(input_dir):
            if file.endswith('_model.joblib'):
                model_name = file.replace('_model.joblib', '')
                model_path = os.path.join(input_dir, file)
                
                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    self.logger.info(f"Model {model_name} loaded from {model_path}")
                except Exception as e:
                    self.logger.error(f"Error loading model {model_name}: {str(e)}")
    
    def plot_model_comparison(self, save_path: str = "results/plots/model_comparison.png"):
        """Plot comparison of model performance."""
        if not self.model_scores:
            self.logger.warning("No model scores available for plotting")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare data for plotting
        models = list(self.model_scores.keys())
        metrics = ['overall_r2', 'overall_rmse', 'overall_mae']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [self.model_scores[model].get(metric, 0) for model in models]
            
            axes[i].bar(models, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model comparison plot saved to {save_path}")
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all model performances."""
        if not self.model_scores:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, scores in self.model_scores.items():
            summary_data.append({
                'Model': model_name,
                'Overall R²': scores.get('overall_r2', 0),
                'Overall RMSE': scores.get('overall_rmse', 0),
                'Overall MAE': scores.get('overall_mae', 0),
                'Explained Variance': scores.get('overall_explained_variance', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Overall R²', ascending=False)
        
        return summary_df
