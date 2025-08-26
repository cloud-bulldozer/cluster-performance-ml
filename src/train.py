"""
Training script for cluster performance ML project.
Handles the complete training pipeline from data loading to model evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessor import ClusterDataPreprocessor
from src.multi_output_model import MultiOutputClusterModel
import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime

def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting cluster performance ML training pipeline...")
    
    try:
        # Initialize preprocessor and model
        preprocessor = ClusterDataPreprocessor()
        model = MultiOutputClusterModel()
        
        # Load data
        data_path = "data/raw/cluster_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            logger.info("Please place your CSV data file at data/raw/cluster_data.csv")
            return
        
        df = preprocessor.load_data(data_path)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X, y = preprocessor.preprocess_data(df, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(X_train, X_test, y_train, y_test)
        
        # Train models
        logger.info("Training models...")
        model.train_models(X_train, y_train)
        
        # Evaluate models
        logger.info("Evaluating models...")
        evaluation_results = model.evaluate_models(X_test, y_test)
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_results = model.cross_validate_models(X_train, y_train)
        
        # Save models
        model.save_models()
        
        # Generate model summary
        summary_df = model.get_model_summary()
        logger.info("Model Performance Summary:")
        logger.info("\n" + summary_df.to_string(index=False))
        
        # Save results
        results_dir = "results/"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save evaluation results
        with open(os.path.join(results_dir, "evaluation_results.yaml"), 'w') as f:
            yaml.dump(evaluation_results, f)
        
        # Save cross-validation results
        with open(os.path.join(results_dir, "cv_results.yaml"), 'w') as f:
            yaml.dump(cv_results, f)
        
        # Save model summary
        summary_df.to_csv(os.path.join(results_dir, "model_summary.csv"), index=False)
        
        # Generate plots
        model.plot_model_comparison()
        
        # Feature importance for best model
        best_model = summary_df.iloc[0]['Model']
        logger.info(f"Generating feature importance for best model: {best_model}")
        
        try:
            importance_df = model.get_feature_importance(best_model)
            importance_df.to_csv(os.path.join(results_dir, f"{best_model}_feature_importance.csv"), index=False)
            logger.info(f"Top 10 most important features for {best_model}:")
            logger.info("\n" + importance_df.head(10).to_string(index=False))
        except Exception as e:
            logger.warning(f"Could not generate feature importance: {str(e)}")
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Results saved to {results_dir}")
        logger.info(f"Models saved to models/")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
