"""
Prediction script for cluster performance ML project.
Loads trained models and makes predictions on new data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessor import ClusterDataPreprocessor
from src.multi_output_model import MultiOutputClusterModel
import pandas as pd
import numpy as np
import logging
import argparse

def main():
    """Main prediction function.""" 
    parser = argparse.ArgumentParser(description='Make predictions using trained cluster performance models')
    parser.add_argument('--input', '-i', required=True, help='Path to input CSV file')
    parser.add_argument('--output', '-o', default='predictions.csv', help='Path to save predictions')
    parser.add_argument('--model', '-m', default=None, help='Specific model to use (optional)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting prediction pipeline...")
    
    try:
        # Initialize preprocessor and model
        preprocessor = ClusterDataPreprocessor()
        model = MultiOutputClusterModel()
        
        # Load preprocessor state (encoders, scalers, column info)
        logger.info("Loading preprocessor state...")
        preprocessor.load_preprocessor_state()
        
        # Load trained models
        logger.info("Loading trained models...")
        model.load_models()
        
        if not model.models:
            logger.error("No trained models found. Please run training first.")
            return
        
        # Load and preprocess new data
        logger.info(f"Loading data from {args.input}...")
        df = preprocessor.load_data(args.input)
        
        # Preprocess data (without fitting - use existing scalers/encoders)
        X, _ = preprocessor.preprocess_data(df, fit=False)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = model.predict(X, model_name=args.model)
        
        # Add original features for reference
        result_df = pd.concat([df, predictions], axis=1)
        
        # Save predictions
        result_df.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")
        
        # Show summary statistics
        logger.info("Prediction Summary:")
        logger.info(f"Number of samples: {len(predictions)}")
        logger.info(f"Number of predicted metrics: {len(predictions.columns)}")
        
        # Show sample predictions
        logger.info("Sample predictions (first 5 rows, first 10 metrics):")
        sample_pred = predictions.head().iloc[:, :10]
        logger.info("\n" + sample_pred.to_string())
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
