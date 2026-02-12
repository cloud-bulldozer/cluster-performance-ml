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
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

def get_latest_s3_file(bucket_name: str, prefix: str, logger) -> str:
    """
    Get the latest CSV file from an S3 bucket based on last modified time.
    
    Args:
        bucket_name: Name of the S3 bucket
        prefix: S3 prefix (folder path) to search in
        logger: Logger instance for logging
        
    Returns:
        S3 URI of the latest file (s3://bucket/key)
    """
    try:
        logger.info(f"Connecting to S3 bucket: {bucket_name}, prefix: {prefix}")
        s3_client = boto3.client('s3')
        
        # List objects in the bucket with the given prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            raise ValueError(f"No files found in s3://{bucket_name}/{prefix}")
        
        # Filter for CSV files only
        csv_files = [obj for obj in response['Contents'] 
                    if obj['Key'].endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in s3://{bucket_name}/{prefix}")
        
        # Sort by LastModified in descending order to get the latest file
        latest_file = sorted(csv_files, key=lambda x: x['LastModified'], reverse=True)[0]
        
        s3_uri = f"s3://{bucket_name}/{latest_file['Key']}"
        logger.info(f"Latest file found: {s3_uri}")
        logger.info(f"Last modified: {latest_file['LastModified']}")
        logger.info(f"File size: {latest_file['Size']} bytes")
        
        return s3_uri
        
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure AWS credentials.")
        raise
    except ClientError as e:
        logger.error(f"Error accessing S3: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting latest S3 file: {str(e)}")
        raise

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
        # Load configuration
        config_path = "configs/config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize preprocessor and model
        preprocessor = ClusterDataPreprocessor()
        model = MultiOutputClusterModel()
        
        # Load data from S3
        s3_config = config.get('data', {}).get('s3', {})
        s3_bucket = s3_config.get('bucket', 'kube-burner-ai-s3-bucket')
        s3_prefix = s3_config.get('prefix', 'cluster-density-v2/')
        
        logger.info("Fetching latest data file from S3...")
        s3_uri = get_latest_s3_file(s3_bucket, s3_prefix, logger)
        
        # Load data directly from S3
        logger.info(f"Loading data from {s3_uri}")
        df = pd.read_csv(s3_uri)
        logger.info(f"Data loaded successfully from S3. Shape: {df.shape}")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X, y = preprocessor.preprocess_data(df, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(X_train, X_test, y_train, y_test)
        
        # Save preprocessor state (encoders, scalers, etc.) for later use in predictions
        preprocessor.save_preprocessor_state()
        
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
        
        # Convert numpy types to Python native types for safe YAML serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save evaluation results
        with open(os.path.join(results_dir, "evaluation_results.yaml"), 'w') as f:
            yaml.safe_dump(convert_numpy_types(evaluation_results), f)
        
        # Save cross-validation results
        with open(os.path.join(results_dir, "cv_results.yaml"), 'w') as f:
            yaml.safe_dump(convert_numpy_types(cv_results), f)
        
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
