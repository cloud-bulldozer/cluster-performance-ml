# Cluster Performance ML Project

A comprehensive machine learning project for predicting cluster performance metrics using multi-output regression models.

## Overview

This project implements multiple regression models to predict various cluster performance metrics based on cluster configuration metadata. The system uses multi-output regression to simultaneously predict multiple performance metrics from cluster setup parameters.

## Features

- **Multi-output Regression**: Predicts multiple performance metrics simultaneously
- **Multiple Algorithms**: Supports Random Forest, XGBoost, and LightGBM
- **Automated Preprocessing**: Handles categorical encoding, scaling, and missing values
- **Model Comparison**: Evaluates and compares different algorithms
- **Feature Importance**: Analyzes which configuration parameters matter most
- **Cross-validation**: Robust model evaluation with k-fold cross-validation
- **Visualization**: Comprehensive plots and analysis

## Project Structure

```
cluster-performance-ml/
├── src/                          # Source code
│   ├── data_preprocessor.py      # Data preprocessing pipeline
│   ├── multi_output_model.py     # Multi-output regression models
│   ├── train.py                  # Training script
│   ├── predict.py                # Prediction script
│   └── __init__.py               # Package initialization
├── configs/                      # Configuration files
│   └── config.yaml               # Main configuration
├── data/                         # Data directory
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
├── models/                       # Trained models
├── results/                      # Results and evaluation metrics
│   └── plots/                    # Generated plots
├── notebooks/                    # Jupyter notebooks
│   └── exploratory_analysis.ipynb # EDA notebook

```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cluster-performance-ml
```

2. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data

Place your cluster performance CSV file in `data/raw/cluster_data.csv`. The CSV should contain:

- **Metadata columns** (input features): clusterType, controlPlaneArch, k8sVersion, etc.
- **Metrics columns** (target outputs): cpu-, memory-, latency metrics, etc.

### 2. Train Models

Run the training pipeline:
```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Train multiple models (Random Forest, XGBoost, LightGBM)
- Evaluate model performance
- Save trained models and results

### 3. Make Predictions

Use trained models to predict on new data:
```bash
python src/predict.py --input path/to/new_data.csv --output predictions.csv --model XGBoost
```

### 4. Exploratory Data Analysis

Run the Jupyter notebook for data exploration:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

#### Example Execution On Google Colab 
* EDA: https://colab.research.google.com/drive/1I_AqN-m2p0T2sP8gpHtornPpQWCl8Zlk#scrollTo=msdNxbzFslaG 
* Training: https://colab.research.google.com/drive/1trek6cCQhJF-yZ-sSBLIGW86QzeY3ugR#scrollTo=bNTJCy_J9fd3
###### Note: Requires Red Hat, Inc email to access above notebooks


## Configuration

Modify `configs/config.yaml` to customize:

- **Data paths**: Input and output file locations
- **Model parameters**: Algorithm hyperparameters
- **Feature patterns**: Patterns to identify input/output columns
- **Evaluation metrics**: Metrics for model assessment

## Input Data Format

The system expects a CSV file with columns following these patterns:

### Input Features (Metadata)
- `clusterType`: Type of cluster (e.g., self-managed)
- `controlPlaneArch`: Architecture (e.g., amd64)
- `k8sVersion`: Kubernetes version
- `masterNodesCount`: Number of master nodes
- `workerNodesCount`: Number of worker nodes
- `jobConfig.*`: Job configuration parameters
- And more cluster configuration parameters...

### Target Metrics (Outputs)
- `cpu-*`: CPU usage metrics
- `memory-*`: Memory usage metrics
- `*-latency`: API call latency metrics
- `99th*`: 99th percentile metrics
- `cgroup*`: CGroup resource metrics
- And more performance metrics...

## Model Performance

The system evaluates models using:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Explained Variance**: Explained variance score

## Results

After training, find results in:

- `results/model_summary.csv`: Model performance comparison
- `results/evaluation_results.yaml`: Detailed evaluation metrics
- `results/plots/`: Performance comparison plots
- `models/`: Trained model files

## Example Results

```
       Model  Overall R²  Overall RMSE  Overall MAE  Explained Variance
RandomForest    0.938880      0.235042     0.113551            0.938970
     XGBoost    0.934250      0.245289     0.122311            0.934259
    CatBoost    0.921117      0.268978     0.150960            0.921137
    LightGBM    0.898202      0.303874     0.157239            0.898221
```

## Advanced Usage

### Custom Model Configuration

Add new models in `configs/config.yaml`:

```yaml
models:
  - name: "CustomRF"
    type: "RandomForestRegressor"
    params:
      n_estimators: 200
      max_depth: 15
      min_samples_split: 5
```

### Feature Engineering

Modify feature patterns in the config to include/exclude specific columns:

```yaml
features:
  metadata_patterns:
    - clusterType
    - customFeature
  metric_patterns:
    - custom-metric
```

## Troubleshooting

### Common Issues

1. **File not found**: Ensure your CSV is at `data/raw/cluster_data.csv`
2. **Memory issues**: Reduce dataset size or use sample data for testing
3. **Missing dependencies**: Run `pip install -r requirements.txt`

### Logging

Check `training.log` for detailed execution logs and error messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the Apache 2.0 License.

## Support

For questions or issues, please create an issue in the repository.
