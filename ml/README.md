# Solar Inverter Failure Prediction ML Pipeline

This directory contains the machine learning pipeline for predicting solar inverter failures using internal telemetry data.

## How to Run

### 1. Locally
Ensure you have Python installed and run:
```bash
pip install pandas numpy xgboost scikit-learn joblib
python train_model.py
```

### 2. Google Colab
1. Upload your dataset (CSV) to the session storage.
2. Copy the code from `train_model.py` into a Colab cell.
3. Run the cell to perform data cleaning, feature engineering, and training.

## Files
- `train_model.py`: Core machine learning pipeline (Cleaning, Engineering, Training).
- `feature_meta.json`: Metadata for original and engineered features.
- `solar_failure_model.json`: Trained XGBoost model.
- `anomaly_model.pkl`: Isolation Forest model for anomaly detection.
