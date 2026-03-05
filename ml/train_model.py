import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import IsolationForest
import json
import os
import joblib

# Modular Utilities (Also available in ml_utils.py for backend/future use)
from ml_utils import merge_external_weather, engineer_weather_risk

# Define feature groups for documentation and interpretability
ORIGINAL_FEATURES = [
    'ac_power', 'dc_voltage', 'dc_current', 'inverter_temp', 
    'grid_voltage', 'grid_frequency', 'power_factor',
    'kwh_total', 'kwh_today'
]

# We can dynamically add all available raw columns later if needed
# For now, we focus on the core internal telemetry signals mentioned.

def clean_data(df):
    """
    Perform data cleaning: remove invalid readings and handle missing values.
    """
    # 1. Basic Cleaning: Remove impossible sensor readings
    if "dc_voltage" in df.columns:
        df = df[df["dc_voltage"] > 0]
    if "ac_power" in df.columns:
        df = df[df["ac_power"] >= 0]
    if "grid_frequency" in df.columns:
        # Realistic grid limits: 45–65 Hz
        df = df[(df["grid_frequency"] >= 45) & (df["grid_frequency"] <= 65)]
    
    # 2. Handle missing values using medians (robust to outliers)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
            
    # 3. Convert timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
            
    return df

def feature_engineering(df):
    """
    Create derived features representing system behaviors and cycles.
    """
    # 1. Electrical Performance Features
    if "dc_voltage" in df.columns and "dc_current" in df.columns:
        df["dc_power"] = df["dc_voltage"] * df["dc_current"]
    
    if "dc_power" in df.columns and "ac_power" in df.columns:
        # Power Loss: Difference between DC input and AC output
        df["power_loss"] = df["dc_power"] - df["ac_power"]
        
        # Conversion Efficiency: AC / DC ratio
        # Clipped to 0-1.1 range to handle measurement noise without extreme outliers
        df["efficiency"] = np.where(df["dc_power"] > 0, df["ac_power"] / df["dc_power"], 0)
        df["efficiency"] = df["efficiency"].clip(0, 1.1)

    # 2. Thermal Stress
    if "inverter_temp" in df.columns:
        # Thermal stress index (in this case, just the temp, but could be relative to ambient)
        df["thermal_stress"] = df["inverter_temp"]

    # 3. Grid Stability Deviations
    if "grid_voltage" in df.columns:
        df["voltage_deviation"] = abs(df["grid_voltage"] - 230)
    if "grid_frequency" in df.columns:
        df["frequency_deviation"] = abs(df["grid_frequency"] - 50)

    # 4. Time-based Features (Daily and Seasonal cycles)
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

    return df

def generate_target_labels(df):
    """
    Generate rule-based failure labels if not explicitly present.
    Based on high temperature, low efficiency, or alarm codes.
    """
    # Avoid errors if columns are missing
    temp = df.get('inverter_temp', 0)
    eff = df.get('efficiency', 1.0)
    dc_p = df.get('dc_power', 0)
    alarm = df.get('alarm_code', 0)
    
    failure_conditions = (
        (temp > 85) |  # Critical overheating
        ((eff < 0.7) & (dc_p > 500)) | # Significant drop in efficiency during production
        (alarm != 0) # Presence of any hardware alarm
    )
    df['failure_risk'] = failure_conditions.astype(int)
    return df

def train_failure_pipeline():
    file_path = 'Copy of 54-10-EC-8C-14-69.raws.csv'
    if not os.path.exists(file_path):
        print(f"Error: Dataset {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    df_raw = pd.read_csv(file_path)

    # Step 1: Map raw columns to logical names (using available columns in this specific dataset)
    # This ensures we extract the core signals mentioned in the prompt.
    feature_map = {
        'inverters[0].pv1_power': 'ac_power',
        'inverters[0].pv1_voltage': 'dc_voltage',
        'inverters[0].pv1_current': 'dc_current',
        'inverters[0].temp': 'inverter_temp',
        'meters[0].v_r': 'grid_voltage',
        'meters[0].freq': 'grid_frequency',
        'meters[0].pf': 'power_factor',
        'inverters[0].kwh_total': 'kwh_total',
        'inverters[0].kwh_today': 'kwh_today',
        'inverters[0].alarm_code': 'alarm_code',
        'timestamp': 'timestamp'
    }
    
    # Filter columns that exist
    cols_to_use = [col for col in feature_map.keys() if col in df_raw.columns]
    df = df_raw[cols_to_use].rename(columns=feature_map)

    # Convert numeric columns to float, coercing errors to NaN
    # This handles cases where the CSV might have non-numeric placeholders
    numeric_cols = [
        'ac_power', 'dc_voltage', 'dc_current', 'inverter_temp', 
        'grid_voltage', 'grid_frequency', 'power_factor',
        'kwh_total', 'kwh_today'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Step 2: Data Cleaning
    print("Cleaning data...")
    df = clean_data(df)

    # Step 3: Feature Engineering
    print("Engineering features...")
    df = feature_engineering(df)

    # Step 4: Target Generation
    print("Generating target labels...")
    df = generate_target_labels(df)

    # Final feature selection (excluding timestamp and target)
    target = 'failure_risk'
    drop_cols = [target, 'timestamp']
    if 'alarm_code' in df.columns:
        drop_cols.append('alarm_code') # Not used as a feature directly if used for label
        
    X = df.drop(columns=drop_cols)
    y = df[target]

    print(f"Dataset summary: {len(X)} samples, {X.shape[1]} features.")
    
    # Identify engineered vs original for documentation
    engineered_features = [col for col in X.columns if col not in ORIGINAL_FEATURES]
    actual_original_features = [col for col in X.columns if col in ORIGINAL_FEATURES]

    # Step 5: Model Training with Time-Aware Split
    print("Training XGBoost Classifier...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # Perform CV
    fold = 1
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        print(f"Fold {fold} trained.")
        fold += 1

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    try:
        print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    except:
        print("ROC-AUC could not be calculated (possibly only one class in test split).")

    # Step 6: Anomaly Detection (Isolation Forest)
    print("\nTraining Anomaly Detection Layer...")
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X)

    # Step 7: Saving Artifacts
    print("\nSaving models and metadata...")
    model.save_model('solar_failure_model.json')
    joblib.dump(iso_forest, 'anomaly_model.pkl')
    
    metadata = {
        'original_features': actual_original_features,
        'engineered_features': engineered_features,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        },
        'system_metadata': {
            'version': '1.0.0',
            'note': 'Internal telemetry pipeline'
        }
    }
    
    with open('feature_meta.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Pipeline complete. Models and metadata saved.")

if __name__ == "__main__":
    train_failure_pipeline()

