import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json
import os

def train_failure_model():
    file_path = 'Copy of 54-10-EC-8C-14-69.raws.csv'
    print(f"Loading data from {file_path}...")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Preprocessing data...")
    
    # Define features based on CSV analysis
    feature_map = {
        'inverters[0].pv1_power': 'pv1_power',
        'inverters[0].pv2_power': 'pv2_power',
        'inverters[0].pv1_voltage': 'pv1_voltage',
        'inverters[0].pv2_voltage': 'pv2_voltage',
        'inverters[0].temp': 'inverter_temp',
        'sensors[0].ambient_temp': 'ambient_temp',
        'meters[0].meter_active_power': 'grid_active_power'
    }
    
    # Filter columns that actually exist in the CSV
    existing_features = [col for col in feature_map.keys() if col in df.columns]
    
    # Create a working dataframe with renamed columns for clarity
    df_train = df[existing_features].copy()
    df_train = df_train.rename(columns={col: feature_map[col] for col in existing_features})
    
    # Handle Target: failure = status code != 0 or presence of alarm
    if 'inverters[0].alarm_code' in df.columns:
        df_train['failure'] = (df['inverters[0].alarm_code'] != 0).astype(int)
    else:
        # Fallback if alarm_code is missing (unlikely based on analysis)
        print("Warning: inverters[0].alarm_code not found. Using heuristic failure.")
        df_train['failure'] = (df_train['inverter_temp'] > 65).astype(int)

    # Simple Feature Engineering
    if 'pv1_power' in df_train.columns and 'pv2_power' in df_train.columns:
        df_train['total_pv_power'] = df_train['pv1_power'] + df_train['pv2_power']
    
    if 'inverter_temp' in df_train.columns and 'ambient_temp' in df_train.columns:
        df_train['temp_diff'] = df_train['inverter_temp'] - df_train['ambient_temp']

    # Drop rows with NaNs in features
    df_train = df_train.dropna()

    X = df_train.drop(['failure'], axis=1)
    y = df_train['failure']
    
    if len(X) == 0:
        print("Error: No valid data after preprocessing.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training XGBoost model on {len(X_train)} samples...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print("Saving model to solar_failure_model.json...")
    model.save_model('solar_failure_model.json')
    
    # Save feature names for inference
    with open('feature_meta.json', 'w') as f:
        json.dump({
            'features': list(X.columns),
            'feature_map': feature_map,
            'accuracy': accuracy
        }, f)
        
    print("Training complete.")

if __name__ == "__main__":
    train_failure_model()
