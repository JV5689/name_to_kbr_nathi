import random
import math
import os
import json
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from models import InverterTelemetry, PredictionResponse

class TelemetryGenerator:
    def __init__(self):
        # Base realistic values
        self.base_voltage = 230.0
        self.base_frequency = 50.0
        self.base_dc_power = 5000.0  
        
        # State per inverter
        self.inverter_states = {
            "INV-01": {"counter": 0, "seed": 42},
            "INV-02": {"counter": 0, "seed": 1337},
            "INV-03": {"counter": 0, "seed": 99}
        }
        
        # ML Model Paths
        self.model_path = '../ml/solar_failure_model.json'
        self.meta_path = '../ml/feature_meta.json'
        self.anomaly_path = '../ml/anomaly_model.pkl'
        
        self.model = None
        self.features = None
        self.anomaly_model = None
        self.ext_cols = []
        
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.meta_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                with open(self.meta_path, 'r') as f:
                    meta = json.load(f)
                    self.features = meta['features']
                    self.ext_cols = meta.get('mode_detection', {}).get('external_columns', [])
                
                if os.path.exists(self.anomaly_path):
                    self.anomaly_model = joblib.load(self.anomaly_path)
                    
                print("ML Models loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None

    def get_inverter_list(self):
        return list(self.inverter_states.keys())

    def get_current_telemetry(self, inverter_id: str = "INV-01") -> InverterTelemetry:
        if inverter_id not in self.inverter_states:
            inverter_id = "INV-01"
            
        state = self.inverter_states[inverter_id]
        state["counter"] += 1
        counter = state["counter"]
        
        rng = random.Random(state["seed"] + counter)
        noise = rng.uniform(-0.5, 0.5)
        
        time_offset = state["seed"] % 10
        irradiation_factor = max(0, math.sin((counter + time_offset) / 20.0))
        power_scale = 1.0 if inverter_id != "INV-02" else 0.85
        
        pv1_power = (2500.0 * irradiation_factor * power_scale) + (noise * 50)
        pv2_power = (2500.0 * irradiation_factor * power_scale) + (noise * 50)
        pv1_voltage = 400.0 + (noise * 5) if pv1_power > 0 else 0
        pv2_voltage = 400.0 + (noise * 5) if pv2_power > 0 else 0
        
        grid_noise_scale = 1.0 if inverter_id != "INV-03" else 15.0
        voltage = self.base_voltage + (math.sin(counter / 10.0) * 2) + (noise * grid_noise_scale)
        frequency = self.base_frequency + rng.uniform(-0.1, 0.1)
        
        # Weather simulation
        ambient_temp = 25.0 + rng.uniform(-2, 2)
        cloud_cover = max(0, min(1, 0.2 + (math.sin(counter/50) * 0.5) + noise))
        rainfall = 0.1 if cloud_cover > 0.8 else 0.0
        wind_speed = 5.0 + rng.uniform(0, 15)
        storm_prob = 0.8 if cloud_cover > 0.9 and wind_speed > 15 else 0.1

        temp_scale = 1.0 if inverter_id != "INV-02" else 1.3
        inverter_temp = ambient_temp + (((pv1_power + pv2_power) / 1000.0) * 5 * temp_scale) + noise
        
        efficiency = 0.96 + rng.uniform(-0.01, 0.005)
        grid_active_power = (pv1_power + pv2_power) * efficiency

        return InverterTelemetry(
            timestamp=datetime.utcnow().isoformat(),
            grid_voltage=round(voltage, 2),
            grid_frequency=round(frequency, 2),
            dc_power=round(pv1_power + pv2_power, 2),
            ac_power=round(grid_active_power, 2),
            inverter_temperature=round(inverter_temp, 2),
            ambient_temperature=round(ambient_temp, 2),
            inverter_efficiency=round(efficiency * 100, 2),
            pv1_power=round(pv1_power, 2),
            pv2_power=round(pv2_power, 2),
            pv1_voltage=round(pv1_voltage, 2),
            pv2_voltage=round(pv2_voltage, 2),
            grid_active_power=round(grid_active_power, 2),
            cloud_cover=round(cloud_cover, 2),
            rainfall=round(rainfall, 2),
            wind_speed=round(wind_speed, 2),
            storm_probability=round(storm_prob, 2),
            dc_current=round((pv1_power + pv2_power) / max(1, pv1_voltage + pv2_voltage), 2),
            power_factor=0.98,
            kwh_total=12500.0 + counter * 0.5,
            kwh_today=15.0 + counter * 0.1
        )

    def get_prediction(self, telemetry: InverterTelemetry) -> PredictionResponse:
        t_dict = telemetry.dict()
        
        # Automatic Mode Detection
        mode = "internal"
        external_vals = [t_dict.get(c, 0) for c in ["cloud_cover", "rainfall", "wind_speed", "storm_probability"]]
        if any(v > 0 for v in external_vals):
            mode = "internal+external"

        # Feature Engineering (Backend parity with Trainer)
        # Note: We use datetime.utcnow() for time features in real-time
        now = datetime.utcnow()
        input_data = {
            'ac_power': t_dict.get('ac_power', 0),
            'dc_voltage': t_dict.get('pv1_voltage', 0) + t_dict.get('pv2_voltage', 0), # Aggregated for parity
            'dc_current': t_dict.get('dc_current', 0),
            'inverter_temp': t_dict.get('inverter_temperature', 0),
            'grid_voltage': t_dict.get('grid_voltage', 0),
            'grid_frequency': t_dict.get('grid_frequency', 0),
            'power_factor': t_dict.get('power_factor', 1.0),
            'kwh_total': t_dict.get('kwh_total', 0),
            'kwh_today': t_dict.get('kwh_today', 0)
        }
        
        # Derived Features
        input_data['dc_power'] = input_data['dc_voltage'] * input_data['dc_current']
        input_data['power_loss'] = input_data['dc_power'] - input_data['ac_power']
        
        # Efficiency clipped to 0-1.1 range
        input_data['efficiency'] = np.where(input_data['dc_power'] > 0, input_data['ac_power'] / input_data['dc_power'], 0)
        input_data['efficiency'] = float(np.clip(input_data['efficiency'], 0, 1.1))
        
        input_data['thermal_stress'] = input_data['inverter_temp']
        input_data['voltage_deviation'] = abs(input_data['grid_voltage'] - 230)
        input_data['frequency_deviation'] = abs(input_data['grid_frequency'] - 50)
        
        # Time-based features parity
        input_data['hour'] = now.hour
        input_data['day_of_week'] = now.weekday()
        input_data['month'] = now.month

        reasons = []
        risk_level = "Low"
        failure_prob = 0.05
        anomaly_detected = False

        if self.model and self.features:
            try:
                df_input = pd.DataFrame([input_data])[self.features]
                prob = self.model.predict_proba(df_input)[0][1]
                failure_prob = float(prob)
                
                if prob > 0.5: risk_level = "Medium"
                if prob > 0.8: risk_level = "High"
                if prob > 0.95: risk_level = "Critical"
                
                # Check anomaly
                if self.anomaly_model:
                    is_anomaly = self.anomaly_model.predict(df_input)[0] == -1
                    if is_anomaly:
                        anomaly_detected = True
                        reasons.append("Abnormal operational pattern detected")

                # Reason Identification
                if input_data['efficiency'] < 0.9: reasons.append("Efficiency drop detected")
                if input_data['thermal_stress'] > 30: reasons.append("High thermal stress")
                if input_data.get('storm_risk', 0) > 5: reasons.append("Elevated storm risk")
                if input_data['voltage_deviation'] > 15: reasons.append("Grid voltage instability")
                
            except Exception as e:
                print(f"Prediction error: {e}")

        return PredictionResponse(
            timestamp=datetime.utcnow().isoformat(),
            failure_probability=round(failure_prob, 4),
            risk_level=risk_level,
            anomaly_detected=anomaly_detected or (failure_prob > 0.7),
            mode=mode,
            reasons=reasons if reasons else ["Operating within normal parameters"],
            root_cause=reasons[0] if reasons else None,
            maintenance_recommendation="Check cooling system and grid connection" if failure_prob > 0.5 else "Routine maintenance"
        )



