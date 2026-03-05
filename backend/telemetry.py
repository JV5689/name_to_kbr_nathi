import random
import math
import os
import json
import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime
from models import InverterTelemetry, PredictionResponse

class TelemetryGenerator:
    def __init__(self):
        # Base realistic values
        self.base_voltage = 230.0
        self.base_frequency = 50.0
        self.base_dc_power = 5000.0  # 5kW system
        
        # State per inverter
        self.inverter_states = {
            "INV-01": {"counter": 0, "seed": 42},
            "INV-02": {"counter": 0, "seed": 1337},
            "INV-03": {"counter": 0, "seed": 99}
        }
        
        # ML Model Path
        self.model_path = '../ml/solar_failure_model.json'
        self.meta_path = '../ml/feature_meta.json'
        self.model = None
        self.features = None
        
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.meta_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                with open(self.meta_path, 'r') as f:
                    self.features = json.load(f)['features']
                print("ML Model loaded successfully.")
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
        
        # Seed random for consistent but distinct inverter "personality"
        rng = random.Random(state["seed"] + counter)
        
        # Base realistic values
        noise = rng.uniform(-0.5, 0.5)
        
        # Simulated PV Power (Daily cycle)
        # Offset different inverters slightly in time
        time_offset = state["seed"] % 10
        irradiation_factor = max(0, math.sin((counter + time_offset) / 20.0))
        
        # Inverter 2 is slightly less efficient/powerful
        power_scale = 1.0 if inverter_id != "INV-02" else 0.85
        
        pv1_power = (2500.0 * irradiation_factor * power_scale) + (noise * 50)
        pv2_power = (2500.0 * irradiation_factor * power_scale) + (noise * 50)
        
        # Simulated Voltages
        pv1_voltage = 400.0 + (noise * 5) if pv1_power > 0 else 0
        pv2_voltage = 400.0 + (noise * 5) if pv2_power > 0 else 0
        
        # Grid Voltage (INV-03 has more grid instability noise)
        grid_noise_scale = 1.0 if inverter_id != "INV-03" else 15.0
        voltage = self.base_voltage + (math.sin(counter / 10.0) * 2) + (noise * grid_noise_scale)
        frequency = self.base_frequency + rng.uniform(-0.1, 0.1)
        
        # Temperatures
        ambient_temp = 25.0 + rng.uniform(-2, 2)
        # Inverter 2 runs hotter
        temp_scale = 1.0 if inverter_id != "INV-02" else 1.3
        inverter_temp = ambient_temp + (((pv1_power + pv2_power) / 1000.0) * 5 * temp_scale) + noise
        
        # Efficiency
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
            grid_active_power=round(grid_active_power, 2)
        )

    def get_prediction(self, telemetry: InverterTelemetry) -> PredictionResponse:
        # Use ML Model Prediction if available
        if self.model and self.features:
            try:
                t_dict = telemetry.dict()
                input_data = {
                    'pv1_power': t_dict.get('pv1_power', 0),
                    'pv2_power': t_dict.get('pv2_power', 0),
                    'pv1_voltage': t_dict.get('pv1_voltage', 0),
                    'pv2_voltage': t_dict.get('pv2_voltage', 0),
                    'inverter_temp': telemetry.inverter_temperature,
                    'ambient_temp': telemetry.ambient_temperature,
                    'grid_active_power': t_dict.get('grid_active_power', 0),
                    'total_pv_power': t_dict.get('pv1_power', 0) + t_dict.get('pv2_power', 0),
                    'temp_diff': telemetry.inverter_temperature - telemetry.ambient_temperature
                }
                
                df_input = pd.DataFrame([input_data])[self.features]
                prob = self.model.predict_proba(df_input)[0][1]
                
                failure_prob = prob
                anomaly_detected = prob > 0.5
                risk_level = "Low"
                if prob > 0.5: risk_level = "Medium"
                if prob > 0.8: risk_level = "High"
                if prob > 0.95: risk_level = "Critical"
                
                root_cause = "Predicted via Optimized XGBoost (User Data)" if anomaly_detected else None
                recommendation = "Review system logs for voltage or temperature spikes." if anomaly_detected else None
                
                return PredictionResponse(
                    timestamp=datetime.utcnow().isoformat(),
                    failure_probability=round(float(failure_prob), 4),
                    risk_level=risk_level,
                    anomaly_detected=anomaly_detected,
                    root_cause=root_cause,
                    maintenance_recommendation=recommendation
                )
            except Exception as e:
                print(f"Prediction error: {e}")

        # Fallback to Mock HEURISTIC Prediction
        failure_prob = 0.05
        risk_level = "Low"
        anomaly_detected = False
        root_cause = None
        recommendation = None

        if telemetry.grid_voltage > 250 or telemetry.grid_voltage < 210:
            failure_prob += 0.4
            risk_level = "Medium"
            anomaly_detected = True
            root_cause = "Grid Instability"
            recommendation = "Check grid synchronization and voltage range settings."
        
        if telemetry.inverter_temperature > 65:
            failure_prob += 0.5
            risk_level = "High"
            anomaly_detected = True
            root_cause = "Inverter Overheating"
            recommendation = "Check cooling fan, clean heat sinks."

        return PredictionResponse(
            timestamp=datetime.utcnow().isoformat(),
            failure_probability=round(failure_prob, 4),
            risk_level=risk_level,
            anomaly_detected=anomaly_detected,
            root_cause=root_cause,
            maintenance_recommendation=recommendation
        )


