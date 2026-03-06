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

from datetime import datetime, timedelta
import collections
from collections import deque
from typing import Dict, List, Optional
import google.generativeai as genai
from models import InverterTelemetry, PredictionResponse, AppSettings

class TelemetryGenerator:
    def __init__(self):
        # Base realistic values
        self.base_voltage = 230.0
        self.base_frequency = 50.0
        
        # Settings management
        self.settings_path = 'settings.json'
        self.settings = self._load_settings()
        
        # History per inverter (last 48h assuming 2.5s intervals is too many, so we'll store every 10th sample or similar for long trends)
        # For this demo, we'll store last 500 points (~20 mins at 2.5s) and simulate a 48h "virtual" trend
        self.history = {
            "INV-01": deque(maxlen=1000),
            "INV-02": deque(maxlen=1000),
            "INV-03": deque(maxlen=1000)
        }
        
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
        
        self._load_model()
        self._init_genai()

    def _load_settings(self) -> AppSettings:
        if os.path.exists(self.settings_path) and os.path.getsize(self.settings_path) > 0:
            try:
                with open(self.settings_path, 'r') as f:
                    data = json.load(f)
                    return AppSettings(**data)
            except Exception as e:
                print(f"Error loading settings: {e}")
        return AppSettings()

    def save_settings(self, settings: AppSettings):
        self.settings = settings
        with open(self.settings_path, 'w') as f:
            # Use model_dump_json for Pydantic v2 compatibility
            # Fallback to json() for older environments if needed
            try:
                f.write(settings.model_dump_json(indent=4))
            except AttributeError:
                f.write(settings.json(indent=4))
        self._init_genai()

    def _init_genai(self):
        if self.settings.gemini_api_key:
            try:
                genai.configure(api_key=self.settings.gemini_api_key)
                self.genai_model = genai.GenerativeModel('gemini-pro')
                print("Gemini AI initialized.")
            except Exception as e:
                print(f"GenAI Init Error: {e}")
                self.genai_model = None
        else:
            self.genai_model = None

    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.meta_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                with open(self.meta_path, 'r') as f:
                    meta = json.load(f)
                    self.features = meta.get('original_features', []) + meta.get('engineered_features', [])
                    if not self.features:
                        self.features = meta.get('features', [])
                
                if os.path.exists(self.anomaly_path):
                    self.anomaly_model = joblib.load(self.anomaly_path)
                print(f"ML Models loaded. Feature count: {len(self.features)}")
            except Exception as e:
                print(f"Error loading model: {e}")

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
        
        # Simulate a gradual degradation trend for INV-02
        degradation = 1.0 if inverter_id != "INV-02" else max(0.7, 1.0 - (counter / 1000.0))
        
        pv1_power = ((2500.0 * irradiation_factor * power_scale) + (noise * 50)) * degradation
        pv2_power = ((2500.0 * irradiation_factor * power_scale) + (noise * 50)) * degradation
        pv1_voltage = 400.0 + (noise * 5) if pv1_power > 0 else 0
        pv2_voltage = 400.0 + (noise * 5) if pv2_power > 0 else 0
        
        # Simulate grid instability for INV-03
        grid_noise_scale = 1.0 if inverter_id != "INV-03" else 15.0
        voltage = self.base_voltage + (math.sin(counter / 10.0) * 2) + (noise * grid_noise_scale)
        frequency = self.base_frequency + rng.uniform(-0.1, 0.1)
        
        # Weather simulation
        ambient_temp = 25.0 + rng.uniform(-2, 2)
        inverter_temp = ambient_temp + (((pv1_power + pv2_power) / 1000.0) * 5) + noise
        
        # Simulate overheating for INV-01 every 100 cycles
        if inverter_id == "INV-01" and 50 < (counter % 100) < 70:
            inverter_temp += 20.0 # Fan failure simulation

        efficiency = 0.96 + rng.uniform(-0.01, 0.005)
        grid_active_power = (pv1_power + pv2_power) * efficiency

        telemetry = InverterTelemetry(
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
            dc_current=round((pv1_power + pv2_power) / max(1, pv1_voltage + pv2_voltage), 2),
            power_factor=0.98,
            kwh_total=12500.0 + counter * 0.5,
            kwh_today=15.0 + counter * 0.1
        )
        
        # Update history
        self.history[inverter_id].append(telemetry.dict())
        return telemetry

    def get_history(self, inverter_id: str) -> List[Dict]:
        return list(self.history.get(inverter_id, []))

    def _calculate_trends(self, inverter_id: str) -> Dict:
        hist = list(self.history.get(inverter_id, []))
        if len(hist) < 10:
            return {"status": "Insufficient data for trend analysis"}
        
        latest = hist[-1]
        prev_avg = {
            "efficiency": np.mean([h['inverter_efficiency'] for h in hist[-20:]]),
            "temp": np.mean([h['inverter_temperature'] for h in hist[-20:]]),
            "voltage": np.mean([h['grid_voltage'] for h in hist[-20:]])
        }
        
        return {
            "efficiency_delta": round(latest['inverter_efficiency'] - prev_avg['efficiency'], 2),
            "temp_trend": "Increasing" if latest['inverter_temperature'] > prev_avg['temp'] + 2 else "Stable",
            "voltage_stability": "High" if np.std([h['grid_voltage'] for h in hist[-20:]]) < 2 else "Low",
            "samples_count": len(hist)
        }

    def get_prediction(self, telemetry: InverterTelemetry, inverter_id: str = "INV-01") -> PredictionResponse:
        t_dict = telemetry.dict()
        trends = self._calculate_trends(inverter_id)
        
        # Mapping to ML features
        # Note: We must sanitize names to match sanitized model features
        input_data = {
            'ac_power': t_dict.get('ac_power', 0),
            'dc_voltage': t_dict.get('pv1_voltage', 0) + t_dict.get('pv2_voltage', 0),
            'dc_current': t_dict.get('dc_current', 0),
            'inverter_temp': t_dict.get('inverter_temperature', 0),
            'grid_voltage': t_dict.get('grid_voltage', 0),
            'grid_frequency': t_dict.get('grid_frequency', 0),
            'efficiency': (t_dict.get('inverter_efficiency', 0) / 100.0) if t_dict.get('inverter_efficiency', 0) > 0 else 0.96,
            'dc_power': t_dict.get('dc_power', 0),
            'power_loss': t_dict.get('dc_power', 0) - t_dict.get('ac_power', 0),
            'voltage_deviation': abs(t_dict.get('grid_voltage', 230) - 230),
            'frequency_deviation': abs(t_dict.get('grid_frequency', 50) - 50),
            'hour': datetime.utcnow().hour,
            'day_of_week': datetime.utcnow().weekday(),
            'month': datetime.utcnow().month
        }
        
        final_features = []
        for feat in self.features:
            val = input_data.get(feat, t_dict.get(feat, 0.0))
            final_features.append(val)

        df_input = pd.DataFrame([final_features], columns=self.features)
        
        failure_prob = 0.05
        anomaly_detected = False
        reasons = []

        if self.model:
            prob = float(self.model.predict_proba(df_input)[0][1])
            failure_prob = prob
            if self.anomaly_model:
                anomaly_detected = self.anomaly_model.predict(df_input)[0] == -1

        # Heuristic Reasons based on trends
        if trends.get("temp_trend") == "Increasing": reasons.append("Upward temperature trend detected")
        if trends.get("efficiency_delta", 0) < -2: reasons.append("Significant efficiency drop in last 5 mins")
        if t_dict['inverter_temperature'] > 45: reasons.append("Critical thermal threshold exceeded")

        risk_level = "Low"
        for level, thresh in sorted(self.settings.risk_thresholds.items(), key=lambda x: x[1], reverse=True):
            if failure_prob >= thresh:
                risk_level = level
                break

        ai_analysis = None
        if (risk_level != "Low" or anomaly_detected) and self.genai_model:
            ai_analysis = self._get_ai_diagnostic(inverter_id, t_dict, trends)

        return PredictionResponse(
            timestamp=datetime.utcnow().isoformat(),
            failure_probability=round(failure_prob, 4),
            risk_level=risk_level,
            anomaly_detected=anomaly_detected or (failure_prob > 0.7),
            mode="internal",
            reasons=reasons if reasons else ["Operating within normal parameters"],
            root_cause=reasons[0] if reasons else "Stable",
            maintenance_recommendation="Contact technician for inspection" if failure_prob > 0.6 else "Monitor trends",
            trend_analysis=trends,
            ai_analysis=ai_analysis
        )

    def _get_ai_diagnostic(self, inverter_id: str, telemetry: dict, trends: dict) -> str:
        if not self.genai_model:
            return "AI Analysis unavailable (Gemini not initialized)"
        
        prompt = f"""
        Analyze solar inverter telemetry for root causes.
        Inverter: {inverter_id}
        Metrics: {telemetry}
        Trends: {trends}
        
        Provide a concise (2-3 sentence) technical diagnosis of potential issues or confirmation of healthy operation.
        """
        try:
            response = self.genai_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI Analysis failed: {e}"

    def analyze_batch(self, df: pd.DataFrame) -> Dict:
        """
        Analyzes a batch of historical data (CSV).
        Performs feature mapping, validation, and batch prediction.
        """
        try:
            # 0. Preparation: Sanitize original column names to handle [0] and other symbols
            # This must match the train_model.py logic
            def sanitize_col(name):
                return str(name).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            
            df.columns = [sanitize_col(c) for c in df.columns]
            
            # 1. Identification & Mapping (Hardware -> Neutral names)
            # Based on feature_map in train_model.py (REVISED TO MATCH EXACTLY)
            feature_map = {
                'inverters_0_.pv1_power': 'ac_power', # Mapped to ac_power in train_model
                'inverters_0_.pv1_voltage': 'dc_voltage',
                'inverters_0_.pv1_current': 'dc_current',
                'inverters_0_.temp': 'inverter_temp',
                'meters_0_.v_r': 'grid_voltage',
                'meters_0_.freq': 'grid_frequency',
                'meters_0_.pf': 'power_factor',
                'inverters_0_.kwh_total': 'kwh_total',
                'inverters_0_.kwh_today': 'kwh_today',
                'inverters_0_.alarm_code': 'alarm_code'
            }
            
            # Identify columns that exist and rename them
            df = df.rename(columns={k: v for k, v in feature_map.items() if k in df.columns})
            
            # Step 2: Ensure critical columns exist (at least as pillars of features)
            # This mirrors get_prediction and train_model.py logic
            if 'dc_power' not in df.columns:
                # Need consistent series/scalar addition
                pv1 = df.get('pv1_power', pd.Series(0, index=df.index))
                pv2 = df.get('pv2_power', pd.Series(0, index=df.index))
                df['dc_power'] = pv1 + pv2
                
            if 'ac_power' not in df.columns:
                # If we didn't map it from hardware, use heuristic or filler
                df['ac_power'] = df['dc_power'] * 0.96
                
            if 'power_loss' not in df.columns:
                df['power_loss'] = df['dc_power'] - df['ac_power']
                
            if 'efficiency' not in df.columns:
                df['efficiency'] = np.where(df['dc_power'] > 0, df['ac_power'] / df['dc_power'], 0)
                df['efficiency'] = df['efficiency'].clip(0, 1.1)

            if 'thermal_stress' not in df.columns:
                df['thermal_stress'] = df.get('inverter_temp', 0)
                
            if 'voltage_deviation' not in df.columns:
                df['voltage_deviation'] = np.abs(df.get('grid_voltage', 230) - 230)
                
            if 'frequency_deviation' not in df.columns:
                df['frequency_deviation'] = np.abs(df.get('grid_frequency', 50) - 50)

            # Time features
            if 'timestamp' in df.columns:
                ts = pd.to_datetime(df['timestamp'], errors='coerce')
                df['hour'] = ts.dt.hour.fillna(12)
                df['day_of_week'] = ts.dt.dayofweek.fillna(0)
                df['month'] = ts.dt.month.fillna(1)
            else:
                df['hour'], df['day_of_week'], df['month'] = 12, 0, 1

            # Step 3: Feature Validation (What is missing from the 71 original features?)
            missing_critical = []
            if self.features:
                missing_critical = [s for s in self.features if s not in df.columns]
                # We can fulfill the requirement of skipping data if features are missing
                # But we'll try to predict with defaults (0) for non-critical ones first.
            
            # Skip Logic (Dropping rows with critical NAs)
            original_count = len(df)
            # Columns absolutely needed for calculation
            clean_check_cols = [c for c in ['dc_power', 'dc_voltage', 'inverter_temp', 'grid_voltage'] if c in df.columns]
            df = df.dropna(subset=clean_check_cols)
            skipped_count = original_count - len(df)
            
            if len(df) == 0:
                return {
                    "status": "Error",
                    "message": "Dataset empty or all rows skipped. Ensure your columns match hardware labels (e.g. inverters[0].temp).",
                    "missing_signals": missing_critical[:10] # Show top missing
                }

            # Step 4: Batch Prediction
            results = []
            if self.model:
                # Reindex to match EXACT model signature (71+ features)
                X = df.reindex(columns=self.features, fill_value=0.0)
                
                # Coerce X to float to ensure no object types
                X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
                
                probs = self.model.predict_proba(X)[:, 1].tolist()
                anomalies = []
                if self.anomaly_model:
                    anomalies = (self.anomaly_model.predict(X) == -1).tolist()
                else:
                    anomalies = [p > 0.8 for p in probs]

                df['prob'] = probs
                df['is_anomaly'] = anomalies
                
                # Pick up to 50 interesting rows (anomalies or high risk)
                # Sort by risk DESC to show most important first
                highlight_df = df[(df['prob'] > 0.3) | (df['is_anomaly'] == True)].sort_values('prob', ascending=False).head(50)
                
                for idx, row in highlight_df.iterrows():
                    results.append({
                        "timestamp": str(row.get('timestamp', idx)),
                        "failure_probability": round(float(row['prob']), 4),
                        "risk_level": "High" if row['prob'] > 0.7 else ("Medium" if row['prob'] > 0.4 else "Low"),
                        "anomaly_detected": bool(row['is_anomaly']),
                        "dc_power": round(float(row.get('dc_power', 0)), 2),
                        "temp": round(float(row.get('inverter_temp', 0)), 2)
                    })

            return {
                "status": "Success",
                "summary": {
                    "total_rows": original_count,
                    "processed_rows": len(df),
                    "skipped_rows": skipped_count,
                    "critical_points_found": len(results),
                    "missing_signals": missing_critical[:5]
                },
                "highlights": results
            }
        except Exception as e:
            import traceback
            print(f"Barch Analysis Error: {str(e)}\n{traceback.format_exc()}")
            return {
                "status": "Error",
                "message": f"Critical internal error during batch analysis: {str(e)}"
            }



