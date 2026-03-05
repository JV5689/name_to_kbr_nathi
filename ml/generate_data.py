import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_solar_data(num_samples=1000):
    start_time = datetime.now()
    data = []
    
    for i in range(num_samples):
        timestamp = start_time + timedelta(minutes=5*i)
        
        # Environmental factors
        ambient_temp = np.random.normal(25, 5)
        irradiation = max(0, np.random.normal(500, 200) * (1 if 6 <= timestamp.hour <= 18 else 0))
        
        # Inverter metrics
        dc_voltage = np.random.normal(400, 20) if irradiation > 0 else 0
        dc_current = (irradiation / dc_voltage * 10) if dc_voltage > 0 else 0
        dc_power = dc_voltage * dc_current
        
        grid_voltage = np.random.normal(230, 5)
        inverter_temp = ambient_temp + (dc_power / 1000) * 5 + np.random.normal(2, 1)
        
        # Efficiency
        efficiency = 0.96 + np.random.normal(0, 0.01)
        ac_power = dc_power * efficiency
        
        # Target: Failure (simulate anomalies)
        failure = 0
        if inverter_temp > 75: # Overheating
            failure = 1
        if grid_voltage > 260 or grid_voltage < 190: # Grid instability
            failure = 1
        if dc_power > 0 and efficiency < 0.85: # Efficiency drop
            failure = 1
            
        data.append({
            'timestamp': timestamp,
            'ambient_temp': ambient_temp,
            'irradiation': irradiation,
            'dc_voltage': dc_voltage,
            'dc_current': dc_current,
            'dc_power': dc_power,
            'grid_voltage': grid_voltage,
            'inverter_temp': inverter_temp,
            'efficiency': efficiency,
            'ac_power': ac_power,
            'failure': failure
        })
        
    df = pd.DataFrame(data)
    
    # Feature Engineering (Derived features from ChatGPT chat)
    df['temp_diff'] = df['inverter_temp'] - df['ambient_temp']
    df['power_ratio'] = df['ac_power'] / (df['dc_power'] + 0.1)
    df['voltage_stability'] = df['grid_voltage'].rolling(window=5).std().fillna(0)
    
    df.to_csv('solar_telemetry_data.csv', index=False)
    print(f"Generated {num_samples} samples in solar_telemetry_data.csv")

if __name__ == "__main__":
    generate_solar_data()
