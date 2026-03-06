import pandas as pd
import numpy as np

def merge_external_weather(internal_df, weather_df):
    """
    Utility to merge internal telemetry with external weather data using timestamps.
    Follows the architecture requirement for future data expansion.
    """
    if 'timestamp' not in internal_df.columns or 'timestamp' not in weather_df.columns:
        print("Warning: Timestamp missing for merge. Skipping.")
        return internal_df
        
    # Ensure datetime format
    internal_df['timestamp'] = pd.to_datetime(internal_df['timestamp'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
    # Sort for merge_asof if needed, or standard merge
    internal_df = internal_df.sort_values('timestamp')
    weather_df = weather_df.sort_values('timestamp')
    
    # Merge on closest timestamp (common in telemetry)
    merged_df = pd.merge_asof(
        internal_df, 
        weather_df, 
        on='timestamp', 
        direction='nearest',
        tolerance=pd.Timedelta('1 hour')
    )
    
    return merged_df

def engineer_weather_risk(df):
    """
    Creates additional engineered features from weather data.
    """
    # Placeholder for features mentioned in the prompt
    if 'lightning_count' in df.columns:
        df['lightning_risk'] = df['lightning_count'].rolling(window=3).mean()
        
    if 'wind_speed' in df.columns and 'storm_probability' in df.columns:
        df['storm_severity'] = df['wind_speed'] * df['storm_probability']
        
    return df
