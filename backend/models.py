from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class InverterTelemetry(BaseModel):
    timestamp: str
    grid_voltage: float
    grid_frequency: float
    dc_power: float
    ac_power: float
    inverter_temperature: float
    ambient_temperature: float
    inverter_efficiency: float
    # External Weather (Mode 2) - Kept for flexibility but defaulted to 0
    cloud_cover: Optional[float] = 0.0
    rainfall: Optional[float] = 0.0
    wind_speed: Optional[float] = 0.0
    storm_probability: Optional[float] = 0.0
    # Extra fields for ML model
    pv1_power: Optional[float] = 0.0
    pv2_power: Optional[float] = 0.0
    pv1_voltage: Optional[float] = 0.0
    pv2_voltage: Optional[float] = 0.0
    grid_active_power: Optional[float] = 0.0
    # New features for parity
    dc_current: Optional[float] = 0.0
    power_factor: Optional[float] = 1.0
    kwh_total: Optional[float] = 0.0
    kwh_today: Optional[float] = 0.0

class PredictionResponse(BaseModel):
    timestamp: str
    failure_probability: float
    risk_level: str
    anomaly_detected: bool
    mode: str  # "internal" or "internal+external"
    reasons: List[str]
    root_cause: Optional[str]
    maintenance_recommendation: Optional[str]
    # Trend and AI Insights
    trend_analysis: Optional[dict] = None
    ai_analysis: Optional[str] = None

class AppSettings(BaseModel):
    gemini_api_key: str = ""
    trend_window_hours: int = 48
    refresh_rate_ms: int = 2500
    risk_thresholds: Dict[str, float] = {
        "Medium": 0.5,
        "High": 0.8,
        "Critical": 0.95
    }
    simulation_speed: float = 1.0

