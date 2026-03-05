from pydantic import BaseModel
from typing import Optional
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
    # Extra fields for ML model
    pv1_power: Optional[float] = 0.0
    pv2_power: Optional[float] = 0.0
    pv1_voltage: Optional[float] = 0.0
    pv2_voltage: Optional[float] = 0.0
    grid_active_power: Optional[float] = 0.0

class PredictionResponse(BaseModel):
    timestamp: str
    failure_probability: float
    risk_level: str
    anomaly_detected: bool
    root_cause: Optional[str]
    maintenance_recommendation: Optional[str]
