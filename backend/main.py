from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from telemetry import TelemetryGenerator

app = FastAPI(title="Solar Inverter Failure Prediction API")

# Initialize telemetry generator
generator = TelemetryGenerator()

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Solar Inverter Failure Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/inverters")
def get_inverters():
    """
    Returns a list of available inverter IDs.
    """
    return generator.get_inverter_list()

@app.get("/api/telemetry/current")
def get_current_telemetry(inverter_id: Optional[str] = "INV-01"):
    """
    Returns current telemetry and prediction for a specific inverter.
    """
    telemetry = generator.get_current_telemetry(inverter_id)
    prediction = generator.get_prediction(telemetry)
    
    return {
        "telemetry": telemetry,
        "prediction": prediction
    }
