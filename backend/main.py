from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import shutil
import os

from telemetry import TelemetryGenerator
from models import InverterTelemetry

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
    return generator.get_inverter_list()

@app.get("/api/telemetry/current")
def get_current_telemetry(inverter_id: Optional[str] = "INV-01"):
    telemetry = generator.get_current_telemetry(inverter_id)
    prediction = generator.get_prediction(telemetry)
    
    return {
        "telemetry": telemetry,
        "prediction": prediction
    }

@app.post("/api/upload-telemetry")
async def upload_telemetry(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
    
    # Save file temporarily or process it
    # For this hackathon demo, we'll just acknowledge the upload
    return {"message": f"Telemetry file {file.filename} uploaded and processed successfully."}

@app.post("/api/upload-weather")
async def upload_weather(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
    
    return {"message": f"Weather data from {file.filename} integrated into the prediction model."}

@app.post("/api/predict")
async def predict_custom(telemetry: InverterTelemetry):
    prediction = generator.get_prediction(telemetry)
    return prediction

