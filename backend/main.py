from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import shutil
import os
import asyncio
import pandas as pd

from telemetry import TelemetryGenerator
from models import InverterTelemetry, AppSettings

app = FastAPI(title="SolarIQ Pulse API")

# Initialize telemetry generator
generator = TelemetryGenerator()

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "SolarIQ Pulse API is active"}

@app.get("/api/inverters")
def get_inverters():
    return generator.get_inverter_list()

@app.get("/api/settings")
def get_settings():
    return generator.settings

@app.post("/api/settings")
def update_settings(settings: AppSettings):
    generator.save_settings(settings)
    return {"message": "Settings updated successfully"}

@app.get("/api/telemetry/current")
def get_current_telemetry(inverter_id: Optional[str] = "INV-01"):
    telemetry = generator.get_current_telemetry(inverter_id)
    prediction = generator.get_prediction(telemetry, inverter_id)
    
    return {
        "telemetry": telemetry,
        "prediction": prediction
    }

@app.get("/api/telemetry/history")
def get_telemetry_history(inverter_id: str = "INV-01"):
    return generator.get_history(inverter_id)

@app.websocket("/ws/telemetry/{inverter_id}")
async def websocket_endpoint(websocket: WebSocket, inverter_id: str):
    await websocket.accept()
    try:
        while True:
            # Respect user settings for refresh rate
            telemetry = generator.get_current_telemetry(inverter_id)
            prediction = generator.get_prediction(telemetry, inverter_id)
            
            await websocket.send_json({
                "telemetry": telemetry.dict(),
                "prediction": prediction.dict()
            })
            
            await asyncio.sleep(generator.settings.refresh_rate_ms / 1000.0)
    except WebSocketDisconnect:
        print(f"Client disconnected from {inverter_id}")
    except Exception as e:
        print(f"WS Error: {e}")

import io

@app.post("/api/dataset/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
    
    try:
        # Read file content safely
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        if df.empty:
            return {"status": "Error", "message": "The uploaded CSV file is empty."}
            
        results = generator.analyze_batch(df)
        return results
    except Exception as e:
        print(f"Dataset Analysis Detail: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/telemetry/ai-diagnostic")
def get_ai_diagnostic(inverter_id: str = "INV-01"):
    hist = generator.get_history(inverter_id)
    if not hist:
        return {"diagnostic": "No data available."}
    
    latest = hist[-1]
    trends = generator._calculate_trends(inverter_id)
    diagnostic = generator._get_ai_diagnostic(inverter_id, latest, trends)
    return {"diagnostic": diagnostic}

@app.post("/api/upload-telemetry")
async def upload_telemetry(file: UploadFile = File(...)):
    return {"message": "Telemtry processing skipped; system is in REAL-TIME mode."}

