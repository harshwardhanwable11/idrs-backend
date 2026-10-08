from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

# Initialize App
app = FastAPI(title="IDRS Backend Engine")

# Enable CORS for frontend connection (React/Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML Artifacts
MODEL_PATH = "models/best_rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    print("Warning: Model or scaler files not found in the 'models' directory.")

# Define the expected input data structure
class NetworkFlow(BaseModel):
    Flow_Duration: float
    Total_Fwd_Packets: float
    Bwd_Packet_Length_Mean: float
    Flow_Bytes_s: float
    Flow_Packets_s: float
    ACK_Flag_Count: float
    Average_Packet_Size: float

# Threat-to-Response Mapping Matrix
RESPONSE_MAP = {
    "BENIGN": "No action required.",
    "DDoS": "Rate-limit applied. Null-route source range.",
    "PortScan": "Temporary block of source IP executed.",
    "Bot": "Host isolated to quarantine VLAN."
}

@app.get("/")
def health_check():
    return {"status": "IDRS Backend is actively running"}

@app.post("/predict")
def predict_threat(flow: NetworkFlow):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="ML Models are not loaded on the server.")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([flow.dict()])
        
        # Apply Z-score normalization
        scaled_data = scaler.transform(input_data)
        
        # Predict threat class
        prediction = str(model.predict(scaled_data)[0])
        
        # Determine Automated Response
        action = RESPONSE_MAP.get(prediction, "Alert SOC for manual review.")
        
        return {
            "status": "success",
            "threat_detected": prediction,
            "automated_response": action
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
