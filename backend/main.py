from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load model and feature columns only (skip broken scaler)
model           = joblib.load('model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

app = FastAPI(title="Churn Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Hardcoded scaling values from training data
SCALE_PARAMS = {
    'tenure':         {'mean': 32.37, 'std': 24.56},
    'MonthlyCharges': {'mean': 64.76, 'std': 30.09},
    'TotalCharges':   {'mean': 2283.30, 'std': 2266.77}
}

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str
    gender: str
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    PaperlessBilling: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    MultipleLines: int

@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Encode gender
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        # One-hot encode
        df = pd.get_dummies(df, columns=['Contract', 'InternetService', 'PaymentMethod'])

        # Add missing columns
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Scale manually using training stats
        for col, params in SCALE_PARAMS.items():
            if col in df.columns:
                df[col] = (df[col] - params['mean']) / params['std']

        # Predict
        prediction  = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "prediction":  "Churn" if prediction == 1 else "No Churn",
            "probability": round(float(probability), 4),
            "risk_level":  "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
