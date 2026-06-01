from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
# ── Load model and supporting files ──────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

scaler = joblib.load(scaler_path)
feature_columns_path = os.path.join(BASE_DIR, "feature_columns.pkl")

feature_columns = joblib.load(feature_columns_path)

# ── Create the app ────────────────────────────────────────────────────────────
app = FastAPI(title="Churn Predictor API")

# Allow frontend to talk to this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Define what input the API expects ────────────────────────────────────────
class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: str        # "Month-to-month", "One year", "Two year"
    InternetService: str # "Fiber optic", "DSL", "No"
    PaymentMethod: str   # "Electronic check", "Mailed check", etc.
    gender: str          # "Male" or "Female"
    SeniorCitizen: int   # 0 or 1
    Partner: int         # 0 or 1
    Dependents: int      # 0 or 1
    PhoneService: int    # 0 or 1
    PaperlessBilling: int# 0 or 1
    OnlineSecurity: int  # 0 or 1
    OnlineBackup: int    # 0 or 1
    DeviceProtection: int# 0 or 1
    TechSupport: int     # 0 or 1
    StreamingTV: int     # 0 or 1
    StreamingMovies: int # 0 or 1
    MultipleLines: int   # 0 or 1

# ── Health check endpoint ─────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "API is running"}

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: CustomerData):

    # Step 1: convert input to dataframe
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Step 2: encode categorical columns same way as training
    df['gender']  = df['gender'].map({'Male': 1, 'Female': 0})

    df = pd.get_dummies(df, columns=['Contract',
                                     'InternetService',
                                     'PaymentMethod'])

    # Step 3: add any missing columns (model needs exact columns)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Step 4: reorder columns to match training order
    df = df[feature_columns]

    # Step 5: scale numerical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])

    # Step 6: predict
    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction":  "Churn" if prediction == 1 else "No Churn",
        "probability": round(float(probability), 4),
        "risk_level":  "High" if probability > 0.7
                       else "Medium" if probability > 0.4
                       else "Low"
    }