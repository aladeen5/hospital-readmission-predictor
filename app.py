# Hospital Readmission Predictor (FastAPI + HTML Dashboard)

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Initialize app
app = FastAPI(title="Hospital Readmission Predictor",
              description="Predicts 30-day readmission risk for discharged patients",
              version="1.0")

# Load trained model
model_path = "readmission_model.pkl"
clf = joblib.load(model_path)

# Set up HTML templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Simple JSON home endpoint
@app.get("/")
def home():
    return {"message": "✅ Hospital Readmission Predictor API is running. Visit /dashboard to use the web form or /docs to test the API."}


# 🧾 Dashboard route (HTML)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


# 📊 Predict from dashboard form
@app.post("/dashboard", response_class=HTMLResponse)
async def predict_dashboard(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    blood_pressure: str = Form(...),
    cholesterol: int = Form(...),
    bmi: float = Form(...),
    diabetes: str = Form(...),
    hypertension: str = Form(...),
    medication_count: int = Form(...),
    length_of_stay: int = Form(...),
    discharge_destination: str = Form(...)
):
    df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "blood_pressure": blood_pressure,
        "cholesterol": cholesterol,
        "bmi": bmi,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "medication_count": medication_count,
        "length_of_stay": length_of_stay,
        "discharge_destination": discharge_destination
    }])

    # Parse blood pressure
    try:
        systolic, diastolic = map(int, blood_pressure.split("/"))
    except:
        systolic, diastolic = (None, None)
    df["systolic"] = systolic
    df["diastolic"] = diastolic

    # Derived features
    bins = [0, 18, 35, 50, 65, 80, 120]
    labels = ["0-18", "19-35", "36-50", "51-65", "66-80", "81+"]
    df["age_group"] = pd.cut([age], bins=bins, labels=labels)[0]

    def bmi_cat(b):
        if b < 18.5: return "underweight"
        if b < 25: return "normal"
        if b < 30: return "overweight"
        return "obese"
    df["bmi_cat"] = df["bmi"].apply(bmi_cat)

    # Predict probability
    prob = clf.predict_proba(df)[0][1]
    prediction = "High risk" if prob >= 0.5 else "Low risk"

    result = {
        "prediction": prediction,
        "probability": round(float(prob), 3)
    }

    return templates.TemplateResponse("index.html", {"request": request, "result": result})


# 🧠 API version (for JSON requests)
class PatientData(BaseModel):
    age: int
    gender: str
    blood_pressure: str
    cholesterol: int
    bmi: float
    diabetes: str
    hypertension: str
    medication_count: int
    length_of_stay: int
    discharge_destination: str


@app.post("/predict")
def predict_readmission(data: PatientData):
    df = pd.DataFrame([data.dict()])

    try:
        systolic, diastolic = map(int, data.blood_pressure.split("/"))
    except:
        systolic, diastolic = (None, None)
    df["systolic"] = systolic
    df["diastolic"] = diastolic

    bins = [0, 18, 35, 50, 65, 80, 120]
    labels = ["0-18", "19-35", "36-50", "51-65", "66-80", "81+"]
    df["age_group"] = pd.cut([df["age"].iloc[0]], bins=bins, labels=labels)[0]

    def bmi_cat(b):
        if b < 18.5: return "underweight"
        if b < 25: return "normal"
        if b < 30: return "overweight"
        return "obese"
    df["bmi_cat"] = df["bmi"].apply(bmi_cat)

    prob = clf.predict_proba(df)[0][1]
    prediction = "High risk" if prob >= 0.5 else "Low risk"

    return {
        "prediction": prediction,
        "readmission_probability": round(float(prob), 3)
    }

# Run with: uvicorn app:app --reload
