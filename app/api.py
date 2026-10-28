"""
FastAPI Prediction Service for Hospital Readmission Risk.
Serves the trained ML model via a REST API endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

# Load model at startup
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/best_model.pkl')
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. /predict will return 503.")

# Load immediately on module import for fast availability
load_model()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="PulseIQ Readmission Prediction API",
    description="Hospital 30-day readmission risk prediction service",
    version="2.0.0",
    lifespan=lifespan
)


class PatientData(BaseModel):
    """Input schema matching the healthcare_mart feature columns."""
    race: str = Field(default="Caucasian", description="Patient race/ethnicity")
    gender: str = Field(default="Female", description="Patient gender")
    age_bracket: str = Field(default="[50-60)", description="Age bracket")
    admission_type_desc: str = Field(default="1.0", description="Admission type")
    discharge_disposition_desc: str = Field(default="Discharged to home", description="Discharge disposition")
    admission_source_desc: str = Field(default="Emergency Room", description="Admission source")
    time_in_hospital: int = Field(default=3, ge=1, le=14, description="Days in hospital")
    medical_specialty: str = Field(default="Unknown", description="Treating specialty")
    num_lab_procedures: int = Field(default=40, ge=0, description="Number of lab procedures")
    num_procedures: int = Field(default=1, ge=0, description="Number of procedures")
    num_medications: int = Field(default=15, ge=0, description="Number of medications")
    number_outpatient: int = Field(default=0, ge=0, description="Prior outpatient visits")
    number_emergency: int = Field(default=0, ge=0, description="Prior emergency visits")
    number_inpatient: int = Field(default=0, ge=0, description="Prior inpatient stays")
    prior_admissions: int = Field(default=1, ge=0, description="Total prior admissions")
    diag_1_code: str = Field(default="410", description="Primary ICD-9 code")
    diag_1_category: str = Field(default="Circulatory", description="Primary diagnosis category")
    diag_2_category: str = Field(default="Diabetes", description="Secondary diagnosis category")
    diag_3_category: str = Field(default="Other", description="Tertiary diagnosis category")
    number_diagnoses: int = Field(default=5, ge=0, description="Number of diagnoses")
    max_glu_serum: str = Field(default="None", description="Glucose serum test result")
    a1c_result: str = Field(default="None", description="A1C result")
    metformin: str = Field(default="No", description="Metformin prescription")
    insulin: str = Field(default="No", description="Insulin prescription")
    change_in_meds: str = Field(default="No", description="Medication change flag")
    diabetes_med_prescribed: str = Field(default="Yes", description="Diabetes medication prescribed")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict")
def predict(data: PatientData):
    """Predict 30-day readmission risk for a patient."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with: python src/mlops/train.py"
        )

    try:
        # Convert input to DataFrame matching training schema
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])

        # Predict probability
        prob = model.predict_proba(df)[0][1]
        pred = model.predict(df)[0]

        # Risk categorization
        if prob >= 0.7:
            risk_level = "Critical"
        elif prob >= 0.5:
            risk_level = "High"
        elif prob >= 0.3:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return {
            "risk_score": round(float(prob), 4),
            "risk_level": risk_level,
            "prediction": int(pred),
            "input_summary": {
                "age": data.age_bracket,
                "diagnosis": data.diag_1_category,
                "stay_days": data.time_in_hospital,
                "prior_admissions": data.prior_admissions
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
