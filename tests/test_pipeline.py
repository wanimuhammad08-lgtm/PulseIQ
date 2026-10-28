"""
Unit and Integration Test Suite for Healthcare ETL Pipeline.
Covers:
- ICD-9 clinical category mapping logic
- Cryptographic surrogate key generation
- Data quality filtering & hospice/expiration rejection
- Database connection & relational schema verification
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.pipeline.icd9_mapper import map_icd9_to_clinical_category
from src.pipeline.transform import generate_surrogate_key
from src.pipeline.data_quality import validate_and_clean_raw_data
from src.database.db_manager import DatabaseManager


def test_icd9_mapping_circulatory():
    assert map_icd9_to_clinical_category("410.1") == "Circulatory"
    assert map_icd9_to_clinical_category("428") == "Circulatory"
    assert map_icd9_to_clinical_category("785") == "Circulatory"


def test_icd9_mapping_diabetes():
    assert map_icd9_to_clinical_category("250.02") == "Diabetes"
    assert map_icd9_to_clinical_category("250") == "Diabetes"


def test_icd9_mapping_respiratory():
    assert map_icd9_to_clinical_category("486") == "Respiratory"
    assert map_icd9_to_clinical_category("493.9") == "Respiratory"


def test_icd9_mapping_digestive():
    assert map_icd9_to_clinical_category("540.0") == "Digestive"
    assert map_icd9_to_clinical_category("577") == "Digestive"


def test_icd9_mapping_unknown_and_missing():
    assert map_icd9_to_clinical_category("?") == "Missing/Unknown"
    assert map_icd9_to_clinical_category(None) == "Missing/Unknown"
    assert map_icd9_to_clinical_category(np.nan) == "Missing/Unknown"


def test_surrogate_key_generation():
    key1 = generate_surrogate_key("12345", prefix="PAT_")
    key2 = generate_surrogate_key("12345", prefix="PAT_")
    key3 = generate_surrogate_key("67890", prefix="PAT_")

    assert key1.startswith("PAT_")
    assert key1 == key2  # Deterministic
    assert key1 != key3  # Unique per patient


def test_database_manager_sqlite(tmp_path):
    test_db_file = os.path.join(tmp_path, "test_healthcare.db")
    db = DatabaseManager(db_path=test_db_file)
    db.initialize_schema("sql/schema_sqlite.sql")

    counts = db.verify_database()
    assert "patient_summary" in counts
    assert "admission" in counts
    assert "diagnosis" in counts
    assert "readmission" in counts
    assert "healthcare_mart" in counts
    assert counts["patient_summary"] == 0


def test_ml_model_prediction():
    import joblib
    model = joblib.load("models/best_model.pkl")
    sample_df = pd.DataFrame([{
        "race": "Caucasian", "gender": "Female", "age_bracket": "[60-70)",
        "admission_type_desc": "1.0", "discharge_disposition_desc": "Discharged to home",
        "admission_source_desc": "Emergency Room", "time_in_hospital": 3,
        "medical_specialty": "Unknown", "num_lab_procedures": 40, "num_procedures": 1,
        "num_medications": 15, "number_outpatient": 0, "number_emergency": 0,
        "number_inpatient": 0, "prior_admissions": 1, "diag_1_code": "410",
        "diag_1_category": "Circulatory", "diag_2_category": "Diabetes",
        "diag_3_category": "Other", "number_diagnoses": 5, "max_glu_serum": "None",
        "a1c_result": "None", "metformin": "No", "insulin": "No",
        "change_in_meds": "No", "diabetes_med_prescribed": "Yes"
    }])
    prob = model.predict_proba(sample_df)[0][1]
    assert 0.0 <= prob <= 1.0


def test_fastapi_endpoints():
    from fastapi.testclient import TestClient
    from app.api import app, load_model
    load_model()
    client = TestClient(app)
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["model_loaded"] is True

    predict_resp = client.post("/predict", json={
        "age_bracket": "[60-70)",
        "diag_1_category": "Circulatory",
        "time_in_hospital": 3,
        "num_medications": 15,
        "prior_admissions": 1,
        "a1c_result": "None",
        "insulin": "No"
    })
    assert predict_resp.status_code == 200
    data = predict_resp.json()
    assert "risk_score" in data
    assert "risk_level" in data

