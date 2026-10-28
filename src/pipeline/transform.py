"""
Transformation and Feature Engineering Module.
Fulfills Assignment Requirement:
- Create patient-safe surrogate keys using cryptographic hashes.
- Standardize diagnosis (via icd9_mapper), admission, discharge, and medication fields.
- Engineer length-of-stay, prior-admission, diagnosis-count, and medication-count variables.
- Create dimensional and fact tables:
  * patient_summary
  * admission
  * diagnosis
  * readmission
  * healthcare_mart
"""

import os
import hashlib
import logging
from typing import Dict
import numpy as np
import pandas as pd
from .icd9_mapper import map_icd9_to_clinical_category

logger = logging.getLogger(__name__)

# List of 23 monitored diabetic medications in UCI dataset
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
]

SALT = "HealthcareMLOps-2026-SecureSalt"


def generate_surrogate_key(value: str, prefix: str = "PSK_") -> str:
    """Generates a reproducible, anonymized SHA-256 surrogate key."""
    salted = f"{prefix}_{value}_{SALT}".encode("utf-8")
    return f"{prefix}{hashlib.sha256(salted).hexdigest()[:16]}"


def transform_and_create_tables(
    cleaned_df: pd.DataFrame,
    curated_dir: str = "data/curated"
) -> Dict[str, pd.DataFrame]:
    """
    Transforms cleaned encounter records into dimensional tables and analytical mart.
    """
    os.makedirs(curated_dir, exist_ok=True)
    df = cleaned_df.copy()
    logger.info(f"Transforming {len(df):,} cleaned encounter records...")

    # 1. Patient-safe surrogate keys
    df["encounter_sk"] = df["encounter_id"].astype(str).apply(lambda x: generate_surrogate_key(x, "ENC_"))
    df["patient_sk"] = df["patient_nbr"].astype(str).apply(lambda x: generate_surrogate_key(x, "PAT_"))

    # 2. Standardize Demographics
    df["race"] = df["race"].fillna("Missing / Unspecified")
    df["gender"] = df["gender"].fillna("Unknown")
    df["age_bracket"] = df["age"].astype(str)

    # 3. Engineer Variables
    # Length of stay
    df["length_of_stay"] = df["time_in_hospital"].astype(int)

    # Prior admission counts (sum of inpatient, emergency, outpatient)
    df["prior_admissions"] = (
        df["number_outpatient"].fillna(0).astype(int) +
        df["number_emergency"].fillna(0).astype(int) +
        df["number_inpatient"].fillna(0).astype(int)
    )

    # Diagnosis count
    df["diagnosis_count"] = df["number_diagnoses"].fillna(0).astype(int)

    # Active medication count (medications where value is not 'No')
    active_med_cols = [c for c in MEDICATION_COLS if c in df.columns]
    df["active_medication_count"] = df[active_med_cols].apply(
        lambda row: sum(str(v).strip().lower() != "no" for v in row), axis=1
    )

    # Medication change flag
    df["change_in_meds"] = df["change"].map(lambda x: "Yes" if str(x).strip() == "Ch" else "No")
    df["diabetes_med_prescribed"] = df["diabetesMed"].map(lambda x: "Yes" if str(x).strip() == "Yes" else "No")

    # 4. Standardize Diagnoses into Clinical Chapters
    df["diag_1_category"] = df["diag_1"].apply(map_icd9_to_clinical_category)
    df["diag_2_category"] = df["diag_2"].apply(map_icd9_to_clinical_category)
    df["diag_3_category"] = df["diag_3"].apply(map_icd9_to_clinical_category)

    df["has_diabetes_primary"] = (df["diag_1_category"] == "Diabetes").astype(int)
    df["has_circulatory_primary"] = (df["diag_1_category"] == "Circulatory").astype(int)

    # 5. Standardize Readmission Target
    # Original values: '<30', '>30', 'NO'
    raw_readmit = df["readmitted"].astype(str).str.strip()
    df["readmitted_30d"] = (raw_readmit == "<30").astype(int)
    df["readmitted_any"] = (raw_readmit.isin(["<30", ">30"])).astype(int)
    df["readmission_category"] = raw_readmit.map({
        "<30": "<30 Days (High Risk)",
        ">30": ">30 Days (Late Readmission)",
        "NO": "No Readmission"
    }).fillna("No Readmission")

    # ==================== BUILD RELATIONAL TABLES ====================

    # Table 1: patient_summary
    patient_grp = df.groupby("patient_sk")
    patient_summary = pd.DataFrame({
        "patient_sk": patient_grp["patient_sk"].first(),
        "gender": patient_grp["gender"].first(),
        "age_bracket": patient_grp["age_bracket"].last(),
        "race": patient_grp["race"].first(),
        "total_encounters": patient_grp["encounter_sk"].count(),
        "first_admission_id": patient_grp["encounter_sk"].first(),
        "last_admission_id": patient_grp["encounter_sk"].last()
    }).reset_index(drop=True)

    # Table 2: admission
    admission_table = pd.DataFrame({
        "encounter_sk": df["encounter_sk"],
        "patient_sk": df["patient_sk"],
        "admission_type_id": df["admission_type_id"],
        "admission_type_desc": df["admission_type_desc"],
        "discharge_disposition_id": df["discharge_disposition_id"],
        "discharge_disposition_desc": df["discharge_disposition_desc"],
        "admission_source_id": df["admission_source_id"],
        "admission_source_desc": df["admission_source_desc"],
        "time_in_hospital": df["length_of_stay"],
        "payer_code": df["payer_code"].fillna("Unspecified"),
        "medical_specialty": df["medical_specialty"].fillna("Unspecified"),
        "num_lab_procedures": df["num_lab_procedures"],
        "num_procedures": df["num_procedures"],
        "num_medications": df["num_medications"],
        "number_outpatient": df["number_outpatient"],
        "number_emergency": df["number_emergency"],
        "number_inpatient": df["number_inpatient"],
        "prior_admissions": df["prior_admissions"],
        "change_in_meds": df["change_in_meds"],
        "diabetes_med_prescribed": df["diabetes_med_prescribed"]
    })

    # Table 3: diagnosis
    diagnosis_table = pd.DataFrame({
        "encounter_sk": df["encounter_sk"],
        "diag_1_code": df["diag_1"].fillna("None"),
        "diag_1_category": df["diag_1_category"],
        "diag_2_code": df["diag_2"].fillna("None"),
        "diag_2_category": df["diag_2_category"],
        "diag_3_code": df["diag_3"].fillna("None"),
        "diag_3_category": df["diag_3_category"],
        "number_diagnoses": df["diagnosis_count"],
        "has_diabetes_primary": df["has_diabetes_primary"],
        "has_circulatory_primary": df["has_circulatory_primary"]
    })

    # Table 4: readmission
    readmission_table = pd.DataFrame({
        "encounter_sk": df["encounter_sk"],
        "raw_readmitted": df["readmitted"],
        "readmitted_30d": df["readmitted_30d"],
        "readmitted_any": df["readmitted_any"],
        "readmission_category": df["readmission_category"]
    })

    # Table 5: Unified Healthcare Analytical Data Mart
    healthcare_mart = pd.DataFrame({
        "encounter_sk": df["encounter_sk"],
        "patient_sk": df["patient_sk"],
        "race": df["race"],
        "gender": df["gender"],
        "age_bracket": df["age_bracket"],
        "admission_type_desc": df["admission_type_desc"],
        "discharge_disposition_desc": df["discharge_disposition_desc"],
        "admission_source_desc": df["admission_source_desc"],
        "time_in_hospital": df["length_of_stay"],
        "payer_code": df["payer_code"].fillna("Unspecified"),
        "medical_specialty": df["medical_specialty"].fillna("Unspecified"),
        "num_lab_procedures": df["num_lab_procedures"],
        "num_procedures": df["num_procedures"],
        "num_medications": df["num_medications"],
        "number_outpatient": df["number_outpatient"],
        "number_emergency": df["number_emergency"],
        "number_inpatient": df["number_inpatient"],
        "prior_admissions": df["prior_admissions"],
        "diag_1_code": df["diag_1"].fillna("None"),
        "diag_1_category": df["diag_1_category"],
        "diag_2_category": df["diag_2_category"],
        "diag_3_category": df["diag_3_category"],
        "number_diagnoses": df["diagnosis_count"],
        "max_glu_serum": df["max_glu_serum"].fillna("None"),
        "a1c_result": df["A1Cresult"].fillna("None"),
        "metformin": df["metformin"].fillna("No"),
        "insulin": df["insulin"].fillna("No"),
        "change_in_meds": df["change_in_meds"],
        "diabetes_med_prescribed": df["diabetes_med_prescribed"],
        "raw_readmitted": df["readmitted"],
        "readmitted_30d": df["readmitted_30d"],
        "readmitted_any": df["readmitted_any"],
        "readmission_category": df["readmission_category"]
    })

    tables = {
        "patient_summary": patient_summary,
        "admission": admission_table,
        "diagnosis": diagnosis_table,
        "readmission": readmission_table,
        "healthcare_mart": healthcare_mart
    }

    # Save curated CSV files
    for name, tbl in tables.items():
        out_csv = os.path.join(curated_dir, f"{name}.csv")
        tbl.to_csv(out_csv, index=False)
        logger.info(f"Curated table {name} saved to {out_csv} ({len(tbl):,} rows)")

    return tables


if __name__ == "__main__":
    from .data_quality import validate_and_clean_raw_data
    df_clean, _ = validate_and_clean_raw_data()
    transform_and_create_tables(df_clean)
