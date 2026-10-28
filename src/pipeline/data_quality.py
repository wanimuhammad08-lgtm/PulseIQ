"""
Data Quality and Validation Module.
Fulfills Assignment Requirement:
- Identify and replace special missing-value codes ('?', 'None', 'Unknown/Invalid').
- Filter invalid encounters (expired, hospice discharge).
- Log rejected records with explicit rejection reasons to data/rejected/rejected_records.csv.
- Generate structured data quality metric reports.
"""

import os
import json
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Clinical discharge dispositions representing patient expiration or hospice:
# ID 11: Expired
# ID 13: Hospice / home
# ID 14: Hospice / medical facility
# ID 19: Expired at home
# ID 20: Expired in medical facility
# ID 21: Expired, place unknown
INVALID_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]


def parse_id_mapping(mapping_path: str) -> Dict[str, Dict[int, str]]:
    """
    Parses IDs_mapping.csv into dictionaries for admission_type,
    discharge_disposition, and admission_source.
    """
    mappings = {
        "admission_type": {},
        "discharge_disposition": {},
        "admission_source": {}
    }
    if not os.path.exists(mapping_path):
        return mappings

    current_section = None
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "admission_type_id" in line:
                    current_section = "admission_type"
                    continue
                elif "discharge_disposition_id" in line:
                    current_section = "discharge_disposition"
                    continue
                elif "admission_source_id" in line:
                    current_section = "admission_source"
                    continue

                if current_section and "," in line:
                    parts = line.split(",", 1)
                    try:
                        id_val = int(parts[0].strip())
                        desc_val = parts[1].strip() if len(parts) > 1 else "Unknown"
                        mappings[current_section][id_val] = desc_val
                    except ValueError:
                        pass
    except Exception as e:
        logger.warning(f"Failed to parse ID mappings ({e}), using default fallback labels.")

    return mappings


def validate_and_clean_raw_data(
    raw_csv_path: str = "data/raw/diabetic_data.csv",
    mapping_csv_path: str = "data/raw/IDs_mapping.csv",
    staging_dir: str = "data/staging",
    rejected_dir: str = "data/rejected"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates, handles missing values, filters invalid records, and logs rejected encounters.
    Returns (cleaned_df, quality_metrics_dict).
    """
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    logger.info(f"Loading raw dataset from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path, low_memory=False)
    initial_row_count = len(df_raw)

    rejected_records = []

    # 1. Check & Replace special missing value codes
    # In UCI Diabetes dataset, '?' represents missing
    df_clean = df_raw.copy()
    missing_value_indicators = ["?", "Unknown/Invalid", "None"]
    
    # Track missingness per column
    missingness_stats = {}
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            null_count = int(df_clean[col].isin(missing_value_indicators).sum() + df_clean[col].isna().sum())
        else:
            null_count = int(df_clean[col].isna().sum())
        if null_count > 0:
            missingness_stats[col] = {
                "null_count": null_count,
                "null_percentage": round((null_count / initial_row_count) * 100, 2)
            }

    # Standardize '?' to np.nan for object columns
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].replace(["?", "None", "nan", "NAN"], np.nan)

    # 2. Check for invalid gender ('Unknown/Invalid')
    invalid_gender_mask = df_clean["gender"].isin(["Unknown/Invalid", np.nan])
    if invalid_gender_mask.sum() > 0:
        for idx in df_clean[invalid_gender_mask].index:
            rec = df_raw.loc[idx].to_dict()
            rec["rejection_reason"] = "Invalid or Missing Gender"
            rejected_records.append(rec)
        df_clean = df_clean[~invalid_gender_mask]

    # 3. Filter invalid encounters: Expired or Hospice Discharges
    # Clinically, deceased patients cannot experience readmissions
    invalid_discharge_mask = df_clean["discharge_disposition_id"].isin(INVALID_DISCHARGE_IDS)
    if invalid_discharge_mask.sum() > 0:
        for idx in df_clean[invalid_discharge_mask].index:
            rec = df_raw.loc[idx].to_dict()
            rec["rejection_reason"] = f"Patient Expired or Discharged to Hospice (Discharge ID: {rec.get('discharge_disposition_id')})"
            rejected_records.append(rec)
        df_clean = df_clean[~invalid_discharge_mask]

    # 4. Check for duplicate encounters
    dup_mask = df_clean.duplicated(subset=["encounter_id"], keep="first")
    if dup_mask.sum() > 0:
        for idx in df_clean[dup_mask].index:
            rec = df_raw.loc[idx].to_dict()
            rec["rejection_reason"] = "Duplicate Encounter ID"
            rejected_records.append(rec)
        df_clean = df_clean[~dup_mask]

    # 5. Save rejected records
    rejected_df = pd.DataFrame(rejected_records)
    rejected_file = os.path.join(rejected_dir, "rejected_records.csv")
    rejected_df.to_csv(rejected_file, index=False)
    logger.info(f"Rejected {len(rejected_records):,} invalid records. Saved log to {rejected_file}")

    # 6. Parse ID Mapping descriptions and enrich columns
    mappings = parse_id_mapping(mapping_csv_path)

    df_clean["admission_type_desc"] = df_clean["admission_type_id"].map(
        lambda x: mappings["admission_type"].get(x, f"Type {x}") if pd.notna(x) else "Unknown"
    )
    df_clean["discharge_disposition_desc"] = df_clean["discharge_disposition_id"].map(
        lambda x: mappings["discharge_disposition"].get(x, f"Discharge {x}") if pd.notna(x) else "Unknown"
    )
    df_clean["admission_source_desc"] = df_clean["admission_source_id"].map(
        lambda x: mappings["admission_source"].get(x, f"Source {x}") if pd.notna(x) else "Unknown"
    )

    # 7. Quality Report JSON
    quality_report = {
        "initial_raw_records": initial_row_count,
        "valid_curated_records": len(df_clean),
        "total_rejected_records": len(rejected_records),
        "rejection_rate_percent": round((len(rejected_records) / initial_row_count) * 100, 2),
        "missingness_summary": missingness_stats,
        "rejection_reasons_breakdown": rejected_df["rejection_reason"].value_counts().to_dict() if not rejected_df.empty else {}
    }

    quality_report_path = os.path.join(staging_dir, "data_quality_report.json")
    with open(quality_report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    # Save cleaned staging data
    cleaned_staging_path = os.path.join(staging_dir, "cleaned_staging_data.csv")
    df_clean.to_csv(cleaned_staging_path, index=False)
    logger.info(f"Cleaned staging dataset written to {cleaned_staging_path} ({len(df_clean):,} records)")

    return df_clean, quality_report


if __name__ == "__main__":
    validate_and_clean_raw_data()
