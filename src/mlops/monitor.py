"""
Data Drift Monitoring Module.
Uses Evidently to compare production data against training reference data
and detect distribution shifts that could degrade model performance.
"""

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import os


def check_drift(reference_data_path, current_data_path, output_path="reports/drift_report.html"):
    """
    Compare reference (training) data against current (production) data
    to detect feature distribution drift.
    
    Returns True if drift is detected, False otherwise.
    """
    print(f"Loading reference data from {reference_data_path}...")
    ref_df = pd.read_csv(reference_data_path)

    print(f"Loading current data from {current_data_path}...")
    curr_df = pd.read_csv(current_data_path)

    # Drop targets and IDs — we only want to check feature drift
    cols_to_drop = [
        'readmitted_30d', 'raw_readmitted', 'readmitted_any',
        'readmission_category', 'encounter_sk', 'patient_sk'
    ]
    ref_features = ref_df.drop([c for c in cols_to_drop if c in ref_df.columns], axis=1)
    curr_features = curr_df.drop([c for c in cols_to_drop if c in curr_df.columns], axis=1)

    # Ensure both DataFrames have the same columns
    common_cols = list(set(ref_features.columns) & set(curr_features.columns))
    ref_features = ref_features[common_cols]
    curr_features = curr_features[common_cols]

    print(f"Comparing {len(common_cols)} features across {len(ref_features):,} ref rows and {len(curr_features):,} current rows...")
    print("Running Data Drift Report...")

    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    data_drift_report.run(reference_data=ref_features, current_data=curr_features)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    data_drift_report.save_html(output_path)

    # Extract drift detection result safely
    try:
        drift_json = data_drift_report.as_dict()
        # Navigate the Evidently result structure carefully
        metrics = drift_json.get('metrics', [])
        if metrics:
            result = metrics[0].get('result', {})
            drift_detected = result.get('dataset_drift', False)
        else:
            drift_detected = False
    except Exception as e:
        print(f"Warning: Could not parse drift result programmatically: {e}")
        drift_detected = None

    print(f"Data Drift Detected: {drift_detected}")
    print(f"Full HTML report saved to {output_path}")

    return drift_detected


if __name__ == "__main__":
    ref = "data/curated/healthcare_mart.csv"
    curr = "data/sample_tables/sample_healthcare_mart.csv"
    if os.path.exists(ref) and os.path.exists(curr):
        check_drift(ref, curr)
    else:
        print("Data files not found for drift check. Run the pipeline first.")
