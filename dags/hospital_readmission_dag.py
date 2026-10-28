"""
Apache Airflow DAG Definition for Healthcare Readmission ETL Pipeline.
Course: Data Engineering and MLOps
Project 5: Hospital Readmission Analytics and Prediction

DAG Structure:
[download_raw_uci_dataset]
            │
            ▼
[audit_and_clean_data_quality]
            │
            ▼
[transform_icd9_and_engineer_features]
            │
            ▼
[load_relational_storage_and_mart]
            │
            ▼
[verify_database_integrity_and_sample]
"""

from datetime import datetime, timedelta
import os
import sys

# Attempt importing Airflow; provide graceful compatibility if inspecting standalone
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_download_raw():
    from src.ingestion.ingest import download_and_extract_uci_data
    manifest = download_and_extract_uci_data(raw_dir="data/raw")
    print(f"Extraction successful: {manifest['total_records_extracted']} records.")
    return manifest


def task_audit_clean():
    from src.pipeline.data_quality import validate_and_clean_raw_data
    df_clean, report = validate_and_clean_raw_data(
        raw_csv_path="data/raw/diabetic_data.csv",
        mapping_csv_path="data/raw/IDs_mapping.csv",
        staging_dir="data/staging",
        rejected_dir="data/rejected"
    )
    print(f"Data quality audit completed: {len(df_clean)} valid, {report['total_rejected_records']} rejected.")
    return len(df_clean)


def task_transform():
    import pandas as pd
    from src.pipeline.transform import transform_and_create_tables
    df_clean = pd.read_csv("data/staging/cleaned_staging_data.csv", low_memory=False)
    tables = transform_and_create_tables(df_clean, curated_dir="data/curated")
    print(f"Dimensional transformation completed. Generated {len(tables)} tables.")
    return list(tables.keys())


def task_load_db():
    import pandas as pd
    from src.database.db_manager import DatabaseManager
    curated_tables = {
        "patient_summary": pd.read_csv("data/curated/patient_summary.csv", low_memory=False),
        "admission": pd.read_csv("data/curated/admission.csv", low_memory=False),
        "diagnosis": pd.read_csv("data/curated/diagnosis.csv", low_memory=False),
        "readmission": pd.read_csv("data/curated/readmission.csv", low_memory=False),
        "healthcare_mart": pd.read_csv("data/curated/healthcare_mart.csv", low_memory=False),
    }
    db = DatabaseManager(db_path="data/healthcare.db")
    db.initialize_schema()
    counts = db.load_curated_tables(curated_tables, if_exists="replace")
    print(f"Database loading completed: {counts}")
    return counts


def task_verify():
    from src.database.db_manager import DatabaseManager
    db = DatabaseManager(db_path="data/healthcare.db")
    verified = db.verify_database()
    db.export_sample_tables(sample_dir="data/sample_tables")
    print(f"Database verification: {verified}")
    return verified


default_args = {
    "owner": "data_engineering_mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 9, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

if AIRFLOW_AVAILABLE:
    dag = DAG(
        "hospital_readmission_etl_pipeline",
        default_args=default_args,
        description="End-to-End Healthcare Readmission ETL & Dimensional Mart Pipeline",
        schedule_interval="@weekly",
        catchup=False,
        tags=["healthcare", "etl", "readmission", "mlops"],
    )

    t1 = PythonOperator(
        task_id="download_raw_uci_dataset",
        python_callable=task_download_raw,
        dag=dag,
    )

    t2 = PythonOperator(
        task_id="audit_and_clean_data_quality",
        python_callable=task_audit_clean,
        dag=dag,
    )

    t3 = PythonOperator(
        task_id="transform_icd9_and_engineer_features",
        python_callable=task_transform,
        dag=dag,
    )

    t4 = PythonOperator(
        task_id="load_relational_storage_and_mart",
        python_callable=task_load_db,
        dag=dag,
    )

    t5 = PythonOperator(
        task_id="verify_database_integrity_and_sample",
        python_callable=task_verify,
        dag=dag,
    )

    t1 >> t2 >> t3 >> t4 >> t5
