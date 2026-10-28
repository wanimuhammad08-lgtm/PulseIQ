"""
Phase 2 MLOps Test Suite.
Covers:
- Data loading and preparation for ML
- Pipeline construction and feature handling
- Model evaluation metrics
- API schema validation
- Drift monitor file handling
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========================= DATA LOADING TESTS =========================

class TestDataLoading:
    """Test the ML data loading and preparation."""

    def test_healthcare_mart_exists(self):
        """Verify the curated healthcare mart CSV exists."""
        assert os.path.exists("data/curated/healthcare_mart.csv"), \
            "healthcare_mart.csv not found. Run the Phase 1 pipeline first."

    def test_healthcare_mart_columns(self):
        """Verify the mart has all required columns."""
        df = pd.read_csv("data/curated/healthcare_mart.csv", nrows=5)
        required_cols = [
            'encounter_sk', 'patient_sk', 'readmitted_30d',
            'readmitted_any', 'time_in_hospital', 'num_medications',
            'prior_admissions', 'diag_1_category'
        ]
        for col in required_cols:
            assert col in df.columns, f"Required column '{col}' missing from healthcare_mart"

    def test_target_column_binary(self):
        """Verify readmitted_30d is binary (0 or 1)."""
        df = pd.read_csv("data/curated/healthcare_mart.csv", usecols=['readmitted_30d'])
        unique_vals = set(df['readmitted_30d'].unique())
        assert unique_vals.issubset({0, 1}), f"Target column has non-binary values: {unique_vals}"

    def test_load_and_prepare_data(self):
        """Test the load_and_prepare_data function."""
        from src.mlops.train import load_and_prepare_data
        X, y = load_and_prepare_data()

        # Check shapes
        assert len(X) == len(y), "Feature and target lengths don't match"
        assert len(X) > 0, "No data loaded"

        # Verify no leakage columns remain
        leakage_cols = ['readmitted_30d', 'readmitted_any', 'raw_readmitted', 'readmission_category']
        for col in leakage_cols:
            assert col not in X.columns, f"Leakage column '{col}' still in features"

        # Verify IDs removed
        id_cols = ['encounter_sk', 'patient_sk', 'payer_code']
        for col in id_cols:
            assert col not in X.columns, f"ID column '{col}' still in features"

    def test_no_nan_in_prepared_data(self):
        """After preparation, there should be no NaN values."""
        from src.mlops.train import load_and_prepare_data
        X, y = load_and_prepare_data()
        assert X.isna().sum().sum() == 0, f"NaN values found in features: {X.isna().sum()[X.isna().sum() > 0].to_dict()}"
        assert y.isna().sum() == 0, "NaN values found in target"


# ========================= PIPELINE TESTS =========================

class TestPipelineConstruction:
    """Test the sklearn pipeline building."""

    def test_build_pipeline(self):
        """Verify pipeline builds without errors."""
        from src.mlops.train import build_pipeline
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        num_cols = ['time_in_hospital', 'num_medications']
        cat_cols = ['race', 'gender']

        clf = build_pipeline(model, num_cols, cat_cols)
        assert clf is not None
        assert 'preprocessor' in clf.named_steps
        assert 'classifier' in clf.named_steps

    def test_pipeline_fit_predict(self):
        """Test that pipeline can fit and predict on sample data."""
        from src.mlops.train import build_pipeline
        from sklearn.linear_model import LogisticRegression

        # Create small synthetic data
        np.random.seed(42)
        X = pd.DataFrame({
            'time_in_hospital': np.random.randint(1, 14, 100),
            'num_medications': np.random.randint(1, 30, 100),
            'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic'], 100),
            'gender': np.random.choice(['Male', 'Female'], 100)
        })
        y = np.random.randint(0, 2, 100)

        clf = build_pipeline(LogisticRegression(), ['time_in_hospital', 'num_medications'], ['race', 'gender'])
        clf.fit(X, y)

        probs = clf.predict_proba(X)
        assert probs.shape == (100, 2), f"Unexpected probability shape: {probs.shape}"
        assert all(0 <= p <= 1 for p in probs[:, 1]), "Probabilities out of [0, 1] range"


# ========================= METRICS TESTS =========================

class TestEvaluation:
    """Test model evaluation metric computation."""

    def test_evaluate_model_perfect(self):
        """Perfect predictions should give perfect scores."""
        from src.mlops.train import evaluate_model
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        y_pred = np.array([0, 0, 1, 1, 1])

        metrics = evaluate_model(y_true, y_prob, y_pred)
        assert metrics['recall'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['roc_auc'] == 1.0
        assert metrics['brier_score'] == 0.0

    def test_evaluate_model_returns_all_keys(self):
        """All expected metric keys should be present."""
        from src.mlops.train import evaluate_model
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        y_pred = np.array([0, 1, 0, 1])

        metrics = evaluate_model(y_true, y_prob, y_pred)
        expected_keys = {'recall', 'precision', 'f1', 'roc_auc', 'brier_score'}
        assert set(metrics.keys()) == expected_keys


# ========================= API SCHEMA TESTS =========================

class TestAPISchema:
    """Test the FastAPI data models and schema."""

    def test_patient_data_defaults(self):
        """PatientData model should initialize with defaults."""
        from app.api import PatientData
        patient = PatientData()
        assert patient.time_in_hospital == 3
        assert patient.gender == "Female"
        assert patient.age_bracket == "[50-60)"

    def test_patient_data_custom(self):
        """PatientData model should accept custom values."""
        from app.api import PatientData
        patient = PatientData(
            race="AfricanAmerican",
            gender="Male",
            age_bracket="[70-80)",
            time_in_hospital=7,
            num_medications=25,
            prior_admissions=5
        )
        assert patient.race == "AfricanAmerican"
        assert patient.time_in_hospital == 7
        assert patient.prior_admissions == 5

    def test_patient_data_serialization(self):
        """PatientData should serialize to dict correctly."""
        from app.api import PatientData
        patient = PatientData()
        data = patient.model_dump()
        assert isinstance(data, dict)
        assert 'time_in_hospital' in data
        assert 'diag_1_category' in data


# ========================= MONITOR TESTS =========================

class TestMonitor:
    """Test the drift monitoring module."""

    def test_monitor_import(self):
        """Monitor module should import cleanly."""
        from src.mlops.monitor import check_drift
        assert callable(check_drift)

    def test_sample_data_exists_for_drift(self):
        """Sample data for drift comparison should exist."""
        assert os.path.exists("data/sample_tables/sample_healthcare_mart.csv"), \
            "Sample mart not found for drift comparison"


# ========================= INTEGRATION TESTS =========================

class TestIntegration:
    """End-to-end integration tests."""

    def test_database_has_data(self):
        """SQLite database should contain populated tables."""
        import sqlite3
        db_path = "data/healthcare.db"
        assert os.path.exists(db_path), "Database not found"

        conn = sqlite3.connect(db_path)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]
        conn.close()

        assert 'healthcare_mart' in table_names, "healthcare_mart table missing"
        assert 'patient_summary' in table_names, "patient_summary table missing"

    def test_curated_files_exist(self):
        """All curated CSV files should exist."""
        expected_files = [
            'data/curated/healthcare_mart.csv',
            'data/curated/patient_summary.csv',
            'data/curated/admission.csv',
            'data/curated/diagnosis.csv',
            'data/curated/readmission.csv',
        ]
        for f in expected_files:
            assert os.path.exists(f), f"Curated file missing: {f}"

    def test_mart_row_count(self):
        """Healthcare mart should have expected number of rows."""
        df = pd.read_csv("data/curated/healthcare_mart.csv")
        assert len(df) > 90000, f"Healthcare mart has only {len(df)} rows, expected 99,340"
        assert len(df) <= 101000, f"Healthcare mart has {len(df)} rows, too many"
