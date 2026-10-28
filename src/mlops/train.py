"""
MLOps Training Pipeline for Hospital Readmission Prediction.
Trains Logistic Regression, Random Forest, and XGBoost models.
Tracks experiments with MLflow and generates SHAP explainability plots.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os
import tempfile
import joblib


def load_and_prepare_data(filepath='data/curated/healthcare_mart.csv'):
    """Load the curated healthcare mart and prepare features/target."""
    df = pd.read_csv(filepath)

    if 'readmitted_30d' not in df.columns:
        raise ValueError("Target column 'readmitted_30d' not found in the dataset.")

    y = df['readmitted_30d'].astype(int)

    # Drop all readmission-related columns to prevent data leakage
    leakage_cols = ['raw_readmitted', 'readmitted_30d', 'readmitted_any', 'readmission_category']
    X = df.drop(leakage_cols, axis=1, errors='ignore')

    # Drop identifiers and sparse columns
    id_cols = ['encounter_sk', 'patient_sk', 'payer_code']
    X = X.drop(id_cols, axis=1, errors='ignore')

    # Force object columns that pandas may read as numeric
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        if X[col].nunique() < 20 and col not in ['time_in_hospital', 'num_lab_procedures',
                                                   'num_procedures', 'num_medications',
                                                   'number_outpatient', 'number_emergency',
                                                   'number_inpatient', 'prior_admissions',
                                                   'number_diagnoses']:
            X[col] = X[col].astype(str)

    # Fill NaN in object/string columns
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = X[col].fillna('Unknown')

    # Fill NaN in numeric columns
    for col in X.select_dtypes(include=['int64', 'float64']).columns:
        X[col] = X[col].fillna(0)

    return X, y


def build_pipeline(model, numerical_cols, categorical_cols):
    """Build a sklearn Pipeline with preprocessing and a classifier."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return clf


def evaluate_model(y_true, y_prob, y_pred):
    """Calculate classification metrics."""
    return {
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob)
    }


def train_and_log():
    """Train all models, log to MLflow, save best model locally."""
    print("Loading and preparing data...")
    X, y = load_and_prepare_data()
    print(f"  Features: {X.shape[1]} columns, {X.shape[0]:,} rows")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    print(f"  Numerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            eval_metric='logloss', random_state=42,
            n_estimators=100, max_depth=5, n_jobs=-1
        )
    }

    mlflow.set_tracking_uri('sqlite:///mlruns.db')
    mlflow.set_experiment("Hospital_Readmission_Prediction")

    best_model_name = None
    best_roc_auc = 0
    best_run_id = None

    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        with mlflow.start_run(run_name=model_name) as run:
            clf = build_pipeline(model, numerical_cols, categorical_cols)

            # Train
            clf.fit(X_train, y_train)

            # Predict
            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = clf.predict(X_test)

            # Evaluate
            metrics = evaluate_model(y_test, y_prob, y_pred)

            # Log params and metrics
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_metrics(metrics)

            # Log model — trust sklearn tree types for RandomForest/XGBoost
            trusted_types = None
            if model_name in ['RandomForest', 'XGBoost']:
                trusted_types = [
                    "sklearn.tree._tree.Tree",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier"
                ]
            mlflow.sklearn.log_model(
                clf, "model",
                skops_trusted_types=trusted_types
            )

            # Print results
            cm = confusion_matrix(y_test, y_pred)
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Brier:     {metrics['brier_score']:.4f}")
            print(f"  Confusion Matrix:\n{cm}")

            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                best_model_name = model_name
                best_run_id = run.info.run_id

            # SHAP Explainability for tree-based models
            if model_name in ['XGBoost', 'RandomForest']:
                try:
                    preprocessor = clf.named_steps['preprocessor']
                    X_test_transformed = preprocessor.transform(X_test)

                    # Build proper feature names list
                    num_feature_names = list(preprocessor.transformers_[0][2])
                    cat_encoder = preprocessor.transformers_[1][1]
                    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_cols))
                    feature_names = num_feature_names + cat_feature_names

                    explainer = shap.TreeExplainer(clf.named_steps['classifier'])
                    shap_values = explainer.shap_values(X_test_transformed[:100])

                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        shap_values, X_test_transformed[:100],
                        feature_names=feature_names, show=False,
                        max_display=20
                    )
                    plt.tight_layout()

                    # Save to temp dir, log to MLflow, then clean up
                    with tempfile.TemporaryDirectory() as tmpdir:
                        shap_path = os.path.join(tmpdir, f"{model_name}_shap_summary.png")
                        plt.savefig(shap_path, bbox_inches='tight', dpi=150)
                        mlflow.log_artifact(shap_path)
                    plt.close('all')

                    print(f"  SHAP explainability plot logged for {model_name}")
                except Exception as e:
                    print(f"  SHAP logging failed for {model_name}: {e}")

    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name} with ROC AUC: {best_roc_auc:.4f}")

    # Register the best model and save a local copy
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        try:
            mlflow.register_model(model_uri, "ReadmissionPredictor")
            print(f"Model registered as 'ReadmissionPredictor'")
        except Exception as e:
            print(f"Model registration note: {e}")

        # Save a local copy for FastAPI serving
        os.makedirs('models', exist_ok=True)
        best_model = mlflow.sklearn.load_model(model_uri)
        joblib.dump(best_model, 'models/best_model.pkl')
        print(f"Best model saved to models/best_model.pkl")


if __name__ == "__main__":
    train_and_log()
