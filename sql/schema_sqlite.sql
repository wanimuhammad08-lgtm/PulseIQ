-- Schema definition for Healthcare Readmission Analytical Mart (SQLite)
-- Project: Hospital Readmission Analytics & Prediction Using Public Data

DROP TABLE IF EXISTS healthcare_mart;
DROP TABLE IF EXISTS readmission;
DROP TABLE IF EXISTS diagnosis;
DROP TABLE IF EXISTS admission;
DROP TABLE IF EXISTS patient_summary;

-- Patient Summary Table (Unique Patients Dimension)
CREATE TABLE patient_summary (
    patient_sk TEXT PRIMARY KEY,
    gender TEXT,
    age_bracket TEXT,
    race TEXT,
    total_encounters INTEGER,
    first_admission_id TEXT,
    last_admission_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Admission Dimension Table (Encounter level details)
CREATE TABLE admission (
    encounter_sk TEXT PRIMARY KEY,
    patient_sk TEXT NOT NULL,
    admission_type_id INTEGER,
    admission_type_desc TEXT,
    discharge_disposition_id INTEGER,
    discharge_disposition_desc TEXT,
    admission_source_id INTEGER,
    admission_source_desc TEXT,
    time_in_hospital INTEGER NOT NULL,
    payer_code TEXT,
    medical_specialty TEXT,
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,
    prior_admissions INTEGER,
    change_in_meds TEXT,
    diabetes_med_prescribed TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_sk) REFERENCES patient_summary(patient_sk)
);

-- Diagnosis Dimension Table
CREATE TABLE diagnosis (
    encounter_sk TEXT PRIMARY KEY,
    diag_1_code TEXT,
    diag_1_category TEXT,
    diag_2_code TEXT,
    diag_2_category TEXT,
    diag_3_code TEXT,
    diag_3_category TEXT,
    number_diagnoses INTEGER,
    has_diabetes_primary INTEGER,
    has_circulatory_primary INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (encounter_sk) REFERENCES admission(encounter_sk)
);

-- Readmission Fact Table
CREATE TABLE readmission (
    encounter_sk TEXT PRIMARY KEY,
    raw_readmitted TEXT NOT NULL,
    readmitted_30d INTEGER NOT NULL,
    readmitted_any INTEGER NOT NULL,
    readmission_category TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (encounter_sk) REFERENCES admission(encounter_sk)
);

-- Unified Analytical Data Mart (Denormalized for Fast Dashboard Queries)
CREATE TABLE healthcare_mart (
    encounter_sk TEXT PRIMARY KEY,
    patient_sk TEXT NOT NULL,
    race TEXT,
    gender TEXT,
    age_bracket TEXT,
    admission_type_desc TEXT,
    discharge_disposition_desc TEXT,
    admission_source_desc TEXT,
    time_in_hospital INTEGER,
    payer_code TEXT,
    medical_specialty TEXT,
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,
    prior_admissions INTEGER,
    diag_1_code TEXT,
    diag_1_category TEXT,
    diag_2_category TEXT,
    diag_3_category TEXT,
    number_diagnoses INTEGER,
    max_glu_serum TEXT,
    a1c_result TEXT,
    metformin TEXT,
    insulin TEXT,
    change_in_meds TEXT,
    diabetes_med_prescribed TEXT,
    raw_readmitted TEXT,
    readmitted_30d INTEGER,
    readmitted_any INTEGER,
    readmission_category TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_mart_patient ON healthcare_mart(patient_sk);
CREATE INDEX idx_mart_age ON healthcare_mart(age_bracket);
CREATE INDEX idx_mart_readmit30 ON healthcare_mart(readmitted_30d);
CREATE INDEX idx_mart_diag1 ON healthcare_mart(diag_1_category);
