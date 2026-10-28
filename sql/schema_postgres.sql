-- Schema definition for Healthcare Readmission Analytical Mart (PostgreSQL)
-- Project: Hospital Readmission Analytics & Prediction Using Public Data

DROP TABLE IF EXISTS healthcare_mart CASCADE;
DROP TABLE IF EXISTS readmission CASCADE;
DROP TABLE IF EXISTS diagnosis CASCADE;
DROP TABLE IF EXISTS admission CASCADE;
DROP TABLE IF EXISTS patient_summary CASCADE;

-- Patient Summary Table (Unique Patients Dimension)
CREATE TABLE patient_summary (
    patient_sk VARCHAR(64) PRIMARY KEY,
    gender VARCHAR(16),
    age_bracket VARCHAR(16),
    race VARCHAR(32),
    total_encounters INTEGER,
    first_admission_id VARCHAR(64),
    last_admission_id VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Admission Dimension Table (Encounter level details)
CREATE TABLE admission (
    encounter_sk VARCHAR(64) PRIMARY KEY,
    patient_sk VARCHAR(64) NOT NULL REFERENCES patient_summary(patient_sk),
    admission_type_id INTEGER,
    admission_type_desc VARCHAR(128),
    discharge_disposition_id INTEGER,
    discharge_disposition_desc VARCHAR(128),
    admission_source_id INTEGER,
    admission_source_desc VARCHAR(128),
    time_in_hospital INTEGER NOT NULL,
    payer_code VARCHAR(16),
    medical_specialty VARCHAR(64),
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,
    prior_admissions INTEGER,
    change_in_meds VARCHAR(8),
    diabetes_med_prescribed VARCHAR(8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Diagnosis Dimension Table
CREATE TABLE diagnosis (
    encounter_sk VARCHAR(64) PRIMARY KEY REFERENCES admission(encounter_sk),
    diag_1_code VARCHAR(16),
    diag_1_category VARCHAR(64),
    diag_2_code VARCHAR(16),
    diag_2_category VARCHAR(64),
    diag_3_code VARCHAR(16),
    diag_3_category VARCHAR(64),
    number_diagnoses INTEGER,
    has_diabetes_primary SMALLINT,
    has_circulatory_primary SMALLINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Readmission Fact Table
CREATE TABLE readmission (
    encounter_sk VARCHAR(64) PRIMARY KEY REFERENCES admission(encounter_sk),
    raw_readmitted VARCHAR(16) NOT NULL,
    readmitted_30d SMALLINT NOT NULL,
    readmitted_any SMALLINT NOT NULL,
    readmission_category VARCHAR(32) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Unified Analytical Data Mart (Denormalized for Fast Dashboard Queries)
CREATE TABLE healthcare_mart (
    encounter_sk VARCHAR(64) PRIMARY KEY,
    patient_sk VARCHAR(64) NOT NULL,
    race VARCHAR(32),
    gender VARCHAR(16),
    age_bracket VARCHAR(16),
    admission_type_desc VARCHAR(128),
    discharge_disposition_desc VARCHAR(128),
    admission_source_desc VARCHAR(128),
    time_in_hospital INTEGER,
    payer_code VARCHAR(16),
    medical_specialty VARCHAR(64),
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,
    prior_admissions INTEGER,
    diag_1_code VARCHAR(16),
    diag_1_category VARCHAR(64),
    diag_2_category VARCHAR(64),
    diag_3_category VARCHAR(64),
    number_diagnoses INTEGER,
    max_glu_serum VARCHAR(16),
    a1c_result VARCHAR(16),
    metformin VARCHAR(16),
    insulin VARCHAR(16),
    change_in_meds VARCHAR(8),
    diabetes_med_prescribed VARCHAR(8),
    raw_readmitted VARCHAR(16),
    readmitted_30d SMALLINT,
    readmitted_any SMALLINT,
    readmission_category VARCHAR(32),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pg_mart_patient ON healthcare_mart(patient_sk);
CREATE INDEX idx_pg_mart_age ON healthcare_mart(age_bracket);
CREATE INDEX idx_pg_mart_readmit30 ON healthcare_mart(readmitted_30d);
CREATE INDEX idx_pg_mart_diag1 ON healthcare_mart(diag_1_category);
