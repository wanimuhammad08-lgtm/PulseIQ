# Data Dictionary & Data Quality Validation Rules

**Project**: Hospital Readmission Analytics & Prediction Using Public Data  
**Course**: Data Engineering and MLOps (Project 5)  
**Deliverable**: Assignment Part 1 — Submission Item 6  

---

## 1. Raw vs. Transformed Pipeline Data Dictionary

| Field Name | Raw Source Field | Target Table | Data Type | Description / Clinical Meaning | Allowed Values / Formatting | Transformation & Validation Rules |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `encounter_sk` | `encounter_id` | All Tables | VARCHAR(64) | Cryptographic surrogate identifier for the encounter | Hex hash (`ENC_[a-f0-9]{16}`) | SHA-256 hash with salt; guarantees de-identification |
| `patient_sk` | `patient_nbr` | All Tables | VARCHAR(64) | Cryptographic surrogate identifier for the unique patient | Hex hash (`PAT_[a-f0-9]{16}`) | SHA-256 hash with salt; tracks multi-encounter histories |
| `race` | `race` | `patient_summary`, `mart` | VARCHAR(32) | Patient self-reported demographic race | Caucasian, AfricanAmerican, Hispanic, Asian, Other, Missing | `'?'` converted to `Missing / Unspecified` |
| `gender` | `gender` | `patient_summary`, `mart` | VARCHAR(16) | Biological sex | Male, Female | Records with `'Unknown/Invalid'` rejected & logged |
| `age_bracket` | `age` | `patient_summary`, `mart` | VARCHAR(16) | 10-year patient age cohort | `[0-10)`, `[10-20)`, ..., `[90-100)` | Standardized bracket representation |
| `admission_type_desc` | `admission_type_id` | `admission`, `mart` | VARCHAR(128) | Clinical urgency of admission | Emergency, Urgent, Elective, Newborn, Trauma, etc. | Lookup mapped from `IDs_mapping.csv` |
| `discharge_disposition_desc` | `discharge_disposition_id` | `admission`, `mart` | VARCHAR(128) | Patient discharge destination | Discharged to home, SNF, Rehab, etc. | Mapped from lookup; Expired/Hospice records rejected |
| `admission_source_desc` | `admission_source_id` | `admission`, `mart` | VARCHAR(128) | Referral source prior to hospital encounter | Emergency Room, Physician Referral, Clinic, Transfer | Lookup mapped from `IDs_mapping.csv` |
| `time_in_hospital` | `time_in_hospital` | `admission`, `mart` | INTEGER | Total hospitalization duration in days | 1 to 14 days | Range constraint: must be integer `1 <= x <= 14` |
| `num_lab_procedures` | `num_lab_procedures` | `admission`, `mart` | INTEGER | Count of diagnostic lab procedures performed | 1 to 132 | Non-negative integer |
| `num_procedures` | `num_procedures` | `admission`, `mart` | INTEGER | Count of non-lab surgical/diagnostic procedures | 0 to 6 | Non-negative integer |
| `num_medications` | `num_medications` | `admission`, `mart` | INTEGER | Total medications administered during stay | 1 to 81 | Positive integer; indicates acute prescription volume |
| `number_outpatient` | `number_outpatient` | `admission`, `mart` | INTEGER | Outpatient visits in the year prior to encounter | >= 0 | Non-negative integer |
| `number_emergency` | `number_emergency` | `admission`, `mart` | INTEGER | Emergency visits in the year prior to encounter | >= 0 | Non-negative integer |
| `number_inpatient` | `number_inpatient` | `admission`, `mart` | INTEGER | Inpatient visits in the year prior to encounter | >= 0 | Non-negative integer |
| `prior_admissions` | Derived | `admission`, `mart` | INTEGER | Total prior acute/ambulatory healthcare contacts | >= 0 | Engineered: `number_outpatient + number_emergency + number_inpatient` |
| `diag_1_code` | `diag_1` | `diagnosis`, `mart` | VARCHAR(16) | Primary ICD-9 diagnosis code | String code (e.g., '250.01', '428') | Standardized string, `'?'` handled as `None` |
| `diag_1_category` | Derived (`diag_1`) | `diagnosis`, `mart` | VARCHAR(64) | High-level clinical diagnostic category | Circulatory, Respiratory, Digestive, Diabetes, etc. | Mapped via clinical chapter logic (Strack et al., 2014) |
| `diag_2_category` | Derived (`diag_2`) | `diagnosis`, `mart` | VARCHAR(64) | Secondary clinical diagnostic category | Circulatory, Respiratory, Digestive, Diabetes, etc. | Mapped via clinical chapter logic |
| `diag_3_category` | Derived (`diag_3`) | `diagnosis`, `mart` | VARCHAR(64) | Tertiary clinical diagnostic category | Circulatory, Respiratory, Digestive, Diabetes, etc. | Mapped via clinical chapter logic |
| `number_diagnoses` | `number_diagnoses` | `diagnosis`, `mart` | INTEGER | Total diagnoses entered into the clinical record | 1 to 16 | Integer constraint; proxy for patient co-morbidity |
| `max_glu_serum` | `max_glu_serum` | `mart` | VARCHAR(16) | Serum glucose laboratory test outcome | `>200`, `>300`, `Norm`, `None` | Categorical test result |
| `a1c_result` | `A1Cresult` | `mart` | VARCHAR(16) | Glycated hemoglobin (HbA1c) test result | `>7`, `>8`, `Norm`, `None` | Key diabetes chronic management metric |
| `metformin` | `metformin` | `mart` | VARCHAR(16) | Prescribed dosage status of Metformin | `No`, `Steady`, `Up`, `Down` | Monitored oral hypoglycemic drug |
| `insulin` | `insulin` | `mart` | VARCHAR(16) | Prescribed dosage status of Insulin | `No`, `Steady`, `Up`, `Down` | Monitored injectable hormone |
| `change_in_meds` | `change` | `admission`, `mart` | VARCHAR(8) | Whether diabetic medication was altered | `Yes`, `No` | Raw `'Ch'` mapped to `'Yes'`, `'No'` to `'No'` |
| `diabetes_med_prescribed` | `diabetesMed` | `admission`, `mart` | VARCHAR(8) | Whether any diabetic drug was prescribed | `Yes`, `No` | Categorical boolean indicator |
| `raw_readmitted` | `readmitted` | `readmission`, `mart` | VARCHAR(16) | Original readmission outcome label | `<30`, `>30`, `NO` | Preserved for auditing and multi-class analysis |
| `readmitted_30d` | Derived | `readmission`, `mart` | SMALLINT | Binary 30-day acute readmission indicator | 0, 1 | Engineered target: 1 if `<30` days, 0 otherwise |
| `readmitted_any` | Derived | `readmission`, `mart` | SMALLINT | Overall readmission indicator | 0, 1 | Engineered target: 1 if `<30` or `>30`, 0 if `NO` |
| `readmission_category` | Derived | `readmission`, `mart` | VARCHAR(32) | Human-readable clinical classification | `<30 Days (High Risk)`, `>30 Days (Late Readmission)`, `No Readmission` | Descriptive outcome category for presentation |

---

## 2. Clinical Diagnostic Classification Rules (ICD-9 Chapter Mapping)

Diagnoses are categorized based on standard epidemiological grouping guidelines:

| Clinical Diagnostic Chapter | ICD-9 Numerical Range / Match Condition | Description & Typical Conditions |
| :--- | :--- | :--- |
| **Diabetes** | `250.xx` or exact `250` | Type 1 & Type 2 Diabetes Mellitus with/without complications |
| **Circulatory** | `390–459`, `785` | Heart failure, ischemic heart disease, hypertension, arrhythmias |
| **Respiratory** | `460–519`, `786` | Pneumonia, COPD, asthma, acute bronchitis |
| **Digestive** | `520–579`, `787` | Gastroenteritis, gastrointestinal hemorrhage, bowel obstruction |
| **Genitourinary** | `580–629`, `788` | Acute kidney failure, urinary tract infection, nephritis |
| **Injury & Poisoning** | `800–999` | Fractures, open wounds, adverse drug events, poisoning |
| **Musculoskeletal** | `710–739` | Osteoarthritis, arthropathies, spine disorders |
| **Neoplasms** | `140–239` | Malignant and benign neoplasms |
| **Other** | All other codes; `E`, `V` codes | Mental disorders, nervous system, skin, external injury causes |
| **Missing/Unknown** | `'?'`, `None`, `nan` | Unspecified or unrecorded diagnosis |

---

## 3. Data Quality & Rejection Rules

The pipeline enforces rigorous validation checks before staging:

1. **Missing Sentinel Replacement Rule**:
   - In the raw UCI dataset, missing values are represented as string `'?'`.
   - The validation engine replaces `'?'`, `'None'`, and empty strings with `NaN` / `NULL`.
2. **Clinical Infeasibility / Expired Exclusion Rule**:
   - A readmission can only occur if the patient is alive following hospital discharge.
   - Patients with `discharge_disposition_id` in `[11, 13, 14, 19, 20, 21]` (representing deceased patients or transfers to hospice) are clinically ineligible for readmission.
   - These records are excluded from downstream analytical and training sets and logged to `data/rejected/rejected_records.csv` with the reason: `Patient Expired or Discharged to Hospice`.
3. **Invalid Demographic Record Rule**:
   - Records with `gender` value `'Unknown/Invalid'` (or null) are rejected and logged.
4. **Duplicate Encounter Rule**:
   - Encounters must be unique. Any duplicate `encounter_id` entries are removed and logged to the rejection table.
5. **Referential Integrity Rule**:
   - Every record in `admission`, `diagnosis`, and `readmission` must have a corresponding valid `patient_sk` in `patient_summary`.
