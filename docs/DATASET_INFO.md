# Dataset Source Information & Access Protocol

**Project**: Hospital Readmission Analytics & Prediction Using Public Data  
**Course**: Data Engineering and MLOps (Project 5)  
**Deliverable**: Assignment Part 1 — Submission Item 4  

---

## 1. Dataset Identification

- **Dataset Name**: Diabetes 130-US Hospitals for Years 1999–2008 Data Set
- **Primary Host**: UCI Machine Learning Repository
- **DOI / URL**: [https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Direct Download Archive**: `https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip`
- **Fallback Mirror**: `https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip`
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 2. Provenance and Academic Citation

The dataset was originally extracted and contributed by the Center for Clinical and Translational Research at Virginia Commonwealth University.

**Primary Literature Citation**:
> Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014).  
> *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*.  
> BioMed Research International, vol. 2014, Article ID 781670.

---

## 3. Scope and Encounter Characteristics

- **Temporal Coverage**: 10 consecutive years (1999 to 2008).
- **Clinical Setting**: Inpatient encounters across 130 hospitals and integrated delivery networks in the United States.
- **Inclusion Criteria**:
  1. It is an inpatient encounter (a hospital admission).
  2. It is a diabetic encounter: symptoms of diabetes were entered into the system during hospitalization.
  3. The length of stay was at least 1 day and at most 14 days.
  4. Laboratory tests were performed during the encounter.
  5. Medications were administered during the encounter.
- **Dimensionality**: 101,766 hospital encounters spanning 71,518 unique patients and 50 raw clinical/administrative attributes.

---

## 4. Privacy, Anonymization & Ethical Compliance

1. **De-Identification**: In accordance with HIPAA Safe Harbor and Institutional Review Board (IRB) standards, all direct personal identifiers (names, Social Security numbers, dates of birth, street addresses, phone numbers) were removed prior to public deposition.
2. **Surrogate Key Generation**: To guarantee zero cross-system linkage risk, our ETL pipeline applies a cryptographic salt-hashed SHA-256 algorithm generating persistent `encounter_sk` and `patient_sk` keys. Original raw identifiers (`encounter_id`, `patient_nbr`) are stripped from downstream curated tables and the data mart.
3. **No Protected Health Information (PHI)**: The curated and analytical storage layers contain strictly anonymized clinical variables, diagnosis category abstractions, and demographic brackets.

---

## 5. Automated Retrieval & Verification Instructions

Automated programmatic access is embedded directly in `src/ingestion/ingest.py` and executed via:

```bash
python run_pipeline.py
```

### Manual Acquisition Steps:
1. Download the archive:
   ```bash
   curl -O https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip
   ```
2. Unzip the archive contents into `data/raw/`.
3. Confirm that `diabetic_data.csv` and `IDs_mapping.csv` are present in `data/raw/`.
4. Inspect the extraction manifest at `data/raw/extraction_manifest.json` for verified SHA-256 cryptographic hashes and record counts.
