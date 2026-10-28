"""
Clinical ICD-9 Diagnostic Categorization Module.
Maps raw ICD-9 codes from diabetic encounter records into standardized clinical categories
based on clinical literature (Strack et al., 2014) and CDC/WHO ICD-9-CM guidelines.
"""

from typing import Optional, Union
import pandas as pd


def map_icd9_to_clinical_category(code: Union[str, float, None]) -> str:
    """
    Categorizes an ICD-9 code into high-level diagnostic chapters:
    - Circulatory (390-459, 785)
    - Respiratory (460-519, 786)
    - Digestive (520-579, 787)
    - Diabetes (250.xx)
    - Injury & Poisoning (800-999)
    - Musculoskeletal (710-739)
    - Genitourinary (580-629, 788)
    - Neoplasms (140-239)
    - Other (Infectious, Mental, Nervous, Skin, Congenital, Symptoms, E/V codes)
    """
    if pd.isna(code):
        return "Missing/Unknown"
    
    code_str = str(code).strip()
    if not code_str or code_str in ("?", "None", "nan", "Unknown"):
        return "Missing/Unknown"

    # Handle E and V codes (External causes / Supplemental classifications)
    if code_str.upper().startswith("E") or code_str.upper().startswith("V"):
        return "Other"

    # Extract leading numerical component
    try:
        num = float(code_str.split(".")[0])
    except ValueError:
        return "Other"

    # Diabetes specific (250.xx)
    if num == 250 or code_str.startswith("250"):
        return "Diabetes"

    # Circulatory system
    if (390 <= num <= 459) or num == 785:
        return "Circulatory"

    # Respiratory system
    if (460 <= num <= 519) or num == 786:
        return "Respiratory"

    # Digestive system
    if (520 <= num <= 579) or num == 787:
        return "Digestive"

    # Injury and Poisoning
    if 800 <= num <= 999:
        return "Injury & Poisoning"

    # Musculoskeletal system & connective tissue
    if 710 <= num <= 739:
        return "Musculoskeletal"

    # Genitourinary system
    if (580 <= num <= 629) or num == 788:
        return "Genitourinary"

    # Neoplasms
    if 140 <= num <= 239:
        return "Neoplasms"

    # All other ICD-9 chapters
    return "Other"
