"""
Preprocessing pipeline for multiple UCI ML datasets using UCIMLRepo.
Includes encoding, normalization, and handling missing values.
"""

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from encode import encode_datasets
from handle_missing_values import deal_data_with_na
from normalise import normalise
import time

# === High-level Preprocessing ===

def preprocess_all(needs_normalisation=False, needs_encoding=True):
    """
    Preprocesses all selected datasets with the specified options.

    Parameters:
    - needs_normalisation (bool): Whether to normalize numeric features.
    - needs_encoding (bool): Whether to encode categorical variables.

    Returns:
    - Tuple of preprocessed datasets and their column names.
    """
    breast_cancer = preprocess("Breast_cancer", needs_normalisation, needs_encoding)
    heart_disease = preprocess("Heart_disease", needs_normalisation, needs_encoding)
    lung_cancer = preprocess("Lung_cancer", needs_normalisation, needs_encoding)
    diabetes = preprocess("Diabetes", needs_normalisation, needs_encoding)
    obesity = preprocess("Obesity", needs_normalisation, needs_encoding)
    alzheimer = preprocess("Alzheimer", needs_normalisation, needs_encoding)

    return breast_cancer, heart_disease, lung_cancer, diabetes, obesity, alzheimer

# === Core Preprocessing Logic ===

def preprocess(name, needs_normalisation=False, needs_encoding=True, log=True):
    """
    Loads and preprocesses a dataset based on its name.

    Parameters:
    - name (str): Dataset name.
    - needs_normalisation (bool): Apply normalization to numeric columns.
    - needs_encoding (bool): Apply encoding to categorical features.
    - log (bool): Whether to print metadata and variable info.

    Returns:
    - data (pd.DataFrame): Preprocessed data.
    - cols (List[str]): List of feature and target columns.
    """
    name_to_id = {
        "Breast_cancer": 17,
        "Heart_disease": 45,
        "Lung_cancer": 62,
        "Diabetes": 296,
        "Obesity": 544,
        "Alzheimer": 732
    }

    data_raw = fetch_ucirepo(id=name_to_id.get(name))
    print(data_raw)
    X, y = data_raw.data.features, data_raw.data.targets
    # X = pd.concat([data_raw.data.ids[['encounter_id', 'patient_nbr']], X], axis=1)
    print(X.head())
    if log:
        print(f"({name}) Metadata:")
        print(data_raw.metadata)
        print(f"({name}) Variables:")
        print(data_raw.variables)

    data_raw = pd.concat([X, y], axis=1)
    print(list(data_raw.columns))
    # data_raw.columns = data_raw.columns.str.strip()
    # for feature in list(data_raw.columns):

    #     df_known = data_raw[data_raw[feature].notna()]
    #     df_missing = data_raw[data_raw[feature].isna()]
    #     print(feature, "not missing: ", len(df_known))
    #     print(feature, "missing: ", len(df_missing))
    # print("Unique encounters:", len(set(list(data_raw["encounter_id"]))))
    # print("Unique patients:", len(set(list(data_raw["patient_nbr"]))))
    # duplicate_patients = data_raw['patient_nbr'][data_raw['patient_nbr'].duplicated(keep=False)]

    # # Extract all rows with those patient_nbr values
    # duplicate_records = data_raw[data_raw['patient_nbr'].isin(duplicate_patients)]
    # data_sorted = duplicate_records.sort_values(by=['patient_nbr', 'encounter_id', 'number_diagnoses'] if 'encounter_id' in data_raw.columns else ['patient_nbr'])

    # print(data_sorted[['race', 'weight', 'patient_nbr', 'number_diagnoses', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult']].head(50))
    
    data = data_raw.dropna()
    


    if len(data) < 0.8 * len(data_raw):
        # data, ordinal_features = encode_datasets(data_raw, name)
        data = deal_data_with_na(data_raw)
        # numeric_cols.extend(ordinal_features)
    if log:
        print(f"({name}) Dropped rows with NA: {len(data_raw) - len(data)}")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    ordinal_features = []
    if needs_encoding:
        data, ordinal_features = encode_datasets(data, name)
    numeric_cols.extend(ordinal_features)

    if needs_normalisation:
        data = normalise(data, numeric_cols)

    cols = list(data.columns)
    if log:
        print(data.head(n=10))
        print("Preprocessed columns: ", cols)

    return data, cols






preprocess("Diabetes")