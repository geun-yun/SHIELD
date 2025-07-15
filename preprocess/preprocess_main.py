"""
Preprocessing pipeline for multiple UCI ML datasets using UCIMLRepo.
Includes encoding, normalization, and handling missing values.
"""

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from typing import Tuple, List
from preprocess.encode import encode_datasets
from preprocess.handle_missing_values import deal_data_with_na
from preprocess.normalise import normalise


def preprocess(
    name: str, 
    needs_normalisation: bool, 
    needs_encoding: bool, 
    needs_imputation: bool, 
    log: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs dataset-specific preprocessing including encoding, normalization, and handling missing values.

    Parameters:
    - name (str): Dataset name.
    - needs_normalisation (bool): Whether to apply normalization.
    - needs_encoding (bool): Whether to apply encoding.
    - needs_imputation (bool): Whether to handle missing values.
    - log (bool): If True, prints debugging/logging information.

    Returns:
    - Tuple of training and testing DataFrames.
    """
    # Dataset name to UCIMLRepo ID mapping
    name_to_id = {
        "Breast_cancer": 17,
        "Heart_disease": 45,
        "Lung_cancer": 62,
        "Diabetes": 296,
        "Obesity": 544,
        "Alzheimer": 732,
        "Crime": 183
    }

    # Load dataset
    data_raw = fetch_ucirepo(id=name_to_id.get(name))
    X, y = data_raw.data.features, data_raw.data.targets

    if log:
        print(X.head()) 
        print(f"({name}) Variables:")
        print(data_raw.variables)
    
    # Combine features and target into one DataFrame
    data_raw = pd.concat([X, y], axis=1)

    numeric_cols = data_raw.select_dtypes(include=[np.number]).columns.tolist()
    if name == "Heart_disease":
        categorical = ["sex", "cp", "fbs", "restecg", "exang",  "slope" , "ca", "thal", "num"]
        for category in categorical:
            numeric_cols.remove(category)

    filled_data = data_raw
    if needs_encoding:
        # Special handling for Diabetes dataset (due to more missing values)
        if name == "Diabetes":
            filled_data = deal_data_with_na(filled_data)
        filled_data = filled_data.dropna()
        print("filled", filled_data)
        filled_data, ordinal_features = encode_datasets(filled_data, name)
        numeric_cols.extend(ordinal_features)

    if needs_normalisation:
        if y.columns[0] in numeric_cols:
            numeric_cols.remove(y.columns[0])
        filled_data = normalise(filled_data, numeric_cols)
    
    # Determine test fraction based on usable rows
    testing_frac = 0.2 * len(data_raw) / len(filled_data)

    if log:
        print(f"Usable test set percentage: {100 * len(filled_data) / len(data_raw):.2f}%")
        print(f"Testing fraction: {testing_frac}")

    if testing_frac > 1:
        raise ValueError("Clean data is too small to make up the test data.")
    
    # Create test set from clean data and drop from original
    test_data = filled_data.sample(frac=testing_frac, random_state=42)
    train_data = data_raw.drop(test_data.index)

    if needs_imputation and name != "Diabetes":
        train_data = train_data.dropna()
        print(train_data.isnull().any())
        assert not train_data.isnull().any().any(), "NaNs detected in train_data even after imputation"
    elif needs_imputation:
        train_data = deal_data_with_na(train_data)
        assert not train_data.isnull().any().any(), "NaNs detected in train_data even after imputation"

    if needs_encoding:
        train_data, _ = encode_datasets(train_data, name)


    if needs_normalisation:
        if y.columns[0] in numeric_cols:
            numeric_cols.remove(y.columns[0])
        train_data = normalise(train_data, numeric_cols)

    assert set(train_data.index).isdisjoint(test_data.index), "Overlap detected between train and test sets!"

    # Reset indices for cleanliness
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Ensure target column is the last column
    target_col = y.columns[0]
    print(target_col)
    train_cols = [col for col in train_data.columns if col != target_col] + [target_col]
    test_cols = [col for col in test_data.columns if col != target_col] + [target_col]
    train_data = train_data[train_cols]
    test_data = test_data[test_cols]

    if log:
        print(f"Training data proportion: {100 * len(train_data) / len(data_raw):.2f}%")
        print(f"Training data n={len(train_data)}:")
        print(train_data.head(n=10))
        print(f"Testing data proportion: {100 * len(test_data) / len(data_raw):.2f}%")
        print(f"Testing data n={len(test_data)}:")
        print(test_data.head(n=10))
        print("Preprocessed columns: ", list(train_data.columns))

    return train_data, test_data