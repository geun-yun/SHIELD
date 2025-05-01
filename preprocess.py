"""
Preprocessing pipeline for multiple UCI ML datasets using UCIMLRepo.
Includes encoding, normalization, and handling missing values.
"""

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler

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
    X, y = data_raw.data.features, data_raw.data.targets

    if log:
        print(f"({name}) Metadata:")
        print(data_raw.metadata)
        print(f"({name}) Variables:")
        print(data_raw.variables)

    data_raw = pd.concat([X, y], axis=1)
    data = data_raw.dropna()
    data.columns = data.columns.str.strip()
    if log:
        print(f"({name}) Dropped rows with NA: {len(data_raw) - len(data)}")

    if len(data) < 0.8 * len(data_raw):
        data = deal_data_with_na(data_raw)

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


def deal_data_with_na(data):
    """
    Fallback method to handle missing data beyond simple dropping.
    Placeholder for future imputation or hybrid strategy.

    Parameters:
    - data (pd.DataFrame)

    Returns:
    - data (pd.DataFrame)
    """
    return data  # TODO: Implement imputation logic

# === Encoding Utilities ===

def encode_nominal(data, feature, is_target=False):
    """
    Encodes nominal categorical variables using one-hot or label encoding.

    Parameters:
    - data (pd.DataFrame)
    - feature (str): Feature to encode.
    - is_target (bool): If True, label encode as a target variable.

    Returns:
    - data or pd.DataFrame with encoded feature(s)
    """
    if not is_target:
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(data[[feature]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([feature]), index=data.index)
        data = data.drop(columns=[feature])
        data = pd.concat([data, encoded_df], axis=1)
        return data
    else:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
        return data


def encode_ordinal(data, feature, categories, is_target=False):
    """
    Encodes ordinal categorical variables with known order.

    Parameters:
    - feature (str): Feature name
    - categories (List[str]): Ordered list of categories
    - is_target (bool): If True, use integer coding directly

    Returns:
    - data (pd.DataFrame)
    """
    if not is_target:
        encoder = OrdinalEncoder(categories=[categories])
        data[feature] = encoder.fit_transform(data[[feature]])
    else:
        target = pd.Categorical(data[feature], categories=categories, ordered=True)
        data[feature] = target.codes
    return data


def encode_all_nominal(data, nominals):
    """
    Encodes all nominal features listed, using label encoding for targets.

    Parameters:
    - nominals (List[str]): List of nominal feature names, 
      append "!" to encode as target.

    Returns:
    - data (pd.DataFrame)
    """
    for feature in nominals:
        clean_feature = feature.strip()
        if feature[-1] == "!":
            data = encode_nominal(data, clean_feature[:-1], True)
        else:
            data = encode_nominal(data, clean_feature)
    return data

# === Dataset-specific Encoding Logic ===

def encode_datasets(data, name):
    """
    Dispatcher for dataset-specific preprocessing.

    Returns:
    - Tuple of (processed data, ordinal features)
    """
    if name == "Breast_cancer":
        return process_Breast_cancer(data)
    elif name == "Heart_disease":
        return process_Heart_disease(data)
    elif name == "Lung_cancer":
        return process_Lung_cancer(data)
    elif name == "Diabetes":
        return process_Diabetes(data)
    elif name == "Obesity":
        return process_Obesity(data)
    elif name == "Alzheimer":
        return process_Alzheimer(data)

# === Normalization ===

def normalise(data, cols_to_normalize):
    """
    Applies standard scaling to specified columns.

    Parameters:
    - data (pd.DataFrame)
    - cols_to_normalize (List[str])

    Returns:
    - data (pd.DataFrame)
    """
    scaler = StandardScaler()
    data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])
    return data

# === Dataset-specific Handlers ===

def process_Breast_cancer(data):
    nominal_features = ["Diagnosis"]
    data = encode_all_nominal(data, nominal_features)
    return data, []


def process_Heart_disease(data):
    return data, []  # Already processed and numeric


def process_Lung_cancer(data):
    return data, []  # Already processed and numeric


def process_Obesity(data):
    ordinal_features = ["CAEC", "CALC", "NObeyesdad!"]
    nominal_features = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC", "MTRANS"]

    frequency_order = ["no", "Sometimes", "Frequently", "Always"]
    obesity_order = [
        "Insufficient_Weight", 
        "Normal_Weight", 
        "Overweight_Level_I", 
        "Overweight_Level_II", 
        "Obesity_Type_I", 
        "Obesity_Type_II", 
        "Obesity_Type_III"
    ]

    for feature in ordinal_features:
        if feature[-1] == "!":
            data = encode_ordinal(data, feature[:-1], obesity_order)
        else:
            data = encode_ordinal(data, feature, frequency_order)

    data = encode_all_nominal(data, nominal_features)
    return data, ordinal_features


def process_Alzheimer(data):
    # Many features
    raise NotImplementedError("Preprocessing for Alzheimer not yet implemented.")


def process_Diabetes(data):
    # Hard to process
    raise NotImplementedError("Preprocessing for Diabetes not yet implemented.")
