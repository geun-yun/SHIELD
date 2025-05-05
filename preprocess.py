"""
Preprocessing pipeline for multiple UCI ML datasets using UCIMLRepo.
Includes encoding, normalization, and handling missing values.
"""

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
from xgboost import XGBClassifier

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
    X = pd.concat([data_raw.data.ids[['encounter_id', 'patient_nbr']], X], axis=1)
    print(X.head())
    if log:
        print(f"({name}) Metadata:")
        print(data_raw.metadata)
        print(f"({name}) Variables:")
        print(data_raw.variables)

    data_raw = pd.concat([X, y], axis=1)
    print(list(data_raw.columns))
    data_raw.columns = data_raw.columns.str.strip()
    for feature in list(data_raw.columns):

        df_known = data_raw[data_raw[feature].notna()]
        df_missing = data_raw[data_raw[feature].isna()]
        print(feature, "not missing: ", len(df_known))
        print(feature, "missing: ", len(df_missing))
    print("Unique encounters:", len(set(list(data_raw["encounter_id"]))))
    print("Unique patients:", len(set(list(data_raw["patient_nbr"]))))
    duplicate_patients = data_raw['patient_nbr'][data_raw['patient_nbr'].duplicated(keep=False)]

    # Extract all rows with those patient_nbr values
    duplicate_records = data_raw[data_raw['patient_nbr'].isin(duplicate_patients)]
    data_sorted = duplicate_records.sort_values(by=['patient_nbr', 'encounter_id', 'number_diagnoses'] if 'encounter_id' in data_raw.columns else ['patient_nbr'])

    print(data_sorted[['race', 'weight', 'patient_nbr', 'number_diagnoses', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult']].head(50))
    
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

def impute_from_patient_history(data: pd.DataFrame, patient_id_col='patient_nbr') -> pd.DataFrame:
    """
    Imputes missing diagnosis and lab result features using other records of the same patient.

    Parameters:
    - data: DataFrame with patient records
    - patient_id_col: column name identifying patients (e.g., 'patient_nbr')

    Returns:
    - DataFrame with selected missing values filled
    """
    data_sorted = data.sort_values(by=[patient_id_col, 'encounter_id'] if 'encounter_id' in data.columns else [patient_id_col])
    data_filled = data_sorted.copy()

    # Demographics and Diagnoses: use mode per patient
    diag_cols = ['race']
    for col in diag_cols:
        def fill_mode(group):
            mode_val = group.mode(dropna=True)
            return group.fillna(mode_val.iloc[0]) if not mode_val.empty else group
        data_filled[col] = data_filled.groupby(patient_id_col)[col].transform(fill_mode)

    # Lab results: use forward and backward fill per patient
    # lab_cols = ['max_glu_serum', 'A1Cresult']
    # for col in lab_cols:
    #     data_filled[col] = data_filled.groupby(patient_id_col)[col].transform(lambda g: g.ffill().bfill())

    return data_filled


def impute_with_classifier(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Imputes missing values in a nominal feature using an XGBoost classifier
    trained on the other available features.

    Parameters:
    - data (pd.DataFrame): DataFrame with exactly one feature column containing NaNs.
    - feature (str): Name of the column to impute.

    Returns:
    - pd.DataFrame: The DataFrame with missing values in 'feature' imputed.
    """
    # Separate rows with and without missing values in the target feature
    df_known = data[data[feature].notna()]
    df_missing = data[data[feature].isna()]

    # Define input features (excluding the target column to be imputed)
    X_train = df_known.drop(columns=[feature])
    y_train = df_known[feature]
    X_test = df_missing.drop(columns=[feature])
    print(feature, "not missing: ", len(df_known))
    print(feature, "missing: ", len(df_missing))
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

    # Train the classifier
    model = XGBClassifier(enable_categorical=True)
    model.fit(X_train, y_train)

    # Predict and impute
    y_pred = model.predict(X_test)
    data.loc[data[feature].isna(), feature] = y_pred

    for col in list(X_train.columns):
        if data[col].dtype.name in 'category':
            data[col] = data[col].astype(str)
    data[feature] = data[feature].astype(str)

    return data

def deal_data_with_na(data):
    """
    Handles missing data by dropping non-useful columns, imputing key nominal features,
    and then dropping remaining rows with any other missing values.

    Parameters:
    - data (pd.DataFrame)

    Returns:
    - data (pd.DataFrame)
    """
    # data = impute_from_patient_history(data)
    # for feature in list(data.columns):
    #     df_known = data[data[feature].notna()]
    #     df_missing = data[data[feature].isna()]
    #     print(feature, "not missing: ", len(df_known))
    #     print(feature, "missing: ", len(df_missing))

    # Drop problematic or irrelevant columns
    data = data.drop(columns="weight")  # use errors='ignore' to avoid crash if column is missing
    print(len(data))
    # Impute key nominal features using classifier
    print("being imputed...")
    features_need_imputation = ["payer_code", "medical_specialty", "max_glu_serum", "A1Cresult"]
    for feature in features_need_imputation:
        data = impute_with_classifier(data, feature)
    print(len(data))

    # Drop any remaining rows with NaNs
    data = data.dropna()
    print(len(data))
    # nominal_features = ["payer_code", "medical_specialty"]
    # data = encode_all_nominal(nominal_features)

    return data


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
    nominal_features = ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id", "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", 
                        "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed", "readmitted!",
                        "payer_code", "medical_specialty"]

    age_order = [
        "[0-10)",
        "[10-20)",
        "[20-30)",
        "[30-40)",
        "[40-50)",
        "[50-60)",
        "[60-70)",
        "[70-80)",
        "[80-90)",
        "[90-100)"
    ]
    data = encode_all_nominal(data, nominal_features)
    data = encode_ordinal(data, "age", age_order)

    return data, "age"

# preprocess("Diabetes")