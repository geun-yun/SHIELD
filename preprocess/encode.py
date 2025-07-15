import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

def encode_nominal(data: pd.DataFrame, feature: str, is_target=False) -> pd.DataFrame:
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


def encode_ordinal(data: pd.DataFrame, feature: str, categories: List[str], is_target=False) -> pd.DataFrame:
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


def encode_all_nominal(data: pd.DataFrame, nominals: List[str]) -> pd.DataFrame:
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
def encode_datasets(data: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Dispatcher for dataset-specific preprocessing.

    Returns:
    - Tuple of (processed data, ordinal features)
    """
    if name == "Breast_cancer":
        return encode_Breast_cancer(data)
    elif name == "Heart_disease":
        return encode_Heart_disease(data)
    elif name == "Diabetes":
        return encode_Diabetes(data)
    elif name == "Obesity":
        return encode_Obesity(data)
    elif name == "Alzheimer":
        return encode_Alzheimer(data)


def encode_Breast_cancer(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    nominal_features = ["Diagnosis!"]
    data = encode_all_nominal(data, nominal_features)
    return data, []


def encode_Heart_disease(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    return data, []  # Already processed and numeric


def encode_Obesity(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    ordinal_features = ["CAEC", "CALC", "NObeyesdad"]
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

    for i in range(len(ordinal_features)):
        feature = ordinal_features[i]
        if i == 2:
            data = encode_ordinal(data, feature, obesity_order)
        else:
            data = encode_ordinal(data, feature, frequency_order)

    data = encode_all_nominal(data, nominal_features)
    return data, ordinal_features


def encode_Alzheimer(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Many features
    data["ID"] = data["ID"].str[3:].astype('int')
    data = encode_all_nominal(data, ["class!"])
    return data, []


def encode_Diabetes(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    nominal_features = ["race", "gender", "diag_1", "diag_2", "diag_3", 
                        "max_glu_serum", "A1Cresult", "metformin", "repaglinide", 
                        "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", 
                        "glipizide", "glyburide", "tolbutamide", "pioglitazone", 
                        "rosiglitazone", "acarbose", "miglitol", "troglitazone", 
                        "tolazamide", "examide", "citoglipton", "insulin", 
                        "glyburide-metformin", "glipizide-metformin", 
                        "glimepiride-pioglitazone", "metformin-rosiglitazone", 
                        "metformin-pioglitazone", "change", "diabetesMed", 
                        "readmitted!", "payer_code", "medical_specialty"]

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

    return data, ["age"]