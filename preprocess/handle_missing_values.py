import pandas as pd
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import time

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

def impute_with_knn_imputer(data: pd.DataFrame, features) -> pd.DataFrame:
    pipeline = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),  # Encode categories first
        ('imputer', KNNImputer(n_neighbors=3))  # Apply KNN Imputer after encoding
    ])
    preprocessor = ColumnTransformer(transformers=[('cat', pipeline, features)])

    data_imputed = preprocessor.fit_transform(data)

    return data_imputed


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

    # # Train the classifier
    # model = XGBClassifier(enable_categorical=True)
    # model.fit(X_train, y_train)
    model = LabelPropagation()
    model.fit(X_train, y_train)

    # Predict and impute
    # y_pred = model.predict(X_test)
    y_pred = model.transduction_
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
    start = time.time()
    # Drop problematic or irrelevant columns
    data = data.drop(columns="weight")  # use errors='ignore' to avoid crash if column is missing
    print(len(data))
    print(data[["payer_code", "medical_specialty", "max_glu_serum", "A1Cresult"]].head(20))
    # Impute key nominal features using classifier
    print("being imputed...")
    
    features_need_imputation = ["payer_code", "medical_specialty", "max_glu_serum", "A1Cresult"]
    data = impute_with_knn_imputer(data, features_need_imputation)
    # data_tmp = data.loc[:, ~data.columns.isin(features_need_imputation)]
    # for feature in features_need_imputation:
    #     data_tmp = impute_with_classifier(data_tmp, feature)
    print(len(data))

    # Drop any remaining rows with NaNs
    data = data.dropna()
    print(len(data))
    # nominal_features = ["payer_code", "medical_specialty"]
    # data = encode_all_nominal(nominal_features)
    end_time = time.time()
    print(f"computation time: {end_time - start} seconds")
    return data
