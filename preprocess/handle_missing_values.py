import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import time

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


def deal_data_with_na(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing data by dropping non-useful columns, imputing key nominal features,
    and then dropping remaining rows with any other missing values.

    Parameters:
    - data (pd.DataFrame)

    Returns:
    - data (pd.DataFrame)
    """
    # Drop problematic or irrelevant columns
    data = data.drop(columns="weight")

    # Impute key nominal features using classifier
    print("imputation started...")

    start = time.time()
    features_need_imputation = ["payer_code", "medical_specialty", "max_glu_serum", "A1Cresult"]
    for feature in features_need_imputation:
        data = impute_with_classifier(data, feature)
    end = time.time()
    print(f"Imputation time computed: {(end - start):.2f} seconds")
    print(len(data))

    # Drop any remaining rows with NaNs
    data = data.dropna()
    print(f"Diabetes after imputation has {len(data)} rows")

    return data