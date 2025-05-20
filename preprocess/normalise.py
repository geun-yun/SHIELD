import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler

def normalise(data: pd.DataFrame, cols_to_normalize: List[str]) -> pd.DataFrame:
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