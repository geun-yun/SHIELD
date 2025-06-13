import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re

def fairness_metrics(y_true, y_pred, sensitive, privileged_value):
    """
    Compute fairness metrics for binary classification.
    
    Params:
    - y_true: array-like of shape (n_samples,)
    - y_pred: array-like of predicted labels
    - sensitive: array-like of sensitive attribute (e.g. Gender, Race)
    - privileged_value: the value considered as the "privileged" group

    Returns: dict of fairness metrics
    """
    metrics = {}

    # Get unprivileged mask
    privileged_mask = (sensitive == privileged_value)
    unprivileged_mask = ~privileged_mask

    # Positive prediction rates (Statistical Parity)
    p_priv = np.mean(y_pred[privileged_mask])
    p_unpriv = np.mean(y_pred[unprivileged_mask])
    metrics["Statistical Parity"] = abs(p_priv - p_unpriv)
    metrics["Disparate Impact"] = p_unpriv / p_priv if p_priv > 0 else np.nan

    # True labels and predictions for each group
    def rates(y_true, y_pred):
        labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        if cm.shape != (2, 2):
            # If only one class is present, fill missing values with zeros
            padded_cm = np.zeros((2, 2), dtype=int)
            for i, label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    padded_cm[label][pred_label] = cm[i][j]
            cm = padded_cm

        tn, fp, fn, tp = cm.ravel()
        return {
            "TPR": tp / (tp + fn + 1e-6),
            "FPR": fp / (fp + tn + 1e-6),
            "Precision": tp / (tp + fp + 1e-6)
        }
    print(y_true[privileged_mask])
    print(y_pred[privileged_mask])
    rates_priv = rates(y_true[privileged_mask], y_pred[privileged_mask])
    rates_unpriv = rates(y_true[unprivileged_mask], y_pred[unprivileged_mask])

    # Equal Opportunity: TPR difference
    metrics["Equal Opportunity (TPR diff)"] = abs(rates_priv["TPR"] - rates_unpriv["TPR"])

    # Equalized Odds: max diff in TPR and FPR
    metrics["Equalized Odds"] = max(
        abs(rates_priv["TPR"] - rates_unpriv["TPR"]),
        abs(rates_priv["FPR"] - rates_unpriv["FPR"])
    )

    # Predictive Parity (Precision difference)
    metrics["Predictive Parity"] = abs(rates_priv["Precision"] - rates_unpriv["Precision"])

    return metrics