# evaluate.py

import shap
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from train.models import get_models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def evaluate_metrics(X_train, y_train, X_test, y_test, task_name="Unknown"):
    """
    Trains and evaluates each model on classification metrics.

    Returns:
    - results: pd.DataFrame with Accuracy, F1-score, and ROC AUC per model
    """
    results = []
    models = get_models()

    for name, model in models.items():
        print(f"Training: {name} on {task_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = None

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-score": f1,
            "ROC AUC": auc
        })

    return pd.DataFrame(results)


def get_shap_explainer(model, X_train):
    """
    Selects the appropriate SHAP explainer based on model type.
    """
    if isinstance(model, (XGBClassifier, RandomForestClassifier)):
        return shap.TreeExplainer(model)
    elif isinstance(model, MLPClassifier):
        return shap.DeepExplainer(model, X_train)
    elif isinstance(model, (LogisticRegression, SVC)):
        return shap.KernelExplainer(model.predict_proba, X_train)
    else:
        return shap.Explainer(model, X_train)  # fallback


def evaluate_shap(X_train, X_test, models, task_name="Unknown"):
    """
    Computes and displays SHAP values and plots for each model.
    """
    for name, model in models.items():
        print(f"\nSHAP Analysis: {name} on {task_name}")
        explainer = get_shap_explainer(model, X_train)
        try:
            shap_values = explainer(X_test)
        except:
            shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        shap.plots.bar(shap_values, max_display=10)