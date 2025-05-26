# evaluate.py

import shap
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, confusion_matrix
from train.models import get_models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def evaluate_metrics(X_train, y_train, X_test, y_test, task_name="Unknown"):
    """
    Trains and evaluates each model on classification metrics.

    Returns:
    - results: pd.DataFrame with Accuracy, F1-score, and ROC AUC per model
    """
    results = []
    trained_models = {}
    models = get_models()

    for name, model in models.items():
        print(f"Training: {name} on {task_name}")
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro')
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = None

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "F1-score": f1,
            "ROC AUC": auc
        })

    return pd.DataFrame(results), trained_models


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


def aggregate_shap_by_group(shap_values, feature_names):
    """
    Aggregates SHAP values for features belonging to the same group.
    Returns a sorted Series of mean absolute SHAP values per group.
    """
    group_map = {}
    for i, name in enumerate(feature_names):
        match = re.match(r"group_\d+", name)
        group = match.group(0) if match else name
        group_map.setdefault(group, []).append(i)

    mean_shap = np.abs(shap_values).mean(axis=0)

    group_scores = {
        group: mean_shap[indices].sum() for group, indices in group_map.items()
    }

    return pd.Series(group_scores).sort_values(ascending=False)


def get_shap_explainer(model, X_train):
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    if isinstance(model, (XGBClassifier, RandomForestClassifier)):
        return shap.TreeExplainer(model)
    elif isinstance(model, MLPClassifier):
        return shap.KernelExplainer(model.predict, X_train)
    elif isinstance(model, (LogisticRegression, SVC)):
        return shap.KernelExplainer(model.predict_proba, X_train)
    else:
        return shap.Explainer(model, X_train)


def evaluate_shap(X_train, X_test, models, task_name="Unknown", max_display=10):
    """
    Computes and plots aggregated SHAP values per group across models.
    """
    for name, model in models.items():
        print(f"\nSHAP Analysis: {name} on {task_name}")
        explainer = get_shap_explainer(model, X_train)

        # Compute SHAP values
        try:
            shap_values = explainer(X_test)
        except:
            shap_values = explainer.shap_values(X_test)

        # Extract base SHAP array and feature names
        if hasattr(shap_values, "values"):
            sv_array = shap_values.values
        else:
            sv_array = shap_values

        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns.tolist()
        else:
            raise ValueError("X_test must be a DataFrame with named columns for SHAP group aggregation.")

        # Aggregate SHAP values by group
        grouped_shap = aggregate_shap_by_group(sv_array, feature_names)

        # Plot top group importances
        grouped_shap.head(max_display).plot(
            kind='bar',
            title=f"{name} - SHAP Summary by Feature Group",
            color='slateblue'
        )
        plt.ylabel("Mean |SHAP value|")
        plt.xlabel("Feature Group")
        plt.tight_layout()
        plt.show()


def run_kfold_cv(X, y, model, k=5, repeats=10, seed=42):
    accs, f1s, aucs, precs = [], [], [], []

    for r in range(repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + r)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            y_proba = model_clone.predict_proba(X_test)[:, 1] if hasattr(model_clone, "predict_proba") else y_pred

            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average="macro"))
            precs.append(precision_score(y_test, y_pred, average="macro"))
            try:
                aucs.append(roc_auc_score(y_test, y_proba))
            except:
                aucs.append(None)
            # fairness = fairness_metrics(y_test, y_pred, sensitive=X_test["sex"], privileged_value=1)
            # print("fairness:", fairness)
    return {
        "accuracy": np.array(accs),
        "f1_score": np.array(f1s),
        "precision": np.array(precs),
        "roc_auc": np.array(aucs)
    }

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