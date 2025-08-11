import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from utils import save_metrics

def evaluate_metrics(X_train, y_train, X_test, y_test, model, task_name="Unknown", run_id=None, config_name=None):
    """
    Trains and evaluates the given model on classification metrics.
    """
    print(f"Training: {model.__class__.__name__} on {task_name}")

    model_clone = clone(model)
    model_clone.fit(X_train, y_train)
    y_pred = model_clone.predict(X_test)
    y_proba = model_clone.predict_proba(X_test)[:, 1] if hasattr(model_clone, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = None

    results = pd.DataFrame([{
        "Model": model.__class__.__name__,
        "Accuracy": acc,
        "Precision": prec,
        "F1-score": f1,
        "ROC AUC": auc
    }])

    fname = f"performance_metrics_{task_name}_{config_name}.json"
    save_metrics(results.to_dict(orient="records"), fname, run_id=run_id)

    return results, model_clone


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

    return {
        "accuracy": np.array(accs),
        "f1_score": np.array(f1s),
        "precision": np.array(precs),
        "roc_auc": np.array(aucs)
    }

