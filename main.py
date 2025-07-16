import argparse
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from preprocess.preprocess_main import preprocess
from train.feature_group import bicriterion_anticlustering, group_dissimilar, embed_feature_groups
from train.models import get_models, bayes_tune_models
from evaluate.evaluate import evaluate_metrics
from evaluate.SHAP import evaluate_shap_features
from evaluate.fairness import fairness_metrics


def get_sensitive_attr(dataset_name: str) -> str | None:
    mapping = {
        "Obesity": "Gender_Male",
        "Diabetes": "race",
        "Heart_disease": "sex",
        "Breast_cancer": None,
        "Alzheimer": None
    }
    return mapping.get(dataset_name, None)


def run_pipeline(
    model_name: str,
    dataset_name: str,
    grouping_method: str,
    shap_mode: str,
    val_fraction: float = 0.15,
    cv_splits: int = 5,
    n_iter: int = 30
):
    print("=== Running Pipeline ===")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Grouping: {grouping_method}")
    print(f"SHAP Mode: {shap_mode}")

    models = get_models()
    
    if model_name in ["LogisticRegression", "SVM", "MLP"]:
        needs_normalisation, needs_encoding, needs_imputation = True, True, True
    elif model_name == "RandomForest":
        needs_normalisation, needs_encoding, needs_imputation = False, True, True
    elif model_name == "XGBoost":
        needs_normalisation, needs_encoding, needs_imputation = False, False, False
    elif model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Check get_models().")

    train_data, test_data = preprocess(
        name=dataset_name,
        needs_normalisation=needs_normalisation,
        needs_encoding=needs_encoding,
        needs_imputation=needs_imputation
    )

    sensitive_attr = get_sensitive_attr(dataset_name)
    if sensitive_attr and sensitive_attr not in test_data.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attr}' not found in test data!")
    sensitive_test = test_data[sensitive_attr] if sensitive_attr else None

    # Split
    target_col = train_data.columns[-1]
    train_sub, val_data = train_test_split(
        train_data,
        test_size=val_fraction,
        stratify=train_data[target_col],
        random_state=42
    )

    X_train = train_sub.iloc[:, :-1]
    y_train = train_sub.iloc[:, -1]
    X_val   = val_data.iloc[:, :-1]
    y_val   = val_data.iloc[:, -1]
    X_test  = test_data.iloc[:, :-1]
    y_test  = test_data.iloc[:, -1]

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    # Branch by grouping method
    if grouping_method == "ungrouped":
        print("\n[1] Running Ungrouped Baseline")

        # Tune on raw features
        bayes_result = bayes_tune_models(
            X_train, y_train,
            model_name,
            cv_splits=cv_splits,
            n_iter=n_iter
        )
        tuned_model = bayes_result.best_estimator_

        # Retrain on Train+Val
        tuned_model.fit(X_train_val, y_train_val)

        y_test_pred = tuned_model.predict(X_test)

        if hasattr(tuned_model, "predict_proba"):
            y_test_proba = tuned_model.predict_proba(X_test)
        else:
            y_test_proba = np.zeros((len(y_test_pred), 2))
            y_test_proba[np.arange(len(y_test_pred)), y_test_pred] = 1.0

        print("\n[Test Performance (Ungrouped)]")
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print(f"F1: {f1_score(y_test, y_test_pred, average='macro'):.4f}")

        shap_ungrouped = evaluate_shap_features(
            X_train_val, X_test,
            {"Ungrouped": tuned_model},
            encoders={}, feature_mappings={}, mode=shap_mode
        )

        if sensitive_test is not None:
            fair_ungrouped = fairness_metrics(
                y_true=y_test,
                y_pred=y_test_pred,
                sensitive=sensitive_test,
                privileged_value=2 if dataset_name=="Diabetes" else 1,
                shap_values=shap_ungrouped["Ungrouped"]["raw"],
                y_pred_proba=y_test_proba,
                protected_attr=sensitive_attr,
                feature_names=X_test.columns.tolist(),
                dataset_name=dataset_name
            )
            print("Fairness (Ungrouped):", fair_ungrouped)

    elif grouping_method in ["bicriterion", "group_dissimilar"]:
        print("\n[1] Running Grouped Flow")

        if grouping_method == "bicriterion":
            groups = bicriterion_anticlustering(train_data, k=4)
        else:
            groups = group_dissimilar(train_data, num_groups=4)

        X_train_val_emb, encoders, feature_mappings = embed_feature_groups(
            X_train_val, groups, encoding_dim=5, fit=True
        )
        X_test_emb = embed_feature_groups(
            test_data, groups, encoding_dim=5, encoders=encoders, fit=False
        )

        # Split embedded Train for tuning
        X_train_emb = X_train_val_emb.iloc[:len(X_train)]
        y_train_emb = y_train_val.iloc[:len(X_train)]

        # Tune on embedded features
        bayes_result = bayes_tune_models(
            X_train_emb, y_train_emb,
            model_name,
            cv_splits=cv_splits,
            n_iter=n_iter
        )
        tuned_model = bayes_result.best_estimator_

        # Retrain on full Train+Val embedded
        tuned_model.fit(X_train_val_emb, y_train_val)

        y_test_pred = tuned_model.predict(X_test_emb)

        if hasattr(tuned_model, "predict_proba"):
            y_test_proba = tuned_model.predict_proba(X_test_emb)
        else:
            y_test_proba = np.zeros((len(y_test_pred), 2))
            y_test_proba[np.arange(len(y_test_pred)), y_test_pred] = 1.0

        perf_grouped, _ = evaluate_metrics(X_train_val_emb, y_train_val, X_test_emb, y_test, model=tuned_model, task_name="Grouped")
        print(perf_grouped)

        shap_grouped = evaluate_shap_features(
            X_train_val, X_test,
            {"Grouped": tuned_model},
            encoders, feature_mappings,
            max_display=10, mode=shap_mode
        )

        if sensitive_test is not None:
            fair_grouped = fairness_metrics(
                y_true=y_test,
                y_pred=y_test_pred,
                sensitive=sensitive_test,
                privileged_value=2 if dataset_name=="Diabetes" else 1,
                shap_values=shap_grouped["Grouped"]["raw"],
                y_pred_proba=y_test_proba,
                protected_attr=sensitive_attr,
                feature_names=X_test.columns.tolist(),
                dataset_name=dataset_name
            )
            print("Fairness (Grouped):", fair_grouped)

    else:
        raise ValueError(f"Unknown grouping method '{grouping_method}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LogisticRegression", help="Model name: LogisticRegression, SVM, etc.")
    parser.add_argument("--dataset", type=str, default="Obesity", help="Dataset name: Obesity, Heart_disease, etc.")
    parser.add_argument("--grouping_method", type=str, default="ungrouped", help="Grouping: ungrouped, bicriterion, group_dissimilar")
    parser.add_argument("--SHAP", type=str, default="each_label", help="SHAP mode: each_label or aggregated")

    args = parser.parse_args()

    run_pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        grouping_method=args.grouping_method,
        shap_mode=args.SHAP
    )
