import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from preprocess.preprocess_main import preprocess
from train.feature_group import bicriterion_anticlustering, group_dissimilar, embed_feature_groups, random_grouping, k_plus_anticlustering
from train.models import get_models, bayes_tune_models
from evaluate.evaluate import evaluate_metrics
from evaluate.SHAP import evaluate_shap_features
from evaluate.fairness import fairness_metrics
from utils import save_model, save_plot, save_metrics, save_dataframe
import warnings
from train.feature_group import evaluate_k_range, plot_k_selection

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
    n_iter: int = 30,
    run_id: int | None = None,
    train_data: pd.DataFrame | None = None,
    test_data: pd.DataFrame | None = None
):
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    print("=== Running Pipeline ===")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Grouping: {grouping_method}")
    print(f"SHAP Mode: {shap_mode}")

    models = get_models()
    curr_config_name = f"{dataset_name}_{model_name}_{grouping_method}"

    if model_name in ["LogisticRegression", "SVM", "MLP"]:
        needs_normalisation, needs_encoding, needs_imputation = True, True, True
    elif model_name == "RandomForest":
        needs_normalisation, needs_encoding, needs_imputation = False, True, True
    elif model_name == "XGBoost":
        needs_normalisation, needs_encoding, needs_imputation = False, True, False
    elif model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Check get_models().")

    # Run preprocessing if not already provided
    if train_data is None or test_data is None:
        train_data, test_data = preprocess(
            name=dataset_name,
            needs_normalisation=needs_normalisation,
            needs_encoding=needs_encoding,
            needs_imputation=needs_imputation
        )

    sensitive_attr = get_sensitive_attr(dataset_name)
    sensitive_test = test_data[sensitive_attr] if sensitive_attr else None

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

    if grouping_method == "ungrouped":
        print("\n[1] Running Ungrouped Baseline")

        bayes_result = bayes_tune_models(
            X_train, y_train,
            model_name,
            cv_splits=cv_splits,
            n_iter=n_iter
        )
        tuned_model = bayes_result.best_estimator_
        tuned_model.fit(X_train_val, y_train_val)

        y_test_pred = tuned_model.predict(X_test)
        y_test_proba = tuned_model.predict_proba(X_test) if hasattr(tuned_model, "predict_proba") else None

        perf, _ = evaluate_metrics(X_train_val, y_train_val, X_test, y_test, model=tuned_model,
                                   task_name="Ungrouped", run_id=run_id, config_name=curr_config_name)

        shap_ungrouped = evaluate_shap_features(
            X_train_val, X_test,
            {"Ungrouped": tuned_model},
            encoders={}, feature_mappings={}, mode=shap_mode,
            run_id=run_id, config_name=curr_config_name
        )

        fair_ungrouped = {}
        if sensitive_test is not None:
            fair_ungrouped = fairness_metrics(
                y_true=y_test,
                y_pred=y_test_pred,
                sensitive=sensitive_test,
                privileged_value=2 if dataset_name == "Diabetes" else 1,
                shap_values=shap_ungrouped["Ungrouped"]["raw"],
                y_pred_proba=y_test_proba,
                protected_attr=sensitive_attr,
                feature_names=X_test.columns.tolist(),
                dataset_name=dataset_name,
                run_id=run_id,
                config_name=curr_config_name
            )

        save_model(tuned_model, f"artifacts/models/{dataset_name}_{model_name}_ungrouped.joblib", run_id)
        return {"accuracy": accuracy_score(y_test, y_test_pred), "f1_score": f1_score(y_test, y_test_pred, average='macro'), **fair_ungrouped}

    elif grouping_method in ["bicriterion", "group_dissimilar", "kplus", "random"]:
        print("\n[1] Running Grouped Flow")

        if grouping_method == "bicriterion":
            groups = bicriterion_anticlustering(train_data, k=4)
        elif grouping_method == "group_dissimilar":
            groups = group_dissimilar(train_data, num_groups=4)
        elif grouping_method == "kplus":
            groups = k_plus_anticlustering(train_data, k=4)
        else:
            groups = random_grouping(train_data, k=4)

        X_train_val_emb, encoders, feature_mappings = embed_feature_groups(
            X_train_val, groups, encoding_dim=5, fit=True
        )
        X_test_emb = embed_feature_groups(
            test_data, groups, encoding_dim=5, encoders=encoders, fit=False
        )

        bayes_result = bayes_tune_models(
            X_train_val_emb, y_train_val,
            model_name,
            cv_splits=cv_splits,
            n_iter=n_iter
        )
        tuned_model = bayes_result.best_estimator_
        tuned_model.fit(X_train_val_emb, y_train_val)

        y_test_pred = tuned_model.predict(X_test_emb)
        y_test_proba = tuned_model.predict_proba(X_test_emb) if hasattr(tuned_model, "predict_proba") else None

        perf, _ = evaluate_metrics(X_train_val_emb, y_train_val, X_test_emb, y_test,
                                   model=tuned_model, task_name="Grouped", run_id=run_id, config_name=curr_config_name)

        shap_grouped = evaluate_shap_features(
            X_train_val, X_test,
            {"Grouped": tuned_model},
            encoders, feature_mappings,
            max_display=10, mode=shap_mode,
            run_id=run_id, config_name=curr_config_name
        )

        fair_grouped = {}
        if sensitive_test is not None:
            fair_grouped = fairness_metrics(
                y_true=y_test,
                y_pred=y_test_pred,
                sensitive=sensitive_test,
                privileged_value=2 if dataset_name == "Diabetes" else 1,
                shap_values=shap_grouped["Grouped"]["raw"],
                y_pred_proba=y_test_proba,
                protected_attr=sensitive_attr,
                feature_names=X_test.columns.tolist(),
                dataset_name=dataset_name,
                run_id=run_id, config_name=curr_config_name
            )

        save_model(tuned_model, f"artifacts/models/{dataset_name}_{model_name}_{grouping_method}.joblib", run_id)
        return {"accuracy": accuracy_score(y_test, y_test_pred), "f1_score": f1_score(y_test, y_test_pred, average='macro'), **fair_grouped}

    else:
        raise ValueError(f"Unknown grouping method '{grouping_method}'!")


def run_repeated_pipeline(
    model_name: str,
    dataset_name: str,
    grouping_method: str,
    shap_mode: str,
    num_repeats: int = 10,
    save_dir: str = "artifacts"
):
    print(f"\n=== Preprocessing once for all {num_repeats} runs ===")
    if model_name in ["LogisticRegression", "SVM", "MLP"]:
        needs_normalisation, needs_encoding, needs_imputation = True, True, True
    elif model_name == "RandomForest":
        needs_normalisation, needs_encoding, needs_imputation = False, True, True
    elif model_name == "XGBoost":
        needs_normalisation, needs_encoding, needs_imputation = False, True, False
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    train_data, test_data = preprocess(
        name=dataset_name,
        needs_normalisation=needs_normalisation,
        needs_encoding=needs_encoding,
        needs_imputation=needs_imputation
    )

    all_metrics = []

    for i in range(num_repeats):
        print(f"\nRepeat {i + 1}/{num_repeats} for {dataset_name} - {model_name} - {grouping_method}")
        # try:
        metrics = run_pipeline(
                model_name=model_name,
                dataset_name=dataset_name,
                grouping_method=grouping_method,
                shap_mode=shap_mode,
                run_id=i+2,
                train_data=train_data.copy(),
                test_data=test_data.copy()
            )
        all_metrics.append(metrics)
        # except Exception as e:
        #     print(f"[Warning] Failed on repeat {i+1}: {e}")

    if not all_metrics:
        print("No successful runs.")
        return

    df = pd.DataFrame(all_metrics)
    summary = {
        "mean": df.mean(numeric_only=True).to_dict(),
        "std": df.std(numeric_only=True).to_dict(),
        "num_runs": len(df)
    }

    base = f"{dataset_name}_{model_name}_{grouping_method}"
    save_dataframe(df, f"{save_dir}/metrics/{base}_raw.csv")
    save_metrics(summary, f"{save_dir}/metrics/{base}_summary.json")

    print("\nSummary Results:")
    for k, v in summary["mean"].items():
        print(f" - {k}: {v:.4f} ± {summary['std'].get(k, 0):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LogisticRegression")
    parser.add_argument("--dataset", type=str, default="Obesity")
    parser.add_argument("--grouping_method", type=str, default="bicriterion")
    parser.add_argument("--SHAP", type=str, default="each_label")

    args = parser.parse_args()

    train_data, _ = preprocess(
            name=args.dataset,
            needs_normalisation=True,
            needs_encoding=True,
            needs_imputation=True
        )

    # results = evaluate_k_range(train_data, method_name=args.grouping_method, k_range=range(2, 11))
    # fig = plot_k_selection(results, method_name=args.grouping_method)
    # save_plot(fig, f"artifacts/plots/diversity_dispersion_{args.dataset}_{args.grouping_method}.png")
    
    run_repeated_pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        grouping_method=args.grouping_method,
        shap_mode=args.SHAP,
        num_repeats=4
    )
    
