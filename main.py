from preprocess.preprocess_main import preprocess_all
from train.feature_group import evaluate_k_range, plot_k_selection, group_dissimilar, create_group_autoencoders, bicriterion_anticlustering, k_plus_anticlustering, embed_feature_groups
from train.models import get_models
from evaluate import evaluate_metrics, evaluate_shap
from evaluate import run_kfold_cv, evaluate_shap
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def run_full_pipeline():
    models = get_models()
    for model in models.keys():
        datasets = list(preprocess_all(model))
        for train_data, test_data in datasets:
            train_and_test(model, train_data, test_data)

def train_and_test(model, train_data, test_data, num_groups=4, encoding_dim=1, test_size=0.2, random_state=42):
    """
    Complete pipeline that:
    1. Preprocesses the dataset
    2. Groups dissimilar features using CMI
    3. Reduces each group using autoencoders
    4. Splits into training/testing
    5. Evaluates models for performance and SHAP

    Parameters:
    - dataset_name: str, one of the predefined dataset names
    - num_groups: int, number of dissimilar feature groups
    - encoding_dim: int, number of dimensions per group's latent representation
    - test_size: float, test split ratio
    - random_state: int, seed for reproducibility
    """
    print(f"\n==== Running pipeline for: {model} ====")
    train_X = train_data.iloc[:, :-1]
    train_Y = train_data.iloc[:, -1]
    test_X = test_data.iloc[:, :-1]
    test_Y = test_data.iloc[:, -1]
    # Group features and encode
    assert not train_data.isnull().any().any(), "NaNs detected in feature data before KMeans!"
    # groups = group_dissimilar(train_data, num_groups=num_groups)
    groups2 = bicriterion_anticlustering(train_data, num_groups)
    # # dissimilarity_matrix, feature_names = compute_dissimilarity(train_data)
    # # groups3 = anticluster_features(dissimilarity_matrix, feature_names, num_groups)
    # groups3 = k_plus_anticlustering(train_X)

    # results_kplus = evaluate_k_range(train_data, method_name='k_plus')
    # plot_k_selection(results_kplus, method_name='k_plus')

    # results_gd = evaluate_k_range(train_data, method_name='group_dissimilar')
    # plot_k_selection(results_gd, method_name='group_dissimilar')

    # results_bi = evaluate_k_range(train_data, method_name='bicriterion')
    # plot_k_selection(results_bi, method_name='bicriterion')

    # print("Iterative grouping")
    # print(groups)
    # print("Bicriterion")
    # print(groups2)
    # print("K plus anticlustering")
    # print(groups3)
    encoded_train, encoders = embed_feature_groups(train_data, groups2, encoding_dim=encoding_dim, fit=True)
    encoded_test = embed_feature_groups(test_data, groups2, encoding_dim=encoding_dim, encoders=encoders, fit=False)

    encoded_train["target"] = train_Y.values
    encoded_test["target"] = test_Y.values

    # Separate again for clarity
    X_train = encoded_train.drop(columns=["target"])
    y_train = encoded_train["target"]
    X_test = encoded_test.drop(columns=["target"])
    y_test = encoded_test["target"]

    # Evaluate
    print("\n🔍 Evaluating Model Performance")
    results, trained_models  = evaluate_metrics(X_train, y_train, X_test, y_test, task_name=model)
    print(results)

    # Optional: SHAP evaluation
    print("\n🔍 Explaining with SHAP")
    evaluate_shap(X_train, X_test, trained_models, task_name=model)
    # Add the target column back
    # encoded_data["target"] = data.iloc[:, -1].values

    # # Split data
    # X = encoded_data.drop(columns=["target"])
    # y = encoded_data["target"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # # Evaluate performance
    # results = evaluate_metrics(X_train, y_train, X_test, y_test, task_name=dataset_name)
    # print(results)

    # # Evaluate SHAP
    # evaluate_shap(X_train, y_train, X_test, dataset_name)

def compare_grouping_methods(data, base_model, k=4, repeats=3):
    print("\n🚩 Starting fairness-aware evaluation via K-fold CV")

    y = data.iloc[:, -1]
    X_raw = data.iloc[:, :-1]

    # -- Original raw features --
    print("Evaluating on raw features...")
    raw_results = run_kfold_cv(X_raw, y, clone(base_model), k=k, repeats=repeats)

    # -- Grouped + Embedded features --
    print("Evaluating on grouped + embedded features...")
    groups = bicriterion_anticlustering(data, k=4)
    X_emb, _ = embed_feature_groups(data, groups, encoding_dim=5, fit=True)
    y_emb = data.iloc[:, -1]
    emb_results = run_kfold_cv(X_emb, y_emb, clone(base_model), k=k, repeats=repeats)

    # -- Compare
    print("\n📊 Averaged Results (Raw vs Grouped)")
    for metric in raw_results.keys():
        raw_vals = [v for v in raw_results[metric] if v is not None]
        emb_vals = [v for v in emb_results[metric] if v is not None]

        raw_mean = np.mean(raw_vals) if raw_vals else np.nan
        emb_mean = np.mean(emb_vals) if emb_vals else np.nan
        print(f"{metric:>10}: Raw = {raw_mean:.4f}, Grouped = {emb_mean:.4f}")

    return raw_results, emb_results

def run_comparison_pipeline():
    models = get_models()
    for model_name, model in models.items():
        datasets = list(preprocess_all(model_name))
        for dataset_id, (train_data, test_data) in enumerate(datasets):
            print(f"\n=== Dataset {dataset_id + 1} for model: {model_name} ===")
            full_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
            compare_grouping_methods(full_data, base_model=model)

run_comparison_pipeline()
# run_full_pipeline()